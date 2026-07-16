"""Mean-Teacher (EMA self-ensembling) nnU-Net trainer — NoDeepSupervision + NoMirroring.

The ONLINE semi-supervised arm of the SSL label-efficiency study, counterpart to the OFFLINE
self-training rung (Dataset121). Trains on Dataset122_ATM22MT: 20 real-GT cases + 90 all-IGNORE
cases. nnU-Net's DC+CE loss skips the 90's voxels automatically (ignore label); this trainer adds
an EMA-teacher consistency loss so the 90 (and the labelled patches too) contribute online.

Design (grounded in the literature, see PROJECT_STATE / the plan):
  - Init: warm-start from the @20 model via ``nnUNetv2_train ... -pretrained_weights <@20 ckpt>``.
    The EMA teacher is deep-copied from the student AFTER those weights load, so it is the strong,
    few-FP @20 model from step 1 — the fix for the from-scratch MONAI MT confirmation-bias null.
  - EMA:  θ'_t = α θ'_{t-1} + (1-α) θ_t, with UA-MT's step clamp α = min(1 - 1/(t+1), 0.99), which
    is the original Mean-Teacher "fast-early / slow-later" schedule (Tarvainen & Valpola 2017).
  - Consistency: plain MSE on softmax probabilities over the WHOLE batch (UA-MT, Yu et al. MICCAI
    2019). Whole-batch (not unlabelled-only) means nnU-Net's foreground oversampling on the 20
    labelled cases feeds airway-rich patches into the consistency term — the fix for the MONAI
    starved-signal null. A MONAI-style hard confidence mask is available as an ablation (off).
  - Ramp: sigmoid w = 0.1 * exp(-5 (1-x)^2), x = min(epoch/rampup, 1), rampup = 40 epochs.
  - Views: student = strong intensity perturbation, teacher = clean; both share nnU-Net's spatial
    augmentation upstream, so voxel correspondence (required for consistency) is exact.
  - Inference model = the EMA teacher: at train end the teacher weights are copied into the network
    so checkpoint_final holds the teacher (predict with the default checkpoint_final).

Deploy on HPC into the ctfm nnU-Net site-packages, alongside the cbDice trainers:
    cp nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py \
       "$NNUNET/training/nnUNetTrainer/variants/network_architecture/"
Verify discovery with recursive_find_python_class (see hpc-nnunet-cbdice-install).
"""

from __future__ import annotations

import math
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import autocast

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import dummy_context

# Resolve the NoDeepSupervision + NoMirroring base class (installed from the cbDice repo). Try the
# direct import first; fall back to nnU-Net's own discovery so we don't hard-depend on the filename.
try:  # pragma: no cover - exercised only inside the HPC nnU-Net env
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_NoDeepSupervision_NoMirroring as _Base,
    )
except Exception:  # pragma: no cover
    _Base = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        "nnUNetTrainer_NoDeepSupervision_NoMirroring",
        "nnunetv2.training.nnUNetTrainer",
    )
    if _Base is None:
        raise ImportError(
            "Could not locate base trainer 'nnUNetTrainer_NoDeepSupervision_NoMirroring'. "
            "Is the cbDice trainer install present? (see hpc-nnunet-cbdice-install)"
        )


class nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring(_Base):
    """EMA Mean-Teacher on top of the NoDeepSupervision + NoMirroring nnU-Net recipe."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- schedule: a short warm-start fine-tune, NOT the 1000-epoch from-scratch run ---
        self.num_epochs = 100
        self.initial_lr = 1e-3  # nnU-Net's default 1e-2 poly would blow up a converged @20

        # --- Mean-Teacher hyperparameters (UA-MT defaults) ---
        self.ema_decay = 0.99            # with the step clamp below == "fast-early / slow-later"
        self.consistency_max = 0.1       # w_max for MSE consistency
        self.consistency_rampup = 40.0   # epochs of the sigmoid ramp

        # --- student(strong) / teacher(weak) intensity perturbation on nnU-Net-normalised CT ---
        self.student_noise_std = 0.1
        self.student_scale = 0.1
        self.student_shift = 0.1
        self.teacher_noise_std = 0.0     # teacher sees the clean (weak) view

        # --- optional MONAI-style hard confidence mask (ablation; UA-MT uses none) ---
        self.use_confidence_mask = False
        self.fg_threshold = 0.8
        self.bg_threshold = 0.2

        # --- state ---
        self.teacher = None
        self._mt_step = 0
        self._log_cons = 0.0
        self._log_sup = 0.0
        self._log_w = 0.0
        self._log_n = 0

    # torch.compile complicates deepcopy + EMA + state_dict swapping; disable it for this arm.
    def _do_i_compile(self) -> bool:
        return False

    # ------------------------------------------------------------------ teacher / EMA -------
    def _build_teacher(self) -> None:
        """Deep-copy the (warm-started) student into a frozen EMA teacher."""
        self.teacher = deepcopy(self.network)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def _update_ema(self) -> None:
        alpha = min(1.0 - 1.0 / (self._mt_step + 1), self.ema_decay)
        student_params = dict(self.network.named_parameters())
        student_buffers = dict(self.network.named_buffers())
        with torch.no_grad():
            for name, tp in self.teacher.named_parameters():
                tp.mul_(alpha).add_(student_params[name].detach(), alpha=1.0 - alpha)
            for name, tb in self.teacher.named_buffers():  # BN stats: hard copy, not EMA
                tb.copy_(student_buffers[name])

    # ------------------------------------------------------------------ consistency --------
    def _consistency_weight(self) -> float:
        x = min(self.current_epoch / max(self.consistency_rampup, 1e-8), 1.0)
        return self.consistency_max * math.exp(-5.0 * (1.0 - x) ** 2)

    def _consistency(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student = torch.softmax(student_logits.float(), dim=1)
        teacher = torch.softmax(teacher_logits.float(), dim=1).detach()
        squared_error = (student - teacher).pow(2)
        if not self.use_confidence_mask:
            return squared_error.mean()
        # Ablation: restrict to voxels where the teacher's airway prob is confident.
        fg = teacher[:, 1:2]
        confident = (fg >= self.fg_threshold) | (fg <= self.bg_threshold)
        if not confident.any():
            return student.sum() * 0.0
        return squared_error.mean(dim=1, keepdim=True)[confident].mean()

    def _perturb(self, data: torch.Tensor, noise_std: float, scale_amp: float, shift_amp: float) -> torch.Tensor:
        out = data
        view = (data.shape[0],) + (1,) * (data.ndim - 1)
        if scale_amp > 0:
            out = out * (1.0 + (torch.rand(view, device=data.device) * 2 - 1) * scale_amp)
        if shift_amp > 0:
            out = out + (torch.rand(view, device=data.device) * 2 - 1) * shift_amp
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std
        return out

    # ------------------------------------------------------------------ hooks --------------
    def on_train_start(self) -> None:
        super().on_train_start()  # ensures initialize() ran; pretrained weights already loaded
        self._build_teacher()
        self.print_to_log_file(
            f"[MeanTeacher] teacher built from student. ema_decay={self.ema_decay} "
            f"consistency_max={self.consistency_max} rampup={self.consistency_rampup} "
            f"epochs={self.num_epochs} lr={self.initial_lr} "
            f"confidence_mask={self.use_confidence_mask}"
        )

    def on_train_end(self) -> None:
        # Deploy the EMA teacher as the inference model (UA-MT / MONAI convention): checkpoint_final
        # and the in-memory network (used by perform_actual_validation) become the teacher.
        if self.teacher is not None:
            self.network.load_state_dict(self.teacher.state_dict())
            self.print_to_log_file("[MeanTeacher] deployed EMA teacher weights as the final network.")
        super().on_train_end()

    def on_train_epoch_end(self, train_outputs) -> None:
        super().on_train_epoch_end(train_outputs)
        if self._log_n > 0:
            cons = self._log_cons / self._log_n
            sup = self._log_sup / self._log_n
            fraction = (self._log_w * cons) / max(abs(sup), 1e-8)
            self.print_to_log_file(
                f"[MeanTeacher] consistency={cons:.5f} weight={self._log_w:.4f} "
                f"sup={sup:.4f} consistency_fraction={fraction:.3f}"
            )
        self._log_cons = self._log_sup = 0.0
        self._log_n = 0

    # ------------------------------------------------------------------ train step ---------
    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        if self.teacher is None:  # safety net if on_train_start ordering ever changes
            self._build_teacher()

        student_in = self._perturb(data, self.student_noise_std, self.student_scale, self.student_shift)
        teacher_in = data if self.teacher_noise_std <= 0 else self._perturb(data, self.teacher_noise_std, 0.0, 0.0)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(student_in)
            supervised_loss = self.loss(output, target)  # ignore label masks the 90 automatically
            with torch.no_grad():
                self.teacher.eval()
                teacher_output = self.teacher(teacher_in)
            consistency = self._consistency(output, teacher_output)
            weight = self._consistency_weight()
            loss = supervised_loss + weight * consistency

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        self._mt_step += 1
        self._update_ema()

        self._log_cons += float(consistency.detach())
        self._log_sup += float(supervised_loss.detach())
        self._log_w = weight
        self._log_n += 1
        return {"loss": loss.detach().cpu().numpy()}
