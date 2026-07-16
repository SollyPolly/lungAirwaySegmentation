"""Mean-Teacher (EMA self-ensembling) nnU-Net trainer — NoDeepSupervision + NoMirroring.

The ONLINE semi-supervised arm of the SSL label-efficiency study, counterpart to the OFFLINE
self-training rung (Dataset121). Trains on Dataset122_ATM22MT: 20 real-GT cases + 90 all-IGNORE
cases. nnU-Net's DC+CE loss skips the 90's voxels automatically (ignore label); this trainer adds
an EMA-teacher consistency loss so the 90 (and the labelled patches too) contribute online.

PROTOCOL — matched to the @121 self-training rung so the arms are comparable:
  from-scratch, 1000 epochs, fold 0, NoDeepSupervision + NoMirroring, nnU-Net's default poly LR.
  NO warm-start: the EMA teacher co-evolves from a random-init student. This is (a) canonical Mean
  Teacher (UA-MT trains from scratch), and (b) the clean counterpart to self-training's FIXED @20
  teacher — so "@122 (MT) vs @121 (self-training)" isolates online-EMA vs offline-frozen targets,
  nothing else. (An @20-warm-started, short-schedule variant is possible but confounds the compare;
  keep it only as a fast pilot, not the reportable run.)

Design (grounded in the literature — UA-MT / Yu et al. MICCAI 2019, Tarvainen & Valpola 2017):
  - EMA:  θ'_t = α θ'_{t-1} + (1-α) θ_t, with UA-MT's step clamp α = min(1 - 1/(t+1), 0.99), which
    is the original Mean-Teacher "fast-early / slow-later" schedule.
  - Consistency: plain MSE on softmax probabilities over the WHOLE batch (UA-MT). Whole-batch (not
    unlabelled-only) means nnU-Net's foreground oversampling on the 20 labelled cases feeds
    airway-rich patches into the consistency term — the fix for the MONAI starved-signal null. A
    MONAI-style hard confidence mask is available as an ablation (off by default).
  - SUPERVISED WARM-UP: hold consistency at 0 until the supervised task plateaus (~epoch 300), THEN
    sigmoid-ramp w = 0.1 * exp(-5 (1-x)^2) over the next 200 epochs. From scratch the early EMA
    teacher is unreliable; engaging consistency only once the teacher is a good, high-precision
    model is what avoids the from-scratch confirmation-bias window that sank the original MONAI MT.
  - Views: student = strong intensity perturbation, teacher = clean; both share nnU-Net's spatial
    augmentation upstream, so voxel correspondence (required for consistency) is exact. During the
    warm-up the student trains on the clean patch (identical to a standard supervised run).
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
import torch.nn.functional as F  # noqa: F401  (kept for ablation experiments)
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


# ---------------------------------------------------------------------------------------------------
# Differentiable soft-skeleton / soft-clDice (Shit et al. 2021), self-contained so the trainer needs
# no import from the lung_airway_segmentation package inside the ctfm nnU-Net env. Mirrors
# lung_airway_segmentation/losses/topology.py (soft_erode/soft_open/soft_skeleton/soft_cldice_loss);
# used for the geometry-aware "cldice" consistency mode. Reference: https://github.com/jocpae/clDice
# ---------------------------------------------------------------------------------------------------
def _soft_erode3d(x: torch.Tensor) -> torch.Tensor:
    d = -F.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0))
    h = -F.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))
    w = -F.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1))
    return torch.minimum(torch.minimum(d, h), w)


def _soft_open3d(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(_soft_erode3d(x), 3, 1, 1)


def _soft_skeleton3d(x: torch.Tensor, iterations: int) -> torch.Tensor:
    opened = _soft_open3d(x)
    skeleton = F.relu(x - opened)
    for _ in range(int(iterations)):
        x = _soft_erode3d(x)
        opened = _soft_open3d(x)
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def _soft_cldice_consistency(student_fg: torch.Tensor, teacher_fg: torch.Tensor,
                             iterations: int, smooth: float = 1.0) -> torch.Tensor:
    """Soft-clDice between the student's airway prob and the teacher's (binarised) tree.

    student_fg / teacher_fg are (B, 1, D, H, W) in [0, 1]; the teacher is detached and thresholded at
    0.5 to a fixed target tree (an online, geometry-level pseudo-label). Gradient flows only through the
    student's soft skeleton / soft prob. Unlike voxel-MSE this normalises over the CENTRELINE, so the
    sparse airway is at the right scale WITHOUT class-balancing — which removes the ~100x gradient
    amplification that collapsed the balanced-MSE teacher.
    """
    target = (teacher_fg > 0.5).float()
    skel_pred = _soft_skeleton3d(student_fg, iterations)
    skel_true = _soft_skeleton3d(target, iterations)
    batch = student_fg.shape[0]
    skel_p = skel_pred.reshape(batch, -1)
    skel_t = skel_true.reshape(batch, -1)
    pred = student_fg.reshape(batch, -1)
    true = target.reshape(batch, -1)
    t_prec = (torch.sum(skel_p * true, dim=1) + smooth) / (torch.sum(skel_p, dim=1) + smooth)
    t_sens = (torch.sum(skel_t * pred, dim=1) + smooth) / (torch.sum(skel_t, dim=1) + smooth)
    cl_dice = 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens)
    return cl_dice.mean()


class nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring(_Base):
    """EMA Mean-Teacher on top of the NoDeepSupervision + NoMirroring nnU-Net recipe."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        # NOTE: keep these exact parameter names — nnUNetTrainer.__init__ reflects on
        # self.__init__'s signature and does `locals()[name]`, so a (*args, **kwargs)
        # signature raises KeyError: 'args'. Must mirror the base signature exactly.
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- schedule: from-scratch, protocol-matched to @121 (inherit num_epochs=1000, lr=1e-2) ---
        # No overrides here on purpose: same length + LR schedule as the self-training rung.

        # --- Mean-Teacher hyperparameters (UA-MT defaults) ---
        self.ema_decay = 0.99            # with the step clamp below == "fast-early / slow-later"
        self.ema_decay_late = 0.999      # after the consistency ramp completes, slow the teacher down
                                         # (Tarvainen & Valpola: alpha 0.99 during ramp-up -> 0.999 for
                                         # the rest). A slower teacher is a stabiliser once consistency
                                         # is doing real work: it resists being dragged into the MT
                                         # collapse the pilot exhibited at high weight.
        self.consistency_max = 0.1       # w_max. NOTE the right scale depends on consistency_mode:
                                         # the voxel-MSE modes use ~0.3 (and still COLLAPSED — plain MSE
                                         # was diluted to the null, class-balancing then amplified the
                                         # airway gradient ~100x and collapsed the teacher: 0.96->0.28).
                                         # The default "cldice" mode returns 1-clDice in [0,1] (~0.3-0.6
                                         # early), ~5x larger than balanced-MSE, so it wants a ~5x
                                         # SMALLER weight -> start 0.1 and calibrate from the pilot's
                                         # consistency_fraction (target a few %) + pseudo-Dice stability.

        # --- supervised warm-up then sigmoid ramp (tied to the ~epoch-300 supervised plateau) ---
        self.consistency_warmup_epochs = 300  # consistency held at 0 before this epoch
        self.consistency_rampup = 200.0       # then ramp to full weight over this many epochs

        # --- student(strong) / teacher(weak) intensity perturbation on nnU-Net-normalised CT ---
        self.student_noise_std = 0.1
        self.student_scale = 0.1
        self.student_shift = 0.1
        self.teacher_noise_std = 0.0     # teacher sees the clean (weak) view

        # --- consistency reduction mode ---
        #   "cldice" (DEFAULT) — GEOMETRY-AWARE consistency: soft-clDice between the student's airway
        #               prob and the teacher's binarised tree (an online centreline-level pseudo-label).
        #               Enforces agreement on the SKELETON/tree, not per-voxel probability. Motivated by
        #               the tubular-SSL literature (MICCAI 2024, airways): "data-level [voxel] consistency
        #               overlooks geometric shape". Normalises over the centreline, so no class-balancing
        #               and hence NONE of the ~100x gradient amplification that collapsed the MSE modes.
        #   "balanced_confident" — class-balanced MSE over teacher-CONFIDENT voxels only (UA-MT gating +
        #               CBMT balancing). Kept as a voxel-consistency comparison.
        #   "balanced"  — class-balanced MSE over ALL voxels. Escapes the null but COLLAPSED the teacher
        #               (0.96->0.28 at w=1.0; ->0.19 val at w=0.3); the ablation that shows why.
        #   "confident" — hard confidence mask, UNbalanced (bg-dominated); ablation.
        #   "plain"     — UA-MT whole-patch MSE (the null baseline); ablation.
        self.consistency_mode = "cldice"
        self.cldice_iters = 10           # soft-skeleton iterations; >= largest airway radius in voxels
        self.partition_threshold = 0.5   # teacher airway/bg split for "balanced"
        self.fg_conf_threshold = 0.8     # confident-airway cut for the *_confident modes
        self.bg_conf_threshold = 0.2     # confident-bg cut for the *_confident modes

        # --- state ---
        self.teacher = None
        self._mt_step = 0
        self._log_cons = 0.0
        self._log_sup = 0.0
        self._log_w = 0.0
        self._log_n = 0
        self._best_teacher_ema = None    # best EMA pseudo-Dice at which we snapshotted the teacher

    # torch.compile complicates deepcopy + EMA + state_dict swapping; disable it for this arm.
    def _do_i_compile(self) -> bool:
        return False

    # ------------------------------------------------------------------ teacher / EMA -------
    def _build_teacher(self) -> None:
        """Deep-copy the student into a frozen EMA teacher (random-init copy when from scratch)."""
        self.teacher = deepcopy(self.network)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def _ramp_end_epoch(self) -> float:
        return self.consistency_warmup_epochs + self.consistency_rampup

    def _update_ema(self) -> None:
        # 0.99 during warm-up + ramp, 0.999 once consistency is at full weight (Tarvainen & Valpola).
        cap = self.ema_decay_late if self.current_epoch >= self._ramp_end_epoch() else self.ema_decay
        alpha = min(1.0 - 1.0 / (self._mt_step + 1), cap)
        student_params = dict(self.network.named_parameters())
        student_buffers = dict(self.network.named_buffers())
        with torch.no_grad():
            for name, tp in self.teacher.named_parameters():
                tp.mul_(alpha).add_(student_params[name].detach(), alpha=1.0 - alpha)
            for name, tb in self.teacher.named_buffers():  # BN stats: hard copy, not EMA
                tb.copy_(student_buffers[name])

    # ------------------------------------------------------------------ consistency --------
    def _consistency_weight(self) -> float:
        epoch = self.current_epoch - self.consistency_warmup_epochs
        if epoch < 0:
            return 0.0
        x = min(epoch / max(self.consistency_rampup, 1e-8), 1.0)
        return self.consistency_max * math.exp(-5.0 * (1.0 - x) ** 2)

    def _consistency(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student = torch.softmax(student_logits.float(), dim=1)
        teacher = torch.softmax(teacher_logits.float(), dim=1).detach()

        if self.consistency_mode == "cldice":
            # Geometry-aware: soft-clDice on the airway channel (B,1,D,H,W), student vs teacher tree.
            return _soft_cldice_consistency(student[:, 1:2], teacher[:, 1:2], self.cldice_iters)

        student_fg = student[:, 1]                       # [B, *spatial] airway channel
        teacher_fg = teacher[:, 1]
        err = (student_fg - teacher_fg).pow(2)           # per-voxel MSE on the airway prob

        if self.consistency_mode == "plain":
            return err.mean()

        if self.consistency_mode == "confident":
            # Ablation: mask to teacher-confident voxels but do NOT class-balance (bg-dominated).
            confident = (teacher_fg >= self.fg_conf_threshold) | (teacher_fg <= self.bg_conf_threshold)
            if not confident.any():
                return err.sum() * 0.0
            return err[confident].mean()

        if self.consistency_mode == "balanced":
            # Class-balance over ALL voxels. Escapes the null but collapses on sparse airways (see note).
            fg_mask = teacher_fg >= self.partition_threshold
            bg_mask = ~fg_mask
            fg_err = err[fg_mask].mean() if fg_mask.any() else err.sum() * 0.0
            bg_err = err[bg_mask].mean() if bg_mask.any() else err.sum() * 0.0
            return 0.5 * (fg_err + bg_err)

        # "balanced_confident" (default): balance FG/BG but ONLY over teacher-CONFIDENT voxels; the
        # uncertain band (bg_conf < teacher_fg < fg_conf) is excluded. That band is the boundary/distal
        # region where the teacher is unreliable, and the ~100x-amplified FG gradient chasing it is what
        # collapsed "balanced" at w=1.0 and w=0.3. Gating (UA-MT) + balancing (CBMT) keeps the airway
        # signal without the positive-feedback loop.
        fg_mask = teacher_fg >= self.fg_conf_threshold
        bg_mask = teacher_fg <= self.bg_conf_threshold
        fg_err = err[fg_mask].mean() if fg_mask.any() else err.sum() * 0.0
        bg_err = err[bg_mask].mean() if bg_mask.any() else err.sum() * 0.0
        return 0.5 * (fg_err + bg_err)

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
        super().on_train_start()  # ensures initialize() ran (builds the random-init network)
        self._build_teacher()
        self.print_to_log_file(
            f"[MeanTeacher] teacher built from student. ema_decay={self.ema_decay} "
            f"consistency_max={self.consistency_max} warmup={self.consistency_warmup_epochs} "
            f"rampup={self.consistency_rampup} epochs={self.num_epochs} lr={self.initial_lr} "
            f"consistency_mode={self.consistency_mode} cldice_iters={self.cldice_iters} "
            f"fg_conf={self.fg_conf_threshold} bg_conf={self.bg_conf_threshold}"
        )

    def on_train_end(self) -> None:
        # Deploy the EMA teacher as the inference model (UA-MT / MONAI convention): checkpoint_final
        # and the in-memory network (used by perform_actual_validation) become the teacher.
        if self.teacher is not None:
            self.network.load_state_dict(self.teacher.state_dict())
            self.print_to_log_file("[MeanTeacher] deployed EMA teacher weights as the final network.")
        super().on_train_end()

    def on_epoch_end(self) -> None:
        super().on_epoch_end()  # logs metrics, saves checkpoint_best (STUDENT) + periodic, steps epoch
        # Best-TEACHER checkpoint. checkpoint_final holds the FINAL teacher (reportable protocol), but
        # if the teacher collapses late (the pilot's failure mode) that file is poisoned AND nnU-Net's
        # checkpoint_best is a STUDENT, not the teacher we infer with. So snapshot the teacher whenever
        # nnU-Net records a new best EMA pseudo-Dice: a late collapse stays fully recoverable to a good
        # teacher via `nnUNetv2_predict ... -chk checkpoint_best_teacher.pth`. (Uses the student's
        # smoothed pseudo-Dice as the proxy for teacher quality — teacher is its EMA, so they peak
        # together; when the student collapses no new best is recorded and the pre-collapse teacher is
        # retained.)
        if self.teacher is None:
            return
        # Piggyback on the base trainer's OWN running-best signal `self._best_ema` — the exact value it
        # uses to decide checkpoint_best and to print "Yayy! New best EMA pseudo Dice". This avoids the
        # logger internals, whose attribute varies across nnU-Net builds (this env's logger is a
        # `MetaLogger` with no `my_fantastic_logging`). When `_best_ema` advances this epoch, the student
        # (hence its EMA teacher) just peaked -> snapshot the teacher. getattr keeps it non-fatal if the
        # attribute is ever renamed (best-teacher just no-ops rather than crashing the run).
        best_ema = getattr(self, "_best_ema", None)
        if best_ema is None:
            return
        if self._best_teacher_ema is None or best_ema > self._best_teacher_ema:
            self._best_teacher_ema = best_ema
            self._save_teacher_checkpoint('checkpoint_best_teacher.pth')
            self.print_to_log_file(
                f"[MeanTeacher] new best teacher @ EMA pseudo Dice {best_ema:.4f} "
                f"-> checkpoint_best_teacher.pth"
            )

    def _save_teacher_checkpoint(self, filename: str) -> None:
        """Write a fully nnU-Net-loadable checkpoint holding the TEACHER weights, by temporarily
        swapping them into self.network and reusing the base save_checkpoint (so the file structure
        matches checkpoint_final exactly). Student weights are restored in a finally block; load_state_dict
        copies in place, so parameter identity — and thus the optimizer state — is preserved."""
        if self.teacher is None:
            return
        student_state = deepcopy(self.network.state_dict())
        try:
            self.network.load_state_dict(self.teacher.state_dict())
            self.save_checkpoint(os.path.join(self.output_folder, filename))
        finally:
            self.network.load_state_dict(student_state)

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

        weight = self._consistency_weight()
        use_consistency = weight > 0.0
        # During the supervised warm-up: train on the clean patch (== a standard supervised run) and
        # skip the teacher forward (saves ~1/3 compute). The strong student view + consistency only
        # switch on once the teacher is a reliable target.
        student_in = (
            self._perturb(data, self.student_noise_std, self.student_scale, self.student_shift)
            if use_consistency else data
        )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(student_in)
            supervised_loss = self.loss(output, target)  # ignore label masks the 90 automatically
            if use_consistency:
                teacher_in = data if self.teacher_noise_std <= 0 else self._perturb(
                    data, self.teacher_noise_std, 0.0, 0.0)
                with torch.no_grad():
                    self.teacher.eval()
                    teacher_output = self.teacher(teacher_in)
                consistency = self._consistency(output, teacher_output)
                loss = supervised_loss + weight * consistency
            else:
                consistency = supervised_loss.new_zeros(())
                loss = supervised_loss

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


class nnUNetTrainer_MeanTeacher_Pilot_NoDeepSupervision_NoMirroring(
    nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring
):
    """FAST pilot — validate the whole MT machinery end-to-end in ~15-20 min. NOT reportable.

    Meant to be launched WARM-STARTED from the @20 model, so a converged teacher exists from
    step 1 and consistency can engage almost immediately:
        nnUNetv2_train 122 3d_fullres 0 \
          -tr nnUNetTrainer_MeanTeacher_Pilot_NoDeepSupervision_NoMirroring \
          -pretrained_weights <Dataset120 fold_0 checkpoint_final.pth>

    It exercises: pretrained load, teacher build, supervised (ignore-masked) loss, the strong/weak
    views, consistency + sigmoid ramp, `consistency_fraction` logging, EMA update, and the
    teacher-as-checkpoint_final deploy — then you can predict+score it as a dry run of the chain.
    The REPORTABLE run uses the parent trainer (from scratch, 1000 ep, no pretrained).
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10             # ~15-20 min total
        self.initial_lr = 1e-3           # low LR: warm-started from a converged @20
        self.consistency_warmup_epochs = 1   # engage consistency almost immediately
        self.consistency_rampup = 3.0        # full weight by epoch 4, so the pilot exercises it
