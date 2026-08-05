"""Isolated Dataset126 K1 ablations with auditable Mean-Teacher diagnostics.

This module deliberately leaves the established Mean-Teacher and two-stream
trainers untouched.  Both public trainers inherit the same one-GT/one-U
sampling, warm start, augmentation schedule, EMA update, checkpoint sidecar,
and 500-epoch optimisation envelope as the Dataset126 K1 run.

The soft arm changes one training variable: its clDice target is the detached
teacher *probability map*, not ``teacher_probability > 0.5``.  A hard-target
clDice value is evaluated only once per epoch as a counterfactual diagnostic;
it never contributes a gradient.  The control arm has no unlabelled objective
or teacher forward and therefore supplies exactly zero U-stream gradient.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast

if __package__ == "nnunet_trainers":  # repository-local tests
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (
        _soft_skeleton3d,
    )
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _TwoStreamBase,
    )
else:  # installed beside the parent trainers in nnU-Net
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (  # noqa: E501
        _soft_skeleton3d,
    )
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _TwoStreamBase,
    )


DIAGNOSTICS_VERSION = "soft_probability_cldice_diag_v1"


def _cldice_from_skeletons(
    student_fg: torch.Tensor,
    target_fg: torch.Tensor,
    student_skeleton: torch.Tensor,
    target_skeleton: torch.Tensor,
    *,
    smooth: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean ``(1-F_beta)``, topology precision, and topology sensitivity."""
    batch = student_fg.shape[0]
    skel_p = student_skeleton.reshape(batch, -1)
    skel_t = target_skeleton.reshape(batch, -1)
    pred = student_fg.reshape(batch, -1)
    target = target_fg.reshape(batch, -1)
    tprec = (torch.sum(skel_p * target, dim=1) + smooth) / (
        torch.sum(skel_p, dim=1) + smooth
    )
    tsens = (torch.sum(skel_t * pred, dim=1) + smooth) / (
        torch.sum(skel_t, dim=1) + smooth
    )
    beta2 = float(beta) ** 2
    score = (1.0 + beta2) * tprec * tsens / (beta2 * tprec + tsens + 1e-8)
    return (1.0 - score).mean(), tprec.mean(), tsens.mean()


def _soft_probability_cldice_terms(
    student_fg: torch.Tensor,
    teacher_fg: torch.Tensor,
    iterations: int,
    *,
    smooth: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """clDice terms using a detached, unthresholded teacher probability map.

    Inputs are foreground probabilities shaped ``(B, 1, D, H, W)``.  The
    teacher is detached inside this function as a second guard against an
    accidental gradient path into the EMA network.
    """
    if student_fg.shape != teacher_fg.shape or student_fg.ndim != 5:
        raise ValueError(
            "Soft probability clDice expects equal (B,1,D,H,W) tensors; "
            f"got student={tuple(student_fg.shape)} teacher={tuple(teacher_fg.shape)}."
        )
    teacher_target = teacher_fg.detach()
    student_skeleton = _soft_skeleton3d(student_fg, iterations)
    teacher_skeleton = _soft_skeleton3d(teacher_target, iterations)
    loss, tprec, tsens = _cldice_from_skeletons(
        student_fg,
        teacher_target,
        student_skeleton,
        teacher_skeleton,
        smooth=smooth,
        beta=beta,
    )
    return loss, tprec, tsens, student_skeleton, teacher_skeleton


def _soft_probability_cldice_consistency(
    student_fg: torch.Tensor,
    teacher_fg: torch.Tensor,
    iterations: int,
    *,
    smooth: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Scalar wrapper used by tests and ``_consistency``."""
    return _soft_probability_cldice_terms(
        student_fg,
        teacher_fg,
        iterations,
        smooth=smooth,
        beta=beta,
    )[0]


def _gradient_probe_stats(
    supervised_loss: torch.Tensor,
    consistency_loss: torch.Tensor | None,
    parameter: torch.nn.Parameter,
    weight: float,
    *,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Measure raw and weighted objective gradients on one shared parameter.

    ``torch.autograd.grad`` does not populate ``parameter.grad``.  Retaining
    the graph lets the ordinary scaled/unscaled backward pass run unchanged.
    This is a probe, not a claim about the norm over the complete network.
    """
    grad_sup = torch.autograd.grad(
        supervised_loss,
        parameter,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad_sup is None:
        zero = supervised_loss.detach().new_zeros(())
        return {
            "supervised_norm": zero,
            "consistency_norm": zero,
            "raw_ratio": zero,
            "applied_weight": zero.new_tensor(float(weight)),
            "weighted_consistency_norm": zero,
            "weighted_ratio": zero,
            "weighted_fraction": zero,
            "cosine": zero,
        }

    grad_sup = grad_sup.detach().float().reshape(-1)
    supervised_norm = torch.linalg.vector_norm(grad_sup)
    if consistency_loss is None or not consistency_loss.requires_grad:
        consistency_norm = supervised_norm.new_zeros(())
        cosine = supervised_norm.new_zeros(())
    else:
        grad_consistency = torch.autograd.grad(
            consistency_loss,
            parameter,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_consistency is None:
            consistency_norm = supervised_norm.new_zeros(())
            cosine = supervised_norm.new_zeros(())
        else:
            grad_consistency = grad_consistency.detach().float().reshape(-1)
            consistency_norm = torch.linalg.vector_norm(grad_consistency)
            denominator = supervised_norm * consistency_norm
            cosine = torch.where(
                denominator > eps,
                torch.dot(grad_sup, grad_consistency) / denominator.clamp_min(eps),
                denominator.new_zeros(()),
            )

    weighted_norm = consistency_norm * float(weight)
    raw_ratio = consistency_norm / supervised_norm.clamp_min(eps)
    weighted_ratio = weighted_norm / supervised_norm.clamp_min(eps)
    weighted_fraction = weighted_norm / (supervised_norm + weighted_norm).clamp_min(eps)
    return {
        "supervised_norm": supervised_norm,
        "consistency_norm": consistency_norm,
        "raw_ratio": raw_ratio,
        "applied_weight": supervised_norm.new_tensor(float(weight)),
        "weighted_consistency_norm": weighted_norm,
        "weighted_ratio": weighted_ratio,
        "weighted_fraction": weighted_fraction,
        "cosine": cosine,
    }


class _DiagnosticK1TwoStreamBase(_TwoStreamBase):
    """Shared implementation; public subclasses select soft consistency/control."""

    diagnostics_version = DIAGNOSTICS_VERSION
    protocol_exposure = "K1"
    expected_labelled_per_step = 1
    expected_unlabelled_per_step = 1
    expected_local_batch_size = 2
    diagnostic_objective = "abstract"
    enable_soft_probability_consistency = False
    configured_consistency_max = 0.0

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # Keep the explicit nnU-Net signature. Its trainer base reflects on
        # self.__init__, so (*args, **kwargs) is unsafe here.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.consistency_max = float(type(self).configured_consistency_max)
        self.consistency_mode = self.diagnostic_objective
        self._diagnostic_sums: dict[str, torch.Tensor] = {}
        self._diagnostic_steps = 0
        self._diagnostic_hard_epoch: int | None = None
        self._diagnostic_gradient_epoch: int | None = None
        self._diagnostic_gradient_probe_name = "not-run"
        self._diagnostic_gradient_stats: dict[str, torch.Tensor] = {}

    def _ensure_diagnostic_state(self) -> None:
        # Also makes object.__new__-based focused tests straightforward and
        # protects old checkpoints from future additions to diagnostic state.
        if not hasattr(self, "_diagnostic_sums"):
            self._diagnostic_sums = {}
        if not hasattr(self, "_diagnostic_steps"):
            self._diagnostic_steps = 0
        if not hasattr(self, "_diagnostic_hard_epoch"):
            self._diagnostic_hard_epoch = None
        if not hasattr(self, "_diagnostic_gradient_epoch"):
            self._diagnostic_gradient_epoch = None
        if not hasattr(self, "_diagnostic_gradient_probe_name"):
            self._diagnostic_gradient_probe_name = "not-run"
        if not hasattr(self, "_diagnostic_gradient_stats"):
            self._diagnostic_gradient_stats = {}

    def _diagnostic_add(self, values: Mapping[str, torch.Tensor]) -> None:
        for name, value in values.items():
            detached = value.detach().float()
            if name in self._diagnostic_sums:
                self._diagnostic_sums[name] = self._diagnostic_sums[name] + detached
            else:
                self._diagnostic_sums[name] = detached

    def _select_gradient_probe_parameter(self) -> tuple[str, torch.nn.Parameter] | tuple[None, None]:
        candidates = [
            (name, parameter)
            for name, parameter in self.network.named_parameters()
            if parameter.requires_grad and parameter.ndim >= 2
        ]
        if not candidates:
            return None, None
        # Prefer the final segmentation projection. With NoDeepSupervision,
        # the last seg layer is the active full-resolution output head.
        output_heads = [
            pair
            for pair in candidates
            if any(token in pair[0].lower() for token in ("seg_layers", "seg_output", "out_conv"))
        ]
        return (output_heads or candidates)[-1]

    def _consistency(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if not self.enable_soft_probability_consistency:
            # Explicit graph-connected zero: even if called accidentally, this
            # class cannot inject an unlabelled gradient.
            return student_logits.sum() * 0.0
        student = torch.softmax(student_logits.float(), dim=1)
        teacher = torch.softmax(teacher_logits.float(), dim=1).detach()
        return _soft_probability_cldice_consistency(
            student[:, 1:2],
            teacher[:, 1:2],
            self.cldice_iters,
            beta=self.cldice_cons_beta,
        )

    def _record_soft_diagnostics(
        self,
        student_fg: torch.Tensor,
        teacher_fg: torch.Tensor,
        student_skeleton: torch.Tensor,
        teacher_skeleton: torch.Tensor,
        soft_loss: torch.Tensor,
        soft_tprec: torch.Tensor,
        soft_tsens: torch.Tensor,
    ) -> None:
        """Accumulate cheap probability/skeleton summaries without GPU syncs."""
        teacher = teacher_fg.detach()
        student = student_fg.detach()
        teacher_skel = teacher_skeleton.detach()
        student_skel = student_skeleton.detach()
        batch = teacher.shape[0]
        flat_teacher = teacher.reshape(batch, -1)
        flat_skel = teacher_skel.reshape(batch, -1)
        skeleton_mass = flat_skel.sum(dim=1)
        hard_positive = (flat_teacher > 0.5).any(dim=1)
        # The historical hard target is p > 0.5, therefore p == 0.5 is also
        # discarded by thresholding and belongs in these subthreshold totals.
        below_half = flat_teacher <= 0.5
        meaningful_subthreshold = (flat_teacher >= 0.05) & below_half
        meaningful_soft_skeleton_mass = (
            flat_skel * (flat_teacher >= 0.05).float()
        ).sum(dim=1)
        # Avoid counting tiny numerical background tails as an active soft
        # tree: require one voxel-equivalent of skeleton supported by p>=0.05.
        soft_active = meaningful_soft_skeleton_mass >= 1.0
        below_half_mass = (flat_skel * below_half.float()).sum(dim=1)
        meaningful_below_half_mass = (
            flat_skel * meaningful_subthreshold.float()
        ).sum(dim=1)
        skeleton_weighted_probability = (flat_skel * flat_teacher).sum(dim=1)
        skeleton_denominator = skeleton_mass.clamp_min(1e-12)
        probability_denominator = flat_teacher.sum(dim=1).clamp_min(1e-12)
        teacher_self_loss = 1.0 - (
            skeleton_weighted_probability + 1.0
        ) / (skeleton_mass + 1.0)

        self._diagnostic_add(
            {
                "soft_loss": soft_loss,
                "soft_tprec": soft_tprec,
                "soft_tsens": soft_tsens,
                "probability_mse": (student - teacher).square().mean(),
                "teacher_probability_mass": flat_teacher.sum(dim=1).mean(),
                "teacher_hard_voxels": (flat_teacher > 0.5).float().sum(dim=1).mean(),
                "teacher_soft_skeleton_mass": skeleton_mass.mean(),
                "teacher_student_skeleton_mass": student_skel.reshape(batch, -1).sum(dim=1).mean(),
                # Reuses teacher_skeleton from the optimized soft loss: no
                # additional 10-iteration skeleton pass is performed here.
                "soft_self_loss": teacher_self_loss.mean(),
                "subthreshold_teacher_probability_mass_share": (
                    (flat_teacher * below_half.float()).sum(dim=1) / probability_denominator
                ).mean(),
                "p0p05_to_0p5_teacher_probability_mass_share": (
                    (flat_teacher * meaningful_subthreshold.float()).sum(dim=1)
                    / probability_denominator
                ).mean(),
                "subthreshold_teacher_skeleton_mass_share": (
                    below_half_mass / skeleton_denominator
                ).mean(),
                "p0p05_to_0p5_teacher_skeleton_mass_share": (
                    meaningful_below_half_mass / skeleton_denominator
                ).mean(),
                "teacher_skeleton_weighted_probability": (
                    skeleton_weighted_probability / skeleton_denominator
                ).mean(),
                "teacher_p_gt_0p1_fraction": (flat_teacher > 0.1).float().mean(),
                "teacher_p_gt_0p3_fraction": (flat_teacher > 0.3).float().mean(),
                "teacher_p_gt_0p5_fraction": (flat_teacher > 0.5).float().mean(),
                "teacher_p_gt_0p8_fraction": (flat_teacher > 0.8).float().mean(),
                "hard_positive_patch_fraction": hard_positive.float().mean(),
                "soft_skeleton_active_patch_fraction": soft_active.float().mean(),
                "soft_evidence_without_hard_patch_fraction": (
                    soft_active & ~hard_positive
                ).float().mean(),
            }
        )
        self._diagnostic_steps += 1

        # This thresholded comparison is diagnostic only and is deliberately
        # limited to one U patch per epoch: it cannot enter the training loss.
        if self._diagnostic_hard_epoch != int(self.current_epoch):
            hard_target = (teacher > 0.5).float()
            hard_skeleton = _soft_skeleton3d(hard_target, self.cldice_iters)
            hard_loss, hard_tprec, hard_tsens = _cldice_from_skeletons(
                student,
                hard_target,
                student_skel,
                hard_skeleton,
                beta=self.cldice_cons_beta,
            )
            self._diagnostic_add(
                {
                    "counterfactual_soft_loss": soft_loss.detach(),
                    "counterfactual_hard_loss": hard_loss,
                    "counterfactual_hard_tprec": hard_tprec,
                    "counterfactual_hard_tsens": hard_tsens,
                }
            )
            self._diagnostic_hard_epoch = int(self.current_epoch)

    def _run_gradient_probe_once(
        self,
        supervised_loss: torch.Tensor,
        consistency_loss: torch.Tensor | None,
        weight: float,
    ) -> None:
        if self._diagnostic_gradient_epoch == int(self.current_epoch):
            return
        name, parameter = self._select_gradient_probe_parameter()
        if parameter is None:
            self._diagnostic_gradient_probe_name = "no-trainable-weight"
            self._diagnostic_gradient_stats = {}
        else:
            self._diagnostic_gradient_probe_name = str(name)
            self._diagnostic_gradient_stats = _gradient_probe_stats(
                supervised_loss,
                consistency_loss,
                parameter,
                weight,
            )
        self._diagnostic_gradient_epoch = int(self.current_epoch)

    def on_train_start(self) -> None:
        super().on_train_start()
        if self.enable_soft_probability_consistency:
            self.print_to_log_file(
                "[MTDiagnostics] objective=soft_probability_cldice protocol=K1 "
                "teacher_target=unthresholded_probability hard_threshold_in_loss=false "
                "hard_counterfactual=once-per-epoch gradient_probe=once-per-epoch"
            )
        else:
            self.print_to_log_file(
                "[MTDiagnostics] objective=supervised_control protocol=K1 "
                "consistency_enabled=false teacher_forward=false unlabelled_gradient=zero "
                "strong_view_schedule=matched ema_weight_average=enabled"
            )

    def train_step(self, batch: dict) -> dict:
        self._ensure_diagnostic_state()
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            raise RuntimeError("Diagnostic two-stream MT requires NoDeepSupervision.")
        target = target.to(self.device, non_blocking=True)
        labelled_idx, unlabelled_idx = self._batch_stream_indices(list(batch["keys"]))

        if self.teacher is None:
            self._build_teacher()

        weight = self._consistency_weight()
        use_consistency = self.enable_soft_probability_consistency and weight > 0.0
        # Match the established K1 control: strong augmentation begins after
        # warm-up even when consistency_max == 0.
        use_strong_view = self.current_epoch >= self.consistency_warmup_epochs
        student_in = (
            self._perturb(data, self.student_noise_std, self.student_scale, self.student_shift)
            if use_strong_view
            else data
        )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(student_in)
            supervised_loss = self.loss(
                output.index_select(0, labelled_idx),
                target.index_select(0, labelled_idx),
            )
            if use_consistency:
                teacher_in = data.index_select(0, unlabelled_idx)
                if self.teacher_noise_std > 0:
                    teacher_in = self._perturb(teacher_in, self.teacher_noise_std, 0.0, 0.0)
                with torch.no_grad():
                    self.teacher.eval()
                    teacher_output = self.teacher(teacher_in)
                student_probability = torch.softmax(
                    output.index_select(0, unlabelled_idx).float(), dim=1
                )[:, 1:2]
                teacher_probability = torch.softmax(teacher_output.float(), dim=1).detach()[:, 1:2]
                (
                    consistency,
                    soft_tprec,
                    soft_tsens,
                    student_skeleton,
                    teacher_skeleton,
                ) = _soft_probability_cldice_terms(
                    student_probability,
                    teacher_probability,
                    self.cldice_iters,
                    beta=self.cldice_cons_beta,
                )
                loss = supervised_loss + weight * consistency
            else:
                consistency = supervised_loss.new_zeros(())
                loss = supervised_loss

        if use_consistency:
            self._record_soft_diagnostics(
                student_probability,
                teacher_probability,
                student_skeleton,
                teacher_skeleton,
                consistency,
                soft_tprec,
                soft_tsens,
            )
            self._run_gradient_probe_once(supervised_loss, consistency, weight)
        else:
            # The control still reports the supervised probe norm and explicit
            # zero consistency norm. During the soft arm's warm-up this also
            # documents that consistency was genuinely inactive.
            self._run_gradient_probe_once(supervised_loss, None, 0.0)

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
        self._log_w = weight if use_consistency else 0.0
        self._log_n += 1
        self._stream_steps += 1
        self._stream_labelled_samples += int(labelled_idx.numel())
        self._stream_unlabelled_samples += int(unlabelled_idx.numel())
        return {"loss": loss.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs) -> None:
        # Parent emits the standard train loss, MeanTeacher scalar fraction,
        # and TwoStream exposure lines before these richer diagnostics.
        super().on_train_epoch_end(train_outputs)
        self._ensure_diagnostic_state()

        if self._diagnostic_gradient_stats:
            stats = {
                name: float(value.detach().cpu())
                for name, value in self._diagnostic_gradient_stats.items()
            }
            self.print_to_log_file(
                f"[MTGradient] probe={self._diagnostic_gradient_probe_name} "
                f"sup_norm={stats['supervised_norm']:.6g} "
                f"raw_consistency_norm={stats['consistency_norm']:.6g} "
                f"raw_ratio={stats['raw_ratio']:.6g} "
                f"weight={stats['applied_weight']:.6g} "
                f"weighted_consistency_norm={stats['weighted_consistency_norm']:.6g} "
                f"weighted_ratio={stats['weighted_ratio']:.6g} "
                f"weighted_fraction={stats['weighted_fraction']:.6g} "
                f"cosine={stats['cosine']:.6g}"
            )

        if self.enable_soft_probability_consistency and self._diagnostic_steps > 0:
            mean = {
                name: float(value.detach().cpu()) / self._diagnostic_steps
                for name, value in self._diagnostic_sums.items()
                if not name.startswith("counterfactual_")
            }
            # Counterfactuals are recorded once, not divided by the number of
            # training steps used for the probability summaries.
            once = {
                name: float(value.detach().cpu())
                for name, value in self._diagnostic_sums.items()
                if name.startswith("counterfactual_")
            }
            self.print_to_log_file(
                f"[MTSoftCLDice] samples={self._diagnostic_steps} "
                f"soft_loss={mean['soft_loss']:.5f} "
                f"soft_tprec={mean['soft_tprec']:.5f} soft_tsens={mean['soft_tsens']:.5f} "
                f"prob_mse={mean['probability_mse']:.7f} "
                f"teacher_prob_mass={mean['teacher_probability_mass']:.2f} "
                f"teacher_hard_voxels={mean['teacher_hard_voxels']:.2f} "
                f"teacher_soft_skel_mass={mean['teacher_soft_skeleton_mass']:.2f} "
                f"student_soft_skel_mass={mean['teacher_student_skeleton_mass']:.2f} "
                f"soft_self_loss={mean['soft_self_loss']:.5f}"
            )
            self.print_to_log_file(
                f"[MTTeacherEvidence] p>0.1={mean['teacher_p_gt_0p1_fraction']:.7f} "
                f"p>0.3={mean['teacher_p_gt_0p3_fraction']:.7f} "
                f"p>0.5={mean['teacher_p_gt_0p5_fraction']:.7f} "
                f"p>0.8={mean['teacher_p_gt_0p8_fraction']:.7f} "
                f"subthr_prob_mass={mean['subthreshold_teacher_probability_mass_share']:.5f} "
                f"subthr_p>=0.05_prob_mass="
                f"{mean['p0p05_to_0p5_teacher_probability_mass_share']:.5f} "
                f"subthr_skel_mass={mean['subthreshold_teacher_skeleton_mass_share']:.5f} "
                f"subthr_p>=0.05_skel_mass="
                f"{mean['p0p05_to_0p5_teacher_skeleton_mass_share']:.5f} "
                f"skeleton_weighted_p={mean['teacher_skeleton_weighted_probability']:.5f} "
                f"hard_active_patches={mean['hard_positive_patch_fraction']:.3f} "
                f"soft_active_patches={mean['soft_skeleton_active_patch_fraction']:.3f} "
                f"soft_only_patches={mean['soft_evidence_without_hard_patch_fraction']:.3f}"
            )
            if once:
                self.print_to_log_file(
                    "[MTHardCounterfactual] samples=1 used_for_gradient=false "
                    f"same_patch_soft_loss={once['counterfactual_soft_loss']:.5f} "
                    f"hard_loss={once['counterfactual_hard_loss']:.5f} "
                    f"hard_tprec={once['counterfactual_hard_tprec']:.5f} "
                    f"hard_tsens={once['counterfactual_hard_tsens']:.5f}"
                )
        elif not self.enable_soft_probability_consistency:
            self.print_to_log_file(
                "[MTControl] objective=supervised_control consistency=0 "
                "teacher_forwards=0 unlabelled_gradient=0"
            )

        self._diagnostic_sums = {}
        self._diagnostic_steps = 0
        self._diagnostic_gradient_stats = {}


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring(
    _DiagnosticK1TwoStreamBase
):
    """K1 Mean Teacher with an unthresholded probability-map clDice target."""

    protocol_exposure = "K1"
    expected_labelled_per_step = 1
    expected_unlabelled_per_step = 1
    diagnostic_objective = "soft_probability_cldice"
    enable_soft_probability_consistency = True
    configured_consistency_max = 0.1

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_ControlDiagnostics_NoDeepSupervision_NoMirroring(
    _DiagnosticK1TwoStreamBase
):
    """Matched K1 envelope with zero consistency/U-stream gradient."""

    protocol_exposure = "K1"
    expected_labelled_per_step = 1
    expected_unlabelled_per_step = 1
    diagnostic_objective = "supervised_control"
    enable_soft_probability_consistency = False
    configured_consistency_max = 0.0

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
