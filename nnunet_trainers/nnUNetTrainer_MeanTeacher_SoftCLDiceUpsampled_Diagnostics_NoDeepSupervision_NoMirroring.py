"""2x-resolution soft skeleton for the Dataset126 soft-clDice consistency arm.

Separate module so the reviewed ``..._SoftCLDice_Diagnostics_...`` file keeps the Git
blob hash its own launch guards pin, and so the running soft/5-fold/sweep jobs are
untouched.

WHAT CHANGES.  Exactly one training variable: the resolution at which the soft
skeleton is extracted.  Both foreground probability maps are trilinearly upsampled 2x,
the skeleton is taken there with 20 iterations instead of 10 (so the same physical
radius is covered), and the clDice terms are evaluated at that resolution.  Everything
else -- one-GT/one-U sampling, warm start, EMA schedule, augmentation, ramp (warm-up 5,
ramp 20), 500 epochs, beta = 1, w_max = 0.1, checkpoint sidecar -- is inherited
unchanged, so a difference against ``126_softcl`` is attributable to the skeleton
resolution alone.

WHY.  ``_soft_erode3d`` is a 7-voxel cross, so for any structure at most two voxels
thick it returns zero, ``open(x) == 0``, ``relu(x - open(x)) == x``, and the "skeleton"
IS the object.  Measured on the reported teacher over 24 patches
(``dissertation/scripts/measure_soft_skeleton_scale.py``): cross-erosion annihilates
**34.0%** of teacher foreground at 1x against 20.4% at 2x, and soft-skeleton mass falls
from 8.78% to 3.07% of foreground -- i.e. 2x recovers a genuine centreline where 1x
returns the tube.

WHAT THE PROBE PREDICTS, so the run can be read against a pre-registered expectation:

  - The ``t_sens`` channel gets WORSE.  ``d t_sens / d pred = skel_t / sum(skel_t)``
    exactly, and the thinnest bucket's share of teacher skeleton mass FALLS at 2x
    (0.2869 -> 0.2589, x0.903).
  - The full gradient gets BETTER, because ``t_prec`` depends on the student's own
    skeleton, which 2x also thins.  Measured distal/proximal gradient ratio rose
    x1.66 on average (7.38->14.35, 3.52->4.50, 4.92->7.33 across three cases), and the
    loss penalty for amputating the <=2-voxel tree rose x1.402.
  - Distal METRICS are nonetheless expected to move within noise, because the teacher
    supplies almost no sub-threshold evidence to begin with (halo ``p>0.1 / p>0.5`` =
    1.0373, ``soft_only_patches <= 0.004``, student/teacher RMS probability
    disagreement 0.014).  A x1.66 reweighting of a signal that measures ~0 is still
    ~0.  **If TD/BD do move materially, that falsifies the saturated-teacher reading
    and is the more interesting outcome.**

KNOWN CEILING.  A one-voxel-thick branch is still fully annihilated at 2x -- doubled it
is two voxels, still degenerate.  Removing that needs 4x, which is out of reach on
memory grounds.  This arm therefore tests the thickness-2 population, not thickness-1.

MEMORY.  The skeleton loop must be gradient-checkpointed or the backward pass will not
fit: measured peak for the student skeleton alone is 6.53 GiB at 2x/20 checkpointed
against an out-of-memory failure unchecked on an 8 GiB card.  The loop is pure min/max
pooling, so recomputation is cheap relative to storing 20 iterations of 8x-sized
activations.  Budget a 40 GB GPU: the checkpointed skeleton sits on top of the ordinary
nnU-Net graph, it does not replace it.

COST.  Measured on an RTX 4060, per patch forward+backward: 1x/10 checkpointed 0.16 s,
2x/20 checkpointed 3.07-3.17 s -- about 20x the skeleton cost.  Scaling by memory
bandwidth (the loop is bandwidth-bound) suggests ~0.55 s/patch on an A100, i.e. about
+140 s on top of 52.5 s/epoch at 250 unlabelled patches per epoch, so roughly 190
s/epoch and ~26 h for 500 epochs.  **That estimate has soft links and the job script
prints the real per-epoch cost early; check it after epoch 10 rather than trusting the
projection.**  Do NOT shorten ``num_epochs`` to fit a walltime: nnU-Net's polynomial LR
schedule is parameterised by it, so a shorter run changes the optimisation trajectory
as well as the budget.  Resume with ``--c`` instead.

Deploy on HPC into the ctfm nnU-Net site-packages alongside the other trainers; the
job script symlinks it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast
from torch.utils.checkpoint import checkpoint

if __package__ == "nnunet_trainers":  # repository-local tests
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (
        _soft_erode3d,
        _soft_open3d,
        _soft_skeleton3d,
    )
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring import (  # noqa: E501
        _cldice_from_skeletons,
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring as _SoftBase,  # noqa: E501
    )
else:  # installed beside the parent trainers in nnU-Net
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (  # noqa: E501
        _soft_erode3d,
        _soft_open3d,
        _soft_skeleton3d,
    )
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring import (  # noqa: E501
        _cldice_from_skeletons,
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring as _SoftBase,  # noqa: E501
    )


DIAGNOSTICS_VERSION = "soft_probability_cldice_upsampled_diag_v1"

# Iterations at 1x in the reported arm. The upsampled arm scales this with the
# resolution so both cover the same physical radius.
BASE_CLDICE_ITERS = 10


def _skeleton_step(x: torch.Tensor, skeleton: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One iteration of ``_soft_skeleton3d``, transcribed so it can be recomputed.

    Must stay a line-for-line transcription of the loop body in the pinned
    ``_soft_skeleton3d``. ``tests/test_nnunet_softcldice_upsampled_trainer.py`` asserts
    that the checkpointed skeleton is bit-identical to the pinned implementation.
    """
    x = _soft_erode3d(x)
    opened = _soft_open3d(x)
    delta = F.relu(x - opened)
    return x, skeleton + F.relu(delta - skeleton * delta)


def _checkpointed_soft_skeleton3d(x: torch.Tensor, iterations: int) -> torch.Tensor:
    """``_soft_skeleton3d`` with each iteration recomputed in the backward pass.

    Numerically identical to the pinned function; it trades the per-iteration
    activations for one extra forward pass over a loop of pure min/max pooling. There
    is no RNG in the loop, so the RNG state does not need preserving.
    """
    opened = _soft_open3d(x)
    skeleton = F.relu(x - opened)
    for _ in range(int(iterations)):
        x, skeleton = checkpoint(
            _skeleton_step,
            x,
            skeleton,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return skeleton


def _upsampled_soft_probability_cldice_terms(
    student_fg: torch.Tensor,
    teacher_fg: torch.Tensor,
    iterations: int,
    scale: int,
    *,
    smooth: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """clDice terms with both maps skeletonised at ``scale`` x the native resolution.

    Inputs are foreground probabilities shaped ``(B, 1, D, H, W)`` at native
    resolution. Every returned tensor is at the upsampled resolution, which is what the
    loss sees and therefore what the diagnostics should describe.

    The teacher is detached before upsampling as a second guard against a gradient path
    into the EMA network, and its skeleton is taken under ``no_grad`` so the 20-iteration
    pass stores nothing.
    """
    if student_fg.shape != teacher_fg.shape or student_fg.ndim != 5:
        raise ValueError(
            "Upsampled soft clDice expects equal (B,1,D,H,W) tensors; "
            f"got student={tuple(student_fg.shape)} teacher={tuple(teacher_fg.shape)}."
        )
    if int(scale) < 1:
        raise ValueError(f"scale must be at least 1, got {scale}.")

    if int(scale) == 1:
        student = student_fg
        teacher_target = teacher_fg.detach()
    else:
        student = F.interpolate(
            student_fg, scale_factor=int(scale), mode="trilinear", align_corners=False
        )
        teacher_target = F.interpolate(
            teacher_fg.detach(), scale_factor=int(scale), mode="trilinear", align_corners=False
        )

    # Only the student's skeleton needs a graph. Checkpointing it is what makes the
    # backward pass fit at 2x.
    student_skeleton = _checkpointed_soft_skeleton3d(student, iterations)
    with torch.no_grad():
        teacher_skeleton = _soft_skeleton3d(teacher_target, iterations)

    loss, tprec, tsens = _cldice_from_skeletons(
        student,
        teacher_target,
        student_skeleton,
        teacher_skeleton,
        smooth=smooth,
        beta=beta,
    )
    return loss, tprec, tsens, student_skeleton, teacher_skeleton


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceUpsampledDiagnostics_NoDeepSupervision_NoMirroring(  # noqa: E501
    _SoftBase
):
    """Soft-probability clDice consistency with the skeleton taken at 2x resolution."""

    diagnostics_version = DIAGNOSTICS_VERSION
    diagnostic_objective = "soft_probability_cldice_2x_skeleton"
    enable_soft_probability_consistency = True
    # Unchanged from the reported arm: this run varies the skeleton resolution only.
    configured_consistency_max = 0.1

    skeleton_scale = 2

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
        scale = int(type(self).skeleton_scale)
        if scale < 1:
            raise ValueError(f"skeleton_scale must be at least 1, got {scale}.")
        # Cover the same physical radius as the 1x arm. Set here rather than as a
        # class attribute so the two can never drift apart.
        self.cldice_iters = BASE_CLDICE_ITERS * scale
        self.skeleton_scale = scale

    def on_train_start(self) -> None:
        super().on_train_start()
        self.print_to_log_file(
            f"[MTUpsampledSkeleton] skeleton_scale={self.skeleton_scale} "
            f"cldice_iters={self.cldice_iters} (1x baseline is {BASE_CLDICE_ITERS}) "
            "gradient_checkpointing=on interpolation=trilinear "
            "clDice terms and diagnostics are evaluated at the UPSAMPLED resolution, "
            "so skeleton-mass statistics are NOT comparable to the 1x arm's logs."
        )

    def _consistency(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Scalar consistency, kept in step with ``train_step`` for tests and probes."""
        if not self.enable_soft_probability_consistency:
            return student_logits.sum() * 0.0
        student = torch.softmax(student_logits.float(), dim=1)
        teacher = torch.softmax(teacher_logits.float(), dim=1).detach()
        return _upsampled_soft_probability_cldice_terms(
            student[:, 1:2],
            teacher[:, 1:2],
            self.cldice_iters,
            self.skeleton_scale,
            beta=self.cldice_cons_beta,
        )[0]

    def train_step(self, batch: dict) -> dict:
        """Transcription of the parent's ``train_step``.

        The parent calls the module-level ``_soft_probability_cldice_terms`` directly and
        exposes no seam to swap it, and its file is Git-blob-pinned by the launch guards
        of three other job scripts, so it cannot be given one. The ONLY difference from
        the parent is the marked call below; if the parent's ``train_step`` changes, this
        copy must be re-synced. ``tests/test_nnunet_softcldice_upsampled_trainer.py``
        guards the parts that can be checked without a GPU.
        """
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
                # ---- THE ONLY CHANGE FROM THE PARENT ----------------------------------
                # Skeletonise at self.skeleton_scale x resolution with the matching
                # iteration count, gradient-checkpointed so the backward pass fits.
                (
                    consistency,
                    soft_tprec,
                    soft_tsens,
                    student_skeleton,
                    teacher_skeleton,
                ) = _upsampled_soft_probability_cldice_terms(
                    student_probability,
                    teacher_probability,
                    self.cldice_iters,
                    self.skeleton_scale,
                    beta=self.cldice_cons_beta,
                )
                # ---- END CHANGE ------------------------------------------------------
                loss = supervised_loss + weight * consistency
            else:
                consistency = supervised_loss.new_zeros(())
                loss = supervised_loss

        if use_consistency:
            # The probability maps are at native resolution and the skeletons at the
            # upsampled one. _record_soft_diagnostics compares each probability map
            # against its OWN skeleton, so both must be on the same grid.
            scale = int(self.skeleton_scale)
            if scale == 1:
                diagnostic_student = student_probability
                diagnostic_teacher = teacher_probability
            else:
                with torch.no_grad():
                    diagnostic_student = F.interpolate(
                        student_probability.detach(),
                        scale_factor=scale,
                        mode="trilinear",
                        align_corners=False,
                    )
                    diagnostic_teacher = F.interpolate(
                        teacher_probability,
                        scale_factor=scale,
                        mode="trilinear",
                        align_corners=False,
                    )
            self._record_soft_diagnostics(
                diagnostic_student,
                diagnostic_teacher,
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
