"""Soft-skeleton iteration sweep for the Dataset126 soft-clDice arm.

Separate module so the reviewed ``..._SoftCLDice_Diagnostics_...`` file keeps the
Git blob hash its own launch guards pin.  Each trainer below is the symmetric
soft-probability clDice arm with a single changed training variable,
``cldice_iters``.

Why this sweep exists: ``cldice_iters = 10`` entered with the first clDice commit
under the rule "set >= the largest expected airway radius in voxels", and was
never swept.  Three measurements since then show the rule was answering a
different question than the one this project asks.

  1. Iterations are a reach dial over calibre, not a quality knob.  A structure
     below its threshold contributes EXACTLY zero soft-skeleton mass; at or above
     it the contribution saturates and is then invariant to further iterations.
     Measured reach radius is approximately the iteration count.

  2. Reach therefore sets how much PROXIMAL mass enters the denominator of both
     clDice terms, and distal branches need no iterations at all -- radius 1-1.5
     structures are skeletonised by the initial ``relu(x - open(x))`` term.  On
     18 teacher patches the distal (radius < 2.5) share of soft-skeleton mass is
     0.690 at 3 iterations and 0.456 at 10.

  3. Against a true morphological skeleton (skimage) the soft skeleton is 2-6x
     over-massive at every calibre, so no setting is "faithful".  A true skeleton
     puts 0.663 of its mass in the distal buckets, which 3 iterations happens to
     approximate and 10 does not -- a cancellation of two errors, not fidelity.

    index 1 -> cldice_iters = 3   (reach ~radius 3; segmental and finer)
    (reference) cldice_iters = 10 == the reported arm, already trained, NOT re-run
    index 2 -> cldice_iters = 15  (past the measured max GT radius of 13.2; full reach)

Everything else -- one-GT/one-U sampling, warm start, EMA schedule, augmentation,
ramp (warm-up 5, ramp 20), 500 epochs, w_max = 0.1, checkpoint sidecar -- is
inherited unchanged, so a difference against ``126_softcl`` is attributable to
``cldice_iters`` alone.

Cost note: the skeleton loop runs twice per step (student and teacher) and costs
four pooling passes per iteration, so index 1 is cheaper than the reference and
index 2 is dearer.  Both subjobs are self-resuming, so a walltime overrun on
index 2 is recoverable by re-submitting the array.
"""

from __future__ import annotations

import torch

if __package__ == "nnunet_trainers":  # repository-local tests
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring as _SoftBase,
    )
else:  # installed beside the parent trainers in nnU-Net
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring as _SoftBase,
    )


# The value the reported arm trained at, inherited from the base Mean-Teacher
# trainer. Named here so the launch preflight can refuse to re-run it.
REFERENCE_CLDICE_ITERS = 10


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceK03Diagnostics_NoDeepSupervision_NoMirroring(
    _SoftBase
):
    """Soft-clDice consistency with 3 soft-skeleton iterations (reach ~radius 3)."""

    diagnostic_objective = "soft_probability_cldice_k03"
    configured_cldice_iters = 3

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
        # The base Mean-Teacher __init__ sets cldice_iters = 10; override after it.
        self.cldice_iters = int(type(self).configured_cldice_iters)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.print_to_log_file(
            f"[KSweep] cldice_iters={self.cldice_iters} "
            f"(reference arm trains at {REFERENCE_CLDICE_ITERS}); "
            f"consistency_max={self.consistency_max} beta={self.cldice_cons_beta}"
        )


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceK15Diagnostics_NoDeepSupervision_NoMirroring(
    _SoftBase
):
    """Soft-clDice consistency with 15 soft-skeleton iterations (full reach)."""

    diagnostic_objective = "soft_probability_cldice_k15"
    configured_cldice_iters = 15

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.cldice_iters = int(type(self).configured_cldice_iters)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.print_to_log_file(
            f"[KSweep] cldice_iters={self.cldice_iters} "
            f"(reference arm trains at {REFERENCE_CLDICE_ITERS}); "
            f"consistency_max={self.consistency_max} beta={self.cldice_cons_beta}"
        )


# Index -> trainer, mirroring the PBS array indices. Kept here rather than in the
# job script so the mapping is versioned with the trainers it names.
# One-based to match the PBS array indices (#PBS -J 1-2), which follow the
# Imperial RCS documented convention rather than a zero-based range.
SWEEP_TRAINERS = {
    1: nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceK03Diagnostics_NoDeepSupervision_NoMirroring,
    2: nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceK15Diagnostics_NoDeepSupervision_NoMirroring,
}
