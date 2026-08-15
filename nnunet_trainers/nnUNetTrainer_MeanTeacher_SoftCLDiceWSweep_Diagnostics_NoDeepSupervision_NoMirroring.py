"""Consistency-weight sensitivity sweep for the Dataset126 soft-clDice arm.

Separate module so the reviewed ``..._SoftCLDice_Diagnostics_...`` file keeps the
Git blob hash its own launch guards pin.  Each trainer below is the symmetric
(beta=1) soft-probability clDice arm with a single changed training variable,
``consistency_max``.

Why this sweep exists: ``w_max = 0.1`` was chosen by a scale argument -- that
``1 - clDice`` returns values in [0, 1] and sits at roughly 0.3-0.6 early, about
five times larger than the class-balanced voxel term, so it takes a
correspondingly smaller coefficient -- and was then confirmed against the measured
consistency gradient share.  It was never swept for this objective.  The observed
teacher collapses at higher weight belong to the class-balanced voxel-MSE
ablation, which is on a different scale and therefore does NOT validate 0.1 here.
Two runs bracketing a decade turn that assertion into a sensitivity analysis.

    index 0 -> w_max = 0.03
    (reference)  w_max = 0.10  == the reported arm, already trained, NOT re-run
    index 1 -> w_max = 0.30

Everything else -- one-GT/one-U sampling, warm start, EMA schedule, augmentation,
ramp (warm-up 5, ramp 20), 500 epochs, beta = 1, checkpoint sidecar -- is
inherited unchanged, so a difference against ``126_softcl`` is attributable to
``w_max`` alone.
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


REFERENCE_CONSISTENCY_MAX = 0.1


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceW003Diagnostics_NoDeepSupervision_NoMirroring(
    _SoftBase
):
    """Soft-clDice consistency at w_max = 0.03 (reference / 3.3)."""

    diagnostic_objective = "soft_probability_cldice_w0p03"
    configured_consistency_max = 0.03

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


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceW030Diagnostics_NoDeepSupervision_NoMirroring(
    _SoftBase
):
    """Soft-clDice consistency at w_max = 0.30 (reference x 3)."""

    diagnostic_objective = "soft_probability_cldice_w0p30"
    configured_consistency_max = 0.30

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)


# Index -> trainer, mirroring the PBS array indices. Kept here rather than in the
# job script so the mapping is versioned with the trainers it names.
SWEEP_TRAINERS = {
    0: nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceW003Diagnostics_NoDeepSupervision_NoMirroring,
    1: nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceW030Diagnostics_NoDeepSupervision_NoMirroring,
}
