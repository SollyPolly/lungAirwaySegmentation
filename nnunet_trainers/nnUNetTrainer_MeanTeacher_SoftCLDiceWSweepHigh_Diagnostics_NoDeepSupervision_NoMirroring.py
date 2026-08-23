"""High-weight continuation of the Dataset126 soft-clDice consistency sweep.

Separate module for the same reason ``..._SoftCLDiceWSweep_...`` is separate from
the reviewed parent: the launch guards of the completed w_max = 0.03 and 0.30 jobs
pin that file's Git blob, so adding rungs there would make those scripts refuse to
relaunch.

Why these rungs exist: the swept range is MONOTONE in tree length and branch
detection on VAL20 and therefore never brackets a turning point --

    w_max   0.03     0.10     0.30      (control 0.00)
    TD      0.9299   0.9398   0.9526    0.9202
    BD      0.8739   0.8909   0.9142    0.8547

-- with 0.30 beating the zero-consistency control on 20/20 cases for both, so the
reported w_max = 0.1 is not the top of the range. The teacher shows no collapse
precursor at 0.30: final internal validation Dice 0.9068 against 0.9051 at 0.10,
and soft skeleton precision and teacher probability mass flat across all three
rungs. The known teacher collapses at w_max >= 0.3 belong to the class-balanced
voxel-MSE ablation, whose ~100x foreground gradient amplification does not apply
to the geometry-normalised clDice term, so they do not bound this objective.

What to watch, since these are the rungs where a ceiling would first show:
``[MTSoftCLDice] teacher_prob_mass`` and ``soft_tprec`` against the 0.30 run, and
Mean Validation Dice at the end. Note that centreline precision falls
monotonically with w across the swept range (``tprec_raw`` worse than control on
20/20 cases at 0.30), so a further TD/BD gain at these rungs is not by itself an
improvement in tree quality.

    index 1 -> w_max = 0.50
    index 2 -> w_max = 1.00

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


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceW050Diagnostics_NoDeepSupervision_NoMirroring(
    _SoftBase
):
    """Soft-clDice consistency at w_max = 0.50 (reference x 5)."""

    diagnostic_objective = "soft_probability_cldice_w0p50"
    configured_consistency_max = 0.50

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


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceW100Diagnostics_NoDeepSupervision_NoMirroring(
    _SoftBase
):
    """Soft-clDice consistency at w_max = 1.00 (reference x 10; consistency term
    weighted equally with the supervised term at the top of the ramp)."""

    diagnostic_objective = "soft_probability_cldice_w1p00"
    configured_consistency_max = 1.00

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
# One-based to match the PBS array indices (#PBS -J 1-2), which follow the
# Imperial RCS documented convention rather than a zero-based range.
SWEEP_TRAINERS = {
    1: nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceW050Diagnostics_NoDeepSupervision_NoMirroring,
    2: nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceW100Diagnostics_NoDeepSupervision_NoMirroring,
}
