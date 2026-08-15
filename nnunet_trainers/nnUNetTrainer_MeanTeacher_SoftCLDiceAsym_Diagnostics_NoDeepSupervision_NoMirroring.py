"""Recall-directed (beta=2) variant of the Dataset126 soft-clDice diagnostic arm.

This module exists as a separate file purely so that the reviewed
``..._SoftCLDice_Diagnostics_...`` module keeps the Git blob hash its own launch
guards pin.  It adds no implementation: the trainer below is the soft-probability
clDice trainer with a single changed training variable, the F-beta weighting of
the two clDice-consistency directions.

Design of the arm (the missing cell of a 2x2, target representation x directional
weighting):

    target \\ beta   1 (symmetric)                 2 (recall-directed)
    hard  (>0.5)    Dataset126 K1/K3              Dataset124 AsymCLDice
    soft  (p_T)     ...SoftCLDiceDiagnostics...   THIS TRAINER

At beta=1 the soft target was inert against the hard target because the
warm-started teacher is saturated and the symmetric F1 is dominated by the region
where the two already agree.  beta=2 up-weights topology sensitivity (the student
COVERS the teacher's tree) and down-weights topology precision (the penalty for
EXCEEDING it).  Only the soft target has sub-threshold teacher evidence for that
recall pressure to aim at: past 0.5 the hard target is identically zero, so
hard-target beta=2 can only over-paint blindly.  The arm therefore tests whether
the teacher's sub-threshold probability mass is real airway or noise.

Read it on the completeness/precision frontier against Dataset124 AsymCLDice, not
on tree-length detection alone -- beta>1 raises tree-length mechanically, which is
not by itself a result.
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


ASYMMETRIC_CONSISTENCY_BETA = 2.0


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceAsymDiagnostics_NoDeepSupervision_NoMirroring(
    _SoftBase
):
    """Soft-probability clDice consistency at F-beta weight beta=2.

    Every other training variable -- one-GT/one-U sampling, warm start, EMA
    schedule, augmentation, consistency ramp and ``consistency_max`` 0.1, 500
    epochs, checkpoint sidecar -- is inherited unchanged from the soft arm, so a
    difference against it is attributable to beta alone.

    The once-per-epoch hard-target counterfactual inherited from the soft arm is
    evaluated at the same beta, which keeps it the matched counterfactual for
    this arm rather than for the symmetric one.
    """

    diagnostic_objective = "soft_probability_cldice_beta2"
    consistency_beta = ASYMMETRIC_CONSISTENCY_BETA

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
        # Set after super().__init__, which assigns the symmetric default 1.0.
        # _consistency reads this attribute on every unlabelled step, and
        # on_train_start echoes it to the log as cldice_cons_beta=2.0.
        self.cldice_cons_beta = float(type(self).consistency_beta)
