"""Isolated Dataset126 K1 plain voxel-MSE Mean-Teacher arms.

These exist to make one claim reportable that is currently only inferable. The
body asserts that geometry-aware consistency is numerically viable where
per-voxel consistency is not (Discussion, Conclusion). The evidence behind that
is a **ten-epoch pilot on the superseded Dataset122 split**, measured only by
``consistency_fraction`` -- a loss-magnitude ratio, explicitly not a gradient
fraction. Nothing was ever trained out, predicted or scored. These trainers turn
that observation into a scored outcome on the protocol actually reported.

They change exactly one thing against the SoftCLDice arm: ``consistency_mode``
moves from ``"cldice"`` to ``"plain"`` (Tarvainen & Valpola whole-patch MSE on
the airway probability). Warm start, sampler, one-GT/one-unlabelled batching,
augmentation, ramp schedule, EMA update and optimisation envelope are inherited
unchanged from the shared K1 two-stream base, and the existing no-consistency
continuation remains the matched control for both.

Two weights, because there is no single honest "matched" weight
-----------------------------------------------------------------
``1 - clDice`` lies in [0, 1] and runs ~0.3-0.6 early; whole-patch MSE on a
0.29%-foreground target is orders smaller. The parent trainer documents this
directly: voxel-MSE modes want ``w_max`` ~0.3 while the clDice mode wants 0.1.
So matching the nominal weight and matching the effective contribution are
different experiments, and neither alone is safe:

``...PlainMSE...``            ``w_max = 0.3``  -- the voxel-MSE scale this
    project's own notes prescribe. This is the PRIMARY arm, because it gives the
    baseline its documented operating point and cannot be dismissed as a
    crippled comparison.
``...PlainMSEMatchedWeight...``  ``w_max = 0.1`` -- identical nominal weight to
    the SoftCLDice arm, so the ONLY difference is the distance function. Closes
    the complementary objection that 0.3 changed two things at once.

A null at both weights is a much harder result to argue with than a null at
either. Note that ``w_max`` here is the *nominal* ceiling; what should be
compared across arms is the measured consistency contribution, not this number.

Class-balanced MSE is deliberately NOT included -- see the module note below.
"""

from __future__ import annotations

import torch

if __package__ == "nnunet_trainers":  # repository-local tests
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _K1Base,
    )
else:  # installed beside the parent trainers in nnU-Net
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _K1Base,
    )

# WHY THERE IS NO class-balanced ARM HERE.
#
# Class-balanced MSE is the obvious repair to the plain-MSE null, and the pilot
# showed it collapses the teacher (pseudo-Dice 0.96 -> 0.28 at w_max=1.0; val
# Dice 0.19 at 0.3). It is tempting to run it out. It should not be run as part
# of this contrast, because there is no weight at which it is a one-variable
# change:
#
#   * at w_max=0.1 or 0.3 it is not a replication of the pilot collapse, which
#     happened at 1.0 and 0.3 on a different split and duration, so it cannot
#     confirm the collapse story;
#   * at w_max=1.0 it differs from the SoftCLDice arm in BOTH geometry and
#     weight, so it cannot serve as the geometry contrast either.
#
# Methods already cites the collapse honestly, as a pilot observation at a
# different scale. Running a confounded third arm would weaken that, not
# strengthen it.

DIAGNOSTICS_VERSION = "plain_voxel_mse_k1_v1"


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_PlainMSE_NoDeepSupervision_NoMirroring(
    _K1Base
):
    """Whole-patch voxel-MSE consistency at the documented voxel-MSE weight."""

    protocol_exposure = "K1"
    expected_labelled_per_step = 1
    expected_unlabelled_per_step = 1
    expected_local_batch_size = 2
    diagnostic_objective = "plain_voxel_mse"
    configured_consistency_max = 0.3

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.consistency_mode = "plain"
        self.consistency_max = type(self).configured_consistency_max


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_PlainMSEMatchedWeight_NoDeepSupervision_NoMirroring(
    nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_PlainMSE_NoDeepSupervision_NoMirroring
):
    """As above, at the SoftCLDice arm's nominal weight, isolating geometry alone."""

    diagnostic_objective = "plain_voxel_mse_matched_weight"
    configured_consistency_max = 0.1
