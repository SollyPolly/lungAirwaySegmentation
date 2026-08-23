"""Isolated Dataset126 K1 class-balanced voxel-MSE Mean-Teacher arm.

WHY THIS ARM EXISTS, AND WHY IT WAS PREVIOUSLY DECLINED
-------------------------------------------------------
The primary claim is that *geometry-aware* consistency beats *per-voxel*
consistency. As of 2026-08-23 that rests on soft-clDice(w=0.10) beating
PlainMSE(w=0.10) by +0.0122 ATM'22 Score (t=+4.17, TD 20/20) at identical
nominal weight. The open objection is that PlainMSE is INERT --
``consistency_fraction = 0.000`` on all 500 logged epochs at both 0.10 and
0.30 -- so that contrast shows a working loss beating a broken one, not that
geometry is the operative variable. Any consistency term that escaped the
starved-signal null might have done as well.

Class-balanced MSE is the only non-geometric loss known to escape that null
(pilot: fraction ~1-3%). It therefore closes the objection, and it is the
third point of the ladder:

    plain voxel MSE   does NOT escape the null   dATM +0.0022 (t=+1.32, ns)
    class-balanced    escapes the null           <- this arm
    soft-clDice       escapes the null           dATM +0.0144 (t=+5.43)

``nnUNetTrainer_MeanTeacher_VoxelMSE_...`` declined a class-balanced arm
because at w_max 0.1 or 0.3 it "cannot confirm the collapse story", the pilot
collapse having occurred at 1.0 and 0.3. That reasoning was correct for the
goal it had. The goal is now different: this arm is not trying to replicate
the collapse, it is completing the geometry contrast. At w_max = 0.10 it is a
one-variable change in BOTH directions -- against PlainMSE(w=0.10) it isolates
balancing, against SoftCLDice(w=0.10) it isolates geometry with the null
already escaped. Run at 1.0 it would differ from the clDice arm in geometry
AND weight, which is the confound the earlier note rightly refused.

ON THE PILOT COLLAPSE
---------------------
The recorded collapse (pseudo-Dice 0.96 -> 0.28 at w_max=1.0; val Dice 0.19 at
0.3) is a 2026-07-16 ten-epoch observation on the superseded Dataset122 split.
No log, checkpoint or scored output for it survives in this repository, so it
cannot be reproduced or cited from anything on disk. It is also confounded:
the guaranteed ``1 GT + 1 U`` sampler landed 2026-07-20, FOUR DAYS LATER, and
Dataset122 fold 0 trains ~88 cases of which ~16 are labelled at batch size 2,
so roughly two thirds of its batches carried no supervised gradient at all.
A positive-feedback teacher collapse is what one would predict from a missing
supervised anchor alone, with or without the ~100x foreground amplification
that the collapse is usually attributed to. That pilot's learning rate was
also 1e-2 against this envelope's 1e-3.

Consequently BOTH outcomes of this run are reportable:
  * no collapse -> the geometry contrast is closed, and the "balancing
    collapses the teacher" claim is narrowed to the pilot's sampling regime;
  * collapse    -> the claim is finally reproducible under the protocol that
    is actually reported, which is strictly better than the present citation.

WHAT "balanced" DOES
--------------------
``nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring._consistency`` mode
``"balanced"``: partition every voxel by the teacher's airway belief at
``partition_threshold``, average the airway-channel MSE within each side, and
combine 50/50. Equal weight to the sparse airway region and the background, so
the ~0.29%-foreground airway voxels are not diluted ~100:1. That is a
COUNT-based rebalancing. soft-clDice is a GEOMETRY-based rebalancing of the
same imbalance -- skeletonisation strips the r^2 area weighting so a
bronchiole carries comparable weight to a main bronchus. The scientific
content of this arm is the difference between those two ways of answering the
same problem: count-based balancing amplifies every foreground voxel equally,
including the unreliable boundary band where the teacher's distal guesses are
worst, whereas the skeleton concentrates on centreline.

Everything else -- warm start, sampler, 1 GT + 1 U batching, augmentation,
ramp schedule, EMA update and final-EMA deployment -- is inherited unchanged
from the shared K1 two-stream base, and the existing no-consistency
continuation remains the matched control.

PRE-REGISTERED PREDICTION (written before the run, 2026-08-23): escapes the
null (``consistency_fraction`` a few %), does NOT collapse, and lands BETWEEN
the two on dATM vs control -- roughly +0.005 to +0.010 -- with a worse
``tprec`` cost per unit of TD gained than clDice buys.
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

DIAGNOSTICS_VERSION = "class_balanced_voxel_mse_k1_v1"


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_BalancedMSE_NoDeepSupervision_NoMirroring(
    _K1Base
):
    """Class-balanced voxel-MSE consistency at the SoftCLDice arm's weight."""

    protocol_exposure = "K1"
    expected_labelled_per_step = 1
    expected_unlabelled_per_step = 1
    expected_local_batch_size = 2
    diagnostic_objective = "class_balanced_voxel_mse_matched_weight"
    # Pinned here rather than inherited silently, so the launch preflight can
    # assert the operative knobs off the CLASS before any GPU time is spent.
    configured_consistency_mode = "balanced"
    configured_consistency_max = 0.1
    configured_partition_threshold = 0.5

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # nnUNetTrainer reflects over this exact signature; do not replace it
        # with *args/**kwargs.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.consistency_mode = type(self).configured_consistency_mode
        self.consistency_max = type(self).configured_consistency_max
        self.partition_threshold = type(self).configured_partition_threshold
