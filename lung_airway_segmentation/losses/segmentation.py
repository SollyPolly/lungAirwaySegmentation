"""Supervised segmentation losses for the baseline airway model.

This module contains the primary optimization objective used for binary airway
segmentation. The current implementation combines voxel-wise binary
cross-entropy on logits with Dice loss on sigmoid-normalized predictions to
balance stable optimization against overlap quality.
"""

import torch
import torch.nn as nn
from monai.losses.dice import DiceLoss

from lung_airway_segmentation.losses.topology import soft_cldice_loss


class CombinedSegmentationLoss(nn.Module):
    """Weighted sum of BCE-with-logits, Dice, and (optional) soft clDice losses.

    BCE + Dice optimise volumetric overlap, which is dominated by thick proximal
    airways. The optional clDice term adds a centerline (topology) objective that
    weights thin distal branches equally — set ``cldice_weight`` > 0 to enable it.
    ``cldice_weight`` is a mutable attribute so an engine can ramp it across epochs
    (soft skeletons of early-training masks are noisy).
    """

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        positive_class_weight=1.0,
        cldice_weight=0.0,
        cldice_iterations=10,
    ):
        super().__init__()
        if float(positive_class_weight) <= 0.0:
            raise ValueError("positive_class_weight must be positive.")
        if float(cldice_weight) < 0.0:
            raise ValueError("cldice_weight must be non-negative.")
        if int(cldice_iterations) < 1:
            raise ValueError("cldice_iterations must be >= 1.")

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(positive_class_weight)])
        )
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = float(cldice_weight)
        self.cldice_iterations = int(cldice_iterations)

    def forward(self, logits, targets):
        """Compute the weighted segmentation loss for logits and target masks."""
        targets = targets.float()

        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total_loss = self.bce_weight * bce + self.dice_weight * dice

        if self.cldice_weight > 0.0:
            probabilities = torch.sigmoid(logits)
            cldice = soft_cldice_loss(probabilities, targets, self.cldice_iterations)
            total_loss = total_loss + self.cldice_weight * cldice

        return total_loss
