"""Supervised segmentation losses for the baseline airway model.

This module contains the primary optimization objective used for binary airway
segmentation. The current implementation combines voxel-wise binary
cross-entropy on logits with Dice loss on sigmoid-normalized predictions to
balance stable optimization against overlap quality.
"""

import torch
import torch.nn as nn
from monai.losses.dice import DiceLoss


class CombinedSegmentationLoss(nn.Module):
    """Weighted sum of BCE-with-logits loss and Dice loss."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce_weight = bce_weight
        self.dice_weight =  dice_weight

    def forward(self, logits, targets):
        """Compute the weighted segmentation loss for logits and target masks."""
        targets = targets.float()

        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total_loss = self.bce_weight * bce + self.dice_weight * dice

        return total_loss
