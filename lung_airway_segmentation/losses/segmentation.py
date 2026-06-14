"""Supervised segmentation losses for the baseline airway model.

This module contains the primary optimization objective used for binary airway
segmentation. The current implementation combines voxel-wise binary
cross-entropy on logits with Dice loss on sigmoid-normalized predictions to
balance stable optimization against overlap quality.
"""

import torch
import torch.nn as nn
from monai.losses.dice import DiceLoss

from lung_airway_segmentation.losses.topology import (
    persistent_homology_loss,
    soft_cbdice_loss,
    soft_cldice_loss,
)


class CombinedSegmentationLoss(nn.Module):
    """Weighted sum of BCE-with-logits, Dice, (optional) soft clDice, and (optional)
    soft cbDice losses.

    BCE + Dice optimise volumetric overlap, which is dominated by thick proximal
    airways. The optional clDice term adds a centerline (topology) objective that
    weights thin distal branches equally — set ``cldice_weight`` > 0 to enable it.
    ``cbDice`` (``cbdice_weight`` > 0) is the radius-aware variant (Shi et al., MICCAI
    2024): it adds vessel-calibre / wall-thickness sensitivity that clDice lacks (the
    loss-level lever against fat-walled distal predictions). Both weights are mutable
    attributes so an engine can ramp them across epochs (soft skeletons of early-
    training masks are noisy).
    """

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        positive_class_weight=1.0,
        cldice_weight=0.0,
        cldice_iterations=10,
        cbdice_weight=0.0,
        cbdice_iterations=10,
        topo_weight=0.0,
    ):
        super().__init__()
        if float(positive_class_weight) <= 0.0:
            raise ValueError("positive_class_weight must be positive.")
        if float(cldice_weight) < 0.0:
            raise ValueError("cldice_weight must be non-negative.")
        if int(cldice_iterations) < 1:
            raise ValueError("cldice_iterations must be >= 1.")
        if float(cbdice_weight) < 0.0:
            raise ValueError("cbdice_weight must be non-negative.")
        if int(cbdice_iterations) < 1:
            raise ValueError("cbdice_iterations must be >= 1.")
        if float(topo_weight) < 0.0:
            raise ValueError("topo_weight must be non-negative.")

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(positive_class_weight)])
        )
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = float(cldice_weight)
        self.cldice_iterations = int(cldice_iterations)
        # cbDice (centerline boundary Dice, Shi et al. MICCAI 2024) — radius-aware
        # topology loss; mutable so the engine can ramp it like clDice. 0.0 = off.
        self.cbdice_weight = float(cbdice_weight)
        self.cbdice_iterations = int(cbdice_iterations)
        # EXPERIMENTAL persistent-homology term; 0.0 = off (default). Ramped by the
        # engine like clDice. See losses/topology.py::persistent_homology_loss.
        self.topo_weight = float(topo_weight)

    def forward(self, logits, targets):
        """Compute the weighted segmentation loss for logits and target masks."""
        targets = targets.float()

        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total_loss = self.bce_weight * bce + self.dice_weight * dice

        # sigmoid is shared by the clDice and (experimental) topology terms.
        probabilities = None
        if self.cldice_weight > 0.0:
            probabilities = torch.sigmoid(logits)
            cldice = soft_cldice_loss(probabilities, targets, self.cldice_iterations)
            total_loss = total_loss + self.cldice_weight * cldice

        if self.cbdice_weight > 0.0:
            if probabilities is None:
                probabilities = torch.sigmoid(logits)
            cbdice = soft_cbdice_loss(probabilities, targets, self.cbdice_iterations)
            total_loss = total_loss + self.cbdice_weight * cbdice

        if self.topo_weight > 0.0:  # EXPERIMENTAL — off by default (topo_weight=0.0)
            if probabilities is None:
                probabilities = torch.sigmoid(logits)
            topo = persistent_homology_loss(probabilities, targets)
            total_loss = total_loss + self.topo_weight * topo

        return total_loss
