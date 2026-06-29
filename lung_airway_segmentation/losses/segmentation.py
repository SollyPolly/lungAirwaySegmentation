"""Supervised segmentation losses for the baseline airway model.

This module contains the primary optimization objective used for binary airway
segmentation. The current implementation combines voxel-wise binary
cross-entropy on logits with Dice loss on sigmoid-normalized predictions to
balance stable optimization against overlap quality.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses.dice import DiceLoss
from scipy import ndimage

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

    ``calibre_weight_max`` > 1.0 enables **calibre/depth-aware BCE weighting** (GUL /
    WingsNet-style within-class reweighting; ATM'22 timi's centreline-distance weight):
    a per-voxel weight that is 1.0 on background and thick proximal airway and ramps up
    to ``calibre_weight_max`` on the thinnest distal voxels (local GT radius <= 1). This
    over-weights exactly the deepest branches clDice cannot "feel" — its sensitivity is
    skeleton-length-proportional, so short distal twigs barely move it — *on top of*
    clDice/cbDice, rather than tuning their weights. It applies to the dense BCE term on
    the TRAINING path only (``compute_components(weight_bce=True)``, set by ``forward``);
    validation keeps plain BCE so no full-volume EDT is added. ``calibre_weight_max=1.0``
    (default) = OFF / behaviour-preserving.
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
        calibre_weight_max=1.0,
        calibre_radius_voxels=3.0,
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
        if float(calibre_weight_max) < 1.0:
            raise ValueError("calibre_weight_max must be >= 1.0 (1.0 = off).")
        if float(calibre_radius_voxels) <= 1.0:
            raise ValueError("calibre_radius_voxels must be > 1.0.")

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
        # Calibre/depth-aware BCE voxel weighting (GUL-style); 1.0 = off (default).
        # Mutable so an engine could ramp it, though it is typically held constant.
        self.calibre_weight_max = float(calibre_weight_max)
        self.calibre_radius_voxels = float(calibre_radius_voxels)

    def _calibre_weight_map(self, targets):
        """Per-voxel BCE weight: 1.0 everywhere, foreground graded by local GT radius.

        The deepest / thinnest branches (EDT radius <= 1) reach ``calibre_weight_max``;
        radius >= ``calibre_radius_voxels`` stays at 1.0 (thick proximal airway and all
        background). This over-weights exactly the distal voxels clDice is blind to
        without changing the bulk (background) BCE scale, so a clean attribution vs the
        no-calibre run is only the distal-foreground gradient. Radius is the EDT of the
        GT mask on the current (training) patch — same per-patch cost / edge-cut caveat
        as the cbDice term (a branch cut by the patch face reads thinner, i.e. slightly
        over-weighted at the patch edge; benign for a weight).
        """
        target_np = targets.detach().cpu().numpy() > 0.5
        weights = np.ones(target_np.shape, dtype=np.float32)
        reference = float(self.calibre_radius_voxels)
        span = reference - 1.0
        boost = float(self.calibre_weight_max) - 1.0
        for index in range(target_np.shape[0]):
            mask = target_np[index, 0]
            if not mask.any():
                continue
            radius = ndimage.distance_transform_edt(mask)
            fraction = np.clip((reference - radius) / span, 0.0, 1.0)
            weights[index, 0] = np.where(mask, 1.0 + boost * fraction, 1.0)
        return torch.from_numpy(weights).to(device=targets.device, dtype=targets.dtype)

    def compute_components(
        self,
        logits,
        targets,
        force_cldice=False,
        force_cbdice=False,
        voxel_spacing=None,
        weight_bce=False,
    ):
        """Return ``(total_loss, components)`` computing each term at most once.

        By default a soft term is only evaluated when its weight is > 0 (the
        training path — skips clDice/cbDice during warm-up). Validation passes
        ``force_cldice`` / ``force_cbdice`` to evaluate them for *logging* even at
        weight 0, while still folding them into ``total_loss`` only at their
        current weight. ``components`` holds the unweighted terms (BCE, Dice, and
        the soft clDice/cbDice diagnostics) as scalar tensors. Centralising the
        term computation here lets the validation loop log the diagnostics without
        a second forward pass (the expensive cbDice scipy EDT runs once, not twice).

        ``weight_bce`` (set by ``forward`` on the training path) applies the
        calibre/depth-aware per-voxel BCE weight when ``calibre_weight_max`` > 1.0.
        Validation leaves it ``False`` so the logged BCE stays plain (no per-case EDT).
        """
        targets = targets.float()

        # Dense BCE term. On the training path (weight_bce) with calibre weighting on,
        # up-weight distal voxels per the GT-radius map; otherwise plain pos_weight BCE.
        if weight_bce and self.calibre_weight_max > 1.0:
            weight_map = self._calibre_weight_map(targets)
            bce_per_voxel = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.bce_loss.pos_weight, reduction="none"
            )
            bce = (weight_map * bce_per_voxel).mean()
        else:
            bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        components = {"bce": bce, "dice": dice}

        # sigmoid is shared by the clDice, cbDice, and (experimental) topology terms.
        probabilities = None
        if force_cldice or self.cldice_weight > 0.0:
            probabilities = torch.sigmoid(logits)
            cldice = soft_cldice_loss(probabilities, targets, self.cldice_iterations)
            components["soft_cldice"] = cldice
            total_loss = total_loss + self.cldice_weight * cldice

        if force_cbdice or self.cbdice_weight > 0.0:
            if probabilities is None:
                probabilities = torch.sigmoid(logits)
            cbdice = soft_cbdice_loss(
                probabilities,
                targets,
                self.cbdice_iterations,
                voxel_spacing=voxel_spacing,
            )
            components["soft_cbdice"] = cbdice
            total_loss = total_loss + self.cbdice_weight * cbdice

        if self.topo_weight > 0.0:  # EXPERIMENTAL — off by default (topo_weight=0.0)
            if probabilities is None:
                probabilities = torch.sigmoid(logits)
            topo = persistent_homology_loss(probabilities, targets)
            total_loss = total_loss + self.topo_weight * topo

        return total_loss, components

    def forward(self, logits, targets, voxel_spacing=None):
        """Compute the weighted segmentation loss for logits and target masks.

        This is the TRAINING entry point, so calibre-aware BCE weighting is enabled
        (``weight_bce=True``); it is a no-op when ``calibre_weight_max == 1.0``.
        """
        total_loss, _ = self.compute_components(
            logits,
            targets,
            voxel_spacing=voxel_spacing,
            weight_bce=True,
        )
        return total_loss
