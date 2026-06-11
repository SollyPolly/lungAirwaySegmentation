"""Topology-aware and geometry-aware losses.

Soft centerline Dice (clDice, Shit et al. 2021) — a differentiable surrogate for
centerline/tree-length overlap. Volumetric Dice is dominated by thick proximal
airways, so it barely penalises hedging on thin distal branches; clDice weights a
one-voxel branch the same as the trachea by operating on the (soft) skeleton.

Soft skeletonisation uses iterated soft morphology (min/max pooling), so the whole
thing is differentiable and runs on the GPU. Sanity-check against the hard
`metrics.topology.cldice_score_from_masks`: 1 - soft_cldice_loss should track it.

Reference: https://github.com/jocpae/clDice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    """3D soft erosion: min-pool implemented as a negated max-pool."""
    return -F.max_pool3d(-image, kernel_size=3, stride=1, padding=1)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    """3D soft dilation: max-pool with unit stride to preserve shape."""
    return F.max_pool3d(image, kernel_size=3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    """Soft morphological opening (erode then dilate)."""
    return soft_dilate(soft_erode(image))


def soft_skeleton(image: torch.Tensor, iterations: int) -> torch.Tensor:
    """Differentiable soft skeleton of a [0, 1] mask, shape (B, C, D, H, W).

    ``iterations`` bounds the maximum structure radius that can be fully thinned;
    set it >= the largest expected airway radius in voxels (~10 at this resolution).
    """
    if image.ndim != 5:
        raise ValueError(f"soft_skeleton expects a 5D (B,C,D,H,W) tensor, got {tuple(image.shape)}.")
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(int(iterations)):
        image = soft_erode(image)
        opened = soft_open(image)
        delta = F.relu(image - opened)
        # add the newly exposed centerline voxels, without double-counting
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    iterations: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Return ``1 - soft clDice`` averaged over the batch.

    ``probabilities`` and ``targets`` are (B, 1, D, H, W) in [0, 1]
    (probabilities = sigmoid(logits); targets = binary mask).
    """
    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probabilities and targets must match: {tuple(probabilities.shape)} "
            f"!= {tuple(targets.shape)}"
        )

    skeleton_pred = soft_skeleton(probabilities, iterations)
    skeleton_true = soft_skeleton(targets, iterations)

    batch = probabilities.shape[0]
    skel_pred = skeleton_pred.reshape(batch, -1)
    skel_true = skeleton_true.reshape(batch, -1)
    pred = probabilities.reshape(batch, -1)
    true = targets.reshape(batch, -1)

    # topology precision: predicted centerline that lies inside the target
    t_prec = (torch.sum(skel_pred * true, dim=1) + smooth) / (torch.sum(skel_pred, dim=1) + smooth)
    # topology sensitivity: target centerline that lies inside the prediction
    t_sens = (torch.sum(skel_true * pred, dim=1) + smooth) / (torch.sum(skel_true, dim=1) + smooth)

    cl_dice = 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens)
    return cl_dice.mean()


class SoftClDiceLoss(nn.Module):
    """Module wrapper around :func:`soft_cldice_loss` (expects probabilities)."""

    def __init__(self, iterations: int = 10, smooth: float = 1.0):
        super().__init__()
        if int(iterations) < 1:
            raise ValueError("clDice iterations must be >= 1.")
        self.iterations = int(iterations)
        self.smooth = float(smooth)

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return soft_cldice_loss(probabilities, targets.float(), self.iterations, self.smooth)
