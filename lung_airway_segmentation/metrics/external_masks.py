"""Shared measurements for external hard-mask predictions."""

import numpy as np

from lung_airway_segmentation.metrics.topology import (
    _foreground_slices,
    _largest_connected_component,
    _skeletonize,
)


# Foreground wall-distance bins in voxels. These are wall-shell measurements,
# not anatomical branch-generation or distal-airway categories.
RADIUS_BINS = [
    ("r=1 (wall shell; voxel EDT)", 0.5, 1.5),
    ("r=2 (wall distance; voxel EDT)", 1.5, 2.5),
    ("r=3 (wall distance; voxel EDT)", 2.5, 3.5),
    ("r=4-5 (wall distance; voxel EDT)", 3.5, 5.5),
    ("r>=6 (thick core; voxel EDT)", 5.5, 1e9),
]


def gt_centerline(gt: np.ndarray) -> tuple[tuple[slice, ...], np.ndarray, int]:
    """Return the cropped skeleton of the largest ground-truth component."""
    component = _largest_connected_component(gt)
    slices = _foreground_slices(component)
    skeleton = _skeletonize(component[slices])
    return slices, skeleton, int(skeleton.sum())


def cheap_metrics(
    predicted: np.ndarray,
    gt: np.ndarray,
    gt_sum: int,
    td_slices: tuple[slice, ...],
    gt_skeleton: np.ndarray,
    gt_skeleton_sum: int,
) -> tuple[float, float, float]:
    """Return Dice, tree-length detection, and voxel precision."""
    pred_sum = int(predicted.sum())
    true_positive = int((predicted & gt).sum())
    dice = 2 * true_positive / ((pred_sum + gt_sum) or 1)
    precision = true_positive / (pred_sum or 1)
    tree_length = (
        int((gt_skeleton & predicted[td_slices]).sum()) / gt_skeleton_sum
        if gt_skeleton_sum
        else 1.0
    )
    return float(dice), float(tree_length), float(precision)
