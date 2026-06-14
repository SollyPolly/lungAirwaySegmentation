"""Prediction cleanup and heuristic filtering."""

import numpy as np
import torch
from scipy import ndimage


def binarize_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Apply sigmoid then threshold to convert raw model logits to a binary mask."""
    return (torch.sigmoid(logits) >= threshold).float()


def keep_largest_connected_component(
    binary_mask: np.ndarray,
    connectivity: int = 6,
) -> np.ndarray:
    """Keep only the largest connected component of a 3D binary mask.

    This can remove isolated false-positive blobs, but it can also remove true
    airway branches when the prediction is disconnected. Preserve and evaluate
    the raw prediction before adopting this as routine postprocessing.
    """
    if binary_mask.ndim != 3:
        raise ValueError(f"Expected a 3D binary mask, got shape {binary_mask.shape}.")
    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be one of 6, 18, or 26.")

    foreground = binary_mask > 0
    connectivity_rank = {6: 1, 18: 2, 26: 3}[connectivity]
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity_rank)
    labeled, num_components = ndimage.label(foreground, structure=structure)
    if num_components == 0:
        return np.zeros_like(binary_mask)

    component_sizes = np.bincount(labeled.ravel())[1:]
    largest = int(np.argmax(component_sizes)) + 1
    return (labeled == largest).astype(binary_mask.dtype)


def _superior_axis_and_sign(affine: np.ndarray) -> tuple[int, int]:
    """Array axis (and sign) that maps to world-superior (+S).

    ``sign == +1`` means a higher index along that axis is more superior. Derived from
    the affine so it is correct for any file orientation (ATM'22 is loaded in native
    orientation, AeroPath/predict_atm in RAS+).
    """
    superior_row = np.asarray(affine, dtype=np.float64)[2, :3]  # world-z per voxel axis
    axis = int(np.argmax(np.abs(superior_row)))
    sign = 1 if superior_row[axis] >= 0 else -1
    return axis, sign


def keep_component_containing_trachea(
    binary_mask: np.ndarray,
    connectivity: int = 6,
    *,
    affine: np.ndarray | None = None,
    superior_axis: int | None = None,
    superior_sign: int = 1,
    superior_fraction: float = 0.25,
    central_fraction: float = 0.5,
    fallback_to_largest: bool = True,
) -> np.ndarray:
    """Keep the connected component anchored at the trachea, not merely the largest.

    The trachea is the airway in the *superior* slab of the volume, near the in-plane
    centre. We keep the largest component that has a presence in that central-superior
    window. This is robust to the CT table / positioning board — a large, peripheral,
    air-density slab that ``keep_largest_connected_component`` otherwise selects when the
    airway tree fragments (the board outsizes the broken tree).

    The superior axis is taken from ``affine`` (recommended — handles arbitrary file
    orientation) or from ``superior_axis``/``superior_sign``; with neither, it defaults
    to the last array axis (RAS+ convention). Falls back to the largest component when no
    component reaches the central-superior window (e.g. the trachea was not predicted).
    """
    if binary_mask.ndim != 3:
        raise ValueError(f"Expected a 3D binary mask, got shape {binary_mask.shape}.")
    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be one of 6, 18, or 26.")

    foreground = binary_mask > 0
    if not foreground.any():
        return np.zeros_like(binary_mask)

    connectivity_rank = {6: 1, 18: 2, 26: 3}[connectivity]
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity_rank)
    labeled, num_components = ndimage.label(foreground, structure=structure)
    if num_components == 1:
        return (labeled == 1).astype(binary_mask.dtype)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    def _largest() -> np.ndarray:
        return (labeled == int(np.argmax(sizes))).astype(binary_mask.dtype)

    if affine is not None:
        superior_axis, superior_sign = _superior_axis_and_sign(affine)
    if superior_axis is None:
        superior_axis, superior_sign = binary_mask.ndim - 1, 1

    shape = binary_mask.shape
    n_sup = max(1, int(round(shape[superior_axis] * superior_fraction)))
    window = [slice(None), slice(None), slice(None)]
    window[superior_axis] = (
        slice(shape[superior_axis] - n_sup, shape[superior_axis])
        if superior_sign >= 0
        else slice(0, n_sup)
    )
    for ax in range(3):
        if ax == superior_axis:
            continue
        lo = int(round(shape[ax] * (1.0 - central_fraction) / 2.0))
        hi = int(round(shape[ax] * (1.0 + central_fraction) / 2.0))
        window[ax] = slice(lo, hi)

    present = np.unique(labeled[tuple(window)])
    present = present[present != 0]
    if present.size == 0:
        return _largest() if fallback_to_largest else np.zeros_like(binary_mask)

    # Largest component that reaches the central-superior (trachea) window.
    trachea = int(present[np.argmax(sizes[present])])
    return (labeled == trachea).astype(binary_mask.dtype)
