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


def _voxel_line(start: tuple[int, int, int], end: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    """Voxel coordinates of a thin (26-connected) straight line from ``start`` to ``end``.

    Returns a fancy-index tuple of integer arrays. The number of samples is the
    Chebyshev (max-norm) distance + 1, so the line has no gaps under 26-connectivity.
    """
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(end, dtype=np.float64)
    steps = int(np.max(np.abs(p1 - p0)))
    coords = np.rint(np.linspace(p0, p1, steps + 1)).astype(np.intp)
    return tuple(coords.T)


def _reachable_from(mask: np.ndarray, seed: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Union of the connected components of ``mask`` that overlap ``seed``."""
    labeled, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    seed_labels = np.unique(labeled[seed & (labeled > 0)])
    return np.isin(labeled, seed_labels)


def reconnect_components_to_trachea(
    binary_mask: np.ndarray,
    connectivity: int = 6,
    *,
    max_gap_voxels: float = 2.0,
    max_passes: int = 3,
    anchor_mask: np.ndarray | None = None,
    **trachea_kwargs,
) -> np.ndarray:
    """Bridge near-touching components back to the trachea tree, then keep that tree.

    ``keep_component_containing_trachea`` *deletes* every component that is not connected
    to the trachea — including true distal branches the model detected but that partial
    volume / a one-voxel dropout split off the tree (our raw-TD > TLD gap). This instead
    draws a thin voxel bridge from any component whose nearest gap to the trachea-connected
    set is ``<= max_gap_voxels``, iterating up to ``max_passes`` so chains of broken twigs
    reconnect, then returns the trachea-connected tree. Far-away false-positive blobs (gap
    above the threshold) are still dropped. This is the deeptree_damo "breakage connection"
    lever, applied as inference-time postprocessing (no retrain).

    The trachea anchor is found exactly as ``keep_component_containing_trachea`` (pass the
    same ``affine``/``superior_*`` kwargs through ``trachea_kwargs``); ``anchor_mask`` lets
    a caller supply the anchor directly (used in tests). Gaps are in voxel units.
    """
    if binary_mask.ndim != 3:
        raise ValueError(f"Expected a 3D binary mask, got shape {binary_mask.shape}.")
    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be one of 6, 18, or 26.")
    if max_gap_voxels < 0:
        raise ValueError("max_gap_voxels must be >= 0.")

    foreground = binary_mask > 0
    if not foreground.any():
        return np.zeros_like(binary_mask)

    connectivity_rank = {6: 1, 18: 2, 26: 3}[connectivity]
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity_rank)

    if anchor_mask is None:
        anchor = keep_component_containing_trachea(
            foreground.astype(np.uint8, copy=False), connectivity=connectivity, **trachea_kwargs
        ) > 0
    else:
        anchor = (anchor_mask > 0) & foreground
    if not anchor.any():
        return np.zeros_like(binary_mask)

    bridged = foreground.copy()
    connected = _reachable_from(bridged, anchor, structure)
    for _ in range(max_passes):
        remaining = bridged & ~connected
        if not remaining.any():
            break
        # Distance (and nearest-voxel index) from every voxel to the connected tree.
        distance, indices = ndimage.distance_transform_edt(~connected, return_indices=True)
        labeled, num = ndimage.label(remaining, structure=structure)
        added = False
        for label_id in range(1, num + 1):
            component = labeled == label_id
            component_distance = np.where(component, distance, np.inf)
            nearest = np.unravel_index(int(np.argmin(component_distance)), component.shape)
            gap = float(distance[nearest])
            if 0.0 < gap <= max_gap_voxels:
                target = (int(indices[0][nearest]), int(indices[1][nearest]), int(indices[2][nearest]))
                bridged[_voxel_line(nearest, target)] = True
                added = True
        if not added:
            break
        connected = _reachable_from(bridged, anchor, structure)

    return connected.astype(binary_mask.dtype)


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


def lung_bbox_slices(
    lung_mask: np.ndarray,
    *,
    affine: np.ndarray | None = None,
    margin_voxels: int = 8,
    superior_margin_voxels: int = 120,
) -> tuple[slice, slice, slice] | None:
    """Return an inference-valid lung bounding box with a superior trachea extension.

    The box is derived only from a CT-derived lung mask. ``margin_voxels`` is used on
    the in-plane faces and the inferior face; the superior face instead receives
    ``superior_margin_voxels`` so that the cervical trachea is not clipped.  The
    affine determines which array axis/direction is world-superior.  With no affine,
    the final array axis increasing toward superior is used (the ATM native fallback).

    ``None`` is returned for an empty lung mask, allowing callers to fail closed or
    preserve the uncropped volume explicitly.  Axis projections are used instead of
    ``np.where`` so full CT volumes do not allocate three large coordinate arrays.
    """
    if lung_mask.ndim != 3:
        raise ValueError(f"Expected a 3D lung mask, got shape {lung_mask.shape}.")
    if margin_voxels < 0 or superior_margin_voxels < 0:
        raise ValueError("Lung ROI margins must be >= 0.")
    if not np.any(lung_mask):
        return None

    if affine is None:
        superior_axis, superior_sign = lung_mask.ndim - 1, 1
    else:
        superior_axis, superior_sign = _superior_axis_and_sign(affine)

    bounds: list[slice] = []
    for axis, size in enumerate(lung_mask.shape):
        other_axes = tuple(candidate for candidate in range(3) if candidate != axis)
        occupied = np.flatnonzero(np.any(lung_mask, axis=other_axes))
        lo, hi = int(occupied[0]), int(occupied[-1])
        lower_margin = upper_margin = int(margin_voxels)
        if axis == superior_axis:
            if superior_sign > 0:
                upper_margin = int(superior_margin_voxels)
            else:
                lower_margin = int(superior_margin_voxels)
        bounds.append(slice(max(0, lo - lower_margin), min(size, hi + upper_margin + 1)))
    return tuple(bounds)  # type: ignore[return-value]


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
