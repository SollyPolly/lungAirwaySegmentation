"""Post-processing used by the active nnU-Net workflows."""

import numpy as np
from scipy import ndimage


def _superior_axis_and_sign(affine: np.ndarray) -> tuple[int, int]:
    """Return the array axis and direction corresponding to world-superior."""
    superior_row = np.asarray(affine, dtype=np.float64)[2, :3]
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
    """Return a lung bounding box with an affine-aware trachea extension."""
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
    """Keep the component anchored in the central-superior trachea window."""
    if binary_mask.ndim != 3:
        raise ValueError(f"Expected a 3D binary mask, got shape {binary_mask.shape}.")
    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be one of 6, 18, or 26.")

    foreground = binary_mask > 0
    if not foreground.any():
        return np.zeros_like(binary_mask)

    connectivity_rank = {6: 1, 18: 2, 26: 3}[connectivity]
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity_rank)
    labelled, num_components = ndimage.label(foreground, structure=structure)
    if num_components == 1:
        return (labelled == 1).astype(binary_mask.dtype)

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0

    def largest() -> np.ndarray:
        return (labelled == int(np.argmax(sizes))).astype(binary_mask.dtype)

    if affine is not None:
        superior_axis, superior_sign = _superior_axis_and_sign(affine)
    if superior_axis is None:
        superior_axis, superior_sign = binary_mask.ndim - 1, 1

    shape = binary_mask.shape
    superior_count = max(1, int(round(shape[superior_axis] * superior_fraction)))
    window = [slice(None), slice(None), slice(None)]
    window[superior_axis] = (
        slice(shape[superior_axis] - superior_count, shape[superior_axis])
        if superior_sign >= 0
        else slice(0, superior_count)
    )
    for axis in range(3):
        if axis == superior_axis:
            continue
        lo = int(round(shape[axis] * (1.0 - central_fraction) / 2.0))
        hi = int(round(shape[axis] * (1.0 + central_fraction) / 2.0))
        window[axis] = slice(lo, hi)

    present = np.unique(labelled[tuple(window)])
    present = present[present != 0]
    if present.size == 0:
        return largest() if fallback_to_largest else np.zeros_like(binary_mask)

    trachea_label = int(present[np.argmax(sizes[present])])
    return (labelled == trachea_label).astype(binary_mask.dtype)
