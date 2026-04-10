"""Geometry helpers for spatial preprocessing operations.

This module contains the small spatial building blocks used during case
preprocessing: validating 3D volumes, normalizing crop margins, deriving a
bounding box from a mask, converting crop boxes into slices, applying crops,
and updating the affine after spatial cropping.
"""

import numpy as np

from lung_airway_segmentation.schemas import Shape3D, CropBox3D
from lung_airway_segmentation.settings import DEFAULT_CROP_MARGIN

def validate_3d_shape(volume, name) -> Shape3D:
    if volume.ndim != 3:
        raise ValueError(f"{name} must be 3D, but got shape {volume.shape}")
    return (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))

def normalize_margin(margin: int | tuple[int, int, int] ) -> Shape3D:
    if isinstance(margin, int):
        if margin < 0:
            raise ValueError(f"Margin must be non-negative. Margin: {margin}")
        return (margin, margin, margin)
    
    if len(margin) != 3:
        raise ValueError(f"Margin must be an int or a 3-item sequence. Margin: {margin}")
    
    normalzied = (int(margin[0]), int(margin[1]), int(margin[2]))

    if any(value < 0 for value in normalzied):
        raise ValueError(f"Margin values must be non-negative. Margin: {margin}")
    
    return normalzied


def crop_box_from_mask(mask, crop_margin=DEFAULT_CROP_MARGIN) -> CropBox3D:
    mask_shape = validate_3d_shape(mask, "mask")
    foreground = np.argwhere(mask > 0)

    if foreground.size == 0:
        raise ValueError("Mask is empty")
    
    margin_array = np.asarray(normalize_margin(crop_margin))
    mask_shape_array = np.asarray(mask_shape)

    # The crop is clipped to the original volume bounds after the margin is applied.
    starts = np.maximum(foreground.min(axis=0) - margin_array, 0)
    stops = np.minimum(foreground.max(axis=0) + margin_array + 1, mask_shape_array)

    return (
        (int(starts[0]), int(stops[0])),
        (int(starts[1]), int(stops[1])),
        (int(starts[2]), int(stops[2])),
    )

def crop_box_to_slices(crop_box):
    z_bound, y_bound, x_bound = crop_box
    return (
        slice(z_bound[0], z_bound[1]),
        slice(y_bound[0], y_bound[1]),
        slice(x_bound[0], x_bound[1])
    )

def crop_volume(volume, crop_box):
    return volume[crop_box_to_slices(crop_box)]

def affine_after_crop(affine, crop_box):
    crop_start = np.array([bounds[0] for bounds in crop_box])
    voxel_transition = np.eye(4)
    voxel_transition[:3, 3] = crop_start
    return affine @ voxel_transition
