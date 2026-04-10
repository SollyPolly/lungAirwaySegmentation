"""Tests for spatial preprocessing helpers.

This file should verify:
- crop-box creation from binary masks
- crop slices and cropped shapes
- affine updates after cropping
- alignment and shape validation on toy examples

Keep spatial math tests here so that geometry bugs are caught early.
"""

import numpy as np
import pytest

from lung_airway_segmentation.preprocessing.geometry import (
    affine_after_crop,
    crop_box_from_mask,
    crop_box_to_slices,
    crop_volume,
    normalize_margin,
    validate_3d_shape,
)


def test_validate_3d_shape_returns_shape_for_3d_volume():
    volume = np.zeros((3, 4, 5), dtype=np.float32)

    shape = validate_3d_shape(volume, "CT")

    assert shape == (3, 4, 5)


def test_validate_3d_shape_raises_for_non_3d_volume():
    volume = np.zeros((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="must be 3D"):
        validate_3d_shape(volume, "CT")


def test_normalize_margin_expands_integer_to_3d_tuple():
    assert normalize_margin(5) == (5, 5, 5)


def test_normalize_margin_keeps_three_item_margin():
    assert normalize_margin((1, 2, 3)) == (1, 2, 3)


def test_normalize_margin_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        normalize_margin((1, -1, 3))


def test_crop_box_from_mask_returns_expected_box():
    mask = np.zeros((5, 6, 7), dtype=np.uint8)
    mask[1:3, 2:5, 4:6] = 1

    crop_box = crop_box_from_mask(mask, crop_margin=1)

    assert crop_box == ((0, 4), (1, 6), (3, 7))


def test_crop_box_from_mask_clips_margin_to_volume_bounds():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[0:2, 0:2, 0:2] = 1

    crop_box = crop_box_from_mask(mask, crop_margin=3)

    assert crop_box == ((0, 5), (0, 5), (0, 5))


def test_crop_box_from_mask_raises_for_empty_mask():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="Mask is empty"):
        crop_box_from_mask(mask)


def test_crop_box_to_slices_and_crop_volume_match_numpy_slicing():
    volume = np.arange(5 * 6 * 7).reshape(5, 6, 7)
    crop_box = ((1, 4), (2, 6), (3, 7))

    slices = crop_box_to_slices(crop_box)
    cropped = crop_volume(volume, crop_box)

    expected = volume[1:4, 2:6, 3:7]

    assert slices == (slice(1, 4), slice(2, 6), slice(3, 7))
    assert np.array_equal(cropped, expected)


def test_affine_after_crop_updates_world_translation():
    affine = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    crop_box = ((1, 5), (2, 6), (3, 7))

    cropped_affine = affine_after_crop(affine, crop_box)

    expected = np.array(
        [
            [2.0, 0.0, 0.0, 2.0],
            [0.0, 3.0, 0.0, 6.0],
            [0.0, 0.0, 4.0, 12.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.array_equal(cropped_affine, expected)
