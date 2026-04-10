"""Tests for CT intensity preprocessing helpers.

This file should verify:
- HU clipping behavior
- normalization ranges
- edge cases like invalid windows or constant arrays

Keep intensity tests separate from geometry tests to avoid mixed concerns.
"""

import numpy as np
import pytest

from lung_airway_segmentation.preprocessing.intensity import (
    clip_ct_to_hu_window,
    normalize_ct,
)


def test_clip_ct_to_hu_window_clips_values_at_both_bounds():
    volume = np.array([-1200.0, -1024.0, 0.0, 600.0, 1000.0], dtype=np.float32)

    clipped = clip_ct_to_hu_window(volume, hu_window=(-1024.0, 600.0))

    expected = np.array([-1024.0, -1024.0, 0.0, 600.0, 600.0], dtype=np.float32)
    assert np.array_equal(clipped, expected)


def test_clip_ct_to_hu_window_returns_float32():
    volume = np.array([-1200, 0, 1000], dtype=np.int16)

    clipped = clip_ct_to_hu_window(volume, hu_window=(-1024.0, 600.0))

    assert clipped.dtype == np.float32


def test_clip_ct_to_hu_window_rejects_invalid_window():
    volume = np.array([0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="greater than the lower bound"):
        clip_ct_to_hu_window(volume, hu_window=(100.0, 100.0))


def test_normalize_ct_maps_window_bounds_to_zero_and_one():
    volume = np.array([-1024.0, 600.0], dtype=np.float32)

    normalized = normalize_ct(volume, hu_window=(-1024.0, 600.0))

    expected = np.array([0.0, 1.0], dtype=np.float32)
    assert np.allclose(normalized, expected)


def test_normalize_ct_keeps_values_inside_zero_one_range():
    volume = np.array([-1200.0, -1024.0, 0.0, 600.0, 1000.0], dtype=np.float32)

    normalized = normalize_ct(volume, hu_window=(-1024.0, 600.0))

    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_ct_rejects_invalid_window():
    volume = np.array([0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="greater than the lower bound"):
        normalize_ct(volume, hu_window=(200.0, 100.0))
