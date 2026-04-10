"""CT intensity-windowing and normalization helpers.

This module contains the intensity-only transforms used during preprocessing.
It clips CT values to a requested HU window and normalizes cropped CT volumes
into the training range without performing any spatial operations.
"""
import numpy as np

from lung_airway_segmentation.settings import DEFAULT_HU_WINDOW

def clip_ct_to_hu_window(volume, hu_window=DEFAULT_HU_WINDOW):
    lower, upper = hu_window
    if upper <= lower:
        raise ValueError("The upper bound of HU window must be greater than the lower bound.")
    
    return np.clip(volume, lower, upper).astype(np.float32, copy=False)

def normalize_ct(volume, hu_window=DEFAULT_HU_WINDOW):
    lower, upper = hu_window
    if upper <= lower:
        raise ValueError("The upper bound of HU window must be greater than the lower bound.")

    clipped = clip_ct_to_hu_window(volume, hu_window)
    normalized = (clipped - lower) / (upper - lower)

    return normalized.astype(np.float32, copy=False)
