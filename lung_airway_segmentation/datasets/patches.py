"""Generic 3D patch sampling helpers.

This module contains dataset-agnostic utilities for patch-based training,
including patch-size normalization, valid patch coordinate handling, and local
patch extraction from 3D image or mask volumes.
"""

from collections.abc import Sequence

import numpy as np

from lung_airway_segmentation.schemas import Shape3D




def normalize_patch_size(patch_size: int | Sequence[int]) -> Shape3D:
    if isinstance(patch_size, int):
        if patch_size <= 0:
            raise ValueError("Patch size must be positive.")
        return(patch_size, patch_size, patch_size)
    
    if len(patch_size) != 3:
        raise ValueError("Patch size must be an int or a 3 item sequence")
    
    normalized = (int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))
    if any(value <= 0 for value in normalized):
        raise ValueError("Patch dimensions must be positive")
    
    return normalized

def clamp_patch_start(
        start: tuple[int, int, int],
        volume_shape: Shape3D,
        patch_size: Shape3D
) -> tuple[int, int, int]:
    clamped = []
    for axis_start, axis_shape, axis_patch in zip(start, volume_shape, patch_size):
        max_start = axis_shape - axis_patch
        if max_start < 0:
            raise ValueError(f"Patch size {patch_size} is larger than the volume shape {volume_shape}")
        clamped.append(min(max(axis_start, 0), max_start))

    return (clamped[0], clamped[1], clamped[2])

def extract_patch(
        volume: np.ndarray,
        start: tuple[int, int, int],
        patch_size: Shape3D
) -> np.ndarray:
    z0, y0, x0 = start
    dz, dy, dx = patch_size
    return volume[z0:z0+dz, y0:y0+dy, x0:x0+dx]

def sample_random_patch_start(
        volume_shape: Shape3D,
        patch_size: Shape3D,
        rng: np.random.Generator
) -> tuple[int, int, int]:
    starts = []
    for axis_shape, axis_patch in zip(volume_shape, patch_size):
        max_start = axis_shape - axis_patch
        if max_start < 0:
            raise ValueError(f"Patch size {patch_size} is larger than the volume shape {volume_shape}")
        starts.append(int(rng.integers(0, max_start + 1)))

    return (starts[0], starts[1], starts[2])

def sample_foreground_patch_start(
        mask: np.ndarray,
        patch_size: Shape3D,
        rng: np.random.Generator
) -> tuple[int, int, int]:
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        raise ValueError("Cannot sample foreground patch from an empty mask")
    
    center = foreground[int(rng.integers(0, len(foreground)))]
    half = np.array(patch_size) // 2
    start = (
        int(center[0] - half[0]),
        int(center[1] - half[1]),
        int(center[2] - half[2]),
    )

    volume_shape = (int(mask.shape[0]), int(mask.shape[1]), int(mask.shape[2]))
    return clamp_patch_start(start, volume_shape, patch_size)