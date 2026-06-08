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
