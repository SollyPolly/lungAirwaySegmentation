"""Prediction cleanup and heuristic filtering."""

import torch


def binarize_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Apply sigmoid then threshold to convert raw model logits to a binary mask."""
    return (torch.sigmoid(logits) >= threshold).float()


def apply_lung_mask(predictions: torch.Tensor, lung_mask: torch.Tensor) -> torch.Tensor:
    """Zero out airway predictions outside the lung region.

    The sliding window runs over the full lung bounding box. Within that box,
    predictions can leak into vertebrae and chest-wall voxels at the edges.
    Multiplying by the binary lung mask suppresses those false positives before
    computing any metric or saving a result.
    """
    return predictions * (lung_mask > 0).float().to(predictions.device)
