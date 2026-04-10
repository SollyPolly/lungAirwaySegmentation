"""Standard segmentation metrics for binary airway masks.

This module contains overlap-focused evaluation helpers that are separate from
the optimization losses. The current implementation provides a hard-thresholded
binary Dice score computed from model logits.
"""

import torch


def binary_dice_score_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    """Compute mean binary Dice from logits and binary target masks."""
    targets = (targets > 0.5).float()
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.reshape(predictions.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = (predictions * targets).sum(dim=1)
    denominator = predictions.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)

    return dice.mean()
