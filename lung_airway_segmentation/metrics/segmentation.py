"""Standard segmentation metrics for binary airway masks."""

import numpy as np
import torch


def _as_binary_numpy_mask(mask) -> np.ndarray:
    """Convert one mask-like array into a boolean NumPy mask."""
    return np.asarray(mask) > 0


def binary_confusion_counts_from_masks(predictions, targets) -> dict[str, int]:
    """Compute binary confusion counts for one predicted and reference mask."""
    prediction_mask = _as_binary_numpy_mask(predictions)
    target_mask = _as_binary_numpy_mask(targets)

    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "Prediction and target masks must have the same shape: "
            f"{prediction_mask.shape} != {target_mask.shape}"
        )

    true_positive = int(np.logical_and(prediction_mask, target_mask).sum())
    false_positive = int(np.logical_and(prediction_mask, np.logical_not(target_mask)).sum())
    false_negative = int(np.logical_and(np.logical_not(prediction_mask), target_mask).sum())
    true_negative = int(np.logical_and(np.logical_not(prediction_mask), np.logical_not(target_mask)).sum())

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def binary_dice_score_from_masks(predictions, targets, eps=1e-6) -> float:
    """Compute binary Dice from two mask volumes."""
    counts = binary_confusion_counts_from_masks(predictions, targets)
    denominator = (
        2 * counts["true_positive"]
        + counts["false_positive"]
        + counts["false_negative"]
    )
    return float((2.0 * counts["true_positive"] + eps) / (denominator + eps))


def binary_iou_score_from_masks(predictions, targets, eps=1e-6) -> float:
    """Compute binary intersection-over-union from two mask volumes."""
    counts = binary_confusion_counts_from_masks(predictions, targets)
    denominator = (
        counts["true_positive"]
        + counts["false_positive"]
        + counts["false_negative"]
    )
    return float((counts["true_positive"] + eps) / (denominator + eps))


def binary_precision_from_masks(predictions, targets, eps=1e-6) -> float:
    """Compute binary precision from two mask volumes."""
    counts = binary_confusion_counts_from_masks(predictions, targets)
    denominator = counts["true_positive"] + counts["false_positive"]
    return float((counts["true_positive"] + eps) / (denominator + eps))


def binary_recall_from_masks(predictions, targets, eps=1e-6) -> float:
    """Compute binary recall from two mask volumes."""
    counts = binary_confusion_counts_from_masks(predictions, targets)
    denominator = counts["true_positive"] + counts["false_negative"]
    return float((counts["true_positive"] + eps) / (denominator + eps))


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
