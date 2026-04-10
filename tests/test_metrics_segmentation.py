"""Tests for overlap-based segmentation metrics.

This module verifies that the binary Dice helper behaves sensibly on small toy
examples before it is used during training and validation.
"""

import torch

from lung_airway_segmentation.metrics.segmentation import binary_dice_score_from_logits


def test_binary_dice_score_from_logits_is_one_for_perfect_prediction():
    logits = torch.tensor([[[[[10.0, -10.0], [-10.0, 10.0]]]]])
    targets = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])

    dice = binary_dice_score_from_logits(logits, targets)

    assert torch.isclose(dice, torch.tensor(1.0))


def test_binary_dice_score_from_logits_is_zero_for_complete_miss():
    logits = torch.tensor([[[[[-10.0, 10.0], [10.0, -10.0]]]]])
    targets = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])

    dice = binary_dice_score_from_logits(logits, targets)

    assert dice.item() < 1e-5


def test_binary_dice_score_from_logits_is_one_when_both_masks_are_empty():
    logits = torch.full((1, 1, 2, 2, 2), -10.0)
    targets = torch.zeros((1, 1, 2, 2, 2))

    dice = binary_dice_score_from_logits(logits, targets)

    assert torch.isclose(dice, torch.tensor(1.0))
