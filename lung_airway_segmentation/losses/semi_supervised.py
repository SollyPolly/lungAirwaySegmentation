"""Semi-supervised objectives and pseudo-label filtering.

This file should own the losses and weighting rules specific to unlabeled or
partially labeled training data.

Examples:
- consistency losses
- confidence-thresholded pseudo-label losses
- uncertainty-aware weighting
"""

import torch
import torch.nn as nn


class ConsistencyLoss(nn.Module):
    """Confidence-masked soft consistency balanced across foreground/background."""

    def __init__(
        self,
        foreground_threshold: float = 0.8,
        background_threshold: float | None = None,
    ):
        super().__init__()
        foreground_threshold = float(foreground_threshold)
        background_threshold = (
            1.0 - foreground_threshold
            if background_threshold is None
            else float(background_threshold)
        )
        if not 0.0 <= background_threshold < foreground_threshold <= 1.0:
            raise ValueError(
                "Consistency thresholds must satisfy "
                "0 <= background_threshold < foreground_threshold <= 1."
            )
        self.foreground_threshold = foreground_threshold
        self.background_threshold = background_threshold

    def confidence_masks(
        self,
        teacher_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return confident teacher foreground and background masks."""
        foreground = teacher_probabilities >= self.foreground_threshold
        background = teacher_probabilities <= self.background_threshold
        return foreground, background

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        student_probs = torch.sigmoid(student_logits.float())
        teacher_probs = teacher_probabilities.detach().float()
        foreground_mask, background_mask = self.confidence_masks(teacher_probs)
        squared_error = (student_probs - teacher_probs).square()

        # Background-only consistency would reinforce the dominant class on
        # random ATM'22 patches and can collapse sparse-airway predictions.
        if not foreground_mask.any():
            return student_probs.sum() * 0.0

        class_losses = [squared_error[foreground_mask].mean()]
        if background_mask.any():
            class_losses.append(squared_error[background_mask].mean())
        return torch.stack(class_losses).mean()
