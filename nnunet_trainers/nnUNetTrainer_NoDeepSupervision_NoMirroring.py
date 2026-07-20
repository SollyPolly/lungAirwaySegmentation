"""Train-safe nnU-Net v2 base with deep supervision and mirroring disabled.

The local environment also contains similarly named *inference shims*.  Those
classes are deliberately sufficient only for rebuilding old checkpoint
architectures and must not be used for training.  Keeping the real recipe in
this small, self-contained module makes the training contract explicit and
portable to the HPC nnU-Net installation.
"""

from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)


class nnUNetTrainer_NoDeepSupervision_NoMirroring(nnUNetTrainerNoDeepSupervision):
    """Stock nnU-Net training with tensor output and no mirror augmentation/TTA."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # nnUNetTrainer reflects over this exact signature; do not replace it
        # with *args/**kwargs.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch_size, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rotation, dummy_2d, initial_patch_size, None

