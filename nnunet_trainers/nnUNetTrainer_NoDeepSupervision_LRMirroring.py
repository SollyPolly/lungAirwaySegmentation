"""Optional ATM'22 seed trainer with left-right-only mirroring.

For the copied Dataset111 plans, SimpleITK presents spatial axes as z/y/x and
``transpose_forward=[1, 0, 2]``. The anatomical left-right x axis is therefore
network axis 2. We intentionally exclude superior-inferior mirroring.
"""

from __future__ import annotations

import torch

try:
    if __package__ == "nnunet_trainers":
        raise ImportError
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_NoDeepSupervision_NoMirroring as _Base,
    )
except ImportError:
    from nnunet_trainers.nnUNetTrainer_NoDeepSupervision_NoMirroring import (
        nnUNetTrainer_NoDeepSupervision_NoMirroring as _Base,
    )


class nnUNetTrainer_NoDeepSupervision_LRMirroring(_Base):
    """No deep supervision; mirror augmentation and TTA on LR only."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch_size, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        if len(self.configuration_manager.patch_size) != 3:
            raise RuntimeError("The ATM lung-crop experiment expects the 3d_fullres configuration.")
        transpose = list(self.plans_manager.transpose_forward)
        if transpose != [1, 0, 2]:
            raise RuntimeError(
                f"LRMirroring was validated for transpose_forward=[1, 0, 2], got {transpose}. "
                "Re-derive the anatomical LR network axis before training."
            )
        mirror_axes = (2,)
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation, dummy_2d, initial_patch_size, mirror_axes

