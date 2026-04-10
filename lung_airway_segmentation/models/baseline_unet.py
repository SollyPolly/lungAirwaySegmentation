"""Baseline 3D U-Net definition for supervised airway segmentation.

This module holds the simplest strong reference model for the project: a
single-output 3D U-Net configured for binary airway segmentation. It defines
only the architecture and leaves optimization, losses, and training behavior to
other modules.
"""

from monai.networks.nets.unet import UNet

def build_baseline_unet():
    """Construct the baseline MONAI 3D U-Net used for supervised training."""
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
