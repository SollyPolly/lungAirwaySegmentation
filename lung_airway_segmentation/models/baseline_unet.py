"""Baseline 3D U-Net definition for supervised airway segmentation.

This module holds the simplest strong reference model for the project: a
single-output 3D U-Net configured for binary airway segmentation. It defines
only the architecture and leaves optimization, losses, and training behavior to
other modules.
"""

from monai.networks.nets.unet import UNet


def build_baseline_unet(
    *,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.0,
    norm: str = "INSTANCE",
    act: str = "PRELU",
):
    """Construct the baseline MONAI 3D U-Net used for supervised training."""
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        dropout=dropout,
        norm=norm,
        act=act,
    )
