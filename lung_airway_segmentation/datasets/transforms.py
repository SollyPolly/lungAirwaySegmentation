from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import EnsureTyped
from monai.transforms.spatial.dictionary import (
    RandFlipd,
    RandAffined
)
from monai.transforms.intensity.dictionary import (
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd
)


def build_train_patch_transforms(include_lung_mask: bool):
    keys = ["image", "airway_mask"]
    if include_lung_mask:
        keys.append("lung_mask")

    modes = ["bilinear" if key == "image" else "nearest" for key in keys]

    return Compose(
        [
            EnsureTyped(keys=keys),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandAffined(
                keys=keys,
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=modes,
                padding_mode="border"
            ),
            RandGaussianNoised(keys="image", prob=0.15, std=0.01),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        ]
    )