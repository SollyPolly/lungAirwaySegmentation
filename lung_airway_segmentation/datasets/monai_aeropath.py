"""MONAI-native AeroPath datasets and transforms.

This module provides the training data path that mirrors a standard MONAI
segmentation workflow: build case records, load full NIfTI volumes, sample
multiple label-balanced patches from each loaded case, crop to the lung ROI,
and apply patch-level augmentation. It is intended for efficient supervised
training while the explicit AeroPath datasets remain available for custom
sampling experiments.
"""

from pathlib import Path

from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
)

from lung_airway_segmentation.datasets.patches import normalize_patch_size
from lung_airway_segmentation.io.case_layout import resolve_case_paths
from lung_airway_segmentation.settings import DEFAULT_HU_WINDOW, RAW_AEROPATH_ROOT


def build_aeropath_records(
    case_ids,
    *,
    data_root: Path = RAW_AEROPATH_ROOT,
    include_lung_mask: bool = False,
) -> list[dict]:
    """Build MONAI dictionary records for labelled AeroPath cases."""
    records = []
    for case_id in case_ids:
        paths = resolve_case_paths(case_id, data_root=data_root)
        if paths["airway"] is None:
            raise ValueError(f"Case {case_id} is missing an airway mask.")

        record = {
            "case_id": paths["case_id"],
            "image": str(paths["ct"]),
            "airway_mask": str(paths["airway"]),
        }

        if include_lung_mask:
            if paths["lung"] is None:
                raise ValueError(f"Case {case_id} is missing a lung mask.")
            record["lung_mask"] = str(paths["lung"])

        records.append(record)

    return records


def build_train_transforms(
    *,
    patch_size: tuple[int, int, int],
    patches_per_case: int,
    foreground_probability: float,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
    crop_margin_voxels: int = 0,
) -> Compose:
    """Build MONAI transforms for patch-based supervised training."""
    patch_size = normalize_patch_size(patch_size)
    if not 0.0 <= foreground_probability <= 1.0:
        raise ValueError("foreground_probability must be between 0.0 and 1.0.")
    if patches_per_case <= 0:
        raise ValueError("patches_per_case must be positive.")

    keys = ["image", "airway_mask", "lung_mask"]
    modes = ["bilinear" if key == "image" else "nearest" for key in keys]
    lower, upper = hu_window

    return Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
            EnsureTyped(keys=keys),
            CropForegroundd(
                keys=keys,
                source_key="lung_mask",
                margin=crop_margin_voxels,
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=lower,
                a_max=upper,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(keys=keys, spatial_size=patch_size),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="airway_mask",
                spatial_size=patch_size,
                pos=foreground_probability,
                neg=1.0 - foreground_probability,
                num_samples=patches_per_case,
                image_key="image",
                image_threshold=0.0,
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandAffined(
                keys=keys,
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode=modes,
                padding_mode="border",
            ),
            RandGaussianNoised(keys="image", prob=0.15, std=0.01),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        ]
    )


def build_val_transforms(
    *,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
    crop_margin_voxels: int = 0,
) -> Compose:
    """Build MONAI transforms for full-volume validation."""
    lower, upper = hu_window
    keys = ["image", "airway_mask", "lung_mask"]

    return Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
            EnsureTyped(keys=keys),
            CropForegroundd(
                keys=keys,
                source_key="lung_mask",
                margin=crop_margin_voxels,
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=lower,
                a_max=upper,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )


def build_monai_aeropath_datasets(
    *,
    train_ids,
    val_ids,
    data_root: Path = RAW_AEROPATH_ROOT,
    patch_size: tuple[int, int, int],
    patches_per_case: int,
    foreground_probability: float,
    cache_rate: float = 0.0,
    crop_margin_voxels: int = 0,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
) -> tuple[Dataset, Dataset]:
    """Build MONAI train and validation datasets for labelled AeroPath data."""
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError("cache_rate must be between 0.0 and 1.0.")

    train_records = build_aeropath_records(
        train_ids,
        data_root=data_root,
        include_lung_mask=True,
    )
    val_records = build_aeropath_records(
        val_ids,
        data_root=data_root,
        include_lung_mask=True,
    )

    dataset_class = CacheDataset if cache_rate > 0.0 else Dataset
    cache_kwargs = {"cache_rate": cache_rate} if cache_rate > 0.0 else {}

    train_dataset = dataset_class(
        data=train_records,
        transform=build_train_transforms(
            patch_size=patch_size,
            patches_per_case=patches_per_case,
            foreground_probability=foreground_probability,
            crop_margin_voxels=crop_margin_voxels,
            hu_window=hu_window,
        ),
        **cache_kwargs,
    )
    val_dataset = dataset_class(
        data=val_records,
        transform=build_val_transforms(
            crop_margin_voxels=crop_margin_voxels,
            hu_window=hu_window,
        ),
        **cache_kwargs,
    )

    return train_dataset, val_dataset
