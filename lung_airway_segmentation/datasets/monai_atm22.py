"""MONAI dataset for ATM'22 used as unlabelled data in semi-supervised training.

Loads CT volumes only — airway labels are intentionally ignored to simulate
the unlabelled setting. The same HU windowing and patch augmentations as the
supervised pipeline are applied so the teacher and student see consistent inputs.
"""

from pathlib import Path

from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    CopyItemsd,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    SpatialPadd,
)

from lung_airway_segmentation.datasets.patches import normalize_patch_size
from lung_airway_segmentation.io.atm22_layout import resolve_case_paths
from lung_airway_segmentation.settings import DEFAULT_HU_WINDOW

def build_atm22_records(case_ids, *, batch_root: Path) -> list[dict]:
    records = []
    for case_id in case_ids:
        paths = resolve_case_paths(case_id, batch_root=batch_root)
        records.append({
            "case_id": paths["case_id"],
            "image": str(paths["ct"])
        })
    return records

def build_unlabelled_transforms(
        *,
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
) -> Compose:
    patch_size = normalize_patch_size(patch_size)
    lower, upper = hu_window

    return Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False),
            EnsureTyped(keys=["image"]),
            ScaleIntensityRanged(
                keys="image",
                a_min=lower,
                a_max=upper,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
            SpatialPadd(keys=["image"], spatial_size=patch_size),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=patch_size,
                num_samples=patches_per_case,
                random_center=True,
                random_size=False,
            ),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image"],
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                mode="bilinear",
                padding_mode="border",
            ),
            # The teacher receives the weak spatially augmented view. Additional
            # intensity perturbations are applied only to the student's view.
            CopyItemsd(keys=["image"], times=1, names=["teacher_image"]),
            RandGaussianNoised(keys="image", prob=0.15, std=0.01),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        ]
    )

# IMPORTANT: use monai.data.DataLoader (not torch.utils.data.DataLoader) with this
# dataset. RandSpatialCropSamplesd returns a list of dicts (one per patch), and
# MONAI's DataLoader handles that collation correctly via list_data_collate.
def build_monai_atm22_dataset(
        case_ids,
        *,
        batch_root: Path,
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        cache_rate: float = 0.0,
        hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
) -> Dataset:
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError("cache_rate must be between 0.0 and 1.0.")

    records = build_atm22_records(case_ids, batch_root=batch_root)

    dataset_class = CacheDataset if cache_rate > 0.0 else Dataset
    cache_kwargs = {"cache_rate": cache_rate} if cache_rate > 0.0 else {}

    return dataset_class(
        data=records,
        transform=build_unlabelled_transforms(
            patch_size=patch_size,
            patches_per_case=patches_per_case,
            hu_window=hu_window,
        ),
        **cache_kwargs,
    )
