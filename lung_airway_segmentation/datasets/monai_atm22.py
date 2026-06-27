"""MONAI dataset for ATM'22 used as unlabelled data in semi-supervised training.

Loads CT volumes only — airway labels are intentionally ignored to simulate
the unlabelled setting. The same HU windowing and patch augmentations as the
supervised pipeline are applied so the teacher and student see consistent inputs.
"""

from pathlib import Path

import numpy as np
from monai.data import CacheDataset, Dataset
from monai.transforms import (
    Compose,
    CopyItemsd,
    CropForegroundd,
    DeleteItemsd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandAffined,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    SpatialPadd,
)
from scipy import ndimage
from skimage.morphology import skeletonize

from lung_airway_segmentation.datasets.patches import normalize_patch_size
from lung_airway_segmentation.io.atm22_layout import (
    resolve_case_paths,
    resolve_distal_classes_path,
)
from lung_airway_segmentation.settings import DEFAULT_HU_WINDOW


def compute_distal_crop_classes(
        mask: np.ndarray,
        *,
        distal_radius_voxels: float = 2.0,
) -> np.ndarray:
    """3-class crop-guidance map from a binary airway mask (spatial dims only).

    * 0 = background
    * 1 = proximal / non-distal airway
    * 2 = distal airway = skeleton voxels whose EDT radius <= ``distal_radius_voxels``

    Class 2 is never left empty when any airway is present (falls back to the whole
    skeleton, then the mask) so ``RandCropByLabelClassesd`` does not warn / mis-weight.
    Shared by ``ComputeDistalCropClassesd`` and the offline precompute script so the
    on-disk maps are identical to the legacy on-the-fly transform. The pipeline does
    not resample, and the CT-foreground crop never cuts the airway, so skeletonising
    the native-resolution mask offline reproduces the cropped/padded class map exactly.
    """
    if distal_radius_voxels <= 0.0:
        raise ValueError("distal_radius_voxels must be positive.")
    binary = np.asarray(mask) > 0
    classes = np.zeros(binary.shape, dtype=np.uint8)
    if binary.any():
        classes[binary] = 1  # all airway starts as proximal
        radius = ndimage.distance_transform_edt(binary)
        skel = skeletonize(binary)
        distal = skel & (radius <= float(distal_radius_voxels))
        if not distal.any():
            distal = skel if skel.any() else binary
        classes[distal] = 2
    return classes


class ComputeDistalCropClassesd(MapTransform):
    """Derive a 3-class crop-guidance map from a binary airway mask.

    Writes ``crop_classes`` (same spatial shape as the mask, channel-first), used
    as the ``label_key`` of ``RandCropByLabelClassesd`` to bias patch centres
    toward thin distal branches (airway hard-mining):

    * 0 = background
    * 1 = proximal / non-distal airway
    * 2 = distal airway = skeleton voxels whose EDT radius <= ``distal_radius_voxels``

    The radius is ``scipy.ndimage.distance_transform_edt`` in voxels — the same EDT
    the distal-analysis ``r=`` bins use (scripts/analyse_distal.py) — so "distal"
    here means the same thing it does in the evaluation. Restricting class 2 to the
    *skeleton* of thin branches keeps it selective (raw EDT<=1.5 is ~half of all
    airway voxels and would barely bias sampling).

    NB the training pipeline no longer runs this on the fly — it loads precomputed
    maps from disk (``scripts/precompute_distal_classes.py``) to avoid the per-epoch
    skeletonize + EDT (a memory/walltime cost). This transform is retained for the
    precompute script's equivalence tests and as a reference implementation; it wraps
    the shared :func:`compute_distal_crop_classes`.
    """

    def __init__(self, keys, *, distal_radius_voxels: float = 2.0, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if distal_radius_voxels <= 0.0:
            raise ValueError("distal_radius_voxels must be positive.")
        self.distal_radius_voxels = float(distal_radius_voxels)
        self.output_key = "crop_classes"

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask_np = np.asarray(d[key])
            # Channel-first (ensure_channel_first=True): operate on channel 0.
            has_channel = mask_np.ndim == 4
            spatial = mask_np[0] if has_channel else mask_np
            classes = compute_distal_crop_classes(
                spatial, distal_radius_voxels=self.distal_radius_voxels
            )
            # Restore the channel axis to match the other channel-first keys.
            d[self.output_key] = classes[None] if has_channel else classes
        return d

def build_atm22_records(case_ids, *, batch_root: Path) -> list[dict]:
    records = []
    for case_id in case_ids:
        paths = resolve_case_paths(case_id, batch_root=batch_root)
        records.append({
            "case_id": paths["case_id"],
            "image": str(paths["ct"])
        })
    return records


def build_atm22_labelled_records(
        case_ids,
        *,
        batch_root: Path,
        distal_radius: float | None = None,
        classes_root: Path | None = None,
) -> list[dict]:
    """Build CT + airway-mask records for ATM'22 used as *labelled* training data.

    When ``distal_radius`` is set, attach the precomputed distal crop-class map path
    (``crop_classes``) for distal-skeleton patch sampling. The map must already exist
    on disk (run ``scripts/precompute_distal_classes.py`` once per dataset+radius) —
    a missing file raises with the exact command to generate it.
    """
    records = []
    for case_id in case_ids:
        paths = resolve_case_paths(case_id, batch_root=batch_root)
        if paths["airway"] is None:
            raise ValueError(f"ATM'22 case {case_id} is missing an airway mask.")
        record = {
            "case_id": paths["case_id"],
            "image": str(paths["ct"]),
            "airway_mask": str(paths["airway"]),
        }
        if distal_radius is not None:
            classes_path = resolve_distal_classes_path(
                paths["case_id"], batch_root=batch_root, radius=distal_radius, classes_root=classes_root
            )
            if not classes_path.is_file():
                raise FileNotFoundError(
                    f"Precomputed distal class map missing for case {paths['case_id']}: {classes_path}\n"
                    f"Generate it once with:\n"
                    f"  python -u -m scripts.precompute_distal_classes "
                    f"--batch-root <BATCH_ROOT> --radius {distal_radius:g}"
                )
            record["crop_classes"] = str(classes_path)
        records.append(record)
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


def build_monai_atm22_selftraining_dataset(
        records,
        *,
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        foreground_probability: float,
        cache_rate: float = 0.0,
        hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
) -> Dataset:
    """Build a labelled-style train dataset from explicit CT + mask records.

    Used for topology-filtered self-training: ``records`` mixes real labelled
    cases (``airway_mask`` = the ground-truth file, optionally duplicated to
    up-weight them) with accepted pseudo-labelled cases (``airway_mask`` = a
    generated pseudo-mask file). Pseudo-masks are treated exactly like real masks
    (hard labels), so the supervised ``build_atm22_labelled_transforms`` pipeline
    is reused unchanged.
    """
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError("cache_rate must be between 0.0 and 1.0.")
    if not records:
        raise ValueError("build_monai_atm22_selftraining_dataset received no records.")

    dataset_class = CacheDataset if cache_rate > 0.0 else Dataset
    cache_kwargs = {"cache_rate": cache_rate} if cache_rate > 0.0 else {}

    return dataset_class(
        data=list(records),
        transform=build_atm22_labelled_transforms(
            patch_size=patch_size,
            patches_per_case=patches_per_case,
            foreground_probability=foreground_probability,
            hu_window=hu_window,
        ),
        **cache_kwargs,
    )


def _build_crop_transforms(
        *,
        keys: list[str],
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        foreground_probability: float,
        distal_sampling: dict | None,
) -> list:
    """Build the patch-cropping stage: distal-biased class sampling or pos/neg.

    Default (``distal_sampling`` None/disabled): the original
    ``RandCropByPosNegLabeld`` keyed on the whole airway mask — bit-identical to
    the prior behaviour. When enabled: sample crop centres from the precomputed
    ``crop_classes`` map (loaded + cropped/padded upstream) by ``ratios`` =
    [background, proximal-airway, distal-skeleton] so thin distal branches are
    over-sampled (airway hard-mining), then drop the map before collation.
    """
    if not distal_sampling or not distal_sampling.get("enabled", False):
        return [
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
        ]

    ratios = [float(r) for r in distal_sampling.get("ratios", [0.3, 0.3, 0.4])]
    if len(ratios) != 3 or any(r < 0.0 for r in ratios) or sum(ratios) <= 0.0:
        raise ValueError(
            "distal_sampling.ratios must be three non-negative numbers summing to > 0 "
            "([background, proximal-airway, distal-skeleton])."
        )
    # ``crop_classes`` is loaded from disk and carried through the deterministic
    # spatial stages (see ``build_atm22_labelled_transforms``); here it is only the
    # sampling label for the random crop, then dropped.
    return [
        RandCropByLabelClassesd(
            keys=keys,
            label_key="crop_classes",
            num_classes=3,
            ratios=ratios,
            spatial_size=patch_size,
            num_samples=patches_per_case,
            image_key="image",
            image_threshold=0.0,
            warn=False,
        ),
        # The class map is full-volume and unused downstream — drop it before
        # collation (mismatched spatial sizes across cases break list_data_collate).
        DeleteItemsd(keys="crop_classes"),
    ]


def build_atm22_labelled_transforms(
        *,
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        foreground_probability: float,
        hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
        distal_sampling: dict | None = None,
) -> Compose:
    """Patch-based supervised transforms for labelled ATM'22 (CT + airway mask).

    Mirrors the AeroPath patch pipeline, but crops on CT-intensity foreground
    (ATM'22 has no lung masks) and normalises before cropping so the foreground
    selector operates on the [0, 1] image like the unlabelled ATM'22 pipeline.

    ``distal_sampling`` (optional) switches the crop stage to distal-biased class
    sampling — see ``_build_crop_transforms``. When enabled, the precomputed
    ``crop_classes`` map is loaded and carried through the deterministic spatial
    stages (load/crop/pad) so it stays aligned with the mask, then used as the crop
    label and dropped before augmentation/collation.
    """
    patch_size = normalize_patch_size(patch_size)
    if not 0.0 <= foreground_probability <= 1.0:
        raise ValueError("foreground_probability must be between 0.0 and 1.0.")
    if patches_per_case <= 0:
        raise ValueError("patches_per_case must be positive.")

    # ``aug_keys`` are cropped + augmented and collated; ``crop_classes`` (when distal
    # sampling is on) is only loaded/cropped/padded as the sampling label and deleted
    # by the crop stage, so it must NOT appear in the post-crop augmentation keys.
    aug_keys = ["image", "airway_mask"]
    modes = ["bilinear", "nearest"]
    distal_enabled = bool(distal_sampling and distal_sampling.get("enabled", False))
    load_keys = [*aug_keys, "crop_classes"] if distal_enabled else aug_keys
    lower, upper = hu_window

    crop_transforms = _build_crop_transforms(
        keys=aug_keys,
        patch_size=patch_size,
        patches_per_case=patches_per_case,
        foreground_probability=foreground_probability,
        distal_sampling=distal_sampling,
    )

    return Compose(
        [
            LoadImaged(keys=load_keys, ensure_channel_first=True, image_only=False),
            EnsureTyped(keys=load_keys),
            ScaleIntensityRanged(
                keys="image",
                a_min=lower,
                a_max=upper,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=load_keys, source_key="image", allow_smaller=True),
            SpatialPadd(keys=load_keys, spatial_size=patch_size),
            *crop_transforms,
            RandFlipd(keys=aug_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=aug_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=aug_keys, prob=0.5, spatial_axis=2),
            RandAffined(
                keys=aug_keys,
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


def build_atm22_labelled_val_transforms(
        *,
        hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
) -> Compose:
    """Full-volume validation transforms for labelled ATM'22 (CT + airway mask)."""
    keys = ["image", "airway_mask"]
    lower, upper = hu_window

    return Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
            EnsureTyped(keys=keys),
            ScaleIntensityRanged(
                keys="image",
                a_min=lower,
                a_max=upper,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
        ]
    )


def build_monai_atm22_labelled_datasets(
        *,
        train_ids,
        val_ids,
        batch_root: Path,
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        foreground_probability: float,
        cache_rate: float = 0.0,
        hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
        distal_sampling: dict | None = None,
) -> tuple[Dataset, Dataset]:
    """Build MONAI train and validation datasets for labelled ATM'22 data."""
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError("cache_rate must be between 0.0 and 1.0.")

    # Distal sampling needs the precomputed crop-class map on the TRAIN records only;
    # validation runs full-volume transforms that never reference ``crop_classes``.
    distal_radius: float | None = None
    classes_root: Path | None = None
    if distal_sampling and distal_sampling.get("enabled", False):
        distal_radius = float(distal_sampling.get("distal_radius_voxels", 2.0))
        configured_root = distal_sampling.get("classes_root")
        classes_root = Path(configured_root) if configured_root else None

    train_records = build_atm22_labelled_records(
        train_ids, batch_root=batch_root, distal_radius=distal_radius, classes_root=classes_root
    )
    val_records = build_atm22_labelled_records(val_ids, batch_root=batch_root)

    dataset_class = CacheDataset if cache_rate > 0.0 else Dataset
    cache_kwargs = {"cache_rate": cache_rate} if cache_rate > 0.0 else {}

    train_dataset = dataset_class(
        data=train_records,
        transform=build_atm22_labelled_transforms(
            patch_size=patch_size,
            patches_per_case=patches_per_case,
            foreground_probability=foreground_probability,
            hu_window=hu_window,
            distal_sampling=distal_sampling,
        ),
        **cache_kwargs,
    )
    val_dataset = dataset_class(
        data=val_records,
        transform=build_atm22_labelled_val_transforms(hu_window=hu_window),
        **cache_kwargs,
    )

    return train_dataset, val_dataset
