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
    SpatialCrop,
    SpatialPadd,
)
from lung_airway_segmentation.datasets.distal_classes import compute_distal_crop_classes
from lung_airway_segmentation.datasets.patches import normalize_patch_size
from lung_airway_segmentation.inference.postprocess import lung_bbox_slices
from lung_airway_segmentation.io.atm22_layout import (
    resolve_case_paths,
    resolve_distal_classes_path,
    resolve_lung_mask_path,
)
from lung_airway_segmentation.settings import DEFAULT_HU_WINDOW

# ``compute_distal_crop_classes`` is imported above (from the DL-free distal_classes
# module) and re-exported here for the training transform, the precompute script and tests.


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
        lung_crop: bool = False,
        lung_root: Path | None = None,
) -> list[dict]:
    """Build CT + airway-mask records for ATM'22 used as *labelled* training data.

    When ``distal_radius`` is set, attach the precomputed distal crop-class map path
    (``crop_classes``) for distal-skeleton patch sampling. The map must already exist
    on disk (run ``scripts/precompute_distal_classes.py`` once per dataset+radius) —
    a missing file raises with the exact command to generate it.

    When ``lung_crop`` is set, attach the precomputed binary lung-mask path (``lung``)
    used to crop CT + airway to the lung bounding box instead of the whole-body CT
    foreground. The mask must already exist on disk (run
    ``scripts/precompute_lung_masks.py`` once per dataset) — a missing file raises with
    the exact command to generate it.
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
        if lung_crop:
            lung_path = resolve_lung_mask_path(
                paths["case_id"], batch_root=batch_root, lung_root=lung_root
            )
            if not lung_path.is_file():
                raise FileNotFoundError(
                    f"Precomputed lung mask missing for case {paths['case_id']}: {lung_path}\n"
                    f"Generate it once with:\n"
                    f"  python -u -m scripts.precompute_lung_masks --batch-root <BATCH_ROOT>"
                )
            record["lung"] = str(lung_path)
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


# Lung-crop defaults (voxels). Measured over all 150 ATM cases: airway NEVER overhangs
# the lung bbox in-plane/inferiorly (max 0), but the cervical trachea overhangs it
# SUPERIORLY by up to ~100 voxels — hence a large superior margin, modest elsewhere.
_DEFAULT_INPLANE_MARGIN = 8
_DEFAULT_SUPERIOR_MARGIN = 120
_LUNG_CROP_STRATEGIES = ("lung_with_trachea_extension", "lung_union_airway")


def _resolve_lung_crop(lung_crop: dict | None) -> tuple[bool, str | None, int, int]:
    """Interpret ``lung_crop`` -> (enabled, strategy, margin_voxels, superior_margin_voxels).

    Disabled (``None``/``enabled: false``) returns ``(False, None, 0, 0)`` so the crop
    stage falls back to the CT-foreground crop, bit-identical to the prior behaviour.
    """
    if not lung_crop or not lung_crop.get("enabled", False):
        return False, None, 0, 0
    strategy = str(lung_crop.get("strategy", "lung_with_trachea_extension"))
    if strategy not in _LUNG_CROP_STRATEGIES:
        raise ValueError(f"lung_crop.strategy must be one of {_LUNG_CROP_STRATEGIES}, got {strategy!r}.")
    margin = int(lung_crop.get("margin_voxels", _DEFAULT_INPLANE_MARGIN))
    superior_margin = int(lung_crop.get("superior_margin_voxels", _DEFAULT_SUPERIOR_MARGIN))
    if margin < 0 or superior_margin < 0:
        raise ValueError("lung_crop margins must be >= 0.")
    return True, strategy, margin, superior_margin


def _superior_axis(affine) -> tuple[int, int]:
    """(array_axis, sign) that points SUPERIOR, from a RAS affine (world +z = superior).

    The array axis whose voxel step has the largest world-z component is superior/
    inferior; its sign says whether increasing the index moves superior (+1) or
    inferior (-1). Falls back to (last spatial axis, +1) — the ATM native layout — when
    no affine is available (e.g. a plain tensor in a unit test).
    """
    if affine is None:
        return 2, 1
    zrow = np.asarray(affine, dtype=np.float64)[2, :3]
    axis = int(np.argmax(np.abs(zrow)))
    return axis, (1 if zrow[axis] >= 0 else -1)


class LungTracheaCropd(MapTransform):
    """Crop ``keys`` to the lung bounding box, extended SUPERIORLY to keep the trachea.

    Inference-valid (the main method): the crop box needs only the CT-derived lung mask
    (``lung_key``) + the image affine — never the airway GT — so the identical crop is
    computable at train / val / test / prediction / pseudo-labelling time. The lung bbox
    is extended by ``superior_margin_voxels`` on the superior end (the cervical trachea
    overhangs the lungs there) and by ``margin_voxels`` on every other side.

    ``strategy="lung_union_airway"`` additionally unions the airway mask into the box
    source — an ORACLE diagnostic / upper-bound only (it uses the label to set the FOV,
    so it is not valid for val/test claims); kept for ablation.
    """

    def __init__(self, keys, *, lung_key="lung", airway_key="airway_mask",
                 strategy="lung_with_trachea_extension",
                 margin_voxels=_DEFAULT_INPLANE_MARGIN,
                 superior_margin_voxels=_DEFAULT_SUPERIOR_MARGIN,
                 allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if strategy not in _LUNG_CROP_STRATEGIES:
            raise ValueError(f"strategy must be one of {_LUNG_CROP_STRATEGIES}, got {strategy!r}.")
        self.lung_key = lung_key
        self.airway_key = airway_key
        self.strategy = strategy
        self.margin = int(margin_voxels)
        self.superior_margin = int(superior_margin_voxels)

    @staticmethod
    def _spatial(mask) -> np.ndarray:
        arr = np.asarray(mask)
        return (arr[0] if arr.ndim == 4 else arr) > 0

    def __call__(self, data):
        d = dict(data)
        source = self._spatial(d[self.lung_key])
        if self.strategy == "lung_union_airway" and self.airway_key in d:
            source = source | self._spatial(d[self.airway_key])
        if not source.any():
            return d  # no lung found (logged upstream) — leave the volume uncropped

        bounds = lung_bbox_slices(
            source,
            affine=getattr(d[self.lung_key], "affine", None),
            margin_voxels=self.margin,
            superior_margin_voxels=self.superior_margin,
        )
        if bounds is None:
            return d
        starts = [int(bound.start) for bound in bounds]
        ends = [int(bound.stop) for bound in bounds]

        cropper = SpatialCrop(roi_start=starts, roi_end=ends)
        for key in self.key_iterator(d):
            d[key] = cropper(d[key])
        return d


def _build_foreground_crop(*, data_keys: list[str], lung_crop: dict | None) -> list:
    """The crop-to-region stage: lung-bbox+trachea crop or CT intensity foreground.

    Lung crop concentrates capacity on the airway region (the biggest fixable gap vs
    ATM'22 pipelines, PROJECT_STATE 2026-07-01). The ``lung`` mask is loaded upstream
    only to define the box; it is dropped straight after so it never reaches collation.
    """
    enabled, strategy, margin, superior_margin = _resolve_lung_crop(lung_crop)
    if not enabled:
        return [CropForegroundd(keys=data_keys, source_key="image", allow_smaller=True)]
    return [
        LungTracheaCropd(
            keys=data_keys, lung_key="lung", airway_key="airway_mask", strategy=strategy,
            margin_voxels=margin, superior_margin_voxels=superior_margin,
        ),
        DeleteItemsd(keys="lung"),
    ]


def build_atm22_labelled_transforms(
        *,
        patch_size: tuple[int, int, int],
        patches_per_case: int,
        foreground_probability: float,
        hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
        distal_sampling: dict | None = None,
        lung_crop: dict | None = None,
) -> Compose:
    """Patch-based supervised transforms for labelled ATM'22 (CT + airway mask).

    Mirrors the AeroPath patch pipeline. By default crops on CT-intensity foreground
    (whole body) and normalises before cropping so the foreground selector operates on
    the [0, 1] image like the unlabelled ATM'22 pipeline.

    ``distal_sampling`` (optional) switches the crop stage to distal-biased class
    sampling — see ``_build_crop_transforms``. When enabled, the precomputed
    ``crop_classes`` map is loaded and carried through the deterministic spatial
    stages (load/crop/pad) so it stays aligned with the mask, then used as the crop
    label and dropped before augmentation/collation.

    ``lung_crop`` (optional) replaces the CT-foreground crop with a crop to the
    precomputed lung bounding box (+margin), tightening the FOV onto the airway region.
    The ``lung`` mask is loaded, used as the crop ``source_key`` and dropped before
    padding — see ``_build_foreground_crop``.
    """
    patch_size = normalize_patch_size(patch_size)
    if not 0.0 <= foreground_probability <= 1.0:
        raise ValueError("foreground_probability must be between 0.0 and 1.0.")
    if patches_per_case <= 0:
        raise ValueError("patches_per_case must be positive.")

    # ``aug_keys`` are cropped + augmented and collated; ``crop_classes`` (distal
    # sampling) and ``lung`` (lung crop) are load-only helpers deleted before collation.
    # ``data_keys`` flow through crop+pad; ``load_keys`` also pulls the lung mask in.
    aug_keys = ["image", "airway_mask"]
    modes = ["bilinear", "nearest"]
    distal_enabled = bool(distal_sampling and distal_sampling.get("enabled", False))
    lung_enabled = _resolve_lung_crop(lung_crop)[0]
    data_keys = [*aug_keys, "crop_classes"] if distal_enabled else list(aug_keys)
    load_keys = [*data_keys, "lung"] if lung_enabled else data_keys
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
            *_build_foreground_crop(data_keys=data_keys, lung_crop=lung_crop),
            SpatialPadd(keys=data_keys, spatial_size=patch_size),
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
        lung_crop: dict | None = None,
) -> Compose:
    """Full-volume validation transforms for labelled ATM'22 (CT + airway mask).

    ``lung_crop`` (optional) crops CT + airway to the precomputed lung bounding box
    (+margin) instead of the CT-intensity foreground, matching the train FOV.
    """
    data_keys = ["image", "airway_mask"]
    lung_enabled = _resolve_lung_crop(lung_crop)[0]
    load_keys = [*data_keys, "lung"] if lung_enabled else data_keys
    lower, upper = hu_window

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
            *_build_foreground_crop(data_keys=data_keys, lung_crop=lung_crop),
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
        lung_crop: dict | None = None,
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

    # Lung crop needs the precomputed lung mask on BOTH train and val records (both
    # crop CT + airway to the lung bbox so their FOV matches).
    lung_enabled = bool(lung_crop and lung_crop.get("enabled", False))
    lung_root: Path | None = None
    if lung_enabled:
        configured_lung_root = lung_crop.get("lung_root")
        lung_root = Path(configured_lung_root) if configured_lung_root else None

    train_records = build_atm22_labelled_records(
        train_ids, batch_root=batch_root, distal_radius=distal_radius, classes_root=classes_root,
        lung_crop=lung_enabled, lung_root=lung_root,
    )
    val_records = build_atm22_labelled_records(
        val_ids, batch_root=batch_root, lung_crop=lung_enabled, lung_root=lung_root,
    )

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
            lung_crop=lung_crop,
        ),
        **cache_kwargs,
    )
    val_dataset = dataset_class(
        data=val_records,
        transform=build_atm22_labelled_val_transforms(hu_window=hu_window, lung_crop=lung_crop),
        **cache_kwargs,
    )

    return train_dataset, val_dataset
