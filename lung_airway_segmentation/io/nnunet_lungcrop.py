"""Geometry-safe lung-ROI inputs for nnU-Net.

The files stay on their original grid and affine. Voxels outside the lung plus
superior-trachea bounding box are set to exactly zero. nnU-Net's normal
non-zero crop then restricts preprocessing/training to that ROI while retaining
its built-in crop metadata, so predictions are automatically exported back to
the original full image geometry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.inference.postprocess import lung_bbox_slices


def _normalise_case_id(value: object) -> str:
    text = str(value).strip()
    if text.startswith("ATM_"):
        text = text[4:]
    if not text.isdigit():
        raise ValueError(f"Invalid ATM case id in intensity overrides: {value!r}.")
    return f"{int(text):03d}"


def parse_case_intensity_overrides(
    config: Mapping,
    *,
    allowed_case_ids: Iterable[object] | None = None,
) -> dict[str, dict]:
    """Flatten and validate manifest-declared stored-value transforms.

    The returned mapping is keyed by zero-padded ATM case id. Keeping this
    declaration in the frozen split manifest makes exceptional source
    encodings reviewable and prevents an incidental dtype heuristic from
    silently changing the training data.
    """
    raw_groups = config.get("ct_intensity_overrides", {})
    if raw_groups is None:
        raw_groups = {}
    if not isinstance(raw_groups, Mapping):
        raise ValueError("ct_intensity_overrides must be a mapping.")

    allowed = (
        {_normalise_case_id(case_id) for case_id in allowed_case_ids}
        if allowed_case_ids is not None
        else None
    )
    overrides: dict[str, dict] = {}
    for group_name, raw_spec in raw_groups.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Intensity override {group_name!r} must be a mapping.")
        raw_case_ids = raw_spec.get("case_ids")
        if not isinstance(raw_case_ids, list) or not raw_case_ids:
            raise ValueError(
                f"Intensity override {group_name!r} must have a non-empty case_ids list."
            )
        try:
            scale = float(raw_spec["scale"])
            offset = float(raw_spec.get("offset", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Intensity override {group_name!r} needs numeric scale and offset."
            ) from exc
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Intensity override {group_name!r} scale must be finite and positive."
            )
        if not np.isfinite(offset):
            raise ValueError(f"Intensity override {group_name!r} offset must be finite.")

        for raw_case_id in raw_case_ids:
            case_id = _normalise_case_id(raw_case_id)
            if case_id in overrides:
                raise ValueError(f"Duplicate intensity override for ATM_{case_id}.")
            if allowed is not None and case_id not in allowed:
                raise ValueError(
                    f"Intensity override ATM_{case_id} is outside the allowed case set."
                )
            overrides[case_id] = {
                "group": str(group_name),
                "scale": scale,
                "offset": offset,
            }
    return overrides


def assert_same_nifti_grid(
    reference: nib.spatialimages.SpatialImage,
    candidate: nib.spatialimages.SpatialImage,
    *,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
    affine_atol: float = 1e-4,
) -> None:
    """Fail if two NIfTIs cannot be combined voxel-for-voxel."""
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"Grid mismatch: {reference_name} shape {reference.shape} != "
            f"{candidate_name} shape {candidate.shape}."
        )
    if not np.allclose(reference.affine, candidate.affine, rtol=0.0, atol=affine_atol):
        max_delta = float(np.max(np.abs(reference.affine - candidate.affine)))
        raise ValueError(
            f"Affine mismatch between {reference_name} and {candidate_name} "
            f"(max absolute delta {max_delta:g})."
        )


def bbox_to_json(bounds: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(axis.start), int(axis.stop)] for axis in bounds]


def bbox_from_json(bounds: list[list[int]]) -> tuple[slice, slice, slice]:
    if len(bounds) != 3 or any(len(axis) != 2 for axis in bounds):
        raise ValueError(f"Expected three [start, stop] bounds, got {bounds!r}.")
    return tuple(slice(int(start), int(stop)) for start, stop in bounds)  # type: ignore[return-value]


def resolve_lung_roi(
    ct_image: nib.spatialimages.SpatialImage,
    lung_image: nib.spatialimages.SpatialImage,
    *,
    margin_voxels: int = 8,
    superior_margin_voxels: int = 120,
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    """Validate the lung mask and return its affine-aware training ROI."""
    assert_same_nifti_grid(ct_image, lung_image, reference_name="CT", candidate_name="lung mask")
    lung = np.asanyarray(lung_image.dataobj) > 0
    bounds = lung_bbox_slices(
        lung,
        affine=ct_image.affine,
        margin_voxels=margin_voxels,
        superior_margin_voxels=superior_margin_voxels,
    )
    if bounds is None:
        raise ValueError("Lung mask is empty; refusing to silently use a full-volume fallback.")
    return bounds, lung


def _save_like(data: np.ndarray, reference: nib.spatialimages.SpatialImage, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    output = nib.Nifti1Image(data, reference.affine, header)
    output.header.set_slope_inter(1.0, 0.0)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qcode))
    if sform is not None:
        output.set_sform(sform, int(scode))
    nib.save(output, str(destination))


def write_lung_roi_ct(
    ct_path: Path,
    lung_path: Path,
    destination: Path,
    *,
    margin_voxels: int = 8,
    superior_margin_voxels: int = 120,
    intensity_scale: float | None = None,
    intensity_offset: float = 0.0,
) -> dict:
    """Write a full-grid CT that is zero outside the lung/trachea ROI.

    ``intensity_scale`` and ``intensity_offset`` are deliberately opt-in. When
    provided, the source values inside the ROI are decoded as
    ``value * scale + offset`` and written as float32. Voxels outside the ROI
    remain exactly zero rather than receiving the offset.
    """
    ct_image = nib.load(str(ct_path))
    lung_image = nib.load(str(lung_path))
    bounds, lung = resolve_lung_roi(
        ct_image,
        lung_image,
        margin_voxels=margin_voxels,
        superior_margin_voxels=superior_margin_voxels,
    )
    proxy = ct_image.dataobj
    source_dtype = str(np.dtype(ct_image.get_data_dtype()))
    intensity_transform = None
    if intensity_scale is not None:
        scale = float(intensity_scale)
        offset = float(intensity_offset)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"intensity_scale must be finite and positive, got {scale!r}.")
        if not np.isfinite(offset):
            raise ValueError(f"intensity_offset must be finite, got {offset!r}.")
        proxy_slope = getattr(proxy, "slope", 1.0)
        proxy_intercept = getattr(proxy, "inter", 0.0)
        proxy_slope = 1.0 if proxy_slope is None else float(proxy_slope)
        proxy_intercept = 0.0 if proxy_intercept is None else float(proxy_intercept)
        if not np.isclose(proxy_slope, 1.0) or not np.isclose(proxy_intercept, 0.0):
            raise ValueError(
                "Custom stored-value intensity decoding requires identity NIfTI "
                f"scaling, got slope={proxy_slope:g}, intercept={proxy_intercept:g}."
            )
        get_unscaled = getattr(proxy, "get_unscaled", None)
        raw_source = get_unscaled() if get_unscaled is not None else np.asanyarray(proxy)
        source_min = float(np.min(raw_source))
        source_max = float(np.max(raw_source))
        source = (
            np.asarray(raw_source, dtype=np.float32) * np.float32(scale)
            + np.float32(offset)
        )
        intensity_transform = {
            "scale": scale,
            "offset": offset,
            "source_dtype": source_dtype,
            "output_dtype": str(np.dtype(source.dtype)),
            "source_range": [source_min, source_max],
            "decoded_range": [
                source_min * scale + offset,
                source_max * scale + offset,
            ],
        }
    else:
        source = np.asanyarray(proxy)
    gated = np.zeros(source.shape, dtype=source.dtype)
    gated[bounds] = source[bounds]
    _save_like(gated, ct_image, destination)
    roi_shape = tuple(int(axis.stop - axis.start) for axis in bounds)
    return {
        "ct": str(Path(ct_path)),
        "lung": str(Path(lung_path)),
        "output": str(Path(destination)),
        "original_shape": [int(v) for v in source.shape],
        "bbox": bbox_to_json(bounds),
        "roi_shape": [int(v) for v in roi_shape],
        "roi_fraction": float(np.prod(roi_shape) / np.prod(source.shape)),
        "lung_voxels": int(lung.sum()),
        "outside_value": 0,
        "intensity_transform": intensity_transform,
    }


def write_roi_ground_truth(
    gt_path: Path,
    ct_path: Path,
    bounds: tuple[slice, slice, slice],
    destination: Path,
    *,
    fail_on_foreground_loss: bool = True,
) -> dict:
    """Write real GT on the full grid, zeroing only voxels outside the ROI."""
    ct_image = nib.load(str(ct_path))
    gt_image = nib.load(str(gt_path))
    assert_same_nifti_grid(ct_image, gt_image, reference_name="CT", candidate_name="airway GT")
    gt = np.asanyarray(gt_image.dataobj)
    if not np.isin(np.unique(gt), (0, 1)).all():
        raise ValueError(f"Airway GT must be binary 0/1: {gt_path}")
    gated = np.zeros(gt.shape, dtype=np.uint8)
    gated[bounds] = (gt[bounds] > 0).astype(np.uint8)
    original_foreground = int(np.count_nonzero(gt))
    retained_foreground = int(gated.sum())
    lost_foreground = original_foreground - retained_foreground
    if fail_on_foreground_loss and lost_foreground:
        raise ValueError(
            f"Lung ROI would remove {lost_foreground} GT airway voxels from {gt_path}; "
            "increase the margins before building the experiment."
        )
    _save_like(gated, gt_image, destination)
    return {
        "gt": str(Path(gt_path)),
        "foreground_voxels": original_foreground,
        "retained_foreground_voxels": retained_foreground,
        "lost_foreground_voxels": lost_foreground,
    }


def write_ignore_target(ct_path: Path, destination: Path, ignore_index: int = 2) -> dict:
    """Write an all-ignore target without consulting any on-disk airway GT."""
    ct_image = nib.load(str(ct_path))
    target = np.full(ct_image.shape, int(ignore_index), dtype=np.uint8)
    _save_like(target, ct_image, destination)
    return {"ignore_index": int(ignore_index), "shape": [int(v) for v in ct_image.shape]}
