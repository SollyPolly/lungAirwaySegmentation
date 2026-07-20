"""Prediction discovery and rendering helpers for ``mask_visualisation.py``.

The browser viewer intentionally uses Plotly rather than a JavaScript NIfTI
widget.  Plotly receives compact airway surface meshes and individual CT
slices, so a 200--300 MB scan never has to cross the notebook websocket.

Two on-disk layouts are supported:

* viewer exports under ``runs/**/predictions*``;
* native nnU-Net masks under ``data/nnunet/predict_out`` (and a project-level
  ``nnunet`` directory, when present).

The functions in this module are UI-agnostic so discovery, alignment, mesh
generation, and ITK-SNAP launching can be tested without starting Marimo.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    apply_orientation,
    io_orientation,
    inv_ornt_aff,
    ornt_transform,
)
from skimage.measure import marching_cubes


SourceKind = Literal["run", "nnunet"]

_NIFTI_SUFFIXES = (".nii.gz", ".nii")
_CASE_NUMBER = re.compile(r"(?:^|_)(\d+)(?:_0000)?$")
_IGNORED_NATIVE_PARTS = {
    "imagestr",
    "labelstr",
    "lungtr",
    "predict_in",
    "nnunet_raw",
    "nnunet_preprocessed",
    "nnunet_results",
}


@dataclass(frozen=True)
class PredictionSource:
    """A directory containing one prediction collection."""

    key: str
    label: str
    kind: SourceKind
    prediction_dir: Path
    run_dir: Path | None = None
    dataset_hint: str | None = None
    modified_ns: int = 0


@dataclass(frozen=True)
class PredictionMask:
    """One selectable mask variant for a case."""

    label: str
    path: Path


@dataclass(frozen=True)
class PredictionCase:
    """A case and the masks available for it."""

    case_id: str
    masks: tuple[PredictionMask, ...]


@dataclass(frozen=True)
class CroppedMask:
    """A binary mask cropped in the voxel grid of its reference CT."""

    data: np.ndarray
    offset: tuple[int, int, int]
    full_shape: tuple[int, int, int]

    @property
    def voxel_count(self) -> int:
        return int(np.count_nonzero(self.data))


@dataclass
class PredictionBundle:
    """The selected CT, masks, metadata, and lightweight computed metrics."""

    source: PredictionSource
    case_id: str
    prediction_path: Path
    ct_path: Path
    ground_truth_path: Path | None
    lung_mask_path: Path | None
    ct_image: nib.spatialimages.SpatialImage
    affine: np.ndarray
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    prediction: CroppedMask
    ground_truth: CroppedMask | None
    metrics: dict[str, float | int | None]
    metadata: dict[str, Any]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MaskMesh:
    """A marching-cubes surface in world coordinates."""

    vertices: np.ndarray
    faces: np.ndarray
    stride: int


def strip_nifti_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in _NIFTI_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalise_dataset_name(value: object) -> str | None:
    text = str(value or "").strip().lower().replace("'", "")
    if "atm" in text:
        return "atm22"
    if "aeropath" in text or "aero_path" in text:
        return "aeropath"
    return None


def _run_dataset_hint(run_dir: Path) -> str | None:
    metadata = _safe_json(run_dir / "run_metadata.json")
    config = _safe_json(run_dir / "resolved_config.json")
    candidates = (
        config.get("data", {}).get("dataset_name")
        if isinstance(config.get("data"), dict)
        else None,
        metadata.get("dataset_name"),
        metadata.get("data_root"),
        run_dir.parent.name,
    )
    return next(
        (name for value in candidates if (name := _normalise_dataset_name(value))),
        None,
    )


def _run_mask_paths(case_dir: Path) -> list[Path]:
    preferred = [
        path
        for path in case_dir.glob("airway_pred*_full.nii.gz")
        if "prob" not in path.name.lower()
        and "lung_masked" not in path.name.lower()
    ]
    if preferred:
        return sorted(preferred, key=lambda path: _mask_sort_key(path.name))

    fallbacks = [
        path
        for path in case_dir.iterdir()
        if path.is_file()
        and path.name.lower().endswith(_NIFTI_SUFFIXES)
        and "prob" not in path.name.lower()
        and "cropped" not in path.name.lower()
    ]
    return sorted(fallbacks, key=lambda path: _mask_sort_key(path.name))


def _mask_sort_key(name: str) -> tuple[int, tuple[Any, ...]]:
    priorities = {
        "airway_pred_full.nii.gz": 0,
        "airway_pred_lcc_full.nii.gz": 1,
        "airway_pred_lcc18_full.nii.gz": 2,
        "airway_pred_lcc26_full.nii.gz": 3,
    }
    return priorities.get(name.lower(), 10), _natural_key(name)


def _mask_label(path: Path, *, native: bool = False) -> str:
    labels = {
        "airway_pred_full.nii.gz": "Raw prediction",
        "airway_pred_lcc_full.nii.gz": "Largest component (6-connectivity)",
        "airway_pred_lcc18_full.nii.gz": "Largest component (18-connectivity)",
        "airway_pred_lcc26_full.nii.gz": "Largest component (26-connectivity)",
    }
    if path.name.lower() in labels:
        return labels[path.name.lower()]
    return "nnU-Net output" if native else path.name


def _directory_has_run_predictions(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_dir() and _run_mask_paths(child):
                return True
    except OSError:
        return False
    return False


def _native_prediction_files(path: Path) -> list[Path]:
    try:
        files = [
            child
            for child in path.iterdir()
            if child.is_file()
            and child.name.lower().endswith(_NIFTI_SUFFIXES)
            and not strip_nifti_suffix(child.name).lower().endswith("_0000")
            and "prob" not in child.name.lower()
        ]
    except OSError:
        return []
    return sorted(files, key=lambda child: _natural_key(child.name))


def _iter_native_prediction_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    found: list[Path] = []
    for current, dirnames, _ in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name.lower() not in _IGNORED_NATIVE_PARTS and not name.startswith(".")
        ]
        current_path = Path(current)
        if _native_prediction_files(current_path):
            found.append(current_path)
    return found


def discover_prediction_sources(project_root: Path) -> list[PredictionSource]:
    """Discover exported runs and directly usable native nnU-Net outputs."""

    project_root = Path(project_root).resolve()
    sources: list[PredictionSource] = []
    seen_directories: set[Path] = set()

    runs_root = project_root / "runs"
    if runs_root.is_dir():
        for prediction_dir in runs_root.rglob("predictions*"):
            if not prediction_dir.is_dir() or not _directory_has_run_predictions(prediction_dir):
                continue
            resolved = prediction_dir.resolve()
            if resolved in seen_directories:
                continue
            seen_directories.add(resolved)
            run_dir = prediction_dir.parent.resolve()
            relative = prediction_dir.relative_to(runs_root)
            hint = _run_dataset_hint(run_dir)
            prefix = "runs / nnU-Net" if "nnunet" in str(relative).lower() else "runs"
            sources.append(
                PredictionSource(
                    key=f"run::{resolved}",
                    label=f"{prefix} · {relative}",
                    kind="run",
                    prediction_dir=resolved,
                    run_dir=run_dir,
                    dataset_hint=hint,
                    modified_ns=prediction_dir.stat().st_mtime_ns,
                )
            )

    native_roots = [
        project_root / "data" / "nnunet" / "predict_out",
        project_root / "nnunet" / "predict_out",
        project_root / "nnunet",
    ]
    for native_root in native_roots:
        for prediction_dir in _iter_native_prediction_directories(native_root):
            resolved = prediction_dir.resolve()
            if resolved in seen_directories:
                continue
            seen_directories.add(resolved)
            try:
                relative = prediction_dir.relative_to(project_root)
            except ValueError:
                relative = prediction_dir
            sources.append(
                PredictionSource(
                    key=f"nnunet::{resolved}",
                    label=f"nnU-Net native · {relative}",
                    kind="nnunet",
                    prediction_dir=resolved,
                    dataset_hint="atm22",
                    modified_ns=prediction_dir.stat().st_mtime_ns,
                )
            )

    # Native outputs first, then viewer exports; newest first within each group.
    return sorted(
        sources,
        key=lambda source: (source.kind != "nnunet", -source.modified_ns, source.label.lower()),
    )


def case_id_from_prediction(path: Path) -> str:
    stem = strip_nifti_suffix(path.name)
    match = _CASE_NUMBER.search(stem)
    if not match:
        return stem
    digits = match.group(1)
    return digits.zfill(3) if stem.upper().startswith("ATM_") else digits


def list_prediction_cases(source: PredictionSource) -> list[PredictionCase]:
    """List cases without requiring metadata sidecars."""

    if source.kind == "run":
        cases = []
        for case_dir in source.prediction_dir.iterdir():
            if not case_dir.is_dir():
                continue
            masks = tuple(
                PredictionMask(_mask_label(path), path.resolve())
                for path in _run_mask_paths(case_dir)
            )
            if masks:
                cases.append(PredictionCase(case_dir.name, masks))
    else:
        grouped: dict[str, list[Path]] = {}
        for path in _native_prediction_files(source.prediction_dir):
            grouped.setdefault(case_id_from_prediction(path), []).append(path)
        cases = [
            PredictionCase(
                case_id,
                tuple(
                    PredictionMask(_mask_label(path, native=True), path.resolve())
                    for path in paths
                ),
            )
            for case_id, paths in grouped.items()
        ]
    return sorted(cases, key=lambda case: _natural_key(case.case_id))


def _resolve_recorded_root(path_value: object, project_root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _run_data_candidates(source: PredictionSource, project_root: Path) -> list[Path]:
    if source.run_dir is None:
        return []
    metadata = _safe_json(source.run_dir / "run_metadata.json")
    config = _safe_json(source.run_dir / "resolved_config.json")
    data_config = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    candidates = [
        _resolve_recorded_root(metadata.get("data_root"), project_root),
        _resolve_recorded_root(data_config.get("raw_data_root"), project_root),
        _resolve_recorded_root(data_config.get("batch_root"), project_root),
    ]
    return [candidate for candidate in candidates if candidate is not None]


def _atm_paths(root: Path, case_id: str) -> tuple[Path, Path | None, Path | None]:
    padded = str(case_id).zfill(3)
    ct = root / "imagesTr" / f"ATM_{padded}_0000.nii.gz"
    gt_candidates = (
        root / "labelsTr" / f"ATM_{padded}.nii.gz",
        root / "labelsTr" / f"ATM_{padded}_0000.nii.gz",
    )
    ground_truth = next((path for path in gt_candidates if path.is_file()), None)
    lung = root / "lungTr" / f"ATM_{padded}_lung.nii.gz"
    return ct, ground_truth, lung if lung.is_file() else None


def _aeropath_paths(root: Path, case_id: str) -> tuple[Path, Path | None, Path | None]:
    text = str(case_id)
    case_dir = root / text
    ct = case_dir / f"{text}_CT_HR.nii.gz"
    gt = case_dir / f"{text}_CT_HR_label_airways.nii.gz"
    lung = case_dir / f"{text}_CT_HR_label_lungs.nii.gz"
    return ct, gt if gt.is_file() else None, lung if lung.is_file() else None


def resolve_reference_paths(
    source: PredictionSource,
    case_id: str,
    project_root: Path,
) -> tuple[str, Path, Path | None, Path | None, Path]:
    """Resolve CT/GT/lung paths, preferring paths recorded by a run."""

    project_root = Path(project_root).resolve()
    recorded = _run_data_candidates(source, project_root)
    dataset_order = (
        [source.dataset_hint]
        if source.dataset_hint in {"atm22", "aeropath"}
        else ["atm22", "aeropath"]
    )
    attempted: list[Path] = []

    for dataset in dataset_order:
        defaults = (
            [project_root / "data" / "ATM22"]
            if dataset == "atm22"
            else [project_root / "data" / "AeroPath"]
        )
        roots = []
        for root in [*recorded, *defaults]:
            resolved = root.resolve()
            if resolved not in roots:
                roots.append(resolved)
        for root in roots:
            ct, gt, lung = (
                _atm_paths(root, case_id)
                if dataset == "atm22"
                else _aeropath_paths(root, case_id)
            )
            attempted.append(ct)
            if ct.is_file():
                return dataset, ct.resolve(), gt.resolve() if gt else None, lung.resolve() if lung else None, root

    attempted_text = "\n".join(f"  - {path}" for path in attempted)
    raise FileNotFoundError(
        f"Could not find the CT for case {case_id}. Tried:\n{attempted_text}"
    )


def _load_mask_in_reference_grid(
    path: Path,
    reference: nib.spatialimages.SpatialImage,
    *,
    name: str,
) -> tuple[np.ndarray, str | None]:
    image = nib.load(str(path))
    reference_affine = np.asarray(reference.affine, dtype=np.float64)
    image_affine = np.asarray(image.affine, dtype=np.float64)

    if image.shape == reference.shape and np.allclose(
        image_affine, reference_affine, atol=1e-4, rtol=1e-5
    ):
        # Some ATM labels are stored as float64 despite containing only integer
        # labels.  Reading directly as uint8 avoids a transient 1.5 GB array for
        # a 512 x 512 x 714 volume.
        return np.asarray(image.dataobj, dtype=np.uint8) > 0, None

    # Predictions written in canonical RAS orientation and original LPS dataset
    # files describe the same grid with axis flips. Reorient masks losslessly;
    # never interpolate a segmentation silently.
    transform = ornt_transform(io_orientation(image_affine), io_orientation(reference_affine))
    data = apply_orientation(
        np.asarray(image.dataobj, dtype=np.uint8), transform
    )
    reoriented_affine = image_affine @ inv_ornt_aff(transform, image.shape)
    if data.shape != reference.shape or not np.allclose(
        reoriented_affine, reference_affine, atol=1e-4, rtol=1e-5
    ):
        raise ValueError(
            f"{name} is not aligned with the CT: shape {image.shape} -> {data.shape}, "
            f"expected {reference.shape}, and the affine grids differ."
        )
    warning = f"{name} was losslessly reoriented to the CT voxel grid."
    return data > 0, warning


def _crop_masks(
    prediction: np.ndarray,
    ground_truth: np.ndarray | None,
    *,
    padding: int = 2,
) -> tuple[CroppedMask, CroppedMask | None]:
    shape = tuple(int(value) for value in prediction.shape)
    occupied_bounds = []
    for mask in (prediction, ground_truth):
        if mask is None or not mask.any():
            continue
        coordinates = np.where(mask)
        occupied_bounds.append(
            tuple(
                (int(axis_coordinates.min()), int(axis_coordinates.max()))
                for axis_coordinates in coordinates
            )
        )
    if not occupied_bounds:
        slices = tuple(slice(0, min(size, 1)) for size in shape)
    else:
        slices = tuple(
            slice(
                max(0, min(bounds[axis][0] for bounds in occupied_bounds) - padding),
                min(
                    size,
                    max(bounds[axis][1] for bounds in occupied_bounds) + padding + 1,
                ),
            )
            for axis, size in enumerate(shape)
        )
    offset = tuple(int(item.start or 0) for item in slices)
    pred_crop = CroppedMask(
        np.ascontiguousarray(prediction[slices], dtype=bool), offset, shape
    )
    gt_crop = (
        CroppedMask(np.ascontiguousarray(ground_truth[slices], dtype=bool), offset, shape)
        if ground_truth is not None
        else None
    )
    return pred_crop, gt_crop


def segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray | None,
) -> dict[str, float | int | None]:
    predicted = int(np.count_nonzero(prediction))
    if ground_truth is None:
        return {
            "dice": None,
            "iou": None,
            "precision": None,
            "recall": None,
            "predicted_voxels": predicted,
            "ground_truth_voxels": None,
            "true_positive_voxels": None,
            "false_positive_voxels": None,
            "missed_voxels": None,
        }
    truth = int(np.count_nonzero(ground_truth))
    # Indexing prediction only at GT foreground avoids allocating another
    # full-volume boolean array for the intersection.
    true_positive = int(np.count_nonzero(prediction[ground_truth]))
    false_positive = predicted - true_positive
    false_negative = truth - true_positive
    union = predicted + truth - true_positive
    return {
        "dice": (2.0 * true_positive / (predicted + truth)) if predicted + truth else 1.0,
        "iou": (true_positive / union) if union else 1.0,
        "precision": (true_positive / predicted) if predicted else (1.0 if not truth else 0.0),
        "recall": (true_positive / truth) if truth else (1.0 if not predicted else 0.0),
        "predicted_voxels": predicted,
        "ground_truth_voxels": truth,
        "true_positive_voxels": true_positive,
        "false_positive_voxels": false_positive,
        "missed_voxels": false_negative,
    }


def _bundle_metadata(source: PredictionSource, prediction_path: Path) -> dict[str, Any]:
    if source.run_dir is None:
        return {"source_type": "Native nnU-Net output"}
    run_metadata = _safe_json(source.run_dir / "run_metadata.json")
    history = _safe_json(source.run_dir / "history.json")
    prediction_metadata = _safe_json(prediction_path.parent / "prediction_metadata.json")
    best = history.get("best", {}) if isinstance(history.get("best"), dict) else {}
    return {
        "source_type": "Exported run",
        "study_name": run_metadata.get("study_name"),
        "run_label": run_metadata.get("run_label"),
        "experiment_name": run_metadata.get("experiment_name"),
        "threshold": prediction_metadata.get("threshold"),
        "checkpoint_epoch": prediction_metadata.get("checkpoint_epoch"),
        "best_val_dice": best.get("val_dice"),
    }


def load_prediction_bundle(
    source: PredictionSource,
    case_id: str,
    prediction_path: Path,
    project_root: Path,
) -> PredictionBundle:
    """Load masks, but keep the large CT lazy for per-slice reads."""

    prediction_path = Path(prediction_path).resolve(strict=True)
    dataset, ct_path, gt_path, lung_path, _ = resolve_reference_paths(
        source, case_id, project_root
    )
    ct_image = nib.load(str(ct_path))
    if len(ct_image.shape) != 3:
        raise ValueError(f"CT must be 3D, got {ct_image.shape}: {ct_path}")

    warnings: list[str] = []
    prediction, warning = _load_mask_in_reference_grid(
        prediction_path, ct_image, name="Prediction"
    )
    if warning:
        warnings.append(warning)

    if gt_path is None:
        ground_truth = None
        warnings.append("No ground-truth airway mask was found for this case.")
    else:
        ground_truth, warning = _load_mask_in_reference_grid(
            gt_path, ct_image, name="Ground truth"
        )
        if warning:
            warnings.append(warning)

    metrics = segmentation_metrics(prediction, ground_truth)
    prediction_crop, ground_truth_crop = _crop_masks(prediction, ground_truth)
    del prediction, ground_truth

    zooms = ct_image.header.get_zooms()[:3]
    metadata = _bundle_metadata(source, prediction_path)
    metadata["dataset_name"] = dataset
    return PredictionBundle(
        source=source,
        case_id=str(case_id),
        prediction_path=prediction_path,
        ct_path=ct_path,
        ground_truth_path=gt_path,
        lung_mask_path=lung_path,
        ct_image=ct_image,
        affine=np.asarray(ct_image.affine, dtype=np.float64),
        shape=tuple(int(value) for value in ct_image.shape),
        spacing=tuple(float(value) for value in zooms),
        prediction=prediction_crop,
        ground_truth=ground_truth_crop,
        metrics=metrics,
        metadata=metadata,
        warnings=tuple(warnings),
    )


def combine_cropped_masks(
    first: CroppedMask,
    second: CroppedMask,
    operation: Literal["and", "first_only", "second_only"],
) -> CroppedMask:
    if first.full_shape != second.full_shape or first.offset != second.offset:
        raise ValueError("Cropped masks must share a grid and crop.")
    if operation == "and":
        data = first.data & second.data
    elif operation == "first_only":
        data = first.data & ~second.data
    elif operation == "second_only":
        data = second.data & ~first.data
    else:
        raise ValueError(f"Unsupported mask operation: {operation}")
    return CroppedMask(data, first.offset, first.full_shape)


def build_mask_mesh(
    mask: CroppedMask,
    affine: np.ndarray,
    *,
    preferred_stride: int = 1,
    max_sampled_foreground_voxels: int = 450_000,
) -> MaskMesh | None:
    """Create a compact world-coordinate mesh with an adaptive safety stride."""

    data = np.asarray(mask.data, dtype=bool)
    if not data.any():
        return None
    stride = max(1, int(preferred_stride))
    while (
        stride < 8
        and int(np.count_nonzero(data[::stride, ::stride, ::stride]))
        > max_sampled_foreground_voxels
    ):
        stride += 1

    sampled = data[::stride, ::stride, ::stride]
    while stride > 1 and not sampled.any():
        stride -= 1
        sampled = data[::stride, ::stride, ::stride]
    coordinates = np.argwhere(sampled)
    lower = np.maximum(coordinates.min(axis=0) - 1, 0)
    upper = np.minimum(coordinates.max(axis=0) + 2, sampled.shape)
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
    cropped = sampled[slices]
    # Zero padding guarantees a closed isosurface even when foreground reaches
    # the edge of the stored crop.
    padded = np.pad(cropped.astype(np.uint8), 1, mode="constant")
    vertices, faces, _, _ = marching_cubes(padded, level=0.5)
    voxel_vertices = (
        (vertices - 1.0 + lower.astype(np.float32)) * float(stride)
        + np.asarray(mask.offset, dtype=np.float32)
    )
    world_vertices = nib.affines.apply_affine(affine, voxel_vertices).astype(
        np.float32, copy=False
    )
    return MaskMesh(
        vertices=world_vertices,
        faces=faces.astype(np.int32, copy=False),
        stride=stride,
    )


def default_slice_index(masks: Sequence[CroppedMask | None], axis: int) -> int:
    valid = [mask for mask in masks if mask is not None and mask.data.any()]
    if not valid:
        shape = next((mask.full_shape for mask in masks if mask is not None), (1, 1, 1))
        return int(shape[axis]) // 2
    starts = []
    stops = []
    for mask in valid:
        projection_axes = tuple(index for index in range(3) if index != axis)
        occupied = np.flatnonzero(np.any(mask.data, axis=projection_axes))
        if occupied.size:
            starts.append(int(occupied.min()) + mask.offset[axis])
            stops.append(int(occupied.max()) + mask.offset[axis])
    return (min(starts) + max(stops)) // 2 if starts else valid[0].full_shape[axis] // 2


def extract_ct_plane(
    image: nib.spatialimages.SpatialImage,
    axis: int,
    index: int,
) -> np.ndarray:
    slicer: list[int | slice] = [slice(None), slice(None), slice(None)]
    slicer[axis] = int(index)
    plane = np.asarray(image.dataobj[tuple(slicer)], dtype=np.float32)
    return np.rot90(plane)


def extract_mask_plane(mask: CroppedMask | None, axis: int, index: int) -> np.ndarray | None:
    if mask is None:
        return None
    plane_shape = tuple(size for dim, size in enumerate(mask.full_shape) if dim != axis)
    output = np.zeros(plane_shape, dtype=bool)
    local_index = int(index) - mask.offset[axis]
    if 0 <= local_index < mask.data.shape[axis]:
        source_slicer: list[int | slice] = [slice(None), slice(None), slice(None)]
        source_slicer[axis] = local_index
        source = mask.data[tuple(source_slicer)]
        target_slices = tuple(
            slice(mask.offset[dim], mask.offset[dim] + mask.data.shape[dim])
            for dim in range(3)
            if dim != axis
        )
        output[target_slices] = source
    return np.rot90(output)


def find_itksnap_executable(
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Find ITK-SNAP without hard-coding one platform or installation version."""

    environment = os.environ if environ is None else environ
    configured = environment.get("ITKSNAP_PATH")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()

    for command in ("ITK-SNAP.exe", "itksnap.exe", "itksnap"):
        resolved = shutil.which(command)
        if resolved:
            return Path(resolved).resolve()

    candidates = []
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root = environment.get(variable)
        if root:
            candidates.append(Path(root) / "ITK-SNAP 4.4" / "bin" / "ITK-SNAP.exe")
            candidates.append(Path(root) / "ITK-SNAP 4.2" / "bin" / "ITK-SNAP.exe")
    candidates.extend(
        [
            Path("/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP"),
            Path("/usr/bin/itksnap"),
            Path("/usr/local/bin/itksnap"),
        ]
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def build_itksnap_command(
    executable: Path,
    ct_path: Path,
    prediction_path: Path,
    ground_truth_path: Path | None = None,
) -> list[str]:
    """Build a no-shell command with GT and prediction as separate label layers."""

    command = [str(Path(executable)), "-g", str(Path(ct_path))]
    segmentations = [
        path
        for path in (ground_truth_path, prediction_path)
        if path is not None
    ]
    if segmentations:
        command.append("-s")
        command.extend(str(Path(path)) for path in segmentations)
    return command


def launch_itksnap(
    executable: Path,
    ct_path: Path,
    prediction_path: Path,
    ground_truth_path: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Open an independent ITK-SNAP process for the selected case."""

    command = build_itksnap_command(
        executable, ct_path, prediction_path, ground_truth_path
    )
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(
        command,
        cwd=str(Path(ct_path).parent),
        close_fds=os.name != "nt",
        creationflags=creationflags,
    )


__all__ = [
    "CroppedMask",
    "MaskMesh",
    "PredictionBundle",
    "PredictionCase",
    "PredictionMask",
    "PredictionSource",
    "build_itksnap_command",
    "build_mask_mesh",
    "case_id_from_prediction",
    "combine_cropped_masks",
    "default_slice_index",
    "discover_prediction_sources",
    "extract_ct_plane",
    "extract_mask_plane",
    "find_itksnap_executable",
    "launch_itksnap",
    "list_prediction_cases",
    "load_prediction_bundle",
    "resolve_reference_paths",
    "segmentation_metrics",
    "strip_nifti_suffix",
]
