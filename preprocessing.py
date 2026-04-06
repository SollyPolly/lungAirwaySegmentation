from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.spatialimages import SpatialImage

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data" / "Aeropath"
DEFAULT_PROCESSED_ROOT = Path(__file__).resolve().parent / "data" / "processed"
DEFAULT_HU_WINDOW: tuple[float, float] = (-1024.0, 600.0)
DEFAULT_CROP_MARGIN = 5

Spacing3D = tuple[float, float, float]
Shape3D = tuple[int, int, int]
CropBox3D = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
SummaryRow = dict[str, object]

SUMMARY_COLUMNS = [
    "case_id",
    "status",
    "error",
    "original_shape",
    "processed_shape",
    "spacing",
    "crop_box",
    "crop_margin",
    "airway_voxels",
    "lung_voxels",
    "ct_min",
    "ct_max",
    "ct_mean",
    "ct_path",
    "lung_mask_path",
    "airway_mask_path",
]


class CasePaths(TypedDict):
    case_id: str
    case_dir: Path
    ct: Path
    lung: Path
    airway: Path


class PreprocessMetadata(TypedDict):
    case_dir: Path
    ct_path: Path
    lung_mask_path: Path
    airway_mask_path: Path
    original_shape: Shape3D
    processed_shape: Shape3D
    original_spacing: Spacing3D
    original_affine: np.ndarray
    cropped_affine: np.ndarray
    crop_margin: Shape3D
    hu_window: tuple[float, float]


class PreprocessedCase(TypedDict):
    case_id: str
    ct: np.ndarray
    airway_mask: np.ndarray
    lung_mask: np.ndarray | None
    spacing: Spacing3D
    affine: np.ndarray
    crop_box: CropBox3D
    metadata: PreprocessMetadata


class SavedCasePaths(TypedDict):
    case_dir: Path
    ct: Path
    airway_mask: Path
    lung_mask: Path | None
    metadata_json: Path


def resolve_case_paths(
    case_id: str | int,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> CasePaths:
    case_str = str(case_id)
    case_dir = Path(data_root) / case_str
    return {
        "case_id": case_str,
        "case_dir": case_dir,
        "ct": case_dir / f"{case_str}_CT_HR.nii.gz",
        "lung": case_dir / f"{case_str}_CT_HR_label_lungs.nii.gz",
        "airway": case_dir / f"{case_str}_CT_HR_label_airways.nii.gz",
    }


def list_case_ids(data_root: Path = DEFAULT_DATA_ROOT) -> list[str]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    case_ids = [
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not case_ids:
        raise ValueError(f"No numeric case directories were found in {data_root}.")

    return sorted(case_ids, key=int)


def load_canonical_image(path: Path) -> SpatialImage:
    if not path.exists():
        raise FileNotFoundError(f"Missing NIfTI file: {path}")
    return nib.as_closest_canonical(nib.load(str(path)))


def ensure_3d(image: SpatialImage, name: str) -> None:
    if len(image.shape) != 3:
        raise ValueError(f"{name} must be 3D, but got shape {image.shape}.")


def affine_from_image(image: SpatialImage, name: str) -> np.ndarray:
    affine = image.affine
    if affine is None:
        raise ValueError(f"{name} does not have an affine matrix.")
    return np.asarray(affine, dtype=np.float64)


def verify_alignment(
    reference_image: SpatialImage,
    other_image: SpatialImage,
    *,
    reference_name: str,
    other_name: str,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> None:
    if reference_image.shape != other_image.shape:
        raise ValueError(
            f"{other_name} is not aligned with {reference_name}: "
            f"shape {other_image.shape} != {reference_image.shape}."
        )

    reference_affine = affine_from_image(reference_image, reference_name)
    other_affine = affine_from_image(other_image, other_name)
    if not np.allclose(reference_affine, other_affine, atol=atol, rtol=rtol):
        raise ValueError(
            f"{other_name} is not aligned with {reference_name}: affine matrices differ."
        )


def spacing_from_image(image: SpatialImage) -> Spacing3D:
    zooms = image.header.get_zooms()[:3]
    if len(zooms) != 3:
        raise ValueError(f"Expected 3 spatial zooms, but got {zooms}.")
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def shape3d_from_array(array: np.ndarray, name: str) -> Shape3D:
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3D, but got shape {array.shape}.")
    return (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))


def normalize_margin(margin: int | tuple[int, int, int]) -> Shape3D:
    if isinstance(margin, int):
        if margin < 0:
            raise ValueError("crop_margin must be non-negative.")
        return (margin, margin, margin)

    if len(margin) != 3:
        raise ValueError("crop_margin must be an int or a 3-tuple of ints.")

    normalized = (int(margin[0]), int(margin[1]), int(margin[2]))
    if any(value < 0 for value in normalized):
        raise ValueError("crop_margin values must be non-negative.")
    return normalized


def bbox_from_mask(mask: np.ndarray, margin: int | tuple[int, int, int] = 0) -> CropBox3D:
    mask_shape = shape3d_from_array(mask, "mask")
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        raise ValueError("The lung mask is empty, so a crop box cannot be computed.")

    margin_zyx = np.asarray(normalize_margin(margin))
    starts = np.maximum(foreground.min(axis=0) - margin_zyx, 0)
    stops = np.minimum(foreground.max(axis=0) + 1 + margin_zyx, mask_shape)

    return (
        (int(starts[0]), int(stops[0])),
        (int(starts[1]), int(stops[1])),
        (int(starts[2]), int(stops[2])),
    )


def crop_box_to_slices(crop_box: CropBox3D) -> tuple[slice, slice, slice]:
    z_bounds, y_bounds, x_bounds = crop_box
    return (
        slice(z_bounds[0], z_bounds[1]),
        slice(y_bounds[0], y_bounds[1]),
        slice(x_bounds[0], x_bounds[1]),
    )


def crop_volume(volume: np.ndarray, crop_box: CropBox3D) -> np.ndarray:
    return volume[crop_box_to_slices(crop_box)]


def affine_after_crop(affine: np.ndarray, crop_box: CropBox3D) -> np.ndarray:
    crop_start = np.array([bounds[0] for bounds in crop_box], dtype=np.float64)
    voxel_translation = np.eye(4, dtype=np.float64)
    voxel_translation[:3, 3] = crop_start
    return affine @ voxel_translation


def clip_ct_to_window(volume: np.ndarray, hu_window: tuple[float, float]) -> np.ndarray:
    lower, upper = hu_window
    if upper <= lower:
        raise ValueError("hu_window must have upper > lower.")
    return np.clip(volume, lower, upper).astype(np.float32, copy=False)


def normalize_ct(volume: np.ndarray, hu_window: tuple[float, float]) -> np.ndarray:
    lower, upper = hu_window
    clipped = clip_ct_to_window(volume, hu_window)
    normalized = (clipped - lower) / (upper - lower)
    return normalized.astype(np.float32, copy=False)


def preprocess_case(
    case_id: str | int,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    include_lung_mask: bool = False,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
    crop_margin: int | tuple[int, int, int] = DEFAULT_CROP_MARGIN,
) -> PreprocessedCase:
    """
    Load and preprocess one AeroPath case at the original voxel spacing.

    The returned CT is cropped, clipped to the requested HU window, and
    normalized to [0, 1]. Masks remain binary.
    """

    paths = resolve_case_paths(case_id, data_root=data_root)

    ct_img = load_canonical_image(paths["ct"])
    lung_img = load_canonical_image(paths["lung"])
    airway_img = load_canonical_image(paths["airway"])

    ensure_3d(ct_img, "CT")
    ensure_3d(lung_img, "Lung mask")
    ensure_3d(airway_img, "Airway mask")

    verify_alignment(ct_img, lung_img, reference_name="CT", other_name="lung mask")
    verify_alignment(ct_img, airway_img, reference_name="CT", other_name="airway mask")

    spacing = spacing_from_image(ct_img)
    original_affine = affine_from_image(ct_img, "CT")

    ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    lung_mask = np.asarray(lung_img.dataobj) > 0
    airway_mask = np.asarray(airway_img.dataobj) > 0

    crop_box = bbox_from_mask(lung_mask, margin=crop_margin)
    cropped_affine = affine_after_crop(original_affine, crop_box)

    cropped_ct = crop_volume(ct, crop_box).astype(np.float32, copy=False)
    cropped_airway_mask = crop_volume(airway_mask, crop_box).astype(np.uint8, copy=False)
    cropped_lung_mask = crop_volume(lung_mask, crop_box).astype(np.uint8, copy=False)

    metadata: PreprocessMetadata = {
        "case_dir": paths["case_dir"],
        "ct_path": paths["ct"],
        "lung_mask_path": paths["lung"],
        "airway_mask_path": paths["airway"],
        "original_shape": shape3d_from_array(ct, "ct"),
        "processed_shape": shape3d_from_array(cropped_ct, "cropped ct"),
        "original_spacing": spacing,
        "original_affine": np.array(original_affine, copy=True),
        "cropped_affine": np.array(cropped_affine, copy=True),
        "crop_margin": normalize_margin(crop_margin),
        "hu_window": hu_window,
    }

    result: PreprocessedCase = {
        "case_id": paths["case_id"],
        "ct": normalize_ct(cropped_ct, hu_window),
        "airway_mask": (cropped_airway_mask > 0).astype(np.uint8, copy=False),
        "lung_mask": (cropped_lung_mask > 0).astype(np.uint8, copy=False)
        if include_lung_mask
        else None,
        "spacing": spacing,
        "affine": cropped_affine,
        "crop_box": crop_box,
        "metadata": metadata,
    }
    return result


def summarize_preprocessed_case(case: PreprocessedCase) -> SummaryRow:
    metadata = case["metadata"]
    ct = case["ct"]
    lung_mask = case["lung_mask"]

    return {
        "case_id": case["case_id"],
        "status": "ok",
        "error": None,
        "original_shape": metadata["original_shape"],
        "processed_shape": metadata["processed_shape"],
        "spacing": case["spacing"],
        "crop_box": case["crop_box"],
        "crop_margin": metadata["crop_margin"],
        "airway_voxels": int(np.count_nonzero(case["airway_mask"])),
        "lung_voxels": None if lung_mask is None else int(np.count_nonzero(lung_mask)),
        "ct_min": float(ct.min()),
        "ct_max": float(ct.max()),
        "ct_mean": float(ct.mean()),
        "ct_path": str(metadata["ct_path"]),
        "lung_mask_path": str(metadata["lung_mask_path"]),
        "airway_mask_path": str(metadata["airway_mask_path"]),
    }


def summarize_preprocessing_error(case_id: str | int, error: Exception) -> SummaryRow:
    return {
        "case_id": str(case_id),
        "status": "error",
        "error": str(error),
        "original_shape": None,
        "processed_shape": None,
        "spacing": None,
        "crop_box": None,
        "crop_margin": None,
        "airway_voxels": None,
        "lung_voxels": None,
        "ct_min": None,
        "ct_max": None,
        "ct_mean": None,
        "ct_path": None,
        "lung_mask_path": None,
        "airway_mask_path": None,
    }


def save_nifti(volume: np.ndarray, affine: np.ndarray, output_path: Path, dtype: np.dtype) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = nib.Nifti1Header()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(volume, dtype=dtype), affine, header=header)
    nib.save(image, str(output_path))
    return output_path


def metadata_to_json_ready(case: PreprocessedCase) -> dict[str, object]:
    metadata = case["metadata"]
    return {
        "case_id": case["case_id"],
        "spacing": list(case["spacing"]),
        "crop_box": [list(bounds) for bounds in case["crop_box"]],
        "ct_path": str(metadata["ct_path"]),
        "lung_mask_path": str(metadata["lung_mask_path"]),
        "airway_mask_path": str(metadata["airway_mask_path"]),
        "original_shape": list(metadata["original_shape"]),
        "processed_shape": list(metadata["processed_shape"]),
        "original_spacing": list(metadata["original_spacing"]),
        "crop_margin": list(metadata["crop_margin"]),
        "hu_window": list(metadata["hu_window"]),
        "original_affine": metadata["original_affine"].tolist(),
        "cropped_affine": metadata["cropped_affine"].tolist(),
    }


def save_preprocessed_case(
    case: PreprocessedCase,
    *,
    output_root: Path = DEFAULT_PROCESSED_ROOT,
) -> SavedCasePaths:
    case_dir = output_root / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    case_id = case["case_id"]
    ct_path = case_dir / f"{case_id}_ct_processed.nii.gz"
    airway_path = case_dir / f"{case_id}_airway_mask_processed.nii.gz"
    lung_path = case_dir / f"{case_id}_lung_mask_processed.nii.gz"
    metadata_json_path = case_dir / f"{case_id}_preprocessing_metadata.json"

    save_nifti(case["ct"], case["affine"], ct_path, np.float32)
    save_nifti(case["airway_mask"], case["affine"], airway_path, np.uint8)

    saved_lung_path: Path | None = None
    if case["lung_mask"] is not None:
        save_nifti(case["lung_mask"], case["affine"], lung_path, np.uint8)
        saved_lung_path = lung_path

    with metadata_json_path.open("w", encoding="utf-8") as file:
        json.dump(metadata_to_json_ready(case), file, indent=2)

    return {
        "case_dir": case_dir,
        "ct": ct_path,
        "airway_mask": airway_path,
        "lung_mask": saved_lung_path,
        "metadata_json": metadata_json_path,
    }


def build_summary_table(
    case_ids: list[str] | tuple[str, ...],
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    processed_root: Path | None = None,
    include_lung_mask: bool = False,
    hu_window: tuple[float, float] = DEFAULT_HU_WINDOW,
    crop_margin: int | tuple[int, int, int] = DEFAULT_CROP_MARGIN,
    continue_on_error: bool = True,
    show_progress: bool = False,
) -> pd.DataFrame:
    rows: list[SummaryRow] = []
    total_cases = len(case_ids)

    for index, case_id in enumerate(case_ids, start=1):
        if show_progress:
            print(f"[{index}/{total_cases}] Preprocessing case {case_id}...")

        try:
            case = preprocess_case(
                case_id,
                data_root=data_root,
                include_lung_mask=include_lung_mask,
                hu_window=hu_window,
                crop_margin=crop_margin,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            rows.append(summarize_preprocessing_error(case_id, exc))
            continue

        if processed_root is not None:
            save_preprocessed_case(case, output_root=processed_root)
        rows.append(summarize_preprocessed_case(case))

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def resolve_summary_csv_path(
    summary_csv: Path | None,
    *,
    run_all_cases: bool,
    processed_root: Path | None = None,
) -> Path | None:
    if summary_csv is not None:
        return summary_csv
    if run_all_cases:
        if processed_root is not None:
            return processed_root / "preprocessing_summary.csv"
        return Path(__file__).resolve().parent / "preprocessing_summary.csv"
    return None


def save_summary_table(summary_table: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(output_path, index=False)
    return output_path


def preview_summary_table(summary_table: pd.DataFrame) -> pd.DataFrame:
    preview_columns = [
        "case_id",
        "status",
        "error",
        "original_shape",
        "processed_shape",
        "spacing",
        "airway_voxels",
        "lung_voxels",
        "crop_box",
    ]
    return summary_table[preview_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess one AeroPath case or all cases at original spacing, "
            "and build a pandas summary table."
        ),
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--case",
        default="1",
        help="Single case ID to preprocess. Defaults to case 1.",
    )
    selection_group.add_argument(
        "--all-cases",
        action="store_true",
        help="Preprocess every numeric case folder under the data root.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to the AeroPath case root directory.",
    )
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Save processed NIfTI files and metadata JSON under the processed root.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Directory where processed case folders will be written.",
    )
    parser.add_argument(
        "--include-lung-mask",
        action="store_true",
        help="Keep the processed lung mask in each case result and summary.",
    )
    parser.add_argument(
        "--hu-window",
        type=float,
        nargs=2,
        metavar=("LOWER", "UPPER"),
        default=DEFAULT_HU_WINDOW,
        help="HU clipping window applied before CT normalization.",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=DEFAULT_CROP_MARGIN,
        help="Lung bounding-box margin in voxels.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Optional output path for the summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hu_window = (float(args.hu_window[0]), float(args.hu_window[1]))
    processed_root = args.processed_root if args.save_processed else None

    case_ids = list_case_ids(args.data_root) if args.all_cases else [str(args.case)]
    summary_table = build_summary_table(
        case_ids,
        data_root=args.data_root,
        processed_root=processed_root,
        include_lung_mask=args.include_lung_mask,
        hu_window=hu_window,
        crop_margin=args.crop_margin,
        continue_on_error=args.all_cases,
        show_progress=True,
    )

    success_count = int((summary_table["status"] == "ok").sum())
    error_count = int((summary_table["status"] == "error").sum())

    print(f"Processed {success_count}/{len(summary_table)} case(s) successfully.")
    if error_count:
        print(f"Encountered {error_count} case(s) with errors.")
    print(preview_summary_table(summary_table).to_string(index=False))

    summary_csv_path = resolve_summary_csv_path(
        args.summary_csv,
        run_all_cases=args.all_cases,
        processed_root=processed_root,
    )
    if summary_csv_path is not None:
        saved_path = save_summary_table(summary_table, summary_csv_path)
        print(f"Saved summary CSV to: {saved_path}")

    if processed_root is not None:
        print(f"Saved processed cases under: {processed_root}")


__all__ = [
    "CasePaths",
    "CropBox3D",
    "DEFAULT_CROP_MARGIN",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_HU_WINDOW",
    "DEFAULT_PROCESSED_ROOT",
    "PreprocessMetadata",
    "PreprocessedCase",
    "SavedCasePaths",
    "SUMMARY_COLUMNS",
    "Spacing3D",
    "affine_after_crop",
    "affine_from_image",
    "bbox_from_mask",
    "build_summary_table",
    "clip_ct_to_window",
    "crop_box_to_slices",
    "crop_volume",
    "list_case_ids",
    "load_canonical_image",
    "main",
    "normalize_ct",
    "parse_args",
    "preview_summary_table",
    "preprocess_case",
    "resolve_case_paths",
    "resolve_summary_csv_path",
    "save_nifti",
    "save_preprocessed_case",
    "save_summary_table",
    "summarize_preprocessed_case",
    "summarize_preprocessing_error",
    "verify_alignment",
]


if __name__ == "__main__":
    main()
