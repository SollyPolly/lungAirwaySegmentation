"""Apply non-destructive postprocessing to saved full-volume predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.inference.postprocess import keep_largest_connected_component
from lung_airway_segmentation.io.nifti import load_canonical_image


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep the largest connected component of saved airway predictions.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing a predictions directory.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Optional prediction directory. Defaults to <run-dir>/predictions.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Cases to process. Defaults to every saved prediction case.",
    )
    parser.add_argument(
        "--input-filename",
        default="airway_pred_full.nii.gz",
        help="Input prediction filename within each case directory.",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Optional output filename. Defaults to a connectivity-specific filename.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(6, 18, 26),
        default=6,
        help="3D voxel connectivity used to identify connected components.",
    )
    return parser


def list_prediction_case_ids(predictions_dir: Path, input_filename: str) -> list[str]:
    case_ids = [
        case_dir.name
        for case_dir in predictions_dir.iterdir()
        if case_dir.is_dir() and (case_dir / input_filename).is_file()
    ]
    return sorted(case_ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def save_like_reference(volume: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(volume.dtype)
    nib.save(nib.Nifti1Image(volume, reference.affine, header), str(path))


def process_case(
    predictions_dir: Path,
    case_id: str,
    input_filename: str,
    output_filename: str,
    connectivity: int,
) -> None:
    case_dir = predictions_dir / case_id
    prediction_path = case_dir / input_filename
    prediction_image = load_canonical_image(prediction_path)

    prediction = np.asarray(prediction_image.dataobj) > 0
    filtered = keep_largest_connected_component(
        prediction.astype(np.uint8, copy=False),
        connectivity=connectivity,
    )

    output_path = case_dir / output_filename
    save_like_reference(filtered, prediction_image, output_path)

    original_voxels = int(prediction.sum())
    retained_voxels = int(filtered.sum())
    retained_percentage = 100.0 * retained_voxels / original_voxels if original_voxels else 0.0
    print(
        f"Case {case_id}: saved {output_path} | connectivity={connectivity} | "
        f"retained {retained_voxels:,}/{original_voxels:,} voxels ({retained_percentage:.2f}%)"
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    output_filename = args.output_filename or {
        6: "airway_pred_lcc_full.nii.gz",
        18: "airway_pred_lcc18_full.nii.gz",
        26: "airway_pred_lcc26_full.nii.gz",
    }[args.connectivity]
    if args.input_filename == output_filename:
        raise ValueError("--output-filename must differ from --input-filename.")

    run_dir = args.run_dir.resolve()
    predictions_dir = (
        args.predictions_dir.resolve()
        if args.predictions_dir is not None
        else run_dir / "predictions"
    )
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory does not exist: {predictions_dir}")

    case_ids = args.case_ids or list_prediction_case_ids(predictions_dir, args.input_filename)
    if not case_ids:
        raise ValueError(f"No saved prediction cases found under {predictions_dir}")

    for case_id in case_ids:
        process_case(
            predictions_dir,
            str(case_id),
            args.input_filename,
            output_filename,
            args.connectivity,
        )

    print("Original prediction files were not modified.")


if __name__ == "__main__":
    main()
