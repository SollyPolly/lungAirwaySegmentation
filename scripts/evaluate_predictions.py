"""Evaluate saved prediction artifacts from one training run."""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.io.case_layout import resolve_case_paths
from lung_airway_segmentation.io.nifti import load_canonical_image
from lung_airway_segmentation.metrics.segmentation import (
    binary_confusion_counts_from_masks,
    binary_dice_score_from_masks,
    binary_iou_score_from_masks,
    binary_precision_from_masks,
    binary_recall_from_masks,
)
from lung_airway_segmentation.training.config import resolve_project_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the prediction evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved prediction masks for one run directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing a predictions/ folder produced by predict_case.py.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Optional explicit predictions directory. Defaults to <run-dir>/predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional evaluation output directory. Defaults to <run-dir>/evaluation.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional raw dataset root override. Defaults to the saved run config.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Optional subset of case IDs to evaluate. Defaults to all saved prediction cases.",
    )
    return parser


def load_json(path: Path) -> dict:
    """Load one JSON artifact and require a top-level object."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return data


def resolve_data_root(run_dir: Path, data_root_override: Path | None) -> Path:
    """Resolve the dataset root from CLI override, run metadata, or resolved config."""
    if data_root_override is not None:
        return data_root_override.resolve()

    run_metadata_path = run_dir / "run_metadata.json"
    if run_metadata_path.exists():
        run_metadata = load_json(run_metadata_path)
        if run_metadata.get("data_root"):
            return Path(run_metadata["data_root"]).resolve()

    resolved_config = load_json(run_dir / "resolved_config.json")
    return resolve_project_path(resolved_config["data"]["raw_data_root"]).resolve()


def list_prediction_case_dirs(predictions_dir: Path, case_ids: list[str] | None = None) -> list[Path]:
    """List case prediction folders to evaluate."""
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions directory does not exist: {predictions_dir}")

    requested_case_ids = None if case_ids is None else {str(case_id) for case_id in case_ids}
    case_dirs = [
        path
        for path in predictions_dir.iterdir()
        if path.is_dir() and (path / "prediction_metadata.json").is_file()
    ]
    if requested_case_ids is not None:
        case_dirs = [path for path in case_dirs if path.name in requested_case_ids]

    case_dirs.sort(key=lambda path: (not path.name.isdigit(), int(path.name) if path.name.isdigit() else path.name))
    return case_dirs


def load_binary_nifti_mask(path: Path) -> np.ndarray:
    """Load one NIfTI file and return a boolean mask."""
    return np.asarray(nib.load(str(path)).dataobj) > 0


def evaluate_prediction_case(case_id: str, prediction_case_dir: Path, data_root: Path) -> dict:
    """Evaluate one saved prediction case against the reference airway mask."""
    prediction_metadata = load_json(prediction_case_dir / "prediction_metadata.json")
    prediction_mask_path = prediction_case_dir / "airway_pred_full.nii.gz"
    if not prediction_mask_path.is_file():
        raise FileNotFoundError(f"Missing predicted full-volume mask: {prediction_mask_path}")

    case_paths = resolve_case_paths(case_id, data_root=data_root)
    if case_paths["airway"] is None:
        raise ValueError(f"Case {case_id} does not have a reference airway mask.")

    target_mask = np.asarray(load_canonical_image(case_paths["airway"]).dataobj) > 0
    prediction_mask = load_binary_nifti_mask(prediction_mask_path)

    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "Prediction and target mask shapes do not match: "
            f"{prediction_mask.shape} != {target_mask.shape}"
        )

    counts = binary_confusion_counts_from_masks(prediction_mask, target_mask)
    predicted_voxels = int(prediction_mask.sum())
    target_voxels = int(target_mask.sum())

    return {
        "case_id": str(case_id),
        "prediction_dir": str(prediction_case_dir),
        "prediction_mask_path": str(prediction_mask_path),
        "target_mask_path": str(case_paths["airway"]),
        "threshold": float(prediction_metadata["threshold"]),
        "checkpoint_epoch": int(prediction_metadata["checkpoint_epoch"]),
        "dice": binary_dice_score_from_masks(prediction_mask, target_mask),
        "iou": binary_iou_score_from_masks(prediction_mask, target_mask),
        "precision": binary_precision_from_masks(prediction_mask, target_mask),
        "recall": binary_recall_from_masks(prediction_mask, target_mask),
        "predicted_voxels": predicted_voxels,
        "target_voxels": target_voxels,
        "voxel_count_ratio": float(predicted_voxels / target_voxels) if target_voxels > 0 else None,
        **counts,
    }


def summarize_case_metrics(case_metrics: list[dict]) -> dict:
    """Aggregate mean, std, min, and max for the evaluated cases."""
    if not case_metrics:
        raise ValueError("No case metrics were provided for summary.")

    metric_names = [
        "dice",
        "iou",
        "precision",
        "recall",
        "predicted_voxels",
        "target_voxels",
    ]

    summary = {"num_cases": len(case_metrics)}
    for metric_name in metric_names:
        values = np.asarray([float(case_metric[metric_name]) for case_metric in case_metrics], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))
        summary[f"{metric_name}_min"] = float(values.min())
        summary[f"{metric_name}_max"] = float(values.max())

    return summary


def write_json(data, output_path: Path) -> None:
    """Write one JSON artifact to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_case_metrics_csv(case_metrics: list[dict], output_path: Path) -> None:
    """Write the per-case evaluation table as CSV."""
    if not case_metrics:
        raise ValueError("No case metrics were provided for CSV export.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(case_metrics[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_metrics)


def main() -> None:
    args = build_argument_parser().parse_args()

    run_dir = args.run_dir.resolve()
    predictions_dir = args.predictions_dir.resolve() if args.predictions_dir is not None else run_dir / "predictions"
    output_dir = args.output_dir.resolve() if args.output_dir is not None else run_dir / "evaluation"
    data_root = resolve_data_root(run_dir, args.data_root)

    case_dirs = list_prediction_case_dirs(predictions_dir, args.case_ids)
    if not case_dirs:
        raise ValueError(f"No saved prediction cases were found in {predictions_dir}.")

    case_metrics = [
        evaluate_prediction_case(case_dir.name, case_dir, data_root)
        for case_dir in case_dirs
    ]
    summary = summarize_case_metrics(case_metrics)

    write_case_metrics_csv(case_metrics, output_dir / "per_case_metrics.csv")
    write_json(case_metrics, output_dir / "per_case_metrics.json")
    write_json(summary, output_dir / "summary.json")

    print(f"Evaluated {len(case_metrics)} case(s) from {predictions_dir}")
    print(f"Mean Dice: {summary['dice_mean']:.4f}")
    print(f"Mean IoU: {summary['iou_mean']:.4f}")
    print(f"Mean Precision: {summary['precision_mean']:.4f}")
    print(f"Mean Recall: {summary['recall_mean']:.4f}")
    print(f"Evaluation artifacts: {output_dir}")


if __name__ == "__main__":
    main()
