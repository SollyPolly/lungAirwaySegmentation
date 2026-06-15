"""Inference CLI for saved supervised baseline checkpoints."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from lung_airway_segmentation.inference.postprocess import keep_largest_connected_component
from lung_airway_segmentation.inference.sliding_window import predict_logits_for_volume
from lung_airway_segmentation.preprocessing.geometry import normalize_margin
from lung_airway_segmentation.preprocessing.pipeline import preprocess_atm22_case, preprocess_case
from lung_airway_segmentation.training.builders import build_model
from lung_airway_segmentation.training.config import resolve_device, resolve_project_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the prediction CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run sliding-window inference for one saved training case.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to one training run directory containing resolved_config.json and checkpoints.",
    )
    parser.add_argument(
        "--case-id",
        required=True,
        help="Dataset case ID to predict.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Which named checkpoint from the run directory to use.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. Overrides --checkpoint when provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output root. Defaults to <run-dir>/predictions.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional dataset-root override. Defaults to the resolved run config.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold override. Defaults to the saved validation threshold.",
    )
    parser.add_argument(
        "--inference-overlap",
        type=float,
        default=None,
        help="Optional sliding-window overlap override. Defaults to the saved run config.",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional lung bounding-box margin override for each canonical axis. "
            "Use 5 5 1000 to preserve the full superior-inferior airway extent."
        ),
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Also save the probability volume as a NIfTI file.",
    )
    parser.add_argument(
        "--skip-save-ct",
        action="store_true",
        help="Do not save a duplicate cropped CT volume in the prediction directory.",
    )
    parser.add_argument(
        "--largest-component",
        action="store_true",
        help=(
            "Also save a mask containing only the largest connected component. "
            "The raw binary prediction is always preserved."
        ),
    )
    return parser


def load_json(path: Path) -> dict:
    """Load one JSON artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return data


def resolve_checkpoint_path(run_dir: Path, checkpoint_name: str, checkpoint_path: Path | None) -> Path:
    """Resolve the checkpoint path from either an explicit file or a run directory."""
    if checkpoint_path is not None:
        if checkpoint_path.is_absolute():
            resolved = checkpoint_path
        else:
            resolved = run_dir / checkpoint_path
    else:
        filename = "best_model.pt" if checkpoint_name == "best" else "last_model.pt"
        resolved = run_dir / filename

    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {resolved}")
    return resolved


def restore_cropped_volume(cropped_volume, crop_box, original_shape, fill_value=0.0):
    """Place one cropped prediction volume back into the canonical full-volume grid."""
    restored = np.full(original_shape, fill_value, dtype=cropped_volume.dtype)
    z_bounds, y_bounds, x_bounds = crop_box
    restored[
        z_bounds[0]:z_bounds[1],
        y_bounds[0]:y_bounds[1],
        x_bounds[0]:x_bounds[1],
    ] = cropped_volume
    return restored


def save_nifti_volume(volume, affine, output_path: Path) -> None:
    """Save one 3D NumPy volume as a NIfTI image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(volume, affine)
    nib.save(image, str(output_path))


def load_model_from_checkpoint(device, model_config: dict, checkpoint_path: Path):
    """Build the model and load saved weights from one checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(device, model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def build_prediction_metadata(
    *,
    case,
    run_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    threshold: float,
    resolved_config: dict,
    checkpoint: dict,
) -> dict:
    """Assemble the saved prediction metadata record."""
    training_config = resolved_config["training"]
    return {
        "case_id": case.case_id,
        "run_dir": str(run_dir),
        "study_name": training_config.get("study_name"),
        "run_label": training_config.get("run_label"),
        "experiment_name": training_config.get("experiment_name"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "output_dir": str(output_dir),
        "threshold": float(threshold),
        "device": args.device,
        "cropped_shape": list(case.ct.shape),
        "original_shape": list(case.metadata["original_shape"]),
        "crop_box": [list(bounds) for bounds in case.crop_box],
        "cropped_affine": case.affine.tolist(),
        "original_affine": case.metadata["original_affine"].tolist(),
        "roi_size": training_config["validation"]["roi_size"],
        "sw_batch_size": training_config["validation"]["sw_batch_size"],
        "inference_overlap": float(args.inference_overlap)
        if args.inference_overlap is not None
        else training_config["validation"]["inference_overlap"],
        "amp_enabled": bool(training_config.get("amp", {}).get("enabled", False)),
        "largest_component_saved": bool(args.largest_component),
        "crop_margin": list(case.metadata["crop_margin"]),
    }


def main() -> None:
    args = build_argument_parser().parse_args()

    run_dir = args.run_dir.resolve()
    resolved_config = load_json(run_dir / "resolved_config.json")

    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint, args.checkpoint_path)
    device = resolve_device(args.device)

    data_config = resolved_config["data"]
    configured_root = data_config.get("raw_data_root", data_config.get("batch_root"))
    if configured_root is None:
        raise ValueError("Run data config must define raw_data_root or batch_root.")
    data_root = (
        args.data_root.resolve()
        if args.data_root is not None
        else resolve_project_path(configured_root)
    )
    preprocessing_config = data_config["preprocessing"]
    hu_window = tuple(float(value) for value in preprocessing_config["hu_window"])
    dataset_name = str(data_config.get("dataset_name", "aeropath")).lower()
    if dataset_name == "atm22":
        case = preprocess_atm22_case(args.case_id, batch_root=data_root, hu_window=hu_window)
    elif dataset_name == "aeropath":
        crop_margin = normalize_margin(
            args.crop_margin
            if args.crop_margin is not None
            else preprocessing_config["crop_margin_voxels"]
        )
        case = preprocess_case(
            args.case_id,
            data_root=data_root,
            include_lung_mask=True,
            hu_window=hu_window,
            crop_margin=crop_margin,
        )
    else:
        raise ValueError(f"Unsupported prediction dataset: {dataset_name}")

    model, checkpoint = load_model_from_checkpoint(
        device,
        resolved_config["model"],
        checkpoint_path,
    )

    validation_config = resolved_config["training"]["validation"]
    use_amp = bool(resolved_config["training"].get("amp", {}).get("enabled", False)) and device.type == "cuda"
    inference_overlap = (
        float(args.inference_overlap)
        if args.inference_overlap is not None
        else float(validation_config["inference_overlap"])
    )
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(validation_config.get("threshold", 0.5))
    )

    image_tensor = torch.from_numpy(case.ct)
    logits = predict_logits_for_volume(
        model,
        image_tensor,
        device=device,
        roi_size=tuple(int(value) for value in validation_config["roi_size"]),
        sw_batch_size=int(validation_config["sw_batch_size"]),
        overlap=inference_overlap,
        use_amp=use_amp,
    )
    probabilities_cropped = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32, copy=False)
    prediction_cropped = (probabilities_cropped >= threshold).astype(np.uint8, copy=False)

    original_shape = tuple(int(value) for value in case.metadata["original_shape"])
    prediction_full = restore_cropped_volume(
        prediction_cropped,
        case.crop_box,
        original_shape,
        fill_value=0,
    )

    output_root = args.output_dir.resolve() if args.output_dir is not None else run_dir / "predictions"
    case_output_dir = output_root / str(case.case_id)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_save_ct:
        save_nifti_volume(
            case.ct.astype(np.float32, copy=False),
            case.affine,
            case_output_dir / "ct_cropped.nii.gz",
        )
    save_nifti_volume(prediction_cropped, case.affine, case_output_dir / "airway_pred_cropped.nii.gz")
    save_nifti_volume(
        prediction_full,
        case.metadata["original_affine"],
        case_output_dir / "airway_pred_full.nii.gz",
    )

    if args.largest_component:
        prediction_lcc_cropped = keep_largest_connected_component(prediction_cropped)
        prediction_lcc_full = restore_cropped_volume(
            prediction_lcc_cropped,
            case.crop_box,
            original_shape,
            fill_value=0,
        )
        save_nifti_volume(
            prediction_lcc_cropped,
            case.affine,
            case_output_dir / "airway_pred_lcc_cropped.nii.gz",
        )
        save_nifti_volume(
            prediction_lcc_full,
            case.metadata["original_affine"],
            case_output_dir / "airway_pred_lcc_full.nii.gz",
        )

    if args.save_probabilities:
        probabilities_full = restore_cropped_volume(
            probabilities_cropped,
            case.crop_box,
            original_shape,
            fill_value=0.0,
        )
        crop_is_full_volume = tuple(probabilities_cropped.shape) == original_shape
        if not crop_is_full_volume:
            save_nifti_volume(
                probabilities_cropped,
                case.affine,
                case_output_dir / "airway_prob_cropped.nii.gz",
            )
        save_nifti_volume(
            probabilities_full,
            case.metadata["original_affine"],
            case_output_dir / "airway_prob_full.nii.gz",
        )

    metadata = build_prediction_metadata(
        case=case,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        output_dir=case_output_dir,
        args=args,
        threshold=threshold,
        resolved_config=resolved_config,
        checkpoint=checkpoint,
    )
    with (case_output_dir / "prediction_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Saved prediction for case {case.case_id} to {case_output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
