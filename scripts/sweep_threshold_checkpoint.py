"""Sweep binarization thresholds directly from a checkpoint.

Runs sliding-window inference once per validation case and evaluates Dice,
precision, and recall across a range of thresholds — without saving NIfTI files.

Usage:
    python scripts/sweep_threshold_checkpoint.py --run-dir runs/.../my_run
    python scripts/sweep_threshold_checkpoint.py --run-dir runs/.../my_run --overlap 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lung_airway_segmentation.inference.sliding_window import predict_logits_for_volume
from lung_airway_segmentation.preprocessing.geometry import normalize_margin
from lung_airway_segmentation.preprocessing.pipeline import preprocess_case
from lung_airway_segmentation.training.builders import build_model
from lung_airway_segmentation.training.config import resolve_device, resolve_project_path


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]


def dice_precision_recall(prob: np.ndarray, target: np.ndarray, threshold: float):
    pred = prob >= threshold
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return dice, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
        help="Which checkpoint to load (default: best).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Sliding-window overlap. Use 0.25 for speed on CPU, 0.75 to match training (slow).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "resolved_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"No resolved_config.json found in {run_dir}")

    with config_path.open() as f:
        resolved_config = json.load(f)

    checkpoint_name = "best_model.pt" if args.checkpoint == "best" else "last_model.pt"
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open() as f:
        run_metadata = json.load(f)

    val_case_ids = run_metadata["splits"]["val_case_ids"]
    device = resolve_device(args.device)

    data_config = resolved_config["data"]
    data_root = resolve_project_path(data_config["raw_data_root"])
    preprocessing = data_config["preprocessing"]
    hu_window = tuple(float(v) for v in preprocessing["hu_window"])
    crop_margin = normalize_margin(preprocessing["crop_margin_voxels"])

    training_config = resolved_config["training"]
    validation_config = training_config["validation"]
    roi_size = tuple(int(v) for v in validation_config["roi_size"])
    sw_batch_size = int(validation_config["sw_batch_size"])
    use_amp = bool(training_config.get("amp", {}).get("enabled", False)) and device.type == "cuda"

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(device, resolved_config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    print(f"Checkpoint epoch: {epoch}")
    print(f"Device: {device}  |  overlap: {args.overlap}  |  val cases: {val_case_ids}\n")

    per_case_probs = {}
    per_case_targets = {}

    for case_id in val_case_ids:
        print(f"Running inference on case {case_id} ...", flush=True)
        case = preprocess_case(
            case_id,
            data_root=data_root,
            include_lung_mask=False,
            hu_window=hu_window,
            crop_margin=crop_margin,
        )

        image_tensor = torch.from_numpy(case.ct)
        logits = predict_logits_for_volume(
            model,
            image_tensor,
            device=device,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=args.overlap,
            use_amp=use_amp,
        )
        probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
        target = case.airway_mask.astype(bool)

        per_case_probs[case_id] = probs
        per_case_targets[case_id] = target

        print(
            f"  prob range: [{probs.min():.4f}, {probs.max():.4f}]"
            f"  mean: {probs.mean():.4f}"
            f"  airway voxels: {target.sum():,}"
        )

    print("\n--- Dice at each threshold ---")
    header = f"{'case':>6}  " + "  ".join(f"{t:.2f}" for t in THRESHOLDS)
    print(header)

    per_threshold_dice = {t: [] for t in THRESHOLDS}

    for case_id in val_case_ids:
        probs = per_case_probs[case_id]
        target = per_case_targets[case_id]
        row = []
        for t in THRESHOLDS:
            d, _, _ = dice_precision_recall(probs, target, t)
            per_threshold_dice[t].append(d)
            row.append(d)
        print(f"{case_id:>6}  " + "  ".join(f"{d:.4f}" for d in row))

    mean_row = [np.mean(per_threshold_dice[t]) for t in THRESHOLDS]
    print(f"\n  mean  " + "  ".join(f"{d:.4f}" for d in mean_row))

    best_t = THRESHOLDS[int(np.argmax(mean_row))]
    best_mean = max(mean_row)
    print(f"\nBest threshold: {best_t:.2f}  |  mean Dice: {best_mean:.4f}")

    print(f"\n--- Precision / Recall at threshold={best_t:.2f} ---")
    print(f"{'case':>6}  {'dice':>6}  {'prec':>6}  {'recall':>6}")
    for case_id in val_case_ids:
        probs = per_case_probs[case_id]
        target = per_case_targets[case_id]
        d, p, r = dice_precision_recall(probs, target, best_t)
        print(f"{case_id:>6}  {d:.4f}  {p:.4f}  {r:.4f}")


if __name__ == "__main__":
    main()
