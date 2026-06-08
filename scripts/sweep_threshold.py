"""Sweep binarization thresholds over saved probability volumes.

Usage:
    python scripts/sweep_threshold.py --run-dir runs/.../20260417_232126

Loads airway_prob_full.nii.gz + ground-truth label for every case in
the evaluation/ per_case_metrics.json, then reports Dice/precision/recall
at multiple thresholds.
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def load_canonical_array(path: Path, dtype=np.float32) -> np.ndarray:
    """Load a NIfTI file in canonical (RAS+) orientation to match the preprocessing pipeline."""
    return np.asarray(nib.as_closest_canonical(nib.load(str(path))).dataobj, dtype=dtype)


def dice_precision_recall(pred: np.ndarray, target: np.ndarray):
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return dice, precision, recall, tp + fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    eval_json = run_dir / "evaluation" / "per_case_metrics.json"
    if not eval_json.is_file():
        raise FileNotFoundError(f"No evaluation found at {eval_json}")

    with eval_json.open() as f:
        cases = json.load(f)

    # Header
    header = f"{'case':>6}  " + "  ".join(f"{t:.2f}" for t in THRESHOLDS)
    print("\n--- Dice at each threshold ---")
    print(header)

    per_threshold_dice = {t: [] for t in THRESHOLDS}

    for case in cases:
        case_id = case["case_id"]
        pred_dir = Path(case["prediction_dir"])
        prob_path = pred_dir / "airway_prob_full.nii.gz"
        target_path = Path(case["target_mask_path"])

        if not prob_path.is_file():
            print(f"  case {case_id}: probability file missing — skipping")
            continue

        probs = load_canonical_array(prob_path)
        target = load_canonical_array(target_path) > 0.5

        row_dice = []
        for t in THRESHOLDS:
            pred = probs >= t
            d, p, r, pred_vox = dice_precision_recall(pred, target)
            row_dice.append(d)
            per_threshold_dice[t].append(d)

        print(f"{case_id:>6}  " + "  ".join(f"{d:.4f}" for d in row_dice))

    print("\n  mean  " + "  ".join(
        f"{np.mean(per_threshold_dice[t]):.4f}" for t in THRESHOLDS
    ))

    # Find best threshold by mean Dice
    best_t = max(THRESHOLDS, key=lambda t: np.mean(per_threshold_dice[t]))
    print(f"\nBest mean Dice at threshold={best_t:.2f}: "
          f"{np.mean(per_threshold_dice[best_t]):.4f}")

    # Also print precision/recall at best threshold for context
    print(f"\n--- Precision / Recall at threshold={best_t:.2f} ---")
    print(f"{'case':>6}  {'dice':>6}  {'prec':>6}  {'recall':>6}  {'pred_vox':>10}  {'target_vox':>10}")
    for case in cases:
        case_id = case["case_id"]
        pred_dir = Path(case["prediction_dir"])
        prob_path = pred_dir / "airway_prob_full.nii.gz"
        target_path = Path(case["target_mask_path"])
        if not prob_path.is_file():
            continue
        probs = load_canonical_array(prob_path)
        target = load_canonical_array(target_path) > 0.5
        pred = probs >= best_t
        d, p, r, pred_vox = dice_precision_recall(pred, target)
        print(f"{case_id:>6}  {d:.4f}  {p:.4f}  {r:.4f}  {pred_vox:>10,}  {int(target.sum()):>10,}")


if __name__ == "__main__":
    main()
