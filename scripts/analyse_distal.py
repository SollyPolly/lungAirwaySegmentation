"""Stratify a model's predicted airway probability by branch thickness.

Distal (terminal) airways are thin; proximal airways (trachea, mainstem) are
thick. Volumetric Dice is dominated by the thick proximal tree, so it can look
healthy while the model is quietly missing most of the distal branches. This
script exposes that by bucketing every ground-truth airway voxel by its local
radius (distance to the airway wall) and reporting, per bucket:

- mean / median / p90 predicted probability
- recall at the operating threshold (e.g. 0.99) and at 0.50

A large gap between recall@0.50 and recall@threshold in the thin buckets means
the model *sees* the distal airways but isn't confident enough to clear the
threshold — pointing at the loss/threshold rather than model capacity. It also
prints Dice and tree-length-detected (TD, centerline recall) per case for context.

Usage:
    python -m scripts.analyse_distal --run-dir runs/supervised-atm-l20/<run>
    python -m scripts.analyse_distal --run-dir runs/<run> --cases 002,012,014 --overlap 0.5

Defaults: evaluates the run's held-out test cases (from run_metadata.json), the
run's own checkpoint/threshold/data-config, overlap 0.25 (fast). Results print as
a table and are saved to <run-dir>/distal_analysis.json.

Notes / caveats:
- "Radius" is per-voxel distance-to-wall (scipy EDT). It is a cheap proxy for
  airway generation; the thinnest bucket (r=1) is dominated by true distal
  branches but also includes the one-voxel surface shell of thicker airways.
  The recall trend across buckets is the robust signal.
- Loading + HU normalisation are taken from the run's resolved_config so the
  inference matches exactly what the model was trained on.

When to use this:
- After any new run, to see *where* on the tree it improved (did clDice /
  radius-weighting / more data actually lift distal recall, or just proximal?).
- To pick/justify an operating threshold for the topology metrics.
- To produce a dissertation figure decomposing Dice/TD by branch thickness.
- On the AeroPath-OOD cases, to check whether generalisation fails proximally
  or distally.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from lung_airway_segmentation.inference.sliding_window import predict_logits_for_volume
from lung_airway_segmentation.metrics.topology import (
    _foreground_slices,
    _largest_connected_component,
    _skeletonize,
)
from lung_airway_segmentation.preprocessing.geometry import normalize_margin
from lung_airway_segmentation.training.builders import build_model
from lung_airway_segmentation.training.config import (
    load_yaml_config,
    resolve_device,
    resolve_project_path,
)

# radius bins in voxels (distance to wall): (label, lo, hi)
RADIUS_BINS = [
    ("r=1 (distal)", 0.5, 1.5),
    ("r=2", 1.5, 2.5),
    ("r=3", 2.5, 3.5),
    ("r=4-5", 3.5, 5.5),
    ("r>=6 (proximal)", 5.5, 1e9),
]

# thresholds for the Dice/TD sweep (clDice models calibrate lower than pos_weight
# baselines, so their optimal operating point is usually below 0.99).
SWEEP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory with resolved_config.json + checkpoint.")
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated case IDs. Default: the run's test split.")
    parser.add_argument("--max-cases", type=int, default=5, help="Cap on number of cases (default 5, for speed).")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--threshold", type=float, default=None, help="Operating threshold. Default: the run's validation threshold.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Sliding-window overlap (0.25 fast / 0.5+ more accurate).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-config", type=Path, default=None, help="Override dataset to load cases from. Default: the run's data config.")
    return parser.parse_args()


def load_case(dataset_name, data_config, case_id, hu_window):
    """Return (ct_tensor, gt_mask_bool) for one case, normalised as in training."""
    if dataset_name == "atm22":
        from monai.transforms import LoadImage, ScaleIntensityRange

        from lung_airway_segmentation.io.atm22_layout import resolve_case_paths

        batch_root = resolve_project_path(data_config["batch_root"])
        paths = resolve_case_paths(case_id, batch_root=batch_root)
        if paths["airway"] is None:
            raise FileNotFoundError(f"ATM case {case_id} has no airway label.")
        loader = LoadImage(image_only=True, ensure_channel_first=True)
        scaler = ScaleIntensityRange(a_min=hu_window[0], a_max=hu_window[1], b_min=0.0, b_max=1.0, clip=True)
        ct = scaler(loader(paths["ct"]))
        gt = np.asarray(loader(paths["airway"])[0]) > 0
        return ct, gt

    if dataset_name == "aeropath":
        from lung_airway_segmentation.preprocessing.pipeline import preprocess_case

        data_root = resolve_project_path(data_config["raw_data_root"])
        crop_margin = normalize_margin(data_config["preprocessing"]["crop_margin_voxels"])
        case = preprocess_case(
            case_id, data_root=data_root, include_lung_mask=False,
            hu_window=tuple(float(v) for v in hu_window), crop_margin=crop_margin,
        )
        return torch.from_numpy(case.ct), case.airway_mask.astype(bool)

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "resolved_config.json").read_text())
    metadata = json.loads((run_dir / "run_metadata.json").read_text())

    data_config = load_yaml_config(args.data_config) if args.data_config else cfg["data"]
    dataset_name = str(data_config["dataset_name"]).lower()
    hu_window = tuple(float(v) for v in data_config["preprocessing"]["hu_window"])

    validation = cfg["training"]["validation"]
    roi_size = tuple(int(v) for v in validation["roi_size"])
    threshold = args.threshold if args.threshold is not None else float(validation.get("threshold", 0.99))

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = [str(c) for c in metadata.get("splits", {}).get("test_case_ids", [])]
    if not cases:
        raise SystemExit("No cases to analyse: pass --cases or ensure run_metadata has a test split.")
    cases = cases[: args.max_cases]

    device = resolve_device(args.device)
    ckpt = run_dir / ("best_model.pt" if args.checkpoint == "best" else "last_model.pt")
    model = build_model(device, cfg["model"])
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    model.eval()

    print(f"model: {metadata.get('experiment_name')}  | cases: {cases}")
    print(f"dataset: {dataset_name}  | threshold {threshold}  | overlap {args.overlap}\n")

    bin_probs = {label: [] for label, _, _ in RADIUS_BINS}
    sweep_dice = {t: [] for t in SWEEP_THRESHOLDS}
    sweep_td = {t: [] for t in SWEEP_THRESHOLDS}
    per_case = []
    for cid in cases:
        ct, gt = load_case(dataset_name, data_config, cid, hu_window)
        logits = predict_logits_for_volume(
            model, ct, device=device, roi_size=roi_size,
            sw_batch_size=4, overlap=args.overlap, use_amp=False,
        )
        prob = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
        radius = ndimage.distance_transform_edt(gt)
        gt_sum = int(gt.sum())

        # GT centerline once per case (largest connected component, then skeleton)
        # — reused for the run-threshold TD and the whole threshold sweep.
        target_component = _largest_connected_component(gt)
        td_slices = _foreground_slices(target_component)
        gt_skeleton = _skeletonize(target_component[td_slices])
        gt_skeleton_sum = int(gt_skeleton.sum())

        def dice_td(threshold_value):
            predicted = prob >= threshold_value
            intersection = int((predicted & gt).sum())
            dice_value = 2 * intersection / ((int(predicted.sum()) + gt_sum) or 1)
            td_value = (
                int((gt_skeleton & predicted[td_slices]).sum()) / gt_skeleton_sum
                if gt_skeleton_sum
                else 1.0
            )
            return float(dice_value), float(td_value)

        dice, td = dice_td(threshold)
        per_case.append({"case_id": cid, "airway_voxels": gt_sum, "dice": dice, "td": td})
        print(f"case {cid}: airway voxels {gt_sum:,}  | Dice@{threshold} {dice:.3f}  | TD@{threshold} {td:.3f}")

        for t in SWEEP_THRESHOLDS:
            dice_t, td_t = dice_td(t)
            sweep_dice[t].append(dice_t)
            sweep_td[t].append(td_t)

        for label, lo, hi in RADIUS_BINS:
            m = gt & (radius >= lo) & (radius < hi)
            if m.any():
                bin_probs[label].append(prob[m])

    total = sum(sum(a.size for a in v) for v in bin_probs.values())
    header = f"\n{'radius bin':>16}  {'voxels':>9}  {'%airway':>7}  {'meanP':>6}  {'medP':>6}  {'p90P':>6}  {'rec@thr':>7}  {'rec@.5':>6}"
    print(header)
    bins_out = []
    for label, _, _ in RADIUS_BINS:
        if not bin_probs[label]:
            continue
        p = np.concatenate(bin_probs[label])
        row = {
            "bin": label, "voxels": int(p.size), "pct_airway": float(100 * p.size / total),
            "mean_prob": float(p.mean()), "median_prob": float(np.median(p)),
            "p90_prob": float(np.percentile(p, 90)),
            "recall_at_threshold": float((p >= threshold).mean()),
            "recall_at_0.5": float((p >= 0.5).mean()),
        }
        bins_out.append(row)
        print(f"{label:>16}  {p.size:>9,}  {row['pct_airway']:>6.1f}%  "
              f"{row['mean_prob']:>6.3f}  {row['median_prob']:>6.3f}  {row['p90_prob']:>6.3f}  "
              f"{100*row['recall_at_threshold']:>6.1f}%  {100*row['recall_at_0.5']:>5.1f}%")

    # --- threshold sweep (mean over cases; reuses the probabilities above) ---
    print(f"\n--- threshold sweep (mean over {len(cases)} cases) ---")
    print(f"{'thresh':>7}  {'Dice':>6}  {'TD':>6}")
    sweep_out = []
    for t in SWEEP_THRESHOLDS:
        mean_dice = float(np.mean(sweep_dice[t]))
        mean_td = float(np.mean(sweep_td[t]))
        sweep_out.append({"threshold": t, "dice": mean_dice, "td": mean_td})
        print(f"{t:>7.2f}  {mean_dice:>6.3f}  {mean_td:>6.3f}")
    best_dice = max(sweep_out, key=lambda r: r["dice"])
    print(
        f"best Dice {best_dice['dice']:.3f} @ threshold {best_dice['threshold']:.2f}"
        f"   (TD at that threshold: {best_dice['td']:.3f})"
    )
    print(
        "note: TD rises as the threshold drops (it is recall-like and ignores false "
        "positives), so its max is degenerate — compare TD between models at a fixed "
        "operating threshold (e.g. the Dice-optimal one), not at its own peak."
    )

    out_path = run_dir / "distal_analysis.json"
    out_path.write_text(json.dumps({
        "run": metadata.get("experiment_name"),
        "dataset": dataset_name, "cases": cases,
        "threshold": threshold, "overlap": args.overlap,
        "per_case": per_case, "bins": bins_out, "threshold_sweep": sweep_out,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
