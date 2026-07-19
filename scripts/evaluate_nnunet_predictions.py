"""Score external prediction MASKS (nnU-Net et al.) with our topology metrics — Route A.

Why this exists: ``analyse_distal`` RE-INFERS from OUR checkpoint + config and needs a
probability volume (threshold sweep / clDice-optimal op-point / distal bins). nnU-Net is a
different architecture with its own checkpoint format and, by default, emits a HARD mask at
its native argmax operating point. This scorer ingests those masks and computes the SAME
dev-mode metrics ``analyse_distal`` reports (+ trachea-LCC), so a stock-nnU-Net control is a
drop-in comparison to our runs' VAL numbers.

Label it correctly in the write-up: "nnU-Net 3d_fullres, native argmax output, scored by our
topology metrics". nnU-Net is scored at its native op-point; our models at a val-selected one.
That is a fair *baseline-strength control*; for strict threshold parity, export softmax
(`nnUNetv2_predict --save_probabilities`) and wire a probability-ingestion path (Route B) —
only worth it if this comparison lands close.

Metrics (matched to analyse_distal dev-mode):
  raw:  dice_raw, td_raw (GT-skeleton recall), prec_raw (voxel precision)
  +LCC: dice_lcc, td_lcc (=TLD), prec_lcc, tprec_lcc, cldice_lcc = harmonic(tprec_lcc, td_lcc)
  lcc_retained_fraction; foreground wall-distance recall, voxel-pooled in voxel-EDT bins.
  --branch also computes ATM'22 BD (+ its own clDice/TLD) via airway_topology_metrics_from_masks.

Geometry: the nnU-Net export links the ORIGINAL ATM niftis and nnU-Net restores predictions to
the input geometry, so ``ATM_<id>.nii.gz`` aligns voxel-for-voxel with the GT ``load_case``
returns. A shape mismatch raises rather than silently mis-scoring.

Usage:
    python -u -m scripts.evaluate_nnunet_predictions \
      --pred-dir data/nnunet/predictions/Dataset111_val \
      --split-run-dir runs/atm-l110-supervised/2026-06-26__06-12-47__cldice-w1-cbdice-w2-p96-l110__baseline_unet \
      --report-split val \
      --out runs/atm-l110-supervised/.../nnunet111_val_topology.json

    # ad-hoc case list (no split-run-dir needed):
    python -u -m scripts.evaluate_nnunet_predictions --pred-dir <dir> --cases 016,027 --out out.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from monai.transforms import LoadImage
from scipy import ndimage

from scripts.analyse_distal import RADIUS_BINS, cheap_metrics, gt_centerline, load_case
from lung_airway_segmentation.inference.postprocess import keep_component_containing_trachea
from lung_airway_segmentation.io.atm22_layout import resolve_case_paths
from lung_airway_segmentation.metrics.topology import (
    TOPOLOGY_METRIC_VERSION,
    airway_topology_metrics_from_masks,
    topology_precision_from_masks,
)
from lung_airway_segmentation.training.config import load_yaml_config, resolve_project_path

_SPLIT_KEYS = {"val": "val_case_ids", "test": "test_case_ids", "train": "train_case_ids"}


def _load_pred_mask(pred_path: Path) -> np.ndarray:
    """Load a saved prediction nifti as a boolean array in native (file) orientation.

    Uses the SAME loader as load_case's GT (LoadImage, ensure_channel_first, no reorient),
    so pred and GT are voxel-aligned when they share the source geometry.
    """
    loader = LoadImage(image_only=True, ensure_channel_first=True)
    img = loader(str(pred_path))
    return np.asarray(img[0]) > 0.5  # nnU-Net writes integer labels {0,1}


def _resolve_cases(args) -> list[str]:
    if args.cases:
        return [c.strip() for c in args.cases.split(",") if c.strip()]
    if not args.split_run_dir:
        raise SystemExit("Provide --cases or --split-run-dir (to read the split from run_metadata.json).")
    meta = json.loads((Path(args.split_run_dir) / "run_metadata.json").read_text())
    ids = [str(c) for c in meta.get("splits", {}).get(_SPLIT_KEYS[args.report_split], [])]
    if not ids:
        raise SystemExit(f"No {args.report_split} case ids in {args.split_run_dir}/run_metadata.json.")
    return ids


def score_case(cid, data_config, hu_window, pred_dir, batch_root, *, prefix, suffix, compute_branch):
    """Return (per-case metric row, {bin_label: pred-over-bin bool array}, gt_sum)."""
    _ct, gt, affine = load_case("atm22", data_config, cid, hu_window, lung_crop=None)
    padded = resolve_case_paths(cid, batch_root=batch_root)["case_id"]
    pred_path = Path(pred_dir) / f"{prefix}{padded}{suffix}"
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"No prediction for case {cid} at {pred_path}. "
            f"Expected nnU-Net output named {prefix}{padded}{suffix} in --pred-dir."
        )
    pred = _load_pred_mask(pred_path)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Geometry mismatch for case {cid}: pred {pred.shape} vs gt {gt.shape}. "
            f"nnU-Net output must be in the original image geometry."
        )

    gt_sum = int(gt.sum())
    td_slices, gt_skeleton, gt_skeleton_sum = gt_centerline(gt)

    dice_raw, td_raw, prec_raw = cheap_metrics(pred, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
    pred_lcc = keep_component_containing_trachea(pred, affine=affine) > 0
    dice_lcc, td_lcc, prec_lcc = cheap_metrics(pred_lcc, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
    lcc_retained_fraction = float(pred_lcc.sum() / max(int(pred.sum()), 1))

    tprec_lcc = float(topology_precision_from_masks(pred_lcc, gt))  # one skeletonisation
    denom = tprec_lcc + td_lcc
    cldice_lcc = float(2.0 * tprec_lcc * td_lcc / denom) if denom > 0 else 0.0

    row = {
        "case_id": cid, "airway_voxels": gt_sum,
        "dice_raw": dice_raw, "td_raw": td_raw, "prec_raw": prec_raw,
        "dice_lcc": dice_lcc, "td_lcc": td_lcc, "prec_lcc": prec_lcc,
        "tprec_lcc": tprec_lcc, "cldice_lcc": cldice_lcc,
        "lcc_retained_fraction": lcc_retained_fraction,
        "bd_lcc": None, "cldice_atm": None, "td_atm": None,
    }
    if compute_branch:
        atm = airway_topology_metrics_from_masks(pred_lcc, gt, branch_detection_threshold=0.8)
        row["bd_lcc"] = float(atm["branch_detected"])
        row["cldice_atm"] = float(atm["cldice"])
        row["td_atm"] = float(atm["tree_length_detected"])

    # radius-stratified recall on the RAW mask (matches analyse_distal's recall_at_0.5).
    radius = ndimage.distance_transform_edt(gt)
    bin_hits = {}
    for label, lo, hi in RADIUS_BINS:
        m = gt & (radius >= lo) & (radius < hi)
        bin_hits[label] = pred[m] if m.any() else np.array([], dtype=bool)

    print(
        f"case {cid}: voxels {gt_sum:,}  | +LCC  Dice {dice_lcc:.3f}  TLD {td_lcc:.3f}  "
        f"clDice {cldice_lcc:.3f}  TPrec {tprec_lcc:.3f}  lcc-kept {lcc_retained_fraction:.3f}",
        flush=True,
    )
    return row, bin_hits, gt_sum


def _table_mean(rows):
    keys = [
        "dice_raw", "td_raw", "prec_raw",
        "dice_lcc", "td_lcc", "prec_lcc", "tprec_lcc", "cldice_lcc",
        "lcc_retained_fraction", "bd_lcc", "cldice_atm", "td_atm",
    ]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def _build_bins(bin_hits_per_case):
    """Voxel-pooled recall per radius bin across all cases (matches analyse_distal)."""
    pooled = {label: [] for label, _, _ in RADIUS_BINS}
    for per_case in bin_hits_per_case:
        for label, arr in per_case.items():
            if arr.size:
                pooled[label].append(arr)
    total = sum(sum(a.size for a in v) for v in pooled.values())
    bins_out = []
    if not total:
        return bins_out
    for label, _, _ in RADIUS_BINS:
        if not pooled[label]:
            continue
        p = np.concatenate(pooled[label]).astype(np.float32)
        recall = float(p.mean())
        bins_out.append({
            "bin": label, "voxels": int(p.size),
            "pct_airway": float(100 * p.size / total),
            "recall": recall, "recall_at_0.5": recall,  # alias: hard mask has one op-point
        })
    return bins_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True, help="Folder of prediction niftis (nnU-Net output).")
    ap.add_argument("--data-config", type=Path, default=Path("configs/data/atm22.yaml"),
                    help="ATM'22 data YAML (batch_root + hu_window). Default: configs/data/atm22.yaml.")
    ap.add_argument("--split-run-dir", type=Path, default=None,
                    help="A run dir whose run_metadata.json holds the split (val/test/train case ids).")
    ap.add_argument("--report-split", choices=("val", "test", "train"), default="val",
                    help="Which split to score (default val — develop on val, seal test).")
    ap.add_argument("--cases", type=str, default=None, help="Comma-separated case ids (overrides --split-run-dir).")
    ap.add_argument("--prefix", type=str, default="ATM_", help="Prediction filename prefix (default 'ATM_').")
    ap.add_argument("--suffix", type=str, default=".nii.gz", help="Prediction filename suffix (default '.nii.gz').")
    ap.add_argument("--branch", action="store_true", help="Also compute ATM'22 BD (slower branch parse).")
    ap.add_argument("--out", type=str, default=None, help="Output JSON (default <pred-dir>/nnunet_topology.json).")
    args = ap.parse_args()

    if args.report_split == "test" and not args.cases:
        print("WARNING: scoring the SEALED TEST split — final numbers only.", flush=True)

    data_config = load_yaml_config(args.data_config)
    batch_root = resolve_project_path(data_config["batch_root"])
    hu_window = tuple(float(v) for v in data_config["preprocessing"]["hu_window"])
    cases = _resolve_cases(args)
    print(f"scoring {len(cases)} case(s) from {args.pred_dir}  (split={args.report_split})", flush=True)

    rows, bin_hits_per_case = [], []
    for cid in cases:
        row, bin_hits, _ = score_case(
            cid, data_config, hu_window, args.pred_dir, batch_root,
            prefix=args.prefix, suffix=args.suffix, compute_branch=args.branch,
        )
        rows.append(row)
        bin_hits_per_case.append(bin_hits)

    mean_row = _table_mean(rows)
    bins_out = _build_bins(bin_hits_per_case)

    out_path = Path(args.out) if args.out else (args.pred_dir / "nnunet_topology.json")
    out_path.write_text(json.dumps({
        "topology_metric_version": TOPOLOGY_METRIC_VERSION,
        "scorer": "evaluate_nnunet_predictions",
        "pred_dir": str(args.pred_dir),
        "dataset": "atm22",
        "report_split": args.report_split,
        "operating_point": {"threshold": "native_argmax", "selected_on": "none",
                            "note": "nnU-Net hard mask at its native op-point; ours are val-selected."},
        "postprocessing": {"lcc": "trachea"},
        "branch_metrics": bool(args.branch),
        "report_cases": cases,
        "table_per_case": rows,
        "table_mean": mean_row,
        "bins": bins_out,
    }, indent=2))

    print("\n--- mean (+LCC, trachea) ---")
    print(f"  clDice {mean_row['cldice_lcc']:.4f} | TLD {mean_row['td_lcc']:.4f} | "
          f"TPrec {mean_row['tprec_lcc']:.4f} | Dice+LCC {mean_row['dice_lcc']:.4f} | "
          f"prec_lcc {mean_row['prec_lcc']:.4f}")
    print(f"  raw: Dice {mean_row['dice_raw']:.4f} | TD {mean_row['td_raw']:.4f} | "
          f"lcc-kept {mean_row['lcc_retained_fraction']:.4f}")
    if mean_row.get("bd_lcc") is not None:
        print(f"  ATM: BD {mean_row['bd_lcc']:.4f} | clDice(atm) {mean_row['cldice_atm']:.4f} | TLD(atm) {mean_row['td_atm']:.4f}")
    for b in bins_out:
        if b["bin"].startswith("r=1"):
            print(f"  wall-shell r=1 recall {b['recall']:.4f}  ({b['voxels']:,} voxels)")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
