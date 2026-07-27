"""Score nnU-Net hard masks with overlap and topology metrics.

Metrics:
  raw:  dice_raw, td_raw (=TLD), prec_raw, tprec_raw, cldice_raw
  +LCC: dice_lcc, td_lcc (=TLD), prec_lcc, tprec_lcc, cldice_lcc
  clDice is the harmonic mean of topology precision and TLD for each mask.
  --branch computes ATM'22 BD, clDice, and TLD for both raw and LCC masks.
  lcc_retained_fraction; foreground wall-distance recall, voxel-pooled in voxel-EDT bins.
  --branch also computes ATM'22 BD (+ its own clDice/TLD) via airway_topology_metrics_from_masks.

Geometry: the nnU-Net export links the original ATM NIfTIs and nnU-Net restores predictions to
the input geometry, so ``ATM_<id>.nii.gz`` aligns voxel-for-voxel with the stored GT. Shape and
affine mismatches raise rather than silently mis-scoring.

Usage:
    python -u -m scripts.evaluate_nnunet_predictions \
      --pred-dir data/nnunet/predictions/Dataset111_val \
      --split-config configs/nnunet/atm22_split_l20.yaml \
      --report-split val \
      --out data/nnunet/predictions/Dataset111_val/nnunet111_val_topology.json

    # ad-hoc case list (overrides the split config):
    python -u -m scripts.evaluate_nnunet_predictions --pred-dir <dir> --cases 016,027 --out out.json
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from lung_airway_segmentation.config import load_yaml_config, resolve_project_path
from lung_airway_segmentation.datasets.splits import cases_for_split, create_split_from_config
from lung_airway_segmentation.inference.postprocess import keep_component_containing_trachea
from lung_airway_segmentation.io.atm22_layout import list_case_ids, resolve_case_paths
from lung_airway_segmentation.metrics.external_masks import (
    RADIUS_BINS,
    cheap_metrics,
    gt_centerline,
)
from lung_airway_segmentation.metrics.topology import (
    TOPOLOGY_METRIC_VERSION,
    airway_topology_metrics_from_masks,
    topology_precision_from_masks,
)

def _load_pred_mask(pred_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved prediction nifti as a boolean array in native (file) orientation.

    Nibabel's array proxy preserves the stored voxel order and permits a direct uint8
    read rather than materialising a floating-point volume.
    """
    image = nib.load(str(pred_path))
    mask = np.asarray(image.dataobj, dtype=np.uint8) > 0
    return mask, np.asarray(image.affine, dtype=np.float64)


def _load_gt_mask_and_affine(cid: str, batch_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only the GT and image geometry needed for external-mask scoring.

    The previous path called ``analyse_distal.load_case``, which also loaded and
    intensity-scaled the full CT despite never using its voxels. That made a
    connectivity sweep unnecessarily slow and memory-heavy.
    """
    paths = resolve_case_paths(cid, batch_root=batch_root)
    if paths["airway"] is None:
        raise FileNotFoundError(f"ATM case {cid} has no airway label.")

    ct_image = nib.load(str(paths["ct"]))
    gt_image = nib.load(str(paths["airway"]))
    ct_affine = np.asarray(ct_image.affine, dtype=np.float64)
    gt_affine = np.asarray(gt_image.affine, dtype=np.float64)
    if gt_image.shape != ct_image.shape or not np.allclose(
        gt_affine,
        ct_affine,
        atol=1e-4,
        rtol=1e-5,
    ):
        raise ValueError(
            f"Ground truth is not aligned with the CT for case {cid}: "
            f"GT shape {gt_image.shape}, CT shape {ct_image.shape}."
        )
    gt = np.asarray(gt_image.dataobj, dtype=np.uint8) > 0
    return gt, ct_affine


def _resolve_cases(args, batch_root: Path) -> list[str]:
    if args.cases:
        return [c.strip() for c in args.cases.split(",") if c.strip()]
    split_config = load_yaml_config(args.split_config)
    split = create_split_from_config(list_case_ids(batch_root), split_config)
    ids = cases_for_split(split, args.report_split)
    if not ids:
        raise SystemExit(f"No {args.report_split} case ids resolved from {args.split_config}.")
    return ids


def score_case(
    cid,
    pred_dir,
    batch_root,
    *,
    prefix,
    suffix,
    compute_branch,
    lcc_connectivity,
    compute_radius_bins,
):
    """Return (per-case metric row, {bin_label: pred-over-bin bool array}, gt_sum)."""
    gt, affine = _load_gt_mask_and_affine(cid, batch_root)
    padded = resolve_case_paths(cid, batch_root=batch_root)["case_id"]
    pred_path = Path(pred_dir) / f"{prefix}{padded}{suffix}"
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"No prediction for case {cid} at {pred_path}. "
            f"Expected nnU-Net output named {prefix}{padded}{suffix} in --pred-dir."
        )
    pred, pred_affine = _load_pred_mask(pred_path)
    if pred.shape != gt.shape or not np.allclose(
        pred_affine,
        affine,
        atol=1e-4,
        rtol=1e-5,
    ):
        raise ValueError(
            f"Geometry mismatch for case {cid}: pred shape {pred.shape} vs GT {gt.shape}, "
            f"or their affine grids differ. nnU-Net output must be in the original image geometry."
        )

    gt_sum = int(gt.sum())
    td_slices, gt_skeleton, gt_skeleton_sum = gt_centerline(gt)

    dice_raw, td_raw, prec_raw = cheap_metrics(pred, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
    pred_lcc = keep_component_containing_trachea(
        pred,
        connectivity=lcc_connectivity,
        affine=affine,
    ) > 0
    dice_lcc, td_lcc, prec_lcc = cheap_metrics(pred_lcc, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
    lcc_retained_fraction = float(pred_lcc.sum() / max(int(pred.sum()), 1))

    tprec_raw = float(topology_precision_from_masks(pred, gt))
    denom = tprec_raw + td_raw
    cldice_raw = float(2.0 * tprec_raw * td_raw / denom) if denom > 0 else 0.0

    tprec_lcc = float(topology_precision_from_masks(pred_lcc, gt))
    denom = tprec_lcc + td_lcc
    cldice_lcc = float(2.0 * tprec_lcc * td_lcc / denom) if denom > 0 else 0.0

    row = {
        "case_id": cid, "airway_voxels": gt_sum,
        "dice_raw": dice_raw, "td_raw": td_raw, "prec_raw": prec_raw,
        "tprec_raw": tprec_raw, "cldice_raw": cldice_raw,
        "dice_lcc": dice_lcc, "td_lcc": td_lcc, "prec_lcc": prec_lcc,
        "tprec_lcc": tprec_lcc, "cldice_lcc": cldice_lcc,
        "lcc_retained_fraction": lcc_retained_fraction,
        "bd_raw": None, "cldice_atm_raw": None, "td_atm_raw": None,
        "bd_lcc": None, "cldice_atm_lcc": None, "td_atm_lcc": None,
        # Backward-compatible aliases for the historical LCC-only ATM fields.
        "cldice_atm": None, "td_atm": None,
    }
    if compute_branch:
        atm_raw = airway_topology_metrics_from_masks(pred, gt, branch_detection_threshold=0.8)
        atm_lcc = airway_topology_metrics_from_masks(pred_lcc, gt, branch_detection_threshold=0.8)
        row["bd_raw"] = float(atm_raw["branch_detected"])
        row["cldice_atm_raw"] = float(atm_raw["cldice"])
        row["td_atm_raw"] = float(atm_raw["tree_length_detected"])
        row["bd_lcc"] = float(atm_lcc["branch_detected"])
        row["cldice_atm_lcc"] = float(atm_lcc["cldice"])
        row["td_atm_lcc"] = float(atm_lcc["tree_length_detected"])
        row["cldice_atm"] = row["cldice_atm_lcc"]
        row["td_atm"] = row["td_atm_lcc"]

    bin_hits = {}
    if compute_radius_bins:
        # Radius-stratified recall uses the RAW mask and is therefore invariant to
        # LCC connectivity. Connectivity sweeps can skip this expensive full-grid EDT.
        radius = ndimage.distance_transform_edt(gt)
        for label, lo, hi in RADIUS_BINS:
            m = gt & (radius >= lo) & (radius < hi)
            bin_hits[label] = pred[m] if m.any() else np.array([], dtype=bool)

    print(
        f"case {cid}: voxels {gt_sum:,}  | +LCC-{lcc_connectivity}  "
        f"raw Dice {dice_raw:.3f} clDice {cldice_raw:.3f}  | "
        f"LCC Dice {dice_lcc:.3f} TLD {td_lcc:.3f} "
        f"clDice {cldice_lcc:.3f} TPrec {tprec_lcc:.3f}  kept {lcc_retained_fraction:.3f}",
        flush=True,
    )
    return row, bin_hits, gt_sum


def _table_mean(rows):
    keys = [
        "dice_raw", "td_raw", "prec_raw", "tprec_raw", "cldice_raw",
        "dice_lcc", "td_lcc", "prec_lcc", "tprec_lcc", "cldice_lcc",
        "lcc_retained_fraction",
        "bd_raw", "cldice_atm_raw", "td_atm_raw",
        "bd_lcc", "cldice_atm_lcc", "td_atm_lcc",
        "cldice_atm", "td_atm",
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


def build_argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True, help="Folder of prediction niftis (nnU-Net output).")
    ap.add_argument("--data-config", type=Path, default=Path("configs/data/atm22.yaml"),
                    help="ATM'22 data YAML (batch_root). Default: configs/data/atm22.yaml.")
    ap.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/nnunet/atm22_split_l20.yaml"),
        help="Canonical ATM'22 split YAML.",
    )
    ap.add_argument("--report-split", choices=("val", "test", "train"), default="val",
                    help="Which split to score (default val — develop on val, seal test).")
    ap.add_argument("--cases", type=str, default=None, help="Comma-separated case ids (overrides --split-config).")
    ap.add_argument("--prefix", type=str, default="ATM_", help="Prediction filename prefix (default 'ATM_').")
    ap.add_argument("--suffix", type=str, default=".nii.gz", help="Prediction filename suffix (default '.nii.gz').")
    ap.add_argument("--branch", action="store_true", help="Also compute ATM'22 BD (slower branch parse).")
    ap.add_argument(
        "--lcc-connectivity",
        type=int,
        choices=(6, 18, 26),
        default=6,
        help="Voxel connectivity for trachea-anchored LCC postprocessing (default: 6).",
    )
    ap.add_argument(
        "--skip-radius-bins",
        action="store_true",
        help=(
            "Skip the full-grid GT distance transform and radius-bin recall. Use for "
            "LCC sensitivity sweeps because these raw-mask bins do not depend on connectivity."
        ),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output JSON. Defaults to <pred-dir>/nnunet_topology.json for LCC-6 "
            "or nnunet_topology_lcc{18,26}.json for the sensitivity variants."
        ),
    )
    return ap


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.report_split == "test" and not args.cases:
        print("WARNING: scoring the SEALED TEST split — final numbers only.", flush=True)

    data_config = load_yaml_config(args.data_config)
    batch_root = resolve_project_path(data_config["batch_root"])
    cases = _resolve_cases(args, batch_root)
    print(
        f"scoring {len(cases)} case(s) from {args.pred_dir}  "
        f"(split={args.report_split}, trachea-LCC-{args.lcc_connectivity})",
        flush=True,
    )

    rows, bin_hits_per_case = [], []
    for cid in cases:
        row, bin_hits, _ = score_case(
            cid,
            args.pred_dir,
            batch_root,
            prefix=args.prefix,
            suffix=args.suffix,
            compute_branch=args.branch,
            lcc_connectivity=args.lcc_connectivity,
            compute_radius_bins=not args.skip_radius_bins,
        )
        rows.append(row)
        bin_hits_per_case.append(bin_hits)

    mean_row = _table_mean(rows)
    bins_out = _build_bins(bin_hits_per_case)

    default_name = (
        "nnunet_topology.json"
        if args.lcc_connectivity == 6
        else f"nnunet_topology_lcc{args.lcc_connectivity}.json"
    )
    out_path = Path(args.out) if args.out else (args.pred_dir / default_name)
    out_path.write_text(json.dumps({
        "topology_metric_version": TOPOLOGY_METRIC_VERSION,
        "scorer": "evaluate_nnunet_predictions",
        "pred_dir": str(args.pred_dir),
        "dataset": "atm22",
        "report_split": args.report_split,
        "operating_point": {"threshold": "native_argmax", "selected_on": "none",
                            "note": "nnU-Net hard mask at its native op-point; ours are val-selected."},
        "postprocessing": {
            "lcc": "trachea",
            "connectivity": args.lcc_connectivity,
        },
        "branch_metrics": bool(args.branch),
        "radius_bins_computed": not args.skip_radius_bins,
        "report_cases": cases,
        "table_per_case": rows,
        "table_mean": mean_row,
        "bins": bins_out,
    }, indent=2))

    print(f"\n--- mean (+trachea-LCC-{args.lcc_connectivity}) ---")
    print(f"  clDice {mean_row['cldice_lcc']:.4f} | TLD {mean_row['td_lcc']:.4f} | "
          f"TPrec {mean_row['tprec_lcc']:.4f} | Dice+LCC {mean_row['dice_lcc']:.4f} | "
          f"prec_lcc {mean_row['prec_lcc']:.4f}")
    print(f"  raw: Dice {mean_row['dice_raw']:.4f} | TD {mean_row['td_raw']:.4f} | "
          f"TPrec {mean_row['tprec_raw']:.4f} | clDice {mean_row['cldice_raw']:.4f} | "
          f"prec {mean_row['prec_raw']:.4f}")
    print(f"  mean LCC-retained fraction {mean_row['lcc_retained_fraction']:.4f}")
    if mean_row.get("bd_lcc") is not None:
        print(f"  ATM raw: BD {mean_row['bd_raw']:.4f} | clDice {mean_row['cldice_atm_raw']:.4f} | "
              f"TLD {mean_row['td_atm_raw']:.4f}")
        print(f"  ATM LCC: BD {mean_row['bd_lcc']:.4f} | clDice {mean_row['cldice_atm_lcc']:.4f} | "
              f"TLD {mean_row['td_atm_lcc']:.4f}")
    for b in bins_out:
        if b["bin"].startswith("r=1"):
            print(f"  wall-shell r=1 recall {b['recall']:.4f}  ({b['voxels']:,} voxels)")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
