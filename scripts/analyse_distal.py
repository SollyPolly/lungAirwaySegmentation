"""Operating-point selection + the ATM'22 table, plus distal stratification.

Picks the operating threshold and reports Dice / TD / BD / clDice (+LCC) at it, with
**test-set hygiene in the defaults** (develop on val, seal test).

Selection modes (`--select-by`):
- **cldice (default)** — maximise clDice over a small candidate set of low-to-mid
  thresholds (`--cldice-candidates`, default 0.4,0.5,0.6). This finds the
  topology-optimal operating point directly (clDice = harmonic mean of topology
  precision and TD). Warns if the peak sits at the candidate-range edge (widen the
  candidates). The right default for a topology-aware method.
- **voxel-precision** — max TD+LCC subject to voxel-precision+LCC >= floor
  (`--precision-floor`). Cheap (no skeletonisation); a fast first look.

Performance: the threshold *sweep* is cheap (Dice/TD/voxel-precision, no
skeletonisation). clDice / topology precision (which need a prediction skeleton) are
computed only at the few candidate thresholds (cldice mode) or once at the chosen
threshold; BD (ATM'22 branch parse) only on the test report. Budget ~3-4 min/case for
clDice mode (one skeletonisation per candidate) — a 20-case val run is ~1-1.5 h.

Develop on val, seal test: defaults select + report on **val**. Pass
`--report-split test` for the sealed final table (selects on val, reports on test,
computes BD).

Usage:
    # development (default): clDice-optimal op-point on val, reported on val
    python -m scripts.analyse_distal --run-dir runs/<exp>/<run>

    # widen the clDice candidate range (e.g. a heavily de-saturated model)
    python -m scripts.analyse_distal --run-dir runs/<exp>/<run> --cldice-candidates 0.3,0.4,0.5,0.6,0.7

    # FINAL sealed test table (frozen models only)
    python -m scripts.analyse_distal --run-dir runs/<exp>/<run> --report-split test

    # fast cheap look (voxel-precision floor, no skeletonisation)
    python -m scripts.analyse_distal --run-dir runs/<exp>/<run> --select-by voxel-precision

    # fixed threshold, no selection
    python -m scripts.analyse_distal --run-dir runs/<exp>/<run> --threshold 0.5 --select-split none

Use `--out distal_analysis__<tag>.json` to avoid clobbering across runs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from lung_airway_segmentation.inference.postprocess import keep_component_containing_trachea
from lung_airway_segmentation.inference.sliding_window import predict_logits_for_volume
from lung_airway_segmentation.metrics.topology import (
    _foreground_slices,
    _largest_connected_component,
    _skeletonize,
    airway_topology_metrics_from_masks,
    topology_precision_from_masks,
)
from lung_airway_segmentation.preprocessing.geometry import normalize_margin
from lung_airway_segmentation.training.builders import build_model, resolve_checkpoint_path
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

# Cheap sweep grid (Dice/TD/voxel-precision, no skeletonisation). 0.3/0.4 included
# because de-saturated models (low pos_weight) shift the useful range left.
SWEEP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

# clDice is maximised over THIS small set (one skeletonisation per candidate per case).
# Covers the clDice peak for both saturated (~0.5) and mildly de-saturated models; widen
# with --cldice-candidates if the edge warning fires.
CLDICE_CANDIDATES_DEFAULT = [0.4, 0.5, 0.6]

_SPLIT_KEYS = {"val": "val_case_ids", "test": "test_case_ids", "train": "train_case_ids"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory with resolved_config.json + checkpoint.")
    parser.add_argument("--select-by", choices=("cldice", "voxel-precision"), default="cldice",
                        help="How to choose the operating threshold (default cldice: max clDice over candidates).")
    parser.add_argument("--select-split", choices=("val", "test", "train", "none"), default="val",
                        help="Split to choose the operating threshold on (default val). 'none' uses the run threshold or --threshold.")
    parser.add_argument("--report-split", choices=("val", "test", "train"), default="val",
                        help="Split to report the table on (default val — develop here). Pass 'test' for the sealed final table.")
    parser.add_argument("--cldice-candidates", type=str, default="0.4,0.5,0.6",
                        help="Comma-separated thresholds clDice is maximised over (cldice mode). Widen if the peak is at the edge.")
    parser.add_argument("--cldice-max-ratio", type=float, default=6.0,
                        help="cldice mode: skip skeletonising a candidate whose LCC'd prediction exceeds this multiple of the GT voxel count (massive over-segmentation — never the clDice peak, and the slow ones to skeletonise).")
    parser.add_argument("--precision-floor", type=float, default=0.80,
                        help="voxel-precision mode only: min voxel-precision+LCC the chosen threshold must meet.")
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated case IDs to override the *report* split.")
    parser.add_argument("--max-cases", type=int, default=None, help="Cap cases per split (default: all).")
    parser.add_argument("--checkpoint", choices=("best", "dice", "topology", "last"), default="best",
                        help="best = Dice-selected (alias of dice); topology = hard-clDice@0.5 selection; last = final epoch.")
    parser.add_argument("--threshold", type=float, default=None, help="Fix the operating threshold (skips selection).")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap (0.5 default; 0.25 faster for dev).")
    parser.add_argument("--sw-batch", type=int, default=8, help="Sliding-window batch size (raise on big GPUs to speed inference).")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-config", type=Path, default=None, help="Override dataset to load cases from. Default: the run's data config.")
    parser.add_argument("--out", type=str, default="distal_analysis.json",
                        help="Output JSON filename (under run-dir unless absolute). Use distinct names per run to avoid clobbering.")
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
        ct_img = loader(paths["ct"])
        affine = np.asarray(ct_img.affine, dtype=np.float64)  # native orientation → superior axis
        ct = scaler(ct_img)
        gt = np.asarray(loader(paths["airway"])[0]) > 0
        return ct, gt, affine

    if dataset_name == "aeropath":
        from lung_airway_segmentation.preprocessing.pipeline import preprocess_case

        data_root = resolve_project_path(data_config["raw_data_root"])
        crop_margin = normalize_margin(data_config["preprocessing"]["crop_margin_voxels"])
        case = preprocess_case(
            case_id, data_root=data_root, include_lung_mask=False,
            hu_window=tuple(float(v) for v in hu_window), crop_margin=crop_margin,
        )
        return torch.from_numpy(case.ct), case.airway_mask.astype(bool), np.asarray(case.affine, dtype=np.float64)

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def split_case_ids(metadata, split):
    return [str(c) for c in metadata.get("splits", {}).get(_SPLIT_KEYS[split], [])]


def infer_probability(model, ct, *, device, roi_size, overlap, sw_batch):
    logits = predict_logits_for_volume(
        model, ct, device=device, roi_size=roi_size,
        sw_batch_size=sw_batch, overlap=overlap, use_amp=False,
    )
    return torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)


def gt_centerline(gt):
    """Largest-connected-component skeleton of the GT (reused across thresholds)."""
    component = _largest_connected_component(gt)
    slices = _foreground_slices(component)
    skeleton = _skeletonize(component[slices])
    return slices, skeleton, int(skeleton.sum())


def cheap_metrics(predicted, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum):
    """Dice, TD (skeleton recall), and VOXEL precision — no prediction skeletonisation."""
    pred_sum = int(predicted.sum())
    tp = int((predicted & gt).sum())
    dice = 2 * tp / ((pred_sum + gt_sum) or 1)
    precision = tp / (pred_sum or 1)
    td = (
        int((gt_skeleton & predicted[td_slices]).sum()) / gt_skeleton_sum
        if gt_skeleton_sum else 1.0
    )
    return float(dice), float(td), float(precision)


def sweep_case(prob, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum, affine=None):
    """Per-threshold Dice/TD (raw + LCC) and voxel-precision+LCC over SWEEP_THRESHOLDS — all cheap."""
    rows = {}
    for t in SWEEP_THRESHOLDS:
        pred = prob >= t
        dice_raw, td_raw, prec_raw = cheap_metrics(pred, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
        pred_lcc = keep_component_containing_trachea(pred, affine=affine) > 0
        dice_lcc, td_lcc, prec_lcc = cheap_metrics(pred_lcc, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
        rows[t] = {
            "dice": dice_raw, "dice_lcc": dice_lcc,
            "td": td_raw, "td_lcc": td_lcc,
            "prec": prec_raw, "prec_lcc": prec_lcc,
            "lcc_retained_fraction": float(pred_lcc.sum() / max(int(pred.sum()), 1)),
        }
    return rows


def topology_at(prob, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum, thresholds, max_ratio=6.0, affine=None):
    """clDice + topology precision (one skeletonisation each) at the given thresholds.

    A candidate whose LCC'd prediction exceeds ``max_ratio`` x the GT voxel count is
    skipped (tprec/clDice = None): such masks are massive over-segmentations — never
    the clDice peak, and pathologically slow to skeletonise (a near-whole-lung blob on
    a saturated model). This keeps the candidate scan affordable on saturated models.
    """
    out = {}
    for t in thresholds:
        pred = prob >= t
        dice_raw, td_raw, prec_raw = cheap_metrics(pred, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
        pred_lcc = keep_component_containing_trachea(pred, affine=affine) > 0
        dice_lcc, td_lcc, prec_lcc = cheap_metrics(pred_lcc, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
        if gt_sum and int(pred_lcc.sum()) > max_ratio * gt_sum:
            tprec_lcc = cldice_lcc = None  # gated: too large to skeletonise, not the peak
        else:
            tprec_lcc = float(topology_precision_from_masks(pred_lcc, gt))  # skeletonises the prediction
            denom = tprec_lcc + td_lcc
            cldice_lcc = float(2.0 * tprec_lcc * td_lcc / denom) if denom > 0 else 0.0
        out[t] = {
            "dice_raw": dice_raw, "td_raw": td_raw, "prec_raw": prec_raw,
            "dice_lcc": dice_lcc, "td_lcc": td_lcc, "prec_lcc": prec_lcc,
            "tprec_lcc": tprec_lcc, "cldice_lcc": cldice_lcc,
            "lcc_retained_fraction": float(pred_lcc.sum() / max(int(pred.sum()), 1)),
        }
    return out


def mean_sweep(case_rows):
    out = []
    for t in SWEEP_THRESHOLDS:
        vals = [cr[t] for cr in case_rows]
        out.append({
            "threshold": t,
            "dice": float(np.mean([v["dice"] for v in vals])),
            "dice_lcc": float(np.mean([v["dice_lcc"] for v in vals])),
            "td": float(np.mean([v["td"] for v in vals])),
            "td_lcc": float(np.mean([v["td_lcc"] for v in vals])),
            "prec": float(np.mean([v["prec"] for v in vals])),
            "prec_lcc": float(np.mean([v["prec_lcc"] for v in vals])),
            "lcc_retained_fraction": float(np.mean([v["lcc_retained_fraction"] for v in vals])),
        })
    return out


def mean_candidates(candidate_rows, thresholds):
    """Mean clDice / topo-precision / TD per candidate threshold (for selection + display).

    Gated cells (clDice/tprec None) are excluded from those means; a candidate gated
    for every case gets clDice None and is dropped from selection.
    """
    out = []
    for t in thresholds:
        vals = [cr[t] for cr in candidate_rows]
        cldice_vals = [v["cldice_lcc"] for v in vals if v["cldice_lcc"] is not None]
        tprec_vals = [v["tprec_lcc"] for v in vals if v["tprec_lcc"] is not None]
        out.append({
            "threshold": t,
            "cldice_lcc": float(np.mean(cldice_vals)) if cldice_vals else None,
            "tprec_lcc": float(np.mean(tprec_vals)) if tprec_vals else None,
            "td_lcc": float(np.mean([v["td_lcc"] for v in vals])),
            "dice_lcc": float(np.mean([v["dice_lcc"] for v in vals])),
            "prec_raw": float(np.mean([v["prec_raw"] for v in vals])),
            "prec_lcc": float(np.mean([v["prec_lcc"] for v in vals])),
            "lcc_retained_fraction": float(np.mean([v["lcc_retained_fraction"] for v in vals])),
            "n_valid": len(cldice_vals),
        })
    return out


def select_by_voxel_precision(sweep, precision_floor):
    """Max TD+LCC subject to voxel-precision+LCC >= floor; else max precision."""
    eligible = [r for r in sweep if r["prec_lcc"] >= precision_floor]
    if eligible:
        best = max(eligible, key=lambda r: r["td_lcc"])
        reason = f"max TD+LCC s.t. voxel-precision+LCC >= {precision_floor:.2f}"
    else:
        best = max(sweep, key=lambda r: r["prec_lcc"])
        reason = f"no threshold met precision floor {precision_floor:.2f}; fell back to max voxel-precision+LCC"
    return float(best["threshold"]), reason


def select_by_cldice(candidate_means):
    """Threshold with max mean clDice among non-gated candidates; flag edge peaks."""
    valid = [r for r in candidate_means if r["cldice_lcc"] is not None]
    if not valid:
        return None, "all clDice candidates gated as over-segmented — lower the candidate range or raise --cldice-max-ratio", False
    best = max(valid, key=lambda r: r["cldice_lcc"])
    valid_thresholds = [r["threshold"] for r in valid]
    at_edge = len(valid_thresholds) > 1 and best["threshold"] in (min(valid_thresholds), max(valid_thresholds))
    reason = f"max clDice over candidates {[r['threshold'] for r in candidate_means]}"
    return float(best["threshold"]), reason, at_edge


def print_sweep(sweep, title):
    print(f"\n--- {title} ---")
    print(f"{'thresh':>7}  {'Dice':>6}  {'Dice+LCC':>8}  {'TD':>6}  {'TD+LCC':>7}  "
          f"{'Prec':>6}  {'Prec+LCC':>8}  {'LCC kept':>8}")
    for r in sweep:
        print(f"{r['threshold']:>7.2f}  {r['dice']:>6.3f}  {r['dice_lcc']:>8.3f}  "
              f"{r['td']:>6.3f}  {r['td_lcc']:>7.3f}  {r['prec']:>6.3f}  "
              f"{r['prec_lcc']:>8.3f}  {r['lcc_retained_fraction']:>8.3f}")


def print_candidates(candidate_means):
    print("\n--- clDice candidate scan (+LCC) ---")
    print(f"{'thresh':>7}  {'clDice':>8}  {'TPrec':>6}  {'TD':>6}  {'Dice':>6}")
    for r in candidate_means:
        cl = f"{r['cldice_lcc']:.3f}" if r["cldice_lcc"] is not None else "gated"
        tp = f"{r['tprec_lcc']:.3f}" if r["tprec_lcc"] is not None else "—"
        print(f"{r['threshold']:>7.2f}  {cl:>8}  {tp:>6}  {r['td_lcc']:>6.3f}  {r['dice_lcc']:>6.3f}")


def run_split(model, dataset_name, data_config, hu_window, cases, *, device, roi_size, overlap, sw_batch,
              chosen_threshold=None, compute_bd=False, collect_bins=False, cldice_candidates=None, cldice_max_ratio=6.0):
    """Inference over a list of cases.

    Always returns the cheap per-case sweep. If ``cldice_candidates`` is given, also
    returns per-case clDice/topo-precision at those thresholds (one skeletonisation
    each). If ``chosen_threshold`` is given, builds the per-case table at it (BD only
    when ``compute_bd``). ``collect_bins`` accumulates radius-stratified probabilities.
    """
    case_sweeps, table_rows, candidate_rows = [], [], []
    bin_probs = {label: [] for label, _, _ in RADIUS_BINS}

    for cid in cases:
        ct, gt, affine = load_case(dataset_name, data_config, cid, hu_window)
        gt_sum = int(gt.sum())
        td_slices, gt_skeleton, gt_skeleton_sum = gt_centerline(gt)
        prob = infer_probability(model, ct, device=device, roi_size=roi_size, overlap=overlap, sw_batch=sw_batch)

        case_sweeps.append(sweep_case(prob, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum, affine))

        if cldice_candidates:
            candidate_rows.append(topology_at(prob, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum, cldice_candidates, cldice_max_ratio, affine))

        if chosen_threshold is not None:
            pred = prob >= chosen_threshold
            dice_raw, td_raw, prec_raw = cheap_metrics(pred, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
            pred_lcc = keep_component_containing_trachea(pred, affine=affine) > 0
            dice_lcc, td_lcc, prec_lcc = cheap_metrics(pred_lcc, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
            lcc_retained_fraction = float(pred_lcc.sum() / max(int(pred.sum()), 1))
            if compute_bd:
                topo = airway_topology_metrics_from_masks(pred_lcc, gt)  # +BD branch parse (expensive)
                td_lcc = float(topo["tree_length_detected"])
                bd_lcc = float(topo["branch_detected"])
                cldice_lcc = float(topo["cldice"])
                tprec_lcc = float(topo["topology_precision"])
                ref_b, det_b = int(topo["reference_branch_count"]), int(topo["detected_branch_count"])
            else:
                tprec_lcc = float(topology_precision_from_masks(pred_lcc, gt))  # one skeletonisation
                denom = tprec_lcc + td_lcc
                cldice_lcc = float(2.0 * tprec_lcc * td_lcc / denom) if denom > 0 else 0.0
                bd_lcc = ref_b = det_b = None
            table_rows.append({
                "case_id": cid, "airway_voxels": gt_sum,
                "dice_lcc": dice_lcc, "td_lcc": td_lcc, "bd_lcc": bd_lcc,
                "cldice_lcc": cldice_lcc, "tprec_lcc": tprec_lcc, "prec_lcc": prec_lcc,
                "reference_branches": ref_b, "detected_branches": det_b,
                "dice_raw": dice_raw, "td_raw": td_raw, "prec_raw": prec_raw,
                "lcc_retained_fraction": lcc_retained_fraction,
            })
            bd_str = f"BD {bd_lcc:.3f}  " if bd_lcc is not None else ""
            print(f"case {cid}: voxels {gt_sum:,}  | +LCC@{chosen_threshold:.2f}  "
                  f"Dice {dice_lcc:.3f}  TD {td_lcc:.3f}  {bd_str}clDice {cldice_lcc:.3f}  Prec {prec_lcc:.3f}")
        elif cldice_candidates:
            print(f"case {cid}: voxels {gt_sum:,}  (clDice scan)")
        else:
            print(f"case {cid}: voxels {gt_sum:,}  (swept)")

        if collect_bins:
            radius = ndimage.distance_transform_edt(gt)
            for label, lo, hi in RADIUS_BINS:
                m = gt & (radius >= lo) & (radius < hi)
                if m.any():
                    bin_probs[label].append(prob[m])

    return case_sweeps, table_rows, bin_probs, candidate_rows


def table_from_candidates(case_ids, candidate_rows, threshold):
    """Dev-mode table at a clDice candidate threshold (full topology metrics, no BD)."""
    rows = []
    for cid, cr in zip(case_ids, candidate_rows):
        m = cr[threshold]
        rows.append({
            "case_id": cid,
            "dice_lcc": m["dice_lcc"], "td_lcc": m["td_lcc"], "prec_lcc": m["prec_lcc"],
            "tprec_lcc": m["tprec_lcc"], "cldice_lcc": m["cldice_lcc"], "bd_lcc": None,
            "dice_raw": m["dice_raw"], "td_raw": m["td_raw"], "prec_raw": m["prec_raw"],
            "lcc_retained_fraction": m["lcc_retained_fraction"],
        })
    return rows


def table_from_sweep(case_ids, case_sweeps, threshold):
    """Dev-mode table from the cheap sweep (Dice/TD/voxel-prec; no clDice/BD) — voxel-precision mode."""
    rows = []
    for cid, cs in zip(case_ids, case_sweeps):
        r = cs[threshold]
        rows.append({
            "case_id": cid,
            "dice_lcc": r["dice_lcc"], "td_lcc": r["td_lcc"], "prec_lcc": r["prec_lcc"],
            "tprec_lcc": None, "cldice_lcc": None, "bd_lcc": None,
            "dice_raw": r["dice"], "td_raw": r["td"], "prec_raw": r["prec"],
            "lcc_retained_fraction": r["lcc_retained_fraction"],
        })
    return rows


def table_mean(rows):
    keys = [
        "dice_raw", "td_raw", "prec_raw",
        "dice_lcc", "td_lcc", "bd_lcc", "cldice_lcc", "tprec_lcc", "prec_lcc",
        "lcc_retained_fraction",
    ]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def build_bins_output(bin_probs, threshold):
    total = sum(sum(a.size for a in v) for v in bin_probs.values())
    bins_out = []
    if not total:
        return bins_out
    for label, _, _ in RADIUS_BINS:
        if not bin_probs[label]:
            continue
        p = np.concatenate(bin_probs[label])
        bins_out.append({
            "bin": label, "voxels": int(p.size), "pct_airway": float(100 * p.size / total),
            "mean_prob": float(p.mean()), "median_prob": float(np.median(p)),
            "p90_prob": float(np.percentile(p, 90)),
            "recall_at_threshold": float((p >= threshold).mean()),
            "recall_at_0.5": float((p >= 0.5).mean()),
        })
    return bins_out


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "resolved_config.json").read_text())
    metadata = json.loads((run_dir / "run_metadata.json").read_text())

    data_config = load_yaml_config(args.data_config) if args.data_config else cfg["data"]
    dataset_name = str(data_config["dataset_name"]).lower()
    hu_window = tuple(float(v) for v in data_config["preprocessing"]["hu_window"])
    roi_size = tuple(int(v) for v in cfg["training"]["validation"]["roi_size"])
    candidates = [float(x) for x in args.cldice_candidates.split(",") if x.strip()] or CLDICE_CANDIDATES_DEFAULT

    device = resolve_device(args.device)
    ckpt = resolve_checkpoint_path(run_dir, args.checkpoint)
    model = build_model(device, cfg["model"])
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    model.eval()

    def cap(ids):
        return ids if args.max_cases is None else ids[: args.max_cases]

    infer_kw = dict(device=device, roi_size=roi_size, overlap=args.overlap, sw_batch=args.sw_batch)
    print(f"model: {metadata.get('experiment_name')}  | checkpoint {args.checkpoint}  | overlap {args.overlap}  | sw_batch {args.sw_batch}")

    if args.cases:
        report_cases, report_label = [c.strip() for c in args.cases.split(",") if c.strip()], "custom"
    else:
        report_cases, report_label = cap(split_case_ids(metadata, args.report_split)), args.report_split
    if not report_cases:
        raise SystemExit("No report cases: pass --cases or ensure run_metadata has the requested split.")

    # --- 1. choose the operating threshold ---------------------------------------
    val_sweep = candidate_means = None
    sel_cases = sel_sweeps = sel_bins = sel_candidates = None
    same_split = False
    select_mode = None
    if args.threshold is not None:
        chosen_threshold, selection_reason, selected_on = float(args.threshold), "fixed by --threshold", None
    elif args.select_split != "none":
        sel_cases = cap(split_case_ids(metadata, args.select_split))
        if not sel_cases:
            raise SystemExit(f"No {args.select_split} cases in run_metadata to select an operating point on.")
        same_split = (not args.cases) and (args.select_split == args.report_split)
        select_mode = args.select_by
        want_candidates = candidates if args.select_by == "cldice" else None
        print(f"\nselecting operating point on {args.select_split} ({len(sel_cases)} cases) by {args.select_by}...")
        sel_sweeps, _, sel_bins, sel_candidates = run_split(
            model, dataset_name, data_config, hu_window, sel_cases,
            collect_bins=same_split, cldice_candidates=want_candidates,
            cldice_max_ratio=args.cldice_max_ratio, **infer_kw,
        )
        val_sweep = mean_sweep(sel_sweeps)
        print_sweep(val_sweep, f"{args.select_split} sweep (mean over {len(sel_cases)} cases)")
        if args.select_by == "cldice":
            candidate_means = mean_candidates(sel_candidates, candidates)
            print_candidates(candidate_means)
            chosen_threshold, selection_reason, at_edge = select_by_cldice(candidate_means)
            if chosen_threshold is None:
                raise SystemExit("clDice selection failed: " + selection_reason)
            if at_edge:
                print(f"\n[warn] clDice peak is at the edge of the evaluated candidates — the optimum "
                      f"may be beyond it. Re-run with a wider --cldice-candidates. (Candidates gated "
                      f"as over-segmented are skipped, not edges — see the scan above.)")
        else:
            chosen_threshold, selection_reason = select_by_voxel_precision(val_sweep, args.precision_floor)
        selected_on = args.select_split
    else:
        chosen_threshold = float(cfg["training"]["validation"].get("threshold", 0.99))
        selection_reason, selected_on = "run validation threshold (no selection)", None
        print(f"\n[warn] no selection: using run validation threshold {chosen_threshold:.2f}")

    print(f"\n==> operating threshold {chosen_threshold:.2f}  ({selection_reason}"
          + (f", on {selected_on}" if selected_on else "") + ")")

    # --- 2. report the table at the chosen threshold -----------------------------
    reuse_cldice = same_split and select_mode == "cldice" and chosen_threshold in candidates
    reuse_voxel = same_split and select_mode == "voxel-precision" and chosen_threshold in SWEEP_THRESHOLDS
    if reuse_cldice:
        print(f"\nreporting on {report_label} ({len(report_cases)} cases) at {chosen_threshold:.2f}, +LCC "
              "— reusing val inference (dev mode; BD omitted, run --report-split test for it):")
        report_sweeps, bin_probs = sel_sweeps, sel_bins
        table_rows = table_from_candidates(sel_cases, sel_candidates, chosen_threshold)
        for r in table_rows:
            print(f"case {r['case_id']}: +LCC@{chosen_threshold:.2f}  Dice {r['dice_lcc']:.3f}  "
                  f"TD {r['td_lcc']:.3f}  clDice {r['cldice_lcc']:.3f}  TPrec {r['tprec_lcc']:.3f}  Prec {r['prec_lcc']:.3f}")
    elif reuse_voxel:
        print(f"\nreporting on {report_label} ({len(report_cases)} cases) at {chosen_threshold:.2f}, +LCC "
              "— reusing val inference (dev mode; clDice/BD omitted):")
        report_sweeps, bin_probs = sel_sweeps, sel_bins
        table_rows = table_from_sweep(sel_cases, sel_sweeps, chosen_threshold)
        for r in table_rows:
            print(f"case {r['case_id']}: +LCC@{chosen_threshold:.2f}  Dice {r['dice_lcc']:.3f}  "
                  f"TD {r['td_lcc']:.3f}  Prec {r['prec_lcc']:.3f}")
    else:
        if args.report_split == "test" and not args.cases:
            print("\n[note] reporting on the SEALED TEST split — only on frozen, pre-decided models. "
                  "Develop and tune on val.")
        compute_bd = (args.report_split == "test")
        print(f"\nreporting on {report_label} ({len(report_cases)} cases) at {chosen_threshold:.2f}, +LCC"
              + (" (with BD branch parse)" if compute_bd else "") + ":")
        report_sweeps, table_rows, bin_probs, _ = run_split(
            model, dataset_name, data_config, hu_window, report_cases,
            chosen_threshold=chosen_threshold, compute_bd=compute_bd, collect_bins=True, **infer_kw,
        )
    report_sweep = mean_sweep(report_sweeps)
    mean_row = table_mean(table_rows)

    def fmt(value):
        return f"{value:.3f}" if value is not None else "—"

    print(f"\n=== TABLE ({report_label} @ {chosen_threshold:.2f}) ===")
    print(f"  RAW   Dice {fmt(mean_row['dice_raw'])}  TD {fmt(mean_row['td_raw'])}  "
          f"Prec {fmt(mean_row['prec_raw'])}")
    print(f"  +LCC  Dice {fmt(mean_row['dice_lcc'])}  TD {fmt(mean_row['td_lcc'])}  "
          f"BD {fmt(mean_row['bd_lcc'])}  clDice {fmt(mean_row['cldice_lcc'])}  "
          f"TPrec {fmt(mean_row['tprec_lcc'])}  Prec {fmt(mean_row['prec_lcc'])}  "
          f"LCC-kept {fmt(mean_row['lcc_retained_fraction'])}")

    bins_out = build_bins_output(bin_probs, chosen_threshold)
    if bins_out:
        print(f"\n{'radius bin':>16}  {'voxels':>9}  {'%airway':>7}  {'meanP':>6}  {'medP':>6}  {'rec@op':>6}  {'rec@.5':>6}")
        for row in bins_out:
            print(f"{row['bin']:>16}  {row['voxels']:>9,}  {row['pct_airway']:>6.1f}%  "
                  f"{row['mean_prob']:>6.3f}  {row['median_prob']:>6.3f}  "
                  f"{100*row['recall_at_threshold']:>5.1f}%  {100*row['recall_at_0.5']:>5.1f}%")
    if not (reuse_cldice or reuse_voxel):
        print_sweep(report_sweep, f"{report_label} sweep (mean over {len(report_cases)} cases)")

    out_path = Path(args.out) if Path(args.out).is_absolute() else run_dir / args.out
    out_path.write_text(json.dumps({
        "run": metadata.get("experiment_name"),
        "dataset": dataset_name,
        "overlap": args.overlap,
        "checkpoint": args.checkpoint,
        "select_by": args.select_by if args.threshold is None and args.select_split != "none" else "fixed/none",
        "operating_point": {
            "threshold": chosen_threshold,
            "reason": selection_reason,
            "selected_on": selected_on,
            "precision_floor": args.precision_floor,
            "cldice_candidates": candidates if select_mode == "cldice" else None,
        },
        "report_split": report_label,
        "report_cases": report_cases,
        "dev_mode": bool(reuse_cldice or reuse_voxel),
        "bd_computed": any(r.get("bd_lcc") is not None for r in table_rows),
        "selection_sweep": val_sweep,
        "cldice_candidate_scan": candidate_means,
        "threshold_sweep": report_sweep,
        "table_per_case": table_rows,
        "table_mean": mean_row,
        "bins": bins_out,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
