"""Operating-point selection + the ATM'22 table, plus distal stratification.

Two jobs in one script, with **test-set hygiene baked into the defaults**:

1. **Choose a defensible operating threshold** by sweeping thresholds on a split and
   picking the one that maximises tree-length detected (TD, +LCC) subject to a
   precision floor — recover as much tree as possible without leakage exploding.

2. **Report the table** (Dice / TD / BD / clDice, +LCC) at that fixed threshold, per
   case and mean.

**Develop on val, seal test.** By default this selects the op-point on **val** and
reports on **val** — use it to compare configs (pos_weight, clDice weight, patch
borders) without touching test. Run the final table on the sealed **test** split only
when models are frozen, via ``--report-split test``; that path selects on val, reports
on test (no test-set tuning), and is the only mode that computes the expensive ATM'22
branch-detected (BD) metric.

Performance: the threshold sweep is **cheap** — Dice, TD (reuses the GT skeleton), and
**voxel** precision (TP/(TP+FP)); it does NOT skeletonise the prediction. Centerline
metrics that need a prediction skeleton (clDice, topology precision, BD) are computed
**once per case, at the chosen threshold only**, and BD (branch parsing) only on the
test report. This keeps a 20-case val run to minutes, not hours.

Usage:
    # development (default): select + report on val. Test stays sealed.
    python -m scripts.analyse_distal --run-dir runs/<exp>/<run> --overlap 0.5

    # FINAL table on the sealed test split — frozen, pre-decided models only
    python -m scripts.analyse_distal --run-dir runs/<exp>/<run> --report-split test --overlap 0.5

    # quick smoke (cap cases)
    python -m scripts.analyse_distal --run-dir runs/<exp>/<run> --max-cases 2

Defaults: select on val, **report on val**, precision floor 0.80 (voxel precision),
overlap 0.5, the run's own checkpoint/model/data-config. Saved to
<run-dir>/distal_analysis.json.

Notes / caveats:
- "Radius" is per-voxel distance-to-wall (scipy EDT) — a cheap proxy for airway
  generation; r=1 is dominated by true distal branches but includes the one-voxel
  surface shell of thick airways. The recall *trend* across bins is the robust signal.
- TD is recall-like (ignores false positives); read it at the fixed operating
  threshold alongside the precision floor / BD, never its max.
- The val/val dev mode is mildly optimistic (threshold chosen on the same cases it's
  reported on) — fine for *ranking* configs; the unbiased number is the sealed test run.
- The precision FLOOR is on voxel precision (cheap, for selection); the per-case table
  also reports clDice + topology precision at the chosen threshold. BD and clDice are
  "—" in dev mode (they need the prediction skeleton) — run --report-split test for them.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from lung_airway_segmentation.inference.postprocess import keep_largest_connected_component
from lung_airway_segmentation.inference.sliding_window import predict_logits_for_volume
from lung_airway_segmentation.metrics.topology import (
    _foreground_slices,
    _largest_connected_component,
    _skeletonize,
    airway_topology_metrics_from_masks,
    topology_precision_from_masks,
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

# Sweep grid. 0.3/0.4 dropped: degenerate (Dice ~0) and the fattest masks — by far
# the slowest to post-process — so they only cost time. clDice models still calibrate
# below the pos_weight baseline, so the useful operating range is ~0.5-0.9.
SWEEP_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

_SPLIT_KEYS = {"val": "val_case_ids", "test": "test_case_ids", "train": "train_case_ids"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory with resolved_config.json + checkpoint.")
    parser.add_argument("--select-split", choices=("val", "test", "train", "none"), default="val",
                        help="Split to choose the operating threshold on (default val). 'none' uses the run threshold or --threshold.")
    parser.add_argument("--report-split", choices=("val", "test", "train"), default="val",
                        help="Split to report the table on (default val — develop here). Pass 'test' for the sealed final table.")
    parser.add_argument("--precision-floor", type=float, default=0.80,
                        help="Min VOXEL precision (TP/(TP+FP)) +LCC the chosen threshold must meet (default 0.80).")
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated case IDs to override the *report* split.")
    parser.add_argument("--max-cases", type=int, default=None, help="Cap cases per split (default: all).")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--threshold", type=float, default=None, help="Fix the operating threshold (skips selection).")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap (0.5 default; 0.25 faster for dev).")
    parser.add_argument("--sw-batch", type=int, default=8, help="Sliding-window batch size (raise on big GPUs to speed inference).")
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


def sweep_case(prob, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum):
    """Per-threshold Dice/TD (raw + LCC) and voxel-precision+LCC — all cheap."""
    rows = {}
    for t in SWEEP_THRESHOLDS:
        pred = prob >= t
        dice_raw, td_raw, _ = cheap_metrics(pred, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
        pred_lcc = keep_largest_connected_component(pred) > 0
        dice_lcc, td_lcc, prec_lcc = cheap_metrics(pred_lcc, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
        rows[t] = {"dice": dice_raw, "dice_lcc": dice_lcc, "td": td_raw, "td_lcc": td_lcc, "prec_lcc": prec_lcc}
    return rows


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
            "prec_lcc": float(np.mean([v["prec_lcc"] for v in vals])),
        })
    return out


def select_operating_point(sweep, precision_floor):
    """Max TD+LCC subject to voxel-precision+LCC >= floor; else max precision."""
    eligible = [r for r in sweep if r["prec_lcc"] >= precision_floor]
    if eligible:
        best = max(eligible, key=lambda r: r["td_lcc"])
        reason = f"max TD+LCC s.t. voxel-precision+LCC >= {precision_floor:.2f}"
    else:
        best = max(sweep, key=lambda r: r["prec_lcc"])
        reason = f"no threshold met precision floor {precision_floor:.2f}; fell back to max voxel-precision+LCC"
    return float(best["threshold"]), reason


def print_sweep(sweep, title):
    print(f"\n--- {title} ---")
    print(f"{'thresh':>7}  {'Dice':>6}  {'Dice+LCC':>8}  {'TD':>6}  {'TD+LCC':>7}  {'Prec+LCC':>8}")
    for r in sweep:
        print(f"{r['threshold']:>7.2f}  {r['dice']:>6.3f}  {r['dice_lcc']:>8.3f}  "
              f"{r['td']:>6.3f}  {r['td_lcc']:>7.3f}  {r['prec_lcc']:>8.3f}")


def run_split(model, dataset_name, data_config, hu_window, cases, *, device, roi_size, overlap, sw_batch,
              chosen_threshold=None, compute_bd=False, collect_bins=False):
    """Inference over a list of cases.

    Always computes the cheap per-case sweep. If ``chosen_threshold`` is set, also
    builds the per-case table at that threshold: clDice + topology precision via a
    single prediction skeletonisation, and BD (branch parse) only when ``compute_bd``.
    If ``collect_bins`` is set, accumulates radius-stratified probabilities.
    """
    case_sweeps, table_rows = [], []
    bin_probs = {label: [] for label, _, _ in RADIUS_BINS}

    for cid in cases:
        ct, gt = load_case(dataset_name, data_config, cid, hu_window)
        gt_sum = int(gt.sum())
        td_slices, gt_skeleton, gt_skeleton_sum = gt_centerline(gt)
        prob = infer_probability(model, ct, device=device, roi_size=roi_size, overlap=overlap, sw_batch=sw_batch)

        case_sweeps.append(sweep_case(prob, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum))

        if chosen_threshold is not None:
            pred = prob >= chosen_threshold
            dice_raw, td_raw, _ = cheap_metrics(pred, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
            pred_lcc = keep_largest_connected_component(pred) > 0
            dice_lcc, td_lcc, prec_lcc = cheap_metrics(pred_lcc, gt, gt_sum, td_slices, gt_skeleton, gt_skeleton_sum)
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
            row = {
                "case_id": cid, "airway_voxels": gt_sum,
                "dice_lcc": dice_lcc, "td_lcc": td_lcc, "bd_lcc": bd_lcc,
                "cldice_lcc": cldice_lcc, "tprec_lcc": tprec_lcc, "prec_lcc": prec_lcc,
                "reference_branches": ref_b, "detected_branches": det_b,
                "dice_raw": dice_raw, "td_raw": td_raw,
            }
            table_rows.append(row)
            bd_str = f"BD {bd_lcc:.3f}  " if bd_lcc is not None else ""
            print(f"case {cid}: voxels {gt_sum:,}  | +LCC@{chosen_threshold:.2f}  "
                  f"Dice {dice_lcc:.3f}  TD {td_lcc:.3f}  {bd_str}clDice {cldice_lcc:.3f}  Prec {prec_lcc:.3f}")
        else:
            print(f"case {cid}: voxels {gt_sum:,}  (swept)")

        if collect_bins:
            radius = ndimage.distance_transform_edt(gt)
            for label, lo, hi in RADIUS_BINS:
                m = gt & (radius >= lo) & (radius < hi)
                if m.any():
                    bin_probs[label].append(prob[m])

    return case_sweeps, table_rows, bin_probs


def table_from_sweep(case_ids, case_sweeps, threshold):
    """Dev-mode table straight from the cheap sweep (Dice/TD/voxel-prec; no clDice/BD)."""
    rows = []
    for cid, cs in zip(case_ids, case_sweeps):
        r = cs[threshold]
        rows.append({
            "case_id": cid,
            "dice_lcc": r["dice_lcc"], "td_lcc": r["td_lcc"], "prec_lcc": r["prec_lcc"],
            "bd_lcc": None, "cldice_lcc": None, "tprec_lcc": None,
            "dice_raw": r["dice"], "td_raw": r["td"],
        })
    return rows


def table_mean(rows):
    keys = ["dice_lcc", "td_lcc", "bd_lcc", "cldice_lcc", "tprec_lcc", "prec_lcc", "dice_raw", "td_raw"]
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

    device = resolve_device(args.device)
    ckpt = run_dir / ("best_model.pt" if args.checkpoint == "best" else "last_model.pt")
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
    val_sweep = None
    sel_cases = sel_sweeps = sel_bins = None
    same_split = False
    if args.threshold is not None:
        chosen_threshold, selection_reason, selected_on = float(args.threshold), "fixed by --threshold", None
    elif args.select_split != "none":
        sel_cases = cap(split_case_ids(metadata, args.select_split))
        if not sel_cases:
            raise SystemExit(f"No {args.select_split} cases in run_metadata to select an operating point on.")
        same_split = (not args.cases) and (args.select_split == args.report_split)
        print(f"\nselecting operating point on {args.select_split} ({len(sel_cases)} cases)...")
        sel_sweeps, _, sel_bins = run_split(
            model, dataset_name, data_config, hu_window, sel_cases,
            collect_bins=same_split, **infer_kw,
        )
        val_sweep = mean_sweep(sel_sweeps)
        print_sweep(val_sweep, f"{args.select_split} sweep (mean over {len(sel_cases)} cases)")
        chosen_threshold, selection_reason = select_operating_point(val_sweep, args.precision_floor)
        selected_on = args.select_split
    else:
        chosen_threshold = float(cfg["training"]["validation"].get("threshold", 0.99))
        selection_reason, selected_on = "run validation threshold (no selection)", None
        print(f"\n[warn] no selection: using run validation threshold {chosen_threshold:.2f}")

    print(f"\n==> operating threshold {chosen_threshold:.2f}  ({selection_reason}"
          + (f", on {selected_on}" if selected_on else "") + ")")

    # --- 2. report the table at the chosen threshold -----------------------------
    reuse = same_split and chosen_threshold in SWEEP_THRESHOLDS
    if reuse:
        print(f"\nreporting on {report_label} ({len(report_cases)} cases) at {chosen_threshold:.2f}, +LCC "
              "— reusing val inference (dev mode; clDice/BD omitted, run --report-split test for them):")
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
        report_sweeps, table_rows, bin_probs = run_split(
            model, dataset_name, data_config, hu_window, report_cases,
            chosen_threshold=chosen_threshold, compute_bd=compute_bd, collect_bins=True, **infer_kw,
        )
    report_sweep = mean_sweep(report_sweeps)
    mean_row = table_mean(table_rows)

    def fmt(value):
        return f"{value:.3f}" if value is not None else "—"

    print(f"\n=== TABLE ({report_label}, +LCC @ {chosen_threshold:.2f}) ===")
    print(f"  MEAN  Dice {fmt(mean_row['dice_lcc'])}  TD {fmt(mean_row['td_lcc'])}  "
          f"BD {fmt(mean_row['bd_lcc'])}  clDice {fmt(mean_row['cldice_lcc'])}  "
          f"Prec {fmt(mean_row['prec_lcc'])}   (raw Dice {fmt(mean_row['dice_raw'])} / TD {fmt(mean_row['td_raw'])})")

    bins_out = build_bins_output(bin_probs, chosen_threshold)
    if bins_out:
        print(f"\n{'radius bin':>16}  {'voxels':>9}  {'%airway':>7}  {'meanP':>6}  {'medP':>6}  {'rec@op':>6}  {'rec@.5':>6}")
        for row in bins_out:
            print(f"{row['bin']:>16}  {row['voxels']:>9,}  {row['pct_airway']:>6.1f}%  "
                  f"{row['mean_prob']:>6.3f}  {row['median_prob']:>6.3f}  "
                  f"{100*row['recall_at_threshold']:>5.1f}%  {100*row['recall_at_0.5']:>5.1f}%")
    if not reuse:
        print_sweep(report_sweep, f"{report_label} sweep (mean over {len(report_cases)} cases)")
    print(
        "\nnote: TD is recall-like (ignores false positives) — read it at the fixed "
        f"operating threshold above, paired with Prec/BD. Op-point chosen on "
        f"{selected_on or 'a fixed threshold'}; "
        + ("dev mode (reported on the same split — mildly optimistic)." if reuse
           else "reported split is separate from selection." if selected_on and selected_on != report_label
           else "no separate selection split.")
    )

    out_path = run_dir / "distal_analysis.json"
    out_path.write_text(json.dumps({
        "run": metadata.get("experiment_name"),
        "dataset": dataset_name,
        "overlap": args.overlap,
        "checkpoint": args.checkpoint,
        "operating_point": {
            "threshold": chosen_threshold,
            "reason": selection_reason,
            "selected_on": selected_on,
            "precision_floor": args.precision_floor,
        },
        "report_split": report_label,
        "report_cases": report_cases,
        "dev_mode": bool(reuse),
        "bd_computed": any(r.get("bd_lcc") is not None for r in table_rows),
        "selection_sweep": val_sweep,
        "threshold_sweep": report_sweep,
        "table_per_case": table_rows,
        "table_mean": mean_row,
        "bins": bins_out,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
