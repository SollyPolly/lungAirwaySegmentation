"""Model recall stratified by GROUND-TRUTH airway calibre, and paired arm differences.

WHAT THIS ANSWERS. The thickness census
(``measure_airway_thickness.py``) established what the ANNOTATION contains: 15.7% of GT
centreline length sits in structures at most two voxels thick, where the soft-clDice
operator degenerates. It says nothing about whether a model recovers that band. This
script measures that, and -- more usefully -- whether the consistency term changes it.

WHY THIS IS NOT "the same analysis run on the predictions". Running the thickness census
on a prediction gives the thickness COMPOSITION of what the model drew, which does not
say whether it drew it in the right place. Recall is a JOINT quantity: thickness classes
come from the GT, and the prediction is asked how much of each GT class it covers.

    recall(n) = |prediction and GT_class_n| / |GT_class_n|

The centreline-weighted variant restricts to GT centreline voxels, which makes it a
tree-length-detected figure stratified by calibre and therefore directly readable
against the reported TD.

PRECISION IS A DIFFERENT MEASUREMENT and is optional here (``--with-precision``). A false
positive cannot be attributed to a GT class it does not lie in, so precision must be
bucketed by the PREDICTION's own thickness. That is the one place where the census really
is re-run on the prediction.

THE DECISION THIS INFORMS. If the paired MT-minus-control difference is ~0 inside the
degenerate classes, the consistency term has no demonstrated lever there and sharpening
the skeletoniser on that band is refining a mechanism that is not currently moving
anything. If it is clearly positive, the operator fix has somewhere to go. The
supervised-ceiling arm separates "consistency cannot" from "nobody can".

CONVENTIONS. GT is reduced to its largest connected component, matching the scorer's
target-component policy. Predictions are used RAW by default, matching the reporting rule
that RAW is primary and trachea-seeded LCC-6 is the declared sensitivity; ``--prediction-lcc``
switches to the largest component. Thickness classes are imported from
``measure_airway_thickness`` so both scripts cannot drift apart.

Usage, from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\measure_recall_by_calibre.py \\
        --arm control=data/nnunet/predict_out/Dataset126_val_mt240_control_final_teacher \\
        --arm mt_soft=data/nnunet/predict_out/Dataset126_val_mt240_softcldice_final_teacher \\
        --baseline-arm control
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from lung_airway_segmentation.metrics.topology import (
    _largest_connected_component,
    _skeletonize,
)

# Imported, not reimplemented: the thickness definition must be identical to the census.
from measure_airway_thickness import (
    BBOX_MARGIN,
    CLASS_GROUPS,
    _bounding_box,
    _operational_class,
    _to_tensor,
)

DEFAULT_GROUND_TRUTH_DIR = ROOT / "data" / "ATM22" / "labelsTr"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "skeleton_scale_probe" / "results_recall_by_calibre"
VAL20 = (
    "ATM_016", "ATM_027", "ATM_028", "ATM_033", "ATM_034", "ATM_043", "ATM_044",
    "ATM_046", "ATM_056", "ATM_068", "ATM_078", "ATM_081", "ATM_087", "ATM_116",
    "ATM_125", "ATM_126", "ATM_147", "ATM_150", "ATM_151", "ATM_152",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Arm label and its prediction directory. Repeatable.",
    )
    parser.add_argument("--baseline-arm", default=None, help="Arm to take paired differences against.")
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", nargs="*", default=list(VAL20))
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--prediction-lcc",
        action="store_true",
        help="Reduce each prediction to its largest component (default: RAW, the primary rule).",
    )
    parser.add_argument(
        "--with-precision",
        action="store_true",
        help="Also bucket by the PREDICTION's own thickness and report precision per class.",
    )
    return parser.parse_args()


def _parse_arms(raw: list[str]) -> dict[str, Path]:
    arms: dict[str, Path] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--arm expects NAME=DIR, got {item!r}")
        name, _, directory = item.partition("=")
        path = Path(directory)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            raise SystemExit(f"Prediction directory not found for arm {name!r}: {path}")
        arms[name] = path
    if not arms:
        raise SystemExit("At least one --arm NAME=DIR is required.")
    return arms


def _load_mask(path: Path, reference_shape: tuple[int, ...]) -> np.ndarray:
    image = nib.load(path)
    if tuple(image.shape) != tuple(reference_shape):
        raise ValueError(f"{path.name}: shape {image.shape} against ground truth {reference_shape}.")
    return np.asanyarray(image.dataobj) > 0


def main() -> None:
    args = _parse_args()
    arms = _parse_arms(args.arm)
    if args.baseline_arm and args.baseline_arm not in arms:
        raise SystemExit(f"--baseline-arm {args.baseline_arm!r} is not one of {list(arms)}")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for case_id in args.cases:
        gt_path = args.ground_truth_dir / f"{case_id}_0000.nii.gz"
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        gt_image = nib.load(gt_path)
        full_shape = tuple(gt_image.shape)
        truth_full = np.asanyarray(gt_image.dataobj) > 0
        truth_full = _largest_connected_component(truth_full)
        box = _bounding_box(truth_full, BBOX_MARGIN)
        truth = np.ascontiguousarray(truth_full[box])

        tensor = _to_tensor(truth, device)
        classes = _operational_class(tensor)[0, 0].cpu().numpy()
        del tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        centreline = _skeletonize(truth)

        for arm_name, arm_dir in arms.items():
            prediction_path = arm_dir / f"{case_id}.nii.gz"
            if not prediction_path.exists():
                raise FileNotFoundError(f"Arm {arm_name!r} is missing {prediction_path.name}")
            prediction_full = _load_mask(prediction_path, full_shape)
            if args.prediction_lcc:
                prediction_full = _largest_connected_component(prediction_full)
            prediction = np.ascontiguousarray(prediction_full[box])

            row: dict[str, object] = {"case_id": case_id, "arm": arm_name}
            covered_centreline = prediction & centreline
            row["td_all"] = float(covered_centreline.sum()) / max(float(centreline.sum()), 1.0)
            row["voxel_recall_all"] = float((prediction & truth).sum()) / max(float(truth.sum()), 1.0)

            for name, low, high in CLASS_GROUPS:
                in_class = (classes >= low) & (classes <= high)
                gt_class = in_class & truth
                gt_class_line = in_class & centreline
                count_v = float(gt_class.sum())
                count_l = float(gt_class_line.sum())
                row[f"voxel_recall__{name}"] = (
                    float((prediction & gt_class).sum()) / count_v if count_v else float("nan")
                )
                row[f"td__{name}"] = (
                    float((prediction & gt_class_line).sum()) / count_l if count_l else float("nan")
                )
                row[f"gt_length_share__{name}"] = count_l / max(float(centreline.sum()), 1.0)

            if args.with_precision:
                # Precision must be bucketed by the PREDICTION's thickness: a false positive
                # has no GT class. Computed on the full volume so distant false positives are
                # not silently excluded by the GT bounding box.
                pred_tensor = _to_tensor(prediction_full, device)
                pred_classes = _operational_class(pred_tensor)[0, 0].cpu().numpy()
                del pred_tensor
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for name, low, high in CLASS_GROUPS:
                    sel = (pred_classes >= low) & (pred_classes <= high) & prediction_full
                    count = float(sel.sum())
                    row[f"precision__{name}"] = (
                        float((sel & truth_full).sum()) / count if count else float("nan")
                    )
                    row[f"pred_volume_share__{name}"] = count / max(
                        float(prediction_full.sum()), 1.0
                    )
                del pred_classes

            rows.append(row)
            del prediction, prediction_full

        print(
            f"[{case_id}] "
            + "  ".join(
                f"{r['arm']} TD={r['td_all']:.4f}" for r in rows if r["case_id"] == case_id
            ),
            flush=True,
        )

    def mean_for(arm: str, key: str) -> float:
        values = [
            r[key]
            for r in rows
            if r["arm"] == arm and isinstance(r.get(key), float) and np.isfinite(r[key])
        ]
        return float(np.mean(values)) if values else float("nan")

    arm_names = list(arms)
    print("\n=== Tree-length detected, stratified by GT calibre (centreline-weighted) ===")
    header = f"{'thickness':>10} {'GT len %':>9} " + " ".join(f"{a:>12}" for a in arm_names)
    print(header)
    for name, _, _ in CLASS_GROUPS:
        share = 100 * mean_for(arm_names[0], f"gt_length_share__{name}")
        cells = " ".join(f"{mean_for(a, f'td__{name}'):>12.4f}" for a in arm_names)
        print(f"{name:>10} {share:>9.2f} {cells}")
    print(f"{'ALL':>10} {100.0:>9.2f} " + " ".join(f"{mean_for(a, 'td_all'):>12.4f}" for a in arm_names))

    print("\n=== Voxel recall, stratified by GT calibre ===")
    print(header)
    for name, _, _ in CLASS_GROUPS:
        share = 100 * mean_for(arm_names[0], f"gt_length_share__{name}")
        cells = " ".join(f"{mean_for(a, f'voxel_recall__{name}'):>12.4f}" for a in arm_names)
        print(f"{name:>10} {share:>9.2f} {cells}")

    if args.with_precision:
        print("\n=== Precision, bucketed by the PREDICTION's own calibre ===")
        print(header)
        for name, _, _ in CLASS_GROUPS:
            share = 100 * mean_for(arm_names[0], f"pred_volume_share__{name}")
            cells = " ".join(f"{mean_for(a, f'precision__{name}'):>12.4f}" for a in arm_names)
            print(f"{name:>10} {share:>9.2f} {cells}")

    if args.baseline_arm:
        base = args.baseline_arm
        others = [a for a in arm_names if a != base]
        print(f"\n=== Paired TD difference vs '{base}' (per class, n_wins/n_cases) ===")
        print(f"{'thickness':>10} " + " ".join(f"{a:>20}" for a in others))
        for name, _, _ in CLASS_GROUPS:
            cells = []
            for arm in others:
                deltas = []
                for case in args.cases:
                    b = next(
                        (r for r in rows if r["case_id"] == case and r["arm"] == base), None
                    )
                    o = next(
                        (r for r in rows if r["case_id"] == case and r["arm"] == arm), None
                    )
                    if b is None or o is None:
                        continue
                    bv, ov = b.get(f"td__{name}"), o.get(f"td__{name}")
                    if (
                        isinstance(bv, float)
                        and isinstance(ov, float)
                        and np.isfinite(bv)
                        and np.isfinite(ov)
                    ):
                        deltas.append(ov - bv)
                if deltas:
                    wins = sum(1 for d in deltas if d > 0)
                    cells.append(f"{np.mean(deltas):+.4f} ({wins}/{len(deltas)})")
                else:
                    cells.append("n/a")
            print(f"{name:>10} " + " ".join(f"{c:>20}" for c in cells))

    summary = {
        "script": "measure_recall_by_calibre.py",
        "cases": args.cases,
        "arms": {k: str(v) for k, v in arms.items()},
        "baseline_arm": args.baseline_arm,
        "prediction_largest_component": args.prediction_lcc,
        "ground_truth_largest_component": True,
        "class_groups": [[n, a, b] for n, a, b in CLASS_GROUPS],
        "per_case_arm": rows,
    }
    (args.output_dir / "recall_by_calibre.json").write_text(json.dumps(summary, indent=2))
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (args.output_dir / "recall_by_calibre.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output_dir / 'recall_by_calibre.json'}")


if __name__ == "__main__":
    main()
