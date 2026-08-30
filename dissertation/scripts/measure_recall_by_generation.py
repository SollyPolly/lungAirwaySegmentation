"""Model recall stratified by GROUND-TRUTH airway branch depth, and paired arm differences.

WHAT THIS ANSWERS. ``measure_recall_by_calibre.py`` established that the consistency term's
gain is concentrated in structures at most two voxels thick. "Thin" is a geometric
description, and the operational thickness classes are defined by the *loss function's own*
morphological operators, so that result is partly a statement about the instrument. This
script asks the same question with a topological definition that shares no machinery with
the thickness census:

    does the gain increase with branch depth, i.e. is it anatomically PERIPHERAL?

If a geometric and a topological stratification independently locate the effect in the same
place, the distal-recovery claim rests on two legs rather than one. If they disagree, the
honest reading is that the effect is geometric (thin branches wherever they occur) and not
anatomical, and the dissertation must say so.

WHAT "GENERATION" MEANS HERE, AND WHAT IT DOES NOT. Depth is BFS distance from the trachea
over the ATM'22 branch graph, which is the same parser the reported BD metric uses. After the
challenge's refinement the graph contains no single-child chains, so depth increments only at
a bifurcation and the index is a bifurcation order. It is NOT certified anatomical generation:

  * the root is the largest-volume branch, not an anatomically identified trachea;
  * a trifurcation advances depth by one, not two, so asymmetric branching desynchronises
    depth from anatomical generation as depth grows;
  * the tree is skeletonised from a binary mask, so depth inherits skeletonisation artefacts.

It is reported as BRANCH DEPTH throughout for that reason. It is the same surrogate, at the
same settings, that fixed the project's proximal/distal line at depth >= 3, and the first
three depths recover the expected anatomy closely (1 trachea, 2 main, 5-6 lobar).

TOPOLOGY HANDLING, all inherited from the ATM'22 parser rather than invented here:

  root                  largest-volume branch after nearest-branch voxel assignment
  >2 children           permitted; every child takes parent depth + 1
  short spurs           skeleton fragments under ``--minimum-branch-voxels`` (default 5) are
                        dropped before labelling; their voxels rejoin the nearest branch
  loops                 a branch reached twice at equal depth records both parents, and the
                        refinement pass then merges multi-parent branches, collapsing the loop
  single-child chains   absorbed into the parent, which is what makes depth a bifurcation order
  junction voxels       skeleton voxels with more than 3 neighbours in 3x3x3 are removed to
                        split branches, then reassigned by nearest-branch EDT, so a junction
                        several voxels across does not create a spurious branch
  disconnected GT       only the largest connected component is parsed; measured cost is a mean
                        0.34% of foreground over 105 scans (see ``branch_generation_labels``)
  unreachable branches  left at depth -1 and EXCLUDED, never silently binned as proximal
  carina, L/R main      not detected as such; they emerge as the depth 0 -> 1 split

LABELS COME FROM THE GROUND TRUTH ONLY. Every arm is scored against one identical depth
labelling per case. Deriving depth from each prediction would let a model that misses a
bifurcation renumber its own tree, which makes arms incomparable.

PRECISION IS DELIBERATELY NOT REPORTED BY DEPTH. A false positive lies on no ground-truth
branch, so it has no depth, and any assignment rule would be an invention rather than a
measurement. The calibre analysis can bucket precision by the prediction's own thickness
because thickness is intrinsic to the predicted object; depth is not.

Usage, from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\measure_recall_by_generation.py \\
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
import time

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from lung_airway_segmentation.metrics.topology import (  # noqa: E402
    _largest_connected_component,
    _skeletonize,
    parse_reference_skeleton_branches,
)

# Imported, not reimplemented: the depth definition must be the one that fixed the
# project's proximal/distal line, and the thickness definition must be the census's.
from generate_hu_imbalance_histogram import branch_generation_labels  # noqa: E402
from measure_airway_thickness import CLASS_GROUPS, _operational_class, _to_tensor  # noqa: E402

DEFAULT_GROUND_TRUTH_DIR = ROOT / "data" / "ATM22" / "labelsTr"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "skeleton_scale_probe" / "results_recall_by_generation"

# The cohort the calibre analysis used, so the two are directly comparable.
VAL20 = (
    "ATM_016", "ATM_027", "ATM_028", "ATM_033", "ATM_034", "ATM_043", "ATM_044",
    "ATM_046", "ATM_056", "ATM_068", "ATM_078", "ATM_081", "ATM_087", "ATM_116",
    "ATM_125", "ATM_126", "ATM_147", "ATM_150", "ATM_151", "ATM_152",
)

BRANCH_DETECTION_THRESHOLD = 0.8


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
        "--depth-groups-from",
        type=Path,
        default=None,
        help=(
            "Reuse depth_groups from an earlier generation_depth_analysis.json. "
            "A final '+' group is extended or shortened to the current cohort's maximum depth."
        ),
    )
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--prediction-lcc",
        action="store_true",
        help="Reduce each prediction to its largest component (default: RAW, the primary rule).",
    )
    parser.add_argument(
        "--minimum-branch-voxels",
        type=int,
        default=5,
        help="ATM'22 spur-pruning threshold on the split skeleton (default 5, the scorer's value).",
    )
    parser.add_argument(
        "--branch-detection-threshold",
        type=float,
        default=BRANCH_DETECTION_THRESHOLD,
        help="Skeleton coverage fraction at which a reference branch counts as detected.",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="recall_by_calibre.json to cross-check overall TD and calibre length shares against.",
    )
    parser.add_argument(
        "--skip-thickness",
        action="store_true",
        help="Skip the depth-vs-calibre joint census (drops the link to the calibre analysis).",
    )
    parser.add_argument(
        "--allow-skeleton-mismatch",
        action="store_true",
        help=(
            "Exploratory OOD mode: allow the branch-labelled skeleton to differ in size "
            "from the plain full-GT skeleton. Per-depth numerator and denominator still use "
            "the same branch-labelled support, and the discrepancy is recorded per case."
        ),
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


def _thickness_classes(truth: np.ndarray, device: torch.device) -> np.ndarray:
    """Operational thickness class per voxel, falling back to CPU if the card cannot hold it."""
    try:
        tensor = _to_tensor(truth, device)
        classes = _operational_class(tensor)[0, 0].cpu().numpy()
        del tensor
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        tensor = _to_tensor(truth, torch.device("cpu"))
        classes = _operational_class(tensor)[0, 0].numpy()
        del tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return classes


def _case_geometry(
    case_id: str,
    ground_truth_dir: Path,
    device: torch.device,
    minimum_branch_voxels: int,
    with_thickness: bool,
    strict: bool = True,
) -> dict[str, object]:
    """Parse one case's reference tree once: depth, branch identity, calibre, centreline.

    ``strict`` guards the scoring path and must stay True there: it asserts that the
    skeleton the branch parser labelled is the skeleton the TD denominator counts, which
    is what makes a per-depth numerator and denominator comparable. A reference-only
    census has no numerator to keep consistent, so it may pass ``strict=False`` to record
    the discrepancy in ``skeleton_mismatch`` and carry on rather than abort the cohort.
    """
    # ATM'22 source labels retain the channel suffix, whereas the staged AeroPath
    # labels follow nnU-Net's segmentation convention without ``_0000``. Accept both
    # explicit layouts; prediction filenames remain unchanged.
    candidates = (
        ground_truth_dir / f"{case_id}_0000.nii.gz",
        ground_truth_dir / f"{case_id}.nii.gz",
    )
    gt_path = next((path for path in candidates if path.is_file()), None)
    if gt_path is None:
        raise FileNotFoundError(
            f"No ground-truth mask for {case_id}; checked: "
            + ", ".join(str(path) for path in candidates)
        )
    image = nib.load(gt_path)
    full_shape = tuple(image.shape)
    spacing = tuple(float(z) for z in image.header.get_zooms()[:3])
    raw = np.asanyarray(image.dataobj) > 0
    raw_foreground = int(raw.sum())

    component = _largest_connected_component(raw)
    del raw

    # Two entry points into the SAME deterministic parse: one returns depth per voxel, the
    # other branch identity per skeleton voxel. They are cross-checked below rather than
    # assumed consistent.
    voxel_generation, branch_generation, crop = branch_generation_labels(
        component, minimum_branch_voxels=minimum_branch_voxels
    )
    branch_labels = parse_reference_skeleton_branches(
        component, minimum_branch_voxels=minimum_branch_voxels
    )
    truth = np.ascontiguousarray(component[crop])
    del component

    if branch_labels.shape != voxel_generation.shape:
        raise RuntimeError(
            f"{case_id}: branch parse and depth parse disagree on crop "
            f"{branch_labels.shape} vs {voxel_generation.shape}"
        )

    centreline = branch_labels > 0
    lookup = np.concatenate([[-1], branch_generation]).astype(np.int16)
    depth_via_branch = lookup[branch_labels[centreline]]
    depth_via_voxel = voxel_generation[centreline]
    agreement = float((depth_via_branch == depth_via_voxel).mean()) if centreline.any() else 1.0
    if agreement < 1.0:
        raise RuntimeError(
            f"{case_id}: branch depth and voxel depth disagree on {100 * (1 - agreement):.4f}% "
            "of centreline voxels; the two parses are not the same tree."
        )

    # The skeleton the branch parser used must be the skeleton the TD numerator uses.
    skeleton_mismatch = int(centreline.sum()) - int(_skeletonize(truth).sum())
    if skeleton_mismatch != 0 and strict:
        raise RuntimeError(
            f"{case_id}: branch-labelled skeleton differs from the plain skeleton by "
            f"{skeleton_mismatch:+d} voxels. For an explicitly exploratory branch-depth-only "
            "analysis, rerun with --allow-skeleton-mismatch; the discrepancy will be recorded."
        )

    classes = _thickness_classes(truth, device) if with_thickness else None

    return {
        "case_id": case_id,
        "full_shape": full_shape,
        "spacing": spacing,
        "crop": crop,
        "truth": truth,
        "centreline": centreline,
        "voxel_generation": voxel_generation,
        "branch_generation": branch_generation,
        "branch_labels": branch_labels,
        "classes": classes,
        "raw_foreground": raw_foreground,
        "parsed_foreground": int(truth.sum()),
        "branch_count": int(branch_generation.size),
        "skeleton_mismatch": skeleton_mismatch,
        # Centreline voxels the branch parse could not reach from the root. They carry no
        # depth, so every per-depth bin excludes them; a census must report how much
        # length that is rather than let it vanish out of the denominator.
        "unreached_centreline": int((voxel_generation[centreline] < 0).sum()),
        "unreached_branches": int((branch_generation < 0).sum()),
        "max_generation": int(branch_generation.max()) if branch_generation.size else -1,
    }


def _gt_rows(geometry: dict[str, object]) -> list[dict[str, object]]:
    """Arm-independent census: per depth, centreline length, volume, branch count, calibre."""
    voxel_generation = geometry["voxel_generation"]
    centreline = geometry["centreline"]
    truth = geometry["truth"]
    branch_generation = geometry["branch_generation"]
    classes = geometry["classes"]
    max_generation = geometry["max_generation"]

    total_line = float(centreline.sum())
    total_volume = float(truth.sum())
    rows: list[dict[str, object]] = []
    for generation in range(max_generation + 1):
        selected = voxel_generation == generation
        line = selected & centreline
        volume = selected & truth
        row: dict[str, object] = {
            "case_id": geometry["case_id"],
            "generation": generation,
            "gt_centreline_voxels": int(line.sum()),
            "gt_volume_voxels": int(volume.sum()),
            "gt_branch_count": int((branch_generation == generation).sum()),
            "gt_centreline_share": float(line.sum()) / max(total_line, 1.0),
            "gt_volume_share": float(volume.sum()) / max(total_volume, 1.0),
        }
        if classes is not None:
            in_generation_line = classes[line]
            row["median_class_centreline"] = (
                float(np.median(in_generation_line)) if in_generation_line.size else float("nan")
            )
            row["mean_class_centreline"] = (
                float(in_generation_line.mean()) if in_generation_line.size else float("nan")
            )
            for name, low, high in CLASS_GROUPS:
                band = (classes >= low) & (classes <= high)
                row[f"line__{name}"] = int((band & line).sum())
                row[f"volume__{name}"] = int((band & volume).sum())
        rows.append(row)
    return rows


def _arm_rows(
    geometry: dict[str, object],
    arm_name: str,
    prediction: np.ndarray,
    detection_threshold: float,
) -> list[dict[str, object]]:
    """Per depth: centreline recovery, volume recovery, and branch detection."""
    voxel_generation = geometry["voxel_generation"]
    centreline = geometry["centreline"]
    truth = geometry["truth"]
    branch_labels = geometry["branch_labels"]
    branch_generation = geometry["branch_generation"]
    max_generation = geometry["max_generation"]
    branch_count = geometry["branch_count"]

    # ATM'22 branch detection, vectorised: a reference branch is detected when the prediction
    # covers at least `detection_threshold` of its skeleton voxels.
    labelled = branch_labels > 0
    reference_counts = np.bincount(
        branch_labels[labelled], minlength=branch_count + 1
    )[1:].astype(np.float64)
    detected_counts = np.bincount(
        branch_labels[labelled & prediction], minlength=branch_count + 1
    )[1:].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.where(reference_counts > 0, detected_counts / reference_counts, 0.0)
    detected = coverage >= detection_threshold

    classes = geometry["classes"]
    rows: list[dict[str, object]] = []
    for generation in range(max_generation + 1):
        selected = voxel_generation == generation
        line = selected & centreline
        volume = selected & truth
        line_total = float(line.sum())
        volume_total = float(volume.sum())
        in_generation = branch_generation == generation
        branches = int(in_generation.sum())
        row: dict[str, object] = {
            "case_id": geometry["case_id"],
            "arm": arm_name,
            "generation": generation,
            "gt_centreline_voxels": int(line_total),
            "detected_centreline_voxels": int((line & prediction).sum()),
            "td": float((line & prediction).sum()) / line_total if line_total else float("nan"),
            "gt_volume_voxels": int(volume_total),
            "detected_volume_voxels": int((volume & prediction).sum()),
            "voxel_recall": (
                float((volume & prediction).sum()) / volume_total
                if volume_total
                else float("nan")
            ),
            "gt_branch_count": branches,
            "detected_branch_count": int((detected & in_generation).sum()),
            "bd": (
                float((detected & in_generation).sum()) / branches if branches else float("nan")
            ),
        }
        if classes is not None:
            # The falsification test the whole analysis turns on. Depth and calibre are
            # correlated, so a depth trend could be a calibre trend wearing a different
            # label. Storing detected/total centreline COUNTS jointly lets the trend be
            # re-examined at FIXED calibre afterwards, without re-reading any image.
            for name, low, high in CLASS_GROUPS:
                band_line = line & (classes >= low) & (classes <= high)
                row[f"gt_line__{name}"] = int(band_line.sum())
                row[f"det_line__{name}"] = int((band_line & prediction).sum())
        rows.append(row)
    return rows


def _aggregate_ratio(rows: list[dict], arm: str, generation: int, num: str, den: str) -> float:
    """Cohort ratio pooled over cases, which is the length-weighted reading of TD."""
    numerator = sum(
        int(r[num]) for r in rows if r["arm"] == arm and r["generation"] == generation
    )
    denominator = sum(
        int(r[den]) for r in rows if r["arm"] == arm and r["generation"] == generation
    )
    return float(numerator / denominator) if denominator else float("nan")


def _case_mean(rows: list[dict], arm: str, generation: int, key: str) -> float:
    """Per-case mean, which is the reading the paired tests operate on."""
    values = [
        r[key]
        for r in rows
        if r["arm"] == arm and r["generation"] == generation and np.isfinite(r[key])
    ]
    return float(np.mean(values)) if values else float("nan")


def _paired(
    rows: list[dict], baseline: str, arm: str, generation: int, key: str
) -> dict[str, float]:
    """Paired per-case difference for one depth, on the cases where both arms are defined."""
    by_case_baseline = {
        r["case_id"]: r[key]
        for r in rows
        if r["arm"] == baseline and r["generation"] == generation
    }
    by_case_arm = {
        r["case_id"]: r[key] for r in rows if r["arm"] == arm and r["generation"] == generation
    }
    deltas = [
        by_case_arm[case] - by_case_baseline[case]
        for case in sorted(set(by_case_baseline) & set(by_case_arm))
        if np.isfinite(by_case_baseline[case]) and np.isfinite(by_case_arm[case])
    ]
    if not deltas:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "sd": float("nan"),
                "iqr": float("nan"), "wins": 0, "losses": 0, "p_wilcoxon": float("nan")}
    array = np.asarray(deltas, dtype=np.float64)
    p_value = float("nan")
    if array.size >= 3 and np.any(array != 0):
        from scipy.stats import wilcoxon

        try:
            p_value = float(wilcoxon(array).pvalue)
        except ValueError:
            p_value = float("nan")
    quartiles = np.percentile(array, [25, 75])
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "iqr": float(quartiles[1] - quartiles[0]),
        "wins": int((array > 0).sum()),
        "losses": int((array < 0).sum()),
        "p_wilcoxon": p_value,
    }


def _choose_groups(
    gt_rows: list[dict],
    cases: list[str],
    minimum_share: float = 0.03,
    minimum_cases: int | None = None,
) -> list[tuple[str, int, int]]:
    """Merge depths upward until each group carries enough length and enough cases.

    Deliberately data-driven rather than hard-coded: late depths do not exist in every case,
    and a bin present in 12 of 20 cases cannot carry a paired claim. The default requires a
    group to be present in EVERY case, so every reported bin supports a complete pairing.
    """
    if minimum_cases is None:
        minimum_cases = len(cases)
    max_generation = max(r["generation"] for r in gt_rows)
    total_line = sum(r["gt_centreline_voxels"] for r in gt_rows)
    per_generation_line = {
        g: sum(r["gt_centreline_voxels"] for r in gt_rows if r["generation"] == g)
        for g in range(max_generation + 1)
    }
    per_generation_cases = {
        g: len({r["case_id"] for r in gt_rows if r["generation"] == g and r["gt_centreline_voxels"] > 0})
        for g in range(max_generation + 1)
    }

    groups: list[tuple[str, int, int]] = []
    start = 0
    for generation in range(max_generation + 1):
        span_line = sum(per_generation_line[g] for g in range(start, generation + 1))
        span_cases = min(per_generation_cases[g] for g in range(start, generation + 1))
        share = span_line / max(total_line, 1)
        last = generation == max_generation
        if (share >= minimum_share and span_cases >= minimum_cases) or last:
            if last and groups and (share < minimum_share or span_cases < minimum_cases):
                # Fold an inadequate tail into the previous group rather than report it alone.
                name, low, _ = groups.pop()
                groups.append((f"{low}+" if low < generation else f"{low}", low, generation))
            else:
                label = f"{start}" if start == generation else f"{start}-{generation}"
                if last:
                    label = f"{start}+" if start < generation else f"{start}"
                groups.append((label, start, generation))
            start = generation + 1
    return groups


def _fixed_groups_from(path: Path, max_generation: int) -> list[tuple[str, int, int]]:
    """Load a pre-specified grouping while letting the final ``8+``-style tail span the cohort."""
    payload = json.loads(path.read_text())
    raw_groups = payload.get("depth_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError(f"{path} has no non-empty depth_groups list")

    groups: list[tuple[str, int, int]] = []
    expected_low = 0
    for index, item in enumerate(raw_groups):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"Invalid depth group {item!r} in {path}")
        name, low, high = str(item[0]), int(item[1]), int(item[2])
        if low != expected_low:
            raise ValueError(
                f"Depth groups in {path} are not contiguous: expected {expected_low}, got {low}"
            )
        is_last = index == len(raw_groups) - 1
        if is_last and name.endswith("+"):
            if low > max_generation:
                raise ValueError(
                    f"Fixed tail {name} begins above this cohort's maximum depth {max_generation}"
                )
            high = max_generation
        elif high > max_generation:
            raise ValueError(
                f"Fixed group {name} ends at {high}, above this cohort's maximum depth {max_generation}"
            )
        if high < low:
            raise ValueError(f"Invalid depth group bounds for {name}: {low}>{high}")
        groups.append((name, low, high))
        expected_low = high + 1

    if expected_low <= max_generation:
        raise ValueError(
            f"Fixed depth groups from {path} stop at {expected_low - 1}, "
            f"below this cohort's maximum depth {max_generation}"
        )
    return groups


def main() -> None:
    args = _parse_args()
    arms = _parse_arms(args.arm)
    if args.baseline_arm and args.baseline_arm not in arms:
        raise SystemExit(f"--baseline-arm {args.baseline_arm!r} is not one of {list(arms)}")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with_thickness = not args.skip_thickness

    gt_rows: list[dict[str, object]] = []
    arm_rows: list[dict[str, object]] = []
    case_meta: list[dict[str, object]] = []

    for case_id in args.cases:
        started = time.time()
        geometry = _case_geometry(
            case_id,
            args.ground_truth_dir,
            device,
            args.minimum_branch_voxels,
            with_thickness,
            strict=not args.allow_skeleton_mismatch,
        )
        if geometry["skeleton_mismatch"]:
            branch_line = int(geometry["centreline"].sum())
            plain_line = branch_line - int(geometry["skeleton_mismatch"])
            lost_fraction = abs(int(geometry["skeleton_mismatch"])) / max(plain_line, 1)
            print(
                f"[{case_id}] WARNING branch-labelled vs plain skeleton: "
                f"{int(geometry['skeleton_mismatch']):+d} voxels "
                f"({100 * lost_fraction:.3f}% of plain skeleton)",
                flush=True,
            )
        gt_rows.extend(_gt_rows(geometry))
        case_meta.append(
            {
                key: geometry[key]
                for key in (
                    "case_id", "full_shape", "spacing", "raw_foreground", "parsed_foreground",
                    "branch_count", "skeleton_mismatch", "unreached_centreline",
                    "unreached_branches", "max_generation",
                )
            }
        )

        for arm_name, arm_dir in arms.items():
            prediction_path = arm_dir / f"{case_id}.nii.gz"
            if not prediction_path.exists():
                raise FileNotFoundError(f"Arm {arm_name!r} is missing {prediction_path.name}")
            prediction_full = _load_mask(prediction_path, geometry["full_shape"])
            if args.prediction_lcc:
                prediction_full = _largest_connected_component(prediction_full)
            prediction = np.ascontiguousarray(prediction_full[geometry["crop"]])
            del prediction_full
            arm_rows.extend(
                _arm_rows(geometry, arm_name, prediction, args.branch_detection_threshold)
            )
            del prediction

        overall = {
            arm: (
                sum(
                    r["detected_centreline_voxels"]
                    for r in arm_rows
                    if r["case_id"] == case_id and r["arm"] == arm
                )
                / max(
                    sum(
                        r["gt_centreline_voxels"]
                        for r in arm_rows
                        if r["case_id"] == case_id and r["arm"] == arm
                    ),
                    1,
                )
            )
            for arm in arms
        }
        print(
            f"[{case_id}] {geometry['branch_count']:>4} branches  depth<={geometry['max_generation']:>2}  "
            + "  ".join(f"{a} TD={v:.4f}" for a, v in overall.items())
            + f"  ({time.time() - started:.0f}s)",
            flush=True,
        )
        del geometry

    _report(args, arms, gt_rows, arm_rows, case_meta, with_thickness)


def _report(
    args: argparse.Namespace,
    arms: dict[str, Path],
    gt_rows: list[dict],
    arm_rows: list[dict],
    case_meta: list[dict],
    with_thickness: bool,
) -> None:
    arm_names = list(arms)
    max_generation = max(r["generation"] for r in gt_rows)
    groups = (
        _fixed_groups_from(args.depth_groups_from, max_generation)
        if args.depth_groups_from
        else _choose_groups(gt_rows, args.cases)
    )
    total_line = sum(r["gt_centreline_voxels"] for r in gt_rows)
    total_volume = sum(r["gt_volume_voxels"] for r in gt_rows)

    print("\n=== Reference tree census by branch depth (pooled over cases) ===")
    print(
        f"{'depth':>6} {'cases':>6} {'branches':>9} {'line vox':>10} {'line %':>8} "
        f"{'vol %':>7} {'med class':>10}"
    )
    for generation in range(max_generation + 1):
        subset = [r for r in gt_rows if r["generation"] == generation]
        present = sum(1 for r in subset if r["gt_centreline_voxels"] > 0)
        line = sum(r["gt_centreline_voxels"] for r in subset)
        volume = sum(r["gt_volume_voxels"] for r in subset)
        branches = sum(r["gt_branch_count"] for r in subset)
        medians = [r["median_class_centreline"] for r in subset if with_thickness
                   and np.isfinite(r.get("median_class_centreline", float("nan")))]
        median = f"{np.median(medians):>10.2f}" if medians else f"{'-':>10}"
        print(
            f"{generation:>6} {present:>6} {branches:>9} {line:>10} "
            f"{100 * line / max(total_line, 1):>8.2f} {100 * volume / max(total_volume, 1):>7.2f} {median}"
        )
    print(f"{'ALL':>6} {len(args.cases):>6} "
          f"{sum(r['gt_branch_count'] for r in gt_rows):>9} {total_line:>10} {100.0:>8.2f} {100.0:>7.2f}")

    grouping_rule = "Fixed" if args.depth_groups_from else "Data-driven"
    print(f"\n{grouping_rule} depth groups: {[g[0] for g in groups]}")

    def grouped(rows: list[dict], arm: str, low: int, high: int, num: str, den: str) -> float:
        numerator = sum(
            int(r[num]) for r in rows if r["arm"] == arm and low <= r["generation"] <= high
        )
        denominator = sum(
            int(r[den]) for r in rows if r["arm"] == arm and low <= r["generation"] <= high
        )
        return float(numerator / denominator) if denominator else float("nan")

    def grouped_per_case(rows: list[dict], arm: str, low: int, high: int, num: str, den: str) -> dict:
        per_case: dict[str, tuple[int, int]] = {}
        for row in rows:
            if row["arm"] != arm or not (low <= row["generation"] <= high):
                continue
            got, want = per_case.get(row["case_id"], (0, 0))
            per_case[row["case_id"]] = (got + int(row[num]), want + int(row[den]))
        return {
            case: numerator / denominator
            for case, (numerator, denominator) in per_case.items()
            if denominator > 0
        }

    for title, num, den in (
        ("Tree length detected (centreline recovery)", "detected_centreline_voxels", "gt_centreline_voxels"),
        ("Voxel recall (volume recovery)", "detected_volume_voxels", "gt_volume_voxels"),
        ("Branch detection", "detected_branch_count", "gt_branch_count"),
    ):
        print(f"\n=== {title}, by branch-depth group (pooled) ===")
        print(f"{'depth':>8} {'line %':>8} " + " ".join(f"{a:>12}" for a in arm_names))
        for label, low, high in groups:
            share = sum(
                r["gt_centreline_voxels"] for r in gt_rows if low <= r["generation"] <= high
            ) / max(total_line, 1)
            cells = " ".join(
                f"{grouped(arm_rows, a, low, high, num, den):>12.4f}" for a in arm_names
            )
            print(f"{label:>8} {100 * share:>8.2f} {cells}")
        cells = " ".join(
            f"{grouped(arm_rows, a, 0, max_generation, num, den):>12.4f}" for a in arm_names
        )
        print(f"{'ALL':>8} {100.0:>8.2f} {cells}")

    paired_tables: dict[str, list[dict]] = {}
    if args.baseline_arm:
        base = args.baseline_arm
        others = [a for a in arm_names if a != base]
        for key, label in (("td", "TD"), ("voxel_recall", "voxel recall"), ("bd", "BD")):
            print(f"\n=== Paired {label} difference vs '{base}', per depth (mean, wins/n, p) ===")
            print(f"{'depth':>8} " + " ".join(f"{a:>26}" for a in others))
            records: list[dict] = []
            for generation in range(max_generation + 1):
                cells = []
                for arm in others:
                    stats = _paired(arm_rows, base, arm, generation, key)
                    records.append({"metric": key, "arm": arm, "generation": generation, **stats})
                    cells.append(
                        f"{stats['mean']:+.4f} ({stats['wins']}/{stats['n']}) p={stats['p_wilcoxon']:.3g}"
                        if stats["n"]
                        else "n/a"
                    )
                print(f"{generation:>8} " + " ".join(f"{c:>26}" for c in cells))
            paired_tables[key] = records

        # Grouped paired differences, which is what the dissertation table will quote.
        num_den = {"td": ("detected_centreline_voxels", "gt_centreline_voxels"),
                   "voxel_recall": ("detected_volume_voxels", "gt_volume_voxels"),
                   "bd": ("detected_branch_count", "gt_branch_count")}
        grouped_paired: list[dict] = []
        for key in num_den:
            num, den = num_den[key]
            print(f"\n=== Paired {key} difference vs '{base}', by GROUP (per-case ratios) ===")
            print(f"{'depth':>8} " + " ".join(f"{a:>30}" for a in others))
            for label, low, high in groups:
                cells = []
                for arm in others:
                    baseline_map = grouped_per_case(arm_rows, base, low, high, num, den)
                    arm_map = grouped_per_case(arm_rows, arm, low, high, num, den)
                    shared = sorted(set(baseline_map) & set(arm_map))
                    deltas = np.asarray([arm_map[c] - baseline_map[c] for c in shared])
                    if deltas.size:
                        p_value = float("nan")
                        if deltas.size >= 3 and np.any(deltas != 0):
                            from scipy.stats import wilcoxon

                            try:
                                p_value = float(wilcoxon(deltas).pvalue)
                            except ValueError:
                                p_value = float("nan")
                        quartiles = np.percentile(deltas, [25, 75])
                        record = {
                            "metric": key, "arm": arm, "group": label, "low": low, "high": high,
                            "n": int(deltas.size), "mean": float(deltas.mean()),
                            "median": float(np.median(deltas)),
                            "sd": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
                            "iqr": float(quartiles[1] - quartiles[0]),
                            "wins": int((deltas > 0).sum()), "losses": int((deltas < 0).sum()),
                            "p_wilcoxon": p_value,
                        }
                        grouped_paired.append(record)
                        cells.append(
                            f"{deltas.mean():+.4f} ({int((deltas > 0).sum())}/{deltas.size}) p={p_value:.3g}"
                        )
                    else:
                        cells.append("n/a")
                print(f"{label:>8} " + " ".join(f"{c:>30}" for c in cells))
        paired_tables["grouped"] = grouped_paired

    joint = _joint_census(gt_rows, max_generation) if with_thickness else None
    if joint is not None:
        print("\n=== Depth against calibre: centreline length share (%) within each depth ===")
        band_names = [name for name, _, _ in CLASS_GROUPS]
        print(f"{'depth':>6} " + " ".join(f"{b:>8}" for b in band_names))
        for generation in range(max_generation + 1):
            row = joint["centreline_by_generation"][generation]
            total = sum(row) or 1
            print(f"{generation:>6} " + " ".join(f"{100 * v / total:>8.2f}" for v in row))
        print(
            f"\nSpearman(depth, thickness class) over centreline voxels: "
            f"rho={joint['spearman_rho']:.4f}  p={joint['spearman_p']:.3g}  n={joint['spearman_n']}"
        )
        print(
            f"Share of 1-2 voxel centreline at depth >= {joint['late_threshold']}: "
            f"{100 * joint['thin_in_late']:.2f}%"
        )
        print(
            f"Share of depth >= {joint['late_threshold']} centreline NOT in the 1-2 band: "
            f"{100 * joint['late_not_thin']:.2f}%"
        )

    fixed_calibre = (
        _fixed_calibre_test(arm_rows, args.baseline_arm, arm_names, max_generation)
        if with_thickness and args.baseline_arm
        else None
    )
    if fixed_calibre is not None:
        print(
            "\n=== FALSIFICATION TEST: does the depth trend survive at FIXED calibre? ===\n"
            "    Delta TD against the control, pooled over cases, within one thickness band.\n"
            "    If depth were only a proxy for thinness, every row would be flat."
        )
        for arm in [a for a in arm_names if a != args.baseline_arm]:
            print(f"\n  {arm}")
            print(f"    {'band':>6} " + " ".join(f"{f'd{d}':>8}" for d in range(max_generation + 1))
                  + f" {'rho':>7} {'p':>9}")
            for band in fixed_calibre[arm]:
                cells = " ".join(
                    f"{v:>+8.4f}" if np.isfinite(v) else f"{'-':>8}"
                    for v in band["delta_by_depth"]
                )
                rho = f"{band['spearman_rho']:>+7.3f}" if np.isfinite(band["spearman_rho"]) else f"{'-':>7}"
                p_value = f"{band['spearman_p']:>9.3g}" if np.isfinite(band["spearman_p"]) else f"{'-':>9}"
                print(f"    {band['band']:>6} {cells} {rho} {p_value}")

    verification = _verify(args, arm_rows, gt_rows) if args.verify_against else None
    if verification:
        print("\n=== Cross-check against the calibre analysis ===")
        for line in verification["lines"]:
            print(line)

    summary = {
        "script": "measure_recall_by_generation.py",
        "cases": args.cases,
        "arms": {k: str(v) for k, v in arms.items()},
        "baseline_arm": args.baseline_arm,
        "prediction_largest_component": args.prediction_lcc,
        "ground_truth_largest_component": True,
        "allow_skeleton_mismatch": args.allow_skeleton_mismatch,
        "minimum_branch_voxels": args.minimum_branch_voxels,
        "branch_detection_threshold": args.branch_detection_threshold,
        "depth_definition": (
            "BFS depth from the largest-volume branch over the ATM'22 refined branch graph; "
            "single-child chains absorbed, so depth increments only at a bifurcation"
        ),
        "depth_groups": [[n, a, b] for n, a, b in groups],
        "depth_groups_source": str(args.depth_groups_from) if args.depth_groups_from else None,
        "class_groups": [[n, a, b] for n, a, b in CLASS_GROUPS],
        "case_meta": case_meta,
        "per_case_generation_gt": gt_rows,
        "per_case_arm_generation": arm_rows,
        "paired": paired_tables,
        "joint_depth_calibre": joint,
        "fixed_calibre_depth_trend": fixed_calibre,
        "verification": verification,
    }
    (args.output_dir / "generation_depth_analysis.json").write_text(json.dumps(summary, indent=2))
    _write_csv(args.output_dir / "generation_depth_per_case.csv", arm_rows)
    _write_csv(args.output_dir / "generation_depth_gt_census.csv", gt_rows)
    _write_csv(
        args.output_dir / "generation_depth_per_generation.csv",
        _per_generation_csv(arm_rows, gt_rows, arm_names, max_generation, total_line),
    )
    print(f"\nWrote {args.output_dir / 'generation_depth_analysis.json'}")


def _fixed_calibre_test(
    arm_rows: list[dict], baseline: str, arm_names: list[str], max_generation: int
) -> dict[str, list[dict]]:
    """Hold calibre fixed and ask whether the depth trend in Delta TD survives.

    Depth and thickness are correlated, so the headline "deeper branches gain more" could be
    "thinner branches gain more" restated. Within a single thickness band that confound is
    removed by construction: any remaining trend with depth is a topological effect that
    geometry does not explain. Pooled over cases because a single band at a single depth in a
    single case is often only a few hundred voxels.
    """
    from scipy.stats import spearmanr

    results: dict[str, list[dict]] = {}
    for arm in arm_names:
        if arm == baseline:
            continue
        bands: list[dict] = []
        for name, _, _ in CLASS_GROUPS:
            deltas: list[float] = []
            supports: list[int] = []
            for generation in range(max_generation + 1):
                totals: dict[str, tuple[int, int]] = {}
                for which in (baseline, arm):
                    numerator = sum(
                        int(r[f"det_line__{name}"])
                        for r in arm_rows
                        if r["arm"] == which and r["generation"] == generation
                    )
                    denominator = sum(
                        int(r[f"gt_line__{name}"])
                        for r in arm_rows
                        if r["arm"] == which and r["generation"] == generation
                    )
                    totals[which] = (numerator, denominator)
                base_num, base_den = totals[baseline]
                arm_num, arm_den = totals[arm]
                if base_den and arm_den:
                    deltas.append(arm_num / arm_den - base_num / base_den)
                    supports.append(base_den)
                else:
                    deltas.append(float("nan"))
                    supports.append(0)

            # Rank correlation of the difference against depth, weighted by nothing: each
            # depth counts once, so a huge shallow bin cannot dominate a genuine trend.
            usable = [
                (generation, value)
                for generation, (value, support) in enumerate(zip(deltas, supports))
                if np.isfinite(value) and support >= 200
            ]
            if len(usable) >= 4:
                rho, p_value = spearmanr([u[0] for u in usable], [u[1] for u in usable])
            else:
                rho, p_value = float("nan"), float("nan")
            bands.append(
                {
                    "band": name,
                    "delta_by_depth": deltas,
                    "support_by_depth": supports,
                    "depths_used": [u[0] for u in usable],
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_value),
                }
            )
        results[arm] = bands
    return results


def _joint_census(gt_rows: list[dict], max_generation: int) -> dict:
    """Relate branch depth to the operational thickness class, on centreline voxels."""
    band_names = [name for name, _, _ in CLASS_GROUPS]
    centreline = [[0] * len(band_names) for _ in range(max_generation + 1)]
    volume = [[0] * len(band_names) for _ in range(max_generation + 1)]
    for row in gt_rows:
        generation = row["generation"]
        for index, name in enumerate(band_names):
            centreline[generation][index] += int(row.get(f"line__{name}", 0))
            volume[generation][index] += int(row.get(f"volume__{name}", 0))

    # Spearman on the pooled contingency table, expanded to per-voxel ranks by weight.
    depths: list[int] = []
    classes: list[float] = []
    weights: list[int] = []
    for generation in range(max_generation + 1):
        for index, (_, low, high) in enumerate(CLASS_GROUPS):
            count = centreline[generation][index]
            if count:
                depths.append(generation)
                classes.append((low + high) / 2.0)
                weights.append(count)
    from scipy.stats import spearmanr

    expanded_depth = np.repeat(np.asarray(depths), np.asarray(weights))
    expanded_class = np.repeat(np.asarray(classes), np.asarray(weights))
    rho, p_value = spearmanr(expanded_depth, expanded_class)

    late_threshold = 5
    thin_index = 0  # CLASS_GROUPS[0] is the 1-2 voxel band
    thin_total = sum(centreline[g][thin_index] for g in range(max_generation + 1))
    thin_late = sum(centreline[g][thin_index] for g in range(late_threshold, max_generation + 1))
    late_total = sum(sum(centreline[g]) for g in range(late_threshold, max_generation + 1))
    late_thin = thin_late
    return {
        "band_names": band_names,
        "centreline_by_generation": centreline,
        "volume_by_generation": volume,
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "spearman_n": int(expanded_depth.size),
        "late_threshold": late_threshold,
        "thin_in_late": float(thin_late / thin_total) if thin_total else float("nan"),
        "late_not_thin": float((late_total - late_thin) / late_total) if late_total else float("nan"),
    }


def _verify(args: argparse.Namespace, arm_rows: list[dict], gt_rows: list[dict]) -> dict:
    """Confirm this parse reproduces the calibre analysis's overall TD and calibre shares."""
    reference = json.loads(Path(args.verify_against).read_text())
    lines: list[str] = []
    checks: list[dict] = []
    by_case_arm: dict[tuple[str, str], tuple[int, int]] = {}
    for row in arm_rows:
        key = (row["case_id"], row["arm"])
        got, want = by_case_arm.get(key, (0, 0))
        by_case_arm[key] = (got + row["detected_centreline_voxels"], want + row["gt_centreline_voxels"])

    for arm in {r["arm"] for r in arm_rows}:
        mine = [
            numerator / denominator
            for (case, arm_name), (numerator, denominator) in by_case_arm.items()
            if arm_name == arm and denominator
        ]
        theirs = [
            r["td_all"]
            for r in reference["per_case_arm"]
            if r["arm"] == arm and np.isfinite(r["td_all"])
        ]
        if not mine or not theirs:
            continue
        difference = abs(float(np.mean(mine)) - float(np.mean(theirs)))
        checks.append({"arm": arm, "generation_td": float(np.mean(mine)),
                       "calibre_td": float(np.mean(theirs)), "abs_difference": difference})
        lines.append(
            f"  TD_all {arm:>12}: depth parse {np.mean(mine):.4f}  calibre parse {np.mean(theirs):.4f}"
            f"  |diff| {difference:.4f}"
        )

    total_line = sum(r["gt_centreline_voxels"] for r in gt_rows)
    first_arm = next(iter(reference["arms"]))
    for name, _, _ in CLASS_GROUPS:
        mine_share = sum(int(r.get(f"line__{name}", 0)) for r in gt_rows) / max(total_line, 1)
        theirs = [
            r[f"gt_length_share__{name}"]
            for r in reference["per_case_arm"]
            if r["arm"] == first_arm and np.isfinite(r[f"gt_length_share__{name}"])
        ]
        if theirs:
            lines.append(
                f"  length share {name:>6}: depth parse {100 * mine_share:>6.2f}%  "
                f"calibre parse {100 * np.mean(theirs):>6.2f}%"
            )
    return {"checks": checks, "lines": lines}


def _per_generation_csv(
    arm_rows: list[dict],
    gt_rows: list[dict],
    arm_names: list[str],
    max_generation: int,
    total_line: int,
) -> list[dict]:
    out: list[dict] = []
    for arm in arm_names:
        for generation in range(max_generation + 1):
            subset = [r for r in arm_rows if r["arm"] == arm and r["generation"] == generation]
            gt_subset = [r for r in gt_rows if r["generation"] == generation]
            line = sum(r["gt_centreline_voxels"] for r in subset)
            volume = sum(r["gt_volume_voxels"] for r in subset)
            branches = sum(r["gt_branch_count"] for r in subset)
            out.append(
                {
                    "arm": arm,
                    "generation": generation,
                    "cases_present": sum(1 for r in subset if r["gt_centreline_voxels"] > 0),
                    "gt_centreline_voxels": line,
                    "gt_centreline_share": line / max(total_line, 1),
                    "gt_volume_voxels": volume,
                    "gt_branch_count": branches,
                    "td_pooled": (
                        sum(r["detected_centreline_voxels"] for r in subset) / line if line else float("nan")
                    ),
                    "td_case_mean": _case_mean(arm_rows, arm, generation, "td"),
                    "voxel_recall_pooled": (
                        sum(r["detected_volume_voxels"] for r in subset) / volume if volume else float("nan")
                    ),
                    "voxel_recall_case_mean": _case_mean(arm_rows, arm, generation, "voxel_recall"),
                    "bd_pooled": (
                        sum(r["detected_branch_count"] for r in subset) / branches
                        if branches
                        else float("nan")
                    ),
                    "bd_case_mean": _case_mean(arm_rows, arm, generation, "bd"),
                    "median_class_centreline": float(
                        np.median(
                            [
                                r["median_class_centreline"]
                                for r in gt_subset
                                if np.isfinite(r.get("median_class_centreline", float("nan")))
                            ]
                            or [float("nan")]
                        )
                    ),
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
