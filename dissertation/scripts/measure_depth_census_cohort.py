"""Reference-only branch-depth census over an arbitrary cohort.

``measure_recall_by_generation.py`` computes the same per-case census, but only as a
by-product of scoring predictions: it requires at least one ``--arm`` and walks every
prediction directory. This script runs the arm-independent half alone, so a cohort with
no predictions -- the 260 training cases, say -- can still be described.

It reuses ``_case_geometry`` and ``_gt_rows`` from that module rather than
reimplementing them, so the numbers here and the numbers behind the Results come from one
code path and cannot drift.

Two differences from the scoring path, both deliberate:

* ``strict=False``. The scoring path asserts that the branch-labelled skeleton is exactly
  the plain skeleton, because its per-depth numerator and denominator have to be
  comparable. A census has no numerator. Cases that fail that assertion are recorded in
  ``skeleton_mismatch`` and counted in the summary rather than aborting the cohort, which
  is what the first 260-case attempt did at case 6 of 260 (ATM_019).
* No thickness. The depth-vs-calibre join is the only GPU consumer in ``_case_geometry``
  and is not needed for a census, so this runs CPU-only and parallelises cleanly.

The work is CPU-bound and single-threaded per case at roughly 32 s and 2.1 GB resident,
so ``--workers`` is a straight division of wall time, bounded by RAM rather than by cores.

Output carries the ``cases`` and ``per_case_generation_gt`` keys the plotting script
already reads.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\measure_depth_census_cohort.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_SPLIT = ROOT / "configs" / "nnunet" / "atm22_split_l20_u240.yaml"
DEFAULT_GROUND_TRUTH = ROOT / "data" / "ATM22" / "labelsTr"
DEFAULT_OUTPUT = (
    ROOT / "data" / "skeleton_scale_probe" / "results_recall_by_generation"
    / "generation_depth_census_train260.json"
)
META_KEYS = (
    "case_id", "full_shape", "spacing", "raw_foreground", "parsed_foreground",
    "branch_count", "unreached_branches", "unreached_centreline", "skeleton_mismatch",
    "max_generation",
)


def _one_case(case_id: str, ground_truth_dir: str, minimum_branch_voxels: int):
    """Parse one case in a worker process and return only picklable summaries.

    The geometry dict holds several full-volume arrays, so nothing but the census rows
    and the per-case metadata crosses the process boundary.
    """
    import torch

    from measure_recall_by_generation import _case_geometry, _gt_rows

    geometry = _case_geometry(
        case_id, Path(ground_truth_dir), torch.device("cpu"), minimum_branch_voxels,
        with_thickness=False, strict=False,
    )
    rows = _gt_rows(geometry)
    meta = {key: geometry[key] for key in META_KEYS}
    return rows, meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--roles",
        nargs="*",
        default=["labelled_train", "unlabelled_train"],
        help="Split roles to pool into the cohort.",
    )
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Explicit case ids, overriding --split/--roles.")
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N cases (for timing runs).")
    parser.add_argument("--minimum-branch-voxels", type=int, default=5)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel worker processes. RAM-bound at roughly 2.1 GB each, not core-bound.",
    )
    return parser.parse_args()


def _cohort(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return list(args.cases)
    split = yaml.safe_load(args.split.read_text())
    ids: list[str] = []
    for role in args.roles:
        if role not in split["splits"]:
            raise SystemExit(f"Role {role!r} not in {args.split}: {list(split['splits'])}")
        ids.extend(split["splits"][role])
    return [f"ATM_{case}" for case in ids]


def main() -> None:
    args = _parse_args()
    cases = _cohort(args)
    if args.limit is not None:
        cases = cases[: args.limit]

    present, missing = [], []
    for case_id in cases:
        target = present if (args.ground_truth_dir / f"{case_id}_0000.nii.gz").exists() else missing
        target.append(case_id)
    print(f"Cohort: {len(present)} cases ({len(missing)} missing), roles={args.roles}, "
          f"workers={args.workers}", flush=True)

    gt_rows: list[dict] = []
    meta: list[dict] = []
    failures: list[tuple[str, str]] = []
    started_all = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_one_case, case_id, str(args.ground_truth_dir),
                        args.minimum_branch_voxels): case_id
            for case_id in present
        }
        for index, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            try:
                rows, case_meta = future.result()
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                failures.append((case_id, f"{type(error).__name__}: {error}"))
                print(f"[{index}/{len(present)}] {case_id}  FAILED  {error}", flush=True)
                continue
            gt_rows.extend(rows)
            meta.append(case_meta)
            rate = (time.time() - started_all) / index
            flag = ""
            if case_meta["skeleton_mismatch"]:
                flag += f"  skel{case_meta['skeleton_mismatch']:+d}"
            if case_meta["unreached_branches"]:
                flag += f"  unreached={case_meta['unreached_branches']}"
            print(f"[{index}/{len(present)}] {case_id}  {case_meta['branch_count']:>4} branches  "
                  f"depth<={case_meta['max_generation']:>2}{flag}  "
                  f"(eta {(len(present) - index) * rate / 60:.0f} min)", flush=True)

    meta.sort(key=lambda row: row["case_id"])
    total_line = sum(row["gt_centreline_voxels"] for row in gt_rows)
    unreached_line = sum(row["unreached_centreline"] for row in meta)
    payload = {
        "script": Path(__file__).name,
        "cases": [row["case_id"] for row in meta],
        "roles": args.roles,
        "split": str(args.split),
        "missing_cases": missing,
        "failed_cases": failures,
        "minimum_branch_voxels": args.minimum_branch_voxels,
        "with_thickness": False,
        "strict_skeleton_check": False,
        "cases_with_skeleton_mismatch": [
            row["case_id"] for row in meta if row["skeleton_mismatch"]
        ],
        "cases_with_unreached_branches": [
            row["case_id"] for row in meta if row["unreached_branches"]
        ],
        "unreached_centreline_voxels": unreached_line,
        "per_case_generation_gt": gt_rows,
        "per_case_meta": meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))

    depths = [row["generation"] for row in gt_rows]
    print(f"\nWrote {args.output}")
    print(f"  {len(meta)} cases, depths 0-{max(depths) if depths else 0}, "
          f"{(time.time() - started_all) / 60:.1f} min")
    print(f"  {len(payload['cases_with_skeleton_mismatch'])} cases fail the strict skeleton "
          f"check, {len(payload['cases_with_unreached_branches'])} have branches unreachable "
          f"from the root")
    print(f"  {unreached_line} of {total_line + unreached_line} centreline voxels carry no "
          f"depth ({100 * unreached_line / max(total_line + unreached_line, 1):.2f}%)")
    if missing:
        print(f"  MISSING labels: {missing}")
    if failures:
        print(f"  FAILED: {failures}")


if __name__ == "__main__":
    main()
