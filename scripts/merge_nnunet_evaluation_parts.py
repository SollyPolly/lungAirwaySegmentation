"""Merge disjoint ``evaluate_nnunet_predictions`` JSON reports.

This is useful when full-resolution topology scoring is split into shorter case
batches. Per-case rows are concatenated, macro table means are recomputed, and
radius-bin recall is pooled by voxel count. Duplicate cases are rejected.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np


_MEAN_KEYS = (
    "dice_raw", "td_raw", "prec_raw",
    "dice_lcc", "td_lcc", "prec_lcc", "tprec_lcc", "cldice_lcc",
    "lcc_retained_fraction", "bd_lcc", "cldice_atm", "td_atm",
)


def _macro_means(rows: list[dict]) -> dict:
    means = {}
    for key in _MEAN_KEYS:
        values = [row[key] for row in rows if row.get(key) is not None]
        means[key] = float(np.mean(values)) if values else None
    return means


def _merge_bins(reports: list[dict]) -> list[dict]:
    labels = []
    for report in reports:
        for row in report.get("bins", []):
            if row["bin"] not in labels:
                labels.append(row["bin"])
    total_voxels = sum(int(row["voxels"]) for report in reports for row in report.get("bins", []))
    merged = []
    for label in labels:
        matching = [
            row for report in reports for row in report.get("bins", []) if row["bin"] == label
        ]
        voxels = sum(int(row["voxels"]) for row in matching)
        hits = sum(float(row["recall"]) * int(row["voxels"]) for row in matching)
        recall = hits / voxels
        merged.append({
            "bin": label,
            "voxels": voxels,
            "pct_airway": 100.0 * voxels / total_voxels,
            "recall": recall,
            "recall_at_0.5": recall,
        })
    return merged


def merge_reports(part_paths: list[Path], roi_manifest: Path | None = None) -> dict:
    reports = [json.loads(path.read_text()) for path in part_paths]
    if not reports:
        raise ValueError("At least one report is required.")

    identity_keys = (
        "topology_metric_version", "scorer", "dataset", "report_split", "branch_metrics",
        "operating_point", "postprocessing",
    )
    expected = {key: reports[0].get(key) for key in identity_keys}
    expected_pred_dir = Path(reports[0]["pred_dir"]).resolve()
    for path, report in zip(part_paths[1:], reports[1:]):
        mismatched = [key for key in identity_keys if report.get(key) != expected[key]]
        if Path(report["pred_dir"]).resolve() != expected_pred_dir:
            mismatched.append("pred_dir")
        if mismatched:
            raise ValueError(f"Incompatible report {path}; mismatched: {', '.join(mismatched)}")

    rows = [deepcopy(row) for report in reports for row in report["table_per_case"]]
    for row in rows:
        row["case_id"] = str(row["case_id"]).zfill(3)
    case_ids = [row["case_id"] for row in rows]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate cases across reports: {', '.join(duplicates)}")
    order = np.argsort(np.asarray(case_ids))
    rows = [rows[int(index)] for index in order]
    case_ids = [row["case_id"] for row in rows]

    merged = deepcopy(reports[0])
    merged["report_cases"] = case_ids
    merged["table_per_case"] = rows
    merged["table_mean"] = _macro_means(rows)
    merged["bins"] = _merge_bins(reports)
    merged["merged_from"] = [str(path) for path in part_paths]
    if roi_manifest is not None:
        manifest = json.loads(roi_manifest.read_text())
        manifest_output_dir = Path(manifest["output_dir"]).resolve()
        if manifest_output_dir != expected_pred_dir:
            raise ValueError(
                f"ROI manifest output_dir {manifest_output_dir} does not match report pred_dir "
                f"{expected_pred_dir}."
            )
        manifest_cases = {str(row["case_id"]).zfill(3) for row in manifest["cases"]}
        if manifest_cases != set(case_ids):
            missing = sorted(set(case_ids) - manifest_cases)
            extra = sorted(manifest_cases - set(case_ids))
            raise ValueError(
                f"ROI manifest/report case mismatch; missing={missing}, extra={extra}."
            )
        merged.setdefault("postprocessing", {})["roi"] = {
            "operation": manifest["operation"],
            "parameters": manifest["parameters"],
            "manifest": str(roi_manifest),
            "uses_ground_truth": False,
        }
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roi-manifest", type=Path, default=None)
    args = parser.parse_args()

    report = merge_reports(args.parts, args.roi_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Merged {len(args.parts)} reports / {len(report['report_cases'])} cases -> {args.out}")


if __name__ == "__main__":
    main()
