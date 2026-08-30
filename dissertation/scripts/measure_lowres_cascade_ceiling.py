"""Measure the ceiling on what nnU-Net's 3d_cascade_fullres prior could contribute.

The cascade trains ``3d_lowres`` first, then feeds its prediction, resampled up to the
full-resolution grid, to ``3d_cascade_fullres`` as an extra input channel.  The prior
is therefore bounded by what the low-resolution grid can represent at all.

For Dataset126 the planned ``3d_lowres`` spacing is 2.31 x 1.41 x 2.31 mm against
0.82 x 0.50 x 0.82 mm at full resolution: about 22x the voxel volume.  A distal
bronchiole with a 1-1.6 mm lumen is sub-voxel there.

This script puts a number on that by round-tripping the ground truth itself through
the low-resolution grid using nnU-Net's own segmentation resampler, then scoring the
result against the original ground truth with the project's ATM'22 scorer.  Because
the input is the ground truth, the result is an upper bound: no ``3d_lowres`` model
can supply a better prior than a perfect one, and a real one will be worse.

Usage, from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\measure_lowres_cascade_ceiling.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lung_airway_segmentation.metrics.topology import (
    _skeletonize,
    airway_topology_metrics_from_masks,
)
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape

PROBE_ROOT = ROOT / "data" / "skeleton_scale_probe"
DEFAULT_PLANS = PROBE_ROOT / "teacher_probabilities" / "plans.json"
DEFAULT_GROUND_TRUTH_DIR = ROOT / "data" / "ATM22" / "labelsTr"
DEFAULT_OUTPUT_DIR = PROBE_ROOT / "results"
DEFAULT_CASES = ("ATM_034", "ATM_044", "ATM_046", "ATM_125")

# Same buckets and thickness convention as measure_soft_skeleton_scale.py.
RADIUS_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("thick<=2", 1.0, 1.5),
    ("thick3-4", 1.5, 2.5),
    ("thick5-6", 2.5, 3.5),
    ("thick7-10", 3.5, 5.5),
    ("thick11-16", 5.5, 8.5),
    ("thick>16", 8.5, float("inf")),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    return parser.parse_args()


def _spacing_in_nibabel_order(plan_spacing: list[float], transpose_forward: list[int]) -> tuple[float, float, float]:
    """Map a plan spacing onto nibabel (x, y, z) axis order.

    nnU-Net's array order reverses nibabel's, then ``transpose_forward`` permutes it,
    so plan axis ``j`` corresponds to nibabel axis ``2 - transpose_forward[j]``.
    """
    spacing = [0.0, 0.0, 0.0]
    for plan_axis, value in enumerate(plan_spacing):
        spacing[2 - transpose_forward[plan_axis]] = float(value)
    return tuple(spacing)  # type: ignore[return-value]


def _centreline_radius(ground_truth: np.ndarray) -> np.ndarray:
    """Calibre of the GT structure each voxel belongs to, in full-resolution index units."""
    if not ground_truth.any():
        return np.zeros(ground_truth.shape, dtype=np.float32)
    distance = ndimage.distance_transform_edt(ground_truth)
    centreline = _skeletonize(ground_truth)
    if not centreline.any():
        return np.zeros(ground_truth.shape, dtype=np.float32)
    nearest = tuple(
        ndimage.distance_transform_edt(~centreline, return_distances=False, return_indices=True)
    )
    return np.asarray(np.where(ground_truth, distance[nearest], 0.0), dtype=np.float32)


def _round_trip_through_lowres(
    ground_truth: np.ndarray,
    fullres_spacing: tuple[float, float, float],
    lowres_spacing: tuple[float, float, float],
) -> np.ndarray:
    """Resample GT down to the 3d_lowres grid and back, as the cascade does its prior."""
    original_shape = np.array(ground_truth.shape)
    scale = np.array(fullres_spacing) / np.array(lowres_spacing)
    lowres_shape = np.maximum(np.round(original_shape * scale).astype(int), 1)

    # The plans use resampling_fn_seg with is_seg=True, order=1, order_z=0.
    lowres = resample_data_or_seg_to_shape(
        ground_truth[None].astype(np.float32),
        lowres_shape.tolist(),
        list(fullres_spacing),
        list(lowres_spacing),
        is_seg=True,
        order=1,
        order_z=0,
        force_separate_z=None,
    )
    restored = resample_data_or_seg_to_shape(
        lowres,
        original_shape.tolist(),
        list(lowres_spacing),
        list(fullres_spacing),
        is_seg=True,
        order=1,
        order_z=0,
        force_separate_z=None,
    )
    return np.asarray(restored[0] > 0.5)


def main() -> None:
    args = _parse_args()
    plans = json.loads(args.plans.read_text())
    transpose_forward = plans["transpose_forward"]
    configurations = plans["configurations"]
    if "3d_lowres" not in configurations:
        raise SystemExit("These plans contain no 3d_lowres stage, so no cascade is configured.")

    fullres_spacing = _spacing_in_nibabel_order(configurations["3d_fullres"]["spacing"], transpose_forward)
    lowres_spacing = _spacing_in_nibabel_order(configurations["3d_lowres"]["spacing"], transpose_forward)
    volume_ratio = float(np.prod(lowres_spacing) / np.prod(fullres_spacing))
    print(
        f"3d_fullres spacing {tuple(round(v, 4) for v in fullres_spacing)} mm, "
        f"3d_lowres spacing {tuple(round(v, 4) for v in lowres_spacing)} mm "
        f"({volume_ratio:.1f}x the voxel volume)"
    )

    rows: list[dict[str, object]] = []
    for case_id in args.cases:
        path = args.ground_truth_dir / f"{case_id}_0000.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing ground truth: {path}")
        image = nib.load(path)
        # The plan spacing is the dataset median; nnU-Net resamples each case from its
        # own spacing to the target, so the round trip starts from the case's grid.
        zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
        ground_truth = np.asanyarray(image.dataobj) > 0

        restored = _round_trip_through_lowres(ground_truth, zooms, lowres_spacing)
        metrics = airway_topology_metrics_from_masks(restored, ground_truth)

        radius = _centreline_radius(ground_truth)
        row: dict[str, object] = {
            "case_id": case_id,
            "case_spacing": list(zooms),
            "gt_voxels": int(ground_truth.sum()),
            "restored_voxels": int(restored.sum()),
            "voxel_recall": float((restored & ground_truth).sum() / max(int(ground_truth.sum()), 1)),
        }
        row.update({key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))})
        for name, low, high in RADIUS_BUCKETS:
            bucket = ground_truth & (radius >= low) & (radius < high)
            count = int(bucket.sum())
            row[f"gt_share__{name}"] = count / max(int(ground_truth.sum()), 1)
            row[f"recall__{name}"] = float((restored & bucket).sum() / count) if count else float("nan")
        rows.append(row)
        print(
            f"[{case_id}] perfect-lowres prior ceiling: "
            f"TD {row.get('tree_length_detected', float('nan')):.4f} "
            f"BD {row.get('branch_detected', float('nan')):.4f} "
            f"clDice {row.get('cldice', float('nan')):.4f} "
            f"voxel recall {row['voxel_recall']:.4f}"
        )

    print("\n=== Recall of a perfect 3d_lowres prior, by GT calibre ===")
    header = f"{'bucket':>11} {'GT share':>9} {'recall':>8}"
    print(header)
    for name, _, _ in RADIUS_BUCKETS:
        shares = [row[f"gt_share__{name}"] for row in rows]
        recalls = [row[f"recall__{name}"] for row in rows if np.isfinite(row[f"recall__{name}"])]  # type: ignore[arg-type]
        share_mean = float(np.mean(shares)) if shares else float("nan")  # type: ignore[arg-type]
        recall_mean = float(np.mean(recalls)) if recalls else float("nan")
        print(f"{name:>11} {share_mean:>9.4f} {recall_mean:>8.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "script": "measure_lowres_cascade_ceiling.py",
        "plans": str(args.plans),
        "fullres_spacing_nibabel_order": list(fullres_spacing),
        "lowres_spacing_nibabel_order": list(lowres_spacing),
        "lowres_voxel_volume_ratio": volume_ratio,
        "note": "Ground truth round-tripped through the 3d_lowres grid: an upper bound on the cascade prior.",
        "per_case": rows,
    }
    output = args.output_dir / "lowres_cascade_ceiling.json"
    output.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
