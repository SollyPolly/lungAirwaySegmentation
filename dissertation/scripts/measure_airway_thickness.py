"""Proportion of the GT airway tree by local thickness, without the wall-shell artefact.

THE ARTEFACT THIS AVOIDS. ``distance_transform_edt(GT)`` gives every voxel its distance
to the nearest background voxel, so the entire OUTER SHELL of the trachea scores 1. Any
histogram built on it reports the large airways' surface as "one voxel thick" and is
meaningless. This is the same fault that retired the legacy ``distal r=1`` numbers.

The fix is to measure thickness by what a voxel FITS INSIDE, not by its own distance to
background: the classical local-thickness / granulometry definition,

    tau(x) = size of the largest structuring element S with  x in (GT opened by S).

A trachea wall voxel sits inside the large element that fills the trachea, so it inherits
the trachea's thickness. A one-voxel bronchiole has nothing bigger to sit in.

TWO MEASURES, because there are two questions.

A. OPERATIONAL thickness, in index units, using the SAME operators as ``_soft_skeleton3d``
   -- a 7-voxel cross erosion and a 27-voxel cube dilation, imported from the pinned
   trainer. Class n is the largest n with ``x in dilate^n(erode^n(GT))``.

   Class 0 is exactly the clDice-degenerate set: ``x not in dilate(erode(GT))`` means
   ``open(x) == 0``, hence ``relu(x - open(x)) == x`` and the "skeleton" IS the object.
   So the degeneracy census is not a separate proxy measurement -- it is bucket 0 of this
   histogram, computed with the operator the loss actually applies.

   A slab of thickness t survives floor((t-1)/2) cross erosions, so class n corresponds to
   thickness {2n+1, 2n+2}: the {1-2}, {3-4}, {5-6} pairing falls out of the operator rather
   than being imposed. The synthetic calibration table verifies this on ideal tubes.

   Index units are the correct units here: the cross erosion is three voxels wide whatever
   the spacing, so degeneracy is a property of the grid, not of millimetres.

B. ANATOMICAL thickness, in millimetres, by Euclidean local thickness with
   ``sampling=zooms``. This is the number to quote for anatomy, where the
   0.82 x 0.82 x 0.5 mm grid makes "voxels" ambiguous. Slower; skip with
   ``--no-mm-thickness``.

WEIGHTING. Volume share and centreline-length share are both reported and they tell
OPPOSITE stories: branch depth <= 2 is about 3.4% of branches but 59.7% of voxels. Quoting
volume alone would understate how much of the TREE is thin. Length is counted as centreline
voxels, which treats a diagonal step as one -- a mild underestimate of true arc length,
applied equally to every bucket.

Usage, from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\measure_airway_thickness.py
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
from scipy import ndimage
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lung_airway_segmentation.metrics.topology import (
    _largest_connected_component,
    _skeletonize,
)
from nnunet_trainers.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (
    _soft_erode3d,
    _soft_open3d,
)

DEFAULT_GROUND_TRUTH_DIR = ROOT / "data" / "ATM22" / "labelsTr"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "skeleton_scale_probe" / "results_thickness"
DEFAULT_CASES = ("ATM_034", "ATM_044", "ATM_046", "ATM_125")

BBOX_MARGIN = 8
MAX_EROSION_CLASS = 15

# Millimetre radii for the Euclidean local thickness sweep, ascending.
MM_RADII: tuple[float, ...] = (
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
)

# Reporting groups over the operational class n (thickness = {2n+1, 2n+2} voxels).
CLASS_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("1-2", 0, 0),
    ("3-4", 1, 1),
    ("5-6", 2, 2),
    ("7-8", 3, 3),
    ("9-12", 4, 5),
    ("13-16", 6, 7),
    (">16", 8, MAX_EROSION_CLASS),
)

MM_BINS: tuple[tuple[str, float, float], ...] = (
    ("<=1mm", 0.0, 1.0),
    ("1-2mm", 1.0, 2.0),
    ("2-3mm", 2.0, 3.0),
    ("3-5mm", 3.0, 5.0),
    ("5-8mm", 5.0, 8.0),
    ("8-14mm", 8.0, 14.0),
    (">14mm", 14.0, float("inf")),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--no-mm-thickness", action="store_true", help="Skip the slow Euclidean mm sweep."
    )
    parser.add_argument(
        "--no-largest-component", action="store_true", help="Use raw GT instead of the LCC."
    )
    parser.add_argument("--no-synthetic-check", action="store_true")
    return parser.parse_args()


def _bounding_box(mask: np.ndarray, margin: int) -> tuple[slice, ...]:
    slices = []
    for axis in range(mask.ndim):
        others = tuple(i for i in range(mask.ndim) if i != axis)
        present = np.flatnonzero(mask.any(axis=others))
        slices.append(
            slice(
                max(0, int(present[0]) - margin),
                min(mask.shape[axis], int(present[-1]) + 1 + margin),
            )
        )
    return tuple(slices)


def _to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).to(device)[None, None]


@torch.no_grad()
def _degenerate_set(mask_t: torch.Tensor) -> torch.Tensor:
    """Exact clDice degeneracy: open(x) == 0, so relu(x - open(x)) == x.

    Must use open(), not erode(). Erosion also removes the outer shell of THICK
    structures, but the dilation inside open() restores that shell, so those voxels never
    enter the skeleton. Measuring on erosion alone overstates the degenerate share.
    """
    return (mask_t > 0) & (_soft_open3d(mask_t) <= 0)


@torch.no_grad()
def _operational_class(mask_t: torch.Tensor, max_class: int = MAX_EROSION_CLASS) -> torch.Tensor:
    """Largest n with x in dilate^n(erode^n(mask)), using the trainer's own operators."""
    classes = torch.zeros_like(mask_t)
    eroded = mask_t
    for n in range(1, max_class + 1):
        eroded = _soft_erode3d(eroded)
        if float(eroded.max()) <= 0.0:
            break
        dilated = eroded
        for _ in range(n):
            dilated = F.max_pool3d(dilated, 3, 1, 1)
        classes = torch.where(
            (dilated > 0) & (mask_t > 0), torch.full_like(classes, float(n)), classes
        )
    return classes


def _euclidean_local_thickness(
    mask: np.ndarray, sampling: tuple[float, float, float], radii: tuple[float, ...]
) -> np.ndarray:
    """tau(x) = 2 * max{ r : x in opening(mask, ball_r) }, in the units of `sampling`.

    Opening by a ball of radius r is erode-then-dilate; both are thresholded distance
    transforms, so each radius costs one extra EDT.
    """
    thickness = np.zeros(mask.shape, dtype=np.float32)
    base = ndimage.distance_transform_edt(mask, sampling=sampling)
    for radius in radii:
        eroded = base > radius
        if not eroded.any():
            break
        reachable = ndimage.distance_transform_edt(~eroded, sampling=sampling) <= radius
        thickness[reachable & mask] = 2.0 * radius
    return thickness


def _synthetic_calibration(device: torch.device) -> list[dict[str, float]]:
    """Ideal tubes of known thickness: does class n really mean thickness {2n+1, 2n+2}?"""
    rows = []
    for thickness in range(1, 11):
        volume = np.zeros((40, 40, 40), dtype=bool)
        low = 20 - thickness // 2
        volume[low : low + thickness, low : low + thickness, 5:35] = True
        tensor = _to_tensor(volume, device)
        classes = _operational_class(tensor)
        degenerate = _degenerate_set(tensor)
        inside = tensor > 0
        modal = float(torch.mode(classes[inside].flatten()).values)
        rows.append(
            {
                "true_thickness": float(thickness),
                "modal_class": modal,
                "max_class": float(classes[inside].max()),
                "predicted_thickness_low": 2.0 * modal + 1.0,
                "degenerate_fraction": float(degenerate.sum() / inside.sum()),
            }
        )
    return rows


def _histogram(
    values: np.ndarray, weights_mask: np.ndarray, groups, total: float
) -> dict[str, float]:
    out = {}
    for name, low, high in groups:
        selected = (values >= low) & (values <= high) & weights_mask
        out[name] = float(selected.sum()) / total if total else float("nan")
    return out


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = [] if args.no_synthetic_check else _synthetic_calibration(device)
    if synthetic:
        print("=== Synthetic calibration: class n should mean thickness {2n+1, 2n+2} ===")
        print(f"{'true t':>7} {'class':>6} {'implies':>9} {'degenerate':>11}")
        for row in synthetic:
            low = int(row["predicted_thickness_low"])
            print(
                f"{row['true_thickness']:>7.0f} {row['modal_class']:>6.0f} "
                f"{f'{low}-{low + 1}':>9} {row['degenerate_fraction']:>11.4f}"
            )

    rows: list[dict[str, object]] = []
    for case_id in args.cases:
        path = args.ground_truth_dir / f"{case_id}_0000.nii.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        image = nib.load(path)
        zooms = tuple(float(v) for v in image.header.get_zooms()[:3])
        truth = np.asanyarray(image.dataobj) > 0
        if not args.no_largest_component:
            truth = _largest_connected_component(truth)
        box = _bounding_box(truth, BBOX_MARGIN)
        truth = np.ascontiguousarray(truth[box])

        started = time.perf_counter()
        tensor = _to_tensor(truth, device)
        classes = _operational_class(tensor)[0, 0].cpu().numpy()
        degenerate = _degenerate_set(tensor)[0, 0].cpu().numpy()
        del tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

        centreline = _skeletonize(truth)
        naive_edt = ndimage.distance_transform_edt(truth)

        volume_total = float(truth.sum())
        length_total = float(centreline.sum())
        row: dict[str, object] = {
            "case_id": case_id,
            "zooms_mm": list(zooms),
            "gt_voxels": int(volume_total),
            "centreline_voxels": int(length_total),
            "degenerate_volume_share": float(degenerate.sum()) / volume_total,
            "degenerate_length_share": float((degenerate & centreline).sum()) / length_total,
            "seconds_operational": time.perf_counter() - started,
        }

        volume_hist = _histogram(classes, truth, CLASS_GROUPS, volume_total)
        length_hist = _histogram(classes, centreline, CLASS_GROUPS, length_total)
        for name, low, high in CLASS_GROUPS:
            selected = (classes >= low) & (classes <= high) & truth
            count = float(selected.sum())
            row[f"volume__{name}"] = volume_hist[name]
            row[f"length__{name}"] = length_hist[name]
            row[f"degenerate_within__{name}"] = (
                float((selected & degenerate).sum()) / count if count else float("nan")
            )

        # Validation of the artefact itself: wall voxels are GT voxels with naive EDT <= 1.
        wall = truth & (naive_edt <= 1.0)
        wall_count = float(wall.sum())
        row["wall_voxel_share"] = wall_count / volume_total
        row["wall_naive_edt_mean"] = float(naive_edt[wall].mean()) if wall_count else float("nan")
        row["wall_operational_class_mean"] = (
            float(classes[wall].mean()) if wall_count else float("nan")
        )
        row["wall_class0_share"] = (
            float((wall & (classes == 0)).sum()) / wall_count if wall_count else float("nan")
        )

        if not args.no_mm_thickness:
            started_mm = time.perf_counter()
            thickness_mm = _euclidean_local_thickness(truth, zooms, MM_RADII)
            for name, low, high in MM_BINS:
                selected = truth & (thickness_mm >= low) & (thickness_mm < high)
                row[f"volume_mm__{name}"] = float(selected.sum()) / volume_total
                row[f"length_mm__{name}"] = float((selected & centreline).sum()) / length_total
            centreline_tau = float(np.median(thickness_mm[centreline]))
            centreline_2edt = float(
                np.median(
                    2.0 * ndimage.distance_transform_edt(truth, sampling=zooms)[centreline]
                )
            )
            row["mm_thickness_at_centreline_p50"] = centreline_tau
            # Consistency: at the centreline, local thickness should track 2 x EDT.
            row["centreline_tau_vs_2edt_ratio"] = centreline_tau / max(centreline_2edt, 1e-9)
            row["seconds_mm"] = time.perf_counter() - started_mm

        rows.append(row)
        print(
            f"[{case_id}] {int(volume_total)} vox, {int(length_total)} centreline; "
            f"degenerate {row['degenerate_volume_share']:.4f} by volume, "
            f"{row['degenerate_length_share']:.4f} by length",
            flush=True,
        )

    def mean(key: str) -> float:
        values = [
            r[key] for r in rows if isinstance(r.get(key), float) and np.isfinite(r[key])
        ]
        return float(np.mean(values)) if values else float("nan")

    print("\n=== Operational thickness (index units, clDice's own operators) ===")
    print(f"{'thickness':>10} {'volume %':>9} {'length %':>9} {'clDice-degenerate %':>20}")
    for name, _, _ in CLASS_GROUPS:
        print(
            f"{name:>10} {100 * mean(f'volume__{name}'):>9.2f} "
            f"{100 * mean(f'length__{name}'):>9.2f} "
            f"{100 * mean(f'degenerate_within__{name}'):>20.2f}"
        )
    print(
        f"\n  TOTAL clDice-degenerate: {100 * mean('degenerate_volume_share'):.2f}% of volume, "
        f"{100 * mean('degenerate_length_share'):.2f}% of centreline length"
    )

    print("\n=== Wall-shell artefact check ===")
    print(
        f"  GT voxels with naive EDT <= 1 (the artefact population): "
        f"{100 * mean('wall_voxel_share'):.2f}% of volume"
    )
    print(
        f"  ...their mean naive EDT         : {mean('wall_naive_edt_mean'):.3f}"
        "   (naive reading: 'thickness 1-2')"
    )
    print(
        f"  ...their mean operational class : {mean('wall_operational_class_mean'):.3f}"
        "   (class 0 == thickness 1-2)"
    )
    print(f"  ...actually in class 0          : {100 * mean('wall_class0_share'):.2f}%")
    print("  A large gap between the last two lines is the artefact, measured and avoided.")

    if not args.no_mm_thickness:
        print("\n=== Anatomical thickness (Euclidean local thickness, mm) ===")
        print(f"{'diameter':>10} {'volume %':>9} {'length %':>9}")
        for name, _, _ in MM_BINS:
            print(
                f"{name:>10} {100 * mean(f'volume_mm__{name}'):>9.2f} "
                f"{100 * mean(f'length_mm__{name}'):>9.2f}"
            )
        print(f"\n  median centreline diameter: {mean('mm_thickness_at_centreline_p50'):.2f} mm")
        print(
            f"  consistency tau / 2*EDT at centreline: "
            f"{mean('centreline_tau_vs_2edt_ratio'):.3f} (expect ~1)"
        )

    summary = {
        "script": "measure_airway_thickness.py",
        "cases": args.cases,
        "largest_component": not args.no_largest_component,
        "max_erosion_class": MAX_EROSION_CLASS,
        "class_groups": [[n, a, b] for n, a, b in CLASS_GROUPS],
        "mm_radii": list(MM_RADII),
        "synthetic_calibration": synthetic,
        "per_case": rows,
    }
    (args.output_dir / "airway_thickness.json").write_text(json.dumps(summary, indent=2))
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (args.output_dir / "airway_thickness_per_case.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote {args.output_dir / 'airway_thickness.json'}")


if __name__ == "__main__":
    main()
