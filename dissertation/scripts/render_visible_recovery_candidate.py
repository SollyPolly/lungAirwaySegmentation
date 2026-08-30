"""Render orthogonal CT evidence for a scouted Soft-clDice recovery candidate.

The candidate is read from ``visible_recovery_candidates.csv`` and reconstructed from
the masks, so the view is reproducible.  Each row selects the slice containing the most
voxels from that connected, reference-supported addition.  This diagnostic is intended
to decide whether a 3-D surface difference represents a recognisable airway segment.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import figure_arms as arms  # noqa: E402
import render_tree as rt  # noqa: E402
from figure_theme import apply_theme  # noqa: E402

OUT = ROOT / "dissertation" / "Figures" / "png" / "discussion"
CSV_PATH = ROOT / "dissertation" / "Figures" / "provenance" / "visible_recovery_candidates.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="087")
    parser.add_argument("--candidate-rank", type=int, default=4,
                        help="Rank among rows for this case in the scout CSV.")
    parser.add_argument("--context-mm", type=float, default=16.0)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(path)
    return np.asanyarray(image.dataobj), np.asarray(image.header.get_zooms()[:3], float)


def _prediction_path(directory: Path, case_name: str) -> Path:
    for name in (f"{case_name}.nii.gz", f"{case_name}_0000.nii.gz"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No prediction for {case_name} in {directory}")


def _slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(array, index, axis=axis).T


def _window(array: np.ndarray, low: float = -1350.0, high: float = 150.0) -> np.ndarray:
    return np.clip((array.astype(np.float32) - low) / (high - low), 0.0, 1.0)


def _overlay_contour(ax, mask: np.ndarray, colour: str, linewidth: float = 1.0) -> None:
    if mask.any() and not mask.all():
        ax.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=linewidth)


def main() -> None:
    args = _parse_args()
    apply_theme()
    case_id = str(args.case).zfill(3)
    case_name = f"ATM_{case_id}"
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        candidates = [row for row in csv.DictReader(handle) if row["case_id"] == case_id]
    if len(candidates) < args.candidate_rank:
        raise SystemExit(f"No candidate rank {args.candidate_rank} for case {case_id}")
    chosen = candidates[args.candidate_rank - 1]

    ct, spacing = _load(ROOT / "data" / "ATM22" / "imagesTr" / f"{case_name}_0000.nii.gz")
    reference, _ = _load(ROOT / "data" / "ATM22" / "labelsTr" / f"{case_name}_0000.nii.gz")
    control_dir = arms.prediction_dir("control", "val")
    treatment_dir = arms.prediction_dir("soft_f0", "val")
    if control_dir is None or treatment_dir is None:
        raise SystemExit("Control and Soft-clDice predictions are required.")
    control, _ = _load(_prediction_path(control_dir, case_name))
    treatment, _ = _load(_prediction_path(treatment_dir, case_name))
    reference = reference > 0
    control = control > 0
    treatment = treatment > 0

    added = reference & treatment & ~control
    labels, _ = ndi.label(added, structure=ndi.generate_binary_structure(3, 3))
    centre = np.rint([
        float(chosen["centre_i"]), float(chosen["centre_j"]), float(chosen["centre_k"])
    ]).astype(int)
    component_id = int(labels[tuple(centre)])
    if component_id == 0:
        raise RuntimeError("The stored candidate centre no longer lands in an added component.")
    component = labels == component_id
    coordinates = np.argwhere(component)
    centreline = rt.reference_centreline(reference)
    recovered_centreline_voxels = int((centreline & component).sum())

    padding = np.ceil(args.context_mm / spacing).astype(int)
    start = np.maximum(coordinates.min(axis=0) - padding, 0)
    stop = np.minimum(coordinates.max(axis=0) + padding + 1, ct.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    arrays = {
        "ct": ct[crop], "reference": reference[crop], "control": control[crop],
        "treatment": treatment[crop], "component": component[crop],
    }
    lost = arrays["reference"] & arrays["control"] & ~arrays["treatment"]
    unsupported_added = ~arrays["reference"] & arrays["treatment"] & ~arrays["control"]

    plane_names = ("Sagittal", "Coronal", "Axial")
    columns = ("Reference annotation", "No consistency", "Soft-clDice", "What changed")
    fig, axes = plt.subplots(3, 4, figsize=(10.6, 8.0), constrained_layout=True)
    for axis, row_axes in enumerate(axes):
        counts = np.bincount(coordinates[:, axis], minlength=ct.shape[axis])
        index_global = int(np.argmax(counts))
        index = index_global - start[axis]
        ct_slice = _window(_slice(arrays["ct"], axis, index))
        reference_slice = _slice(arrays["reference"], axis, index)
        control_slice = _slice(arrays["control"], axis, index)
        treatment_slice = _slice(arrays["treatment"], axis, index)
        component_slice = _slice(arrays["component"], axis, index)
        lost_slice = _slice(lost, axis, index)
        unsupported_slice = _slice(unsupported_added, axis, index)

        for column, ax in enumerate(row_axes):
            ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                _overlay_contour(ax, reference_slice, "#CC79A7", 1.4)
            elif column == 1:
                _overlay_contour(ax, reference_slice, "white", 0.8)
                _overlay_contour(ax, control_slice, "#0072B2", 1.5)
            elif column == 2:
                _overlay_contour(ax, reference_slice, "white", 0.8)
                _overlay_contour(ax, treatment_slice, "#009E73", 1.5)
            else:
                classes = np.zeros(component_slice.shape, dtype=np.uint8)
                classes[component_slice] = 1
                classes[lost_slice] = 2
                classes[unsupported_slice] = 3
                overlay = np.ma.masked_where(classes == 0, classes)
                ax.imshow(
                    overlay, cmap=ListedColormap(["#000000", "#009E73", "#E69F00", "#0072B2"]),
                    vmin=0, vmax=3, alpha=0.78, origin="lower", interpolation="nearest",
                )
                _overlay_contour(ax, reference_slice, "white", 0.8)
            if axis == 0:
                ax.set_title(columns[column], fontsize=9.0, fontweight="semibold")
        row_axes[0].set_ylabel(
            f"{plane_names[axis]}\nslice {index_global}", fontsize=8.6, fontweight="semibold"
        )

    handles = [
        Patch(facecolor="#CC79A7", label="Reference outline"),
        Patch(facecolor="#0072B2", label="No-consistency outline / unsupported addition"),
        Patch(facecolor="#009E73", label="Soft-clDice outline / reference-supported addition"),
        Patch(facecolor="#E69F00", label="Reference-supported loss"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.0)
    fig.suptitle(
        f"ATM {case_id}: connected reference-supported Soft-clDice addition "
        f"({int(chosen['added_reference_voxels']):,} voxels; "
        f"{recovered_centreline_voxels} centreline voxels)",
        fontsize=11.0, fontweight="bold",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / f"visible_recovery_candidate_val_{case_id}.png"
    fig.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Recovered reference-centreline voxels in component: {recovered_centreline_voxels}")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
