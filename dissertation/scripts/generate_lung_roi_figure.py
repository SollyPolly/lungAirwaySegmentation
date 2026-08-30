"""Lung and trachea region of interest, drawn on the CT it is actually derived from.

Replaces the earlier ``generate_lung_roi_schematic.py``, which drew the geometry to the
right proportions but with stylised ellipses for anatomy. The geometry was never the part
in doubt; what a schematic could not show is that the superior extension exists to catch a
real structure -- the cervical trachea continues above the lung apex, so a bounding box
fitted to the lungs alone cuts the airway tree at its root.

Nothing here is redrawn. Panel (a) is the acquisition volume and panel (b) is the staged
file the network is given, read off disk; the boxes are the lung mask's own bounding box
and the ROI recorded in the staging manifest, not values retyped from it.

**Coronal slab MinIP, not a single slice.** The trachea shifts anteriorly as it ascends, so
no one coronal plane holds both the cervical trachea and the carina. A minimum-intensity
projection over an anterior-posterior slab is the standard airway view and shows the air
column continuously. The slab is centred on the lung mask's own AP centroid, so the choice
of view uses no annotation -- which matters, because the claim being illustrated is that no
annotation enters the ROI construction either.

Orientation is resolved from the affine rather than assumed: ATM'22 is stored LPS, so array
axis 2 increases superiorly and axis 0 towards the patient's left. Radiological convention,
patient's right on the viewer's left.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\generate_lung_roi_figure.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_theme import (  # noqa: E402
    ANNOTATION_PT, ANNOTATION_RED, INK, LABEL_PT, MUTED, TICK_PT, apply_theme, panel_label,
)

DEFAULT_CASE = "ATM_016"
DEFAULT_MANIFEST = (
    ROOT / "data" / "nnunet" / "predict_in" / "val_lungroi_m8_s120" / "lung_crop_manifest.json"
)
PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "methods" / "lungroi"
PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "methods" / "lungroi"

# Lung window. Wide enough that the mediastinum stays bright against the air column, which
# is the contrast the figure is actually about.
WINDOW = (-1350.0, 150.0)
# Half-thickness of the MinIP slab, in voxels either side of the lung AP centroid. About
# 33 mm at this spacing: enough to hold the trachea through its anterior drift, narrow
# enough that the lungs do not wash out into a single flat shadow.
SLAB_HALF = 40
# Okabe-Ito sky blue. Not assigned to any arm, so a lung outline cannot read as a result.
LUNG_EDGE = "#56B4E9"
SCALE_BAR_MM = 50.0
# Larger than the shared ANNOTATION_PT. Two coronal panels side by side are aspect-limited
# -- they are already as large as a 6.9 in figure allows -- so legibility on the printed
# page has to come from type size rather than from more room.
OVERLAY_PT = 8.0
# A CT panel has near-black air and near-white soft tissue in the same frame, so no single
# ink is legible everywhere. Every annotation gets a thin halo in the opposite tone rather
# than being placed by hand and hoping the background stays put.
HALO = [path_effects.withStroke(linewidth=1.7, foreground="white")]
DARK_HALO = [path_effects.withStroke(linewidth=1.7, foreground="black")]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pdf-output-dir", type=Path, default=PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=PNG_OUT)
    return parser.parse_args()


def _axis_roles(affine: np.ndarray) -> tuple[int, int, int]:
    """Array axes carrying left-right, anterior-posterior and superior-inferior."""
    codes = nib.aff2axcodes(affine)
    role = {code: index for index, code in enumerate(codes)}
    lateral = role.get("L", role.get("R"))
    frontal = role.get("A", role.get("P"))
    vertical = role.get("S", role.get("I"))
    if None in (lateral, frontal, vertical):
        raise ValueError(f"Could not resolve anatomical axes from codes {codes}")
    if codes[vertical] != "S":
        raise ValueError(
            f"Axis {vertical} runs inferior-first ({codes}); this figure assumes a "
            "superior-increasing axis and would render upside down."
        )
    return lateral, frontal, vertical


def _bounding_box(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive per-axis extent of the mask."""
    box = []
    for axis in range(3):
        others = tuple(other for other in range(3) if other != axis)
        occupied = np.flatnonzero(np.any(mask, axis=others))
        box.append((int(occupied[0]), int(occupied[-1])))
    return box


def _minip(volume: np.ndarray, frontal: int, low: int, high: int) -> np.ndarray:
    """Minimum-intensity projection across an anterior-posterior slab."""
    return volume.take(indices=range(low, high), axis=frontal).min(axis=frontal)


def _draw(axis, image, lateral_mm, vertical_mm, lateral_first: bool) -> None:
    """One coronal panel, in millimetres, superior upward."""
    oriented = image if lateral_first else image.T
    axis.imshow(oriented.T, origin="lower", cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1],
                extent=(0.0, lateral_mm, 0.0, vertical_mm), aspect="equal",
                interpolation="bilinear")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _box(axis, extent_mm, colour, **kwargs) -> None:
    (x0, x1), (z0, z1) = extent_mm
    axis.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, facecolor="none", edgecolor=colour,
                             **kwargs))


def main() -> None:
    args = _parse_args()
    apply_theme()

    manifest = json.loads(args.manifest.read_text())
    entry = manifest["cases"][args.case]
    margin = int(manifest["margin_voxels"])
    superior_request = int(manifest["superior_margin_voxels"])

    ct_image = nib.load(entry["ct"])
    lung_image = nib.load(entry["lung"])
    staged_image = nib.load(ROOT / entry["output"])
    lateral, frontal, vertical = _axis_roles(ct_image.affine)
    spacing = tuple(float(z) for z in ct_image.header.get_zooms()[:3])

    ct = np.asanyarray(ct_image.dataobj).astype(np.float32)
    lung = np.asanyarray(lung_image.dataobj) > 0
    staged = np.asanyarray(staged_image.dataobj).astype(np.float32)

    lung_box = _bounding_box(lung)
    roi = [(int(lo), int(hi)) for lo, hi in entry["bbox"]]

    # The manifest is the record, but it is a record of a computation; recompute the box
    # from the mask and the stated margins and require the two to agree, so a figure can
    # never quietly disagree with the staging it claims to illustrate.
    for axis_index, (lo, hi) in enumerate(lung_box):
        upper = superior_request if axis_index == vertical else margin
        expected = (max(0, lo - margin), min(ct.shape[axis_index], hi + upper + 1))
        if expected != roi[axis_index]:
            raise SystemExit(
                f"{args.case}: recomputed ROI on axis {axis_index} is {expected}, "
                f"manifest says {roi[axis_index]}"
            )

    centre = int(np.argwhere(lung)[:, frontal].mean())
    slab = (max(0, centre - SLAB_HALF), min(ct.shape[frontal], centre + SLAB_HALF + 1))
    ct_view = _minip(ct, frontal, *slab)
    staged_view = _minip(staged, frontal, *slab)
    lung_view = np.any(lung.take(indices=range(*slab), axis=frontal), axis=frontal)

    lateral_mm = ct.shape[lateral] * spacing[lateral]
    vertical_mm = ct.shape[vertical] * spacing[vertical]
    to_mm = lambda pair, axis: (pair[0] * spacing[axis], (pair[1] + 1) * spacing[axis])  # noqa: E731
    lung_mm = (to_mm(lung_box[lateral], lateral), to_mm(lung_box[vertical], vertical))
    roi_mm = ((roi[lateral][0] * spacing[lateral], roi[lateral][1] * spacing[lateral]),
              (roi[vertical][0] * spacing[vertical], roi[vertical][1] * spacing[vertical]))
    superior_applied = roi[vertical][1] - (lung_box[vertical][1] + 1)

    lateral_first = lateral < vertical
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.5), layout="constrained")

    axis = axes[0]
    _draw(axis, ct_view, lateral_mm, vertical_mm, lateral_first)
    grid = lung_view if lateral_first else lung_view.T
    axis.contour(grid.T, levels=[0.5], colors=[LUNG_EDGE], linewidths=1.0,
                 extent=(0.0, lateral_mm, 0.0, vertical_mm))
    _box(axis, lung_mm, "#f0f0f0", linewidth=0.9, linestyle=(0, (4, 2.5)), zorder=4)
    _box(axis, roi_mm, ANNOTATION_RED, linewidth=1.5, zorder=5)

    # The superior extension, which is the whole reason the box is not just the lung box.
    mid = 0.5 * (roi_mm[0][0] + roi_mm[0][1])
    axis.add_patch(FancyArrowPatch((mid, lung_mm[1][1]), (mid, roi_mm[1][1]),
                                   arrowstyle="<->", mutation_scale=6, color=ANNOTATION_RED,
                                   linewidth=0.9, zorder=6))
    # Kept inside the data area: text that spills past the axes enlarges this panel's tight
    # bounding box under constrained layout and shrinks its neighbour, so the two CTs stop
    # being drawn at the same size.
    axis.annotate(
        f"superior extension\n$+${superior_applied} of {superior_request} requested",
        xy=(mid, 0.5 * (lung_mm[1][1] + roi_mm[1][1])),
        xytext=(mid + 0.05 * lateral_mm, roi_mm[1][1] - 0.012 * vertical_mm),
        fontsize=OVERLAY_PT, color=ANNOTATION_RED, ha="left", va="top",
        path_effects=HALO,
        arrowprops=dict(arrowstyle="-", color=ANNOTATION_RED, linewidth=0.6,
                        shrinkA=2, shrinkB=2, path_effects=HALO),
    )
    axis.annotate(
        f"$+${margin} voxels", xy=(roi_mm[0][0], 0.36 * vertical_mm),
        xytext=(0.02 * lateral_mm, 0.30 * vertical_mm), fontsize=ANNOTATION_PT,
        color=ANNOTATION_RED, ha="left", va="center", path_effects=HALO,
        arrowprops=dict(arrowstyle="->", color=ANNOTATION_RED, linewidth=0.6,
                        path_effects=HALO),
    )

    bar_x = 0.04 * lateral_mm
    bar_y = 0.045 * vertical_mm
    axis.plot([bar_x, bar_x + SCALE_BAR_MM], [bar_y, bar_y], color="white", linewidth=1.6,
              solid_capstyle="butt", zorder=7, path_effects=DARK_HALO)
    axis.text(bar_x + 0.5 * SCALE_BAR_MM, bar_y + 0.012 * vertical_mm,
              f"{SCALE_BAR_MM:.0f} mm", color="white", fontsize=OVERLAY_PT - 0.8,
              ha="center", va="bottom", zorder=7, path_effects=DARK_HALO)
    dimensions = " × ".join(str(size) for size in ct.shape)
    axis.set_title(f"{args.case}: acquisition volume, {dimensions} voxels",
                   fontsize=OVERLAY_PT, color=MUTED)
    panel_label(axis, "a", x=-0.02)

    axis = axes[1]
    _draw(axis, staged_view, lateral_mm, vertical_mm, lateral_first)
    _box(axis, roi_mm, ANNOTATION_RED, linewidth=1.5, zorder=5)
    axis.set_title(
        f"staged input: outside the ROI set to 0 HU, {100 * entry['roi_fraction']:.1f}% retained",
        fontsize=OVERLAY_PT, color=MUTED,
    )
    panel_label(axis, "b", x=-0.02)

    keys = [
        plt.Line2D([], [], color=LUNG_EDGE, linewidth=0.8, label="lung mask"),
        plt.Line2D([], [], color="#9aa3ab", linewidth=0.9, linestyle=(0, (4, 2.5)),
                   label="lung-mask bounding box"),
        plt.Line2D([], [], color=ANNOTATION_RED, linewidth=1.5, label="retained ROI"),
    ]
    fig.legend(handles=keys, loc="outside lower center", ncol=3, fontsize=OVERLAY_PT,
               frameon=False, handletextpad=0.5, columnspacing=1.8)

    args.pdf_output_dir.mkdir(parents=True, exist_ok=True)
    args.png_output_dir.mkdir(parents=True, exist_ok=True)
    stem = "lung_roi_bbox"
    fig.savefig(args.pdf_output_dir / f"{stem}.pdf", facecolor="white")
    fig.savefig(args.png_output_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  lung bbox {lung_box}  ROI {roi}  superior applied {superior_applied}")
    print(f"  MinIP slab {slab} on axis {frontal}, centred on the lung AP centroid {centre}")
    print("  wrote", args.pdf_output_dir / f"{stem}.pdf")


if __name__ == "__main__":
    main()
