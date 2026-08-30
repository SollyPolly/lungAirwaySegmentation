"""Full-page, reproducibly selected qualitative comparison for the Discussion.

The patient is selected as the largest paired TLD gain before rendering.  Within that
patient, the highlighted region is the largest connected reference-supported addition that
is thin and elongated.  The page then shows the same region as a local 3-D surface and on
CT.  This answers a limitation of a centreline-only rendering: it verifies that at least one
metric gain corresponds to a visible annotated airway segment rather than a one-voxel shift
of an otherwise identical surface.

Run from the repository root::

    .venv\Scripts\python.exe dissertation\scripts\render_qualitative_plate.py
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import figure_arms as arms  # noqa: E402
import render_case_gallery as gallery  # noqa: E402
import render_tree as rt  # noqa: E402
from figure_theme import (  # noqa: E402
    INK, MUTED, apply_theme,
)

PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "discussion"
PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "discussion"
PREDICTION_RED = "#C62828"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=arms.COHORTS, default="val")
    parser.add_argument("--treatment", default="soft_f0")
    parser.add_argument("--comparator", default="control")
    parser.add_argument("--metric", default="td_raw")
    parser.add_argument("--case", default=None,
                        help="Explicit case id; otherwise use the largest paired-TLD gain.")
    parser.add_argument("--component-max-radius-mm", type=float, default=1.0)
    parser.add_argument("--component-min-elongation", type=float, default=3.0)
    parser.add_argument("--component-min-voxels", type=int, default=12)
    parser.add_argument("--context-mm", type=float, default=16.0)
    parser.add_argument("--ct-context-mm", type=float, default=10.0)
    parser.add_argument("--azimuth", type=float, default=-30.0)
    parser.add_argument("--elevation", type=float, default=10.0)
    parser.add_argument("--px-mm", type=float, default=0.32)
    parser.add_argument("--zoom-px-mm", type=float, default=0.10)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--smooth-mm", type=float, default=0.25)
    parser.add_argument("--taubin", type=int, default=16)
    parser.add_argument("--pdf-output-dir", type=Path, default=PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=PNG_OUT)
    return parser.parse_args()


def _component_geometry(coordinates: np.ndarray, zooms: np.ndarray) -> tuple[float, float]:
    """Return (long-axis span in mm, elongation) for one connected component."""
    physical = coordinates.astype(np.float64) * zooms
    if len(physical) < 3:
        return 0.0, 1.0
    centred = physical - physical.mean(axis=0)
    covariance = centred.T @ centred / len(centred)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    principal = eigenvectors[:, -1]
    span = float(np.ptp(centred @ principal))
    elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[-2]))
    return span, elongation


def _select_visible_addition(
    reference: np.ndarray,
    control: np.ndarray,
    treatment: np.ndarray,
    centreline: np.ndarray,
    zooms: np.ndarray,
    *,
    max_radius_mm: float,
    min_elongation: float,
    min_voxels: int,
) -> tuple[np.ndarray, dict]:
    """Select a visible tube segment without choosing a crop by eye.

    Candidates are 26-connected reference voxels present only in the treatment.  The
    mean reference EDT rejects thick-airway wall changes; PCA elongation rejects compact
    patches.  Among eligible candidates the largest volume is selected, with recovered
    centreline count and long-axis span as deterministic tie breakers.
    """
    added = reference & treatment & ~control
    labels, count = ndi.label(added, structure=ndi.generate_binary_structure(3, 3))
    sizes = np.bincount(labels.ravel())
    radius = ndi.distance_transform_edt(reference, sampling=zooms)
    candidates: list[dict] = []
    for component_id, item in enumerate(ndi.find_objects(labels), start=1):
        size = int(sizes[component_id])
        if item is None or size < min_voxels:
            continue
        local = labels[item] == component_id
        offset = np.asarray([part.start for part in item], dtype=int)
        coordinates = np.argwhere(local) + offset
        mean_radius = float(radius[tuple(coordinates.T)].mean())
        span, elongation = _component_geometry(coordinates, zooms)
        recovered_line = int((centreline[tuple(coordinates.T)]).sum())
        candidates.append({
            "component_id": component_id,
            "size": size,
            "mean_radius_mm": mean_radius,
            "long_axis_span_mm": span,
            "elongation": elongation,
            "recovered_centreline_voxels": recovered_line,
            "coordinates": coordinates,
            "eligible": (
                mean_radius <= max_radius_mm
                and elongation >= min_elongation
                and recovered_line > 0
            ),
        })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise SystemExit(
            f"No connected reference-supported addition among {count} components met "
            f"radius <= {max_radius_mm:g} mm, elongation >= {min_elongation:g}, "
            f"size >= {min_voxels} and recovered-centreline support."
        )
    chosen = max(
        eligible,
        key=lambda candidate: (
            candidate["size"], candidate["recovered_centreline_voxels"],
            candidate["long_axis_span_mm"], -candidate["component_id"],
        ),
    )
    selected = np.zeros(reference.shape, dtype=bool)
    selected[tuple(chosen["coordinates"].T)] = True
    record = {key: value for key, value in chosen.items() if key != "coordinates"}
    record["eligible_component_count"] = len(eligible)
    record["all_added_component_count"] = count
    return selected, record


def _crop_around(mask: np.ndarray, zooms: np.ndarray, context_mm: float) -> tuple[slice, ...]:
    coordinates = np.argwhere(mask)
    padding = np.ceil(context_mm / zooms).astype(int)
    start = np.maximum(coordinates.min(axis=0) - padding, 0)
    stop = np.minimum(coordinates.max(axis=0) + padding + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop))


def _cropped_affine(affine: np.ndarray, crop: tuple[slice, ...]) -> np.ndarray:
    shifted = np.asarray(affine, dtype=float).copy()
    start = np.asarray([s.start for s in crop], dtype=float)
    shifted[:3, 3] = (affine @ np.r_[start, 1.0])[:3]
    return shifted


def _surface_zoom_panels(
    reference: np.ndarray,
    control: np.ndarray,
    treatment: np.ndarray,
    crop: tuple[slice, ...],
) -> list[rt.Panel]:
    local_reference = reference[crop]
    local_control = control[crop]
    local_treatment = treatment[crop]
    prediction_palette = {1: (PREDICTION_RED, "Model prediction")}
    return [
        rt.Panel("Reference annotation", rt.reference_classes(local_reference),
                 rt.REFERENCE_PALETTE),
        rt.Panel("Control prediction", rt.reference_classes(local_control),
                 prediction_palette),
        rt.Panel("+ Soft-clDice prediction", rt.reference_classes(local_treatment),
                 prediction_palette),
    ]


def _boundary(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    edge = mask ^ ndi.binary_erosion(mask)
    if not edge.any() or iterations <= 0:
        return edge
    return ndi.binary_dilation(edge, iterations=iterations)


def _resample_plane(array: np.ndarray, spacing: np.ndarray, *, order: int) -> np.ndarray:
    target = float(np.min(spacing))
    factors = spacing / target
    return ndi.zoom(array, factors, order=order, prefilter=order > 1)


def _ct_plane_images(
    ct: np.ndarray,
    reference: np.ndarray,
    control: np.ndarray,
    treatment: np.ndarray,
    component: np.ndarray,
    crop: tuple[slice, ...],
    zooms: np.ndarray,
    affine: np.ndarray,
) -> tuple[list[np.ndarray], list[rt.Panel], dict]:
    """Four CT panels through the slice containing most of the selected component."""
    local_component = component[crop]
    coordinates = np.argwhere(local_component)
    peak_counts = []
    peak_indices = []
    for axis in range(3):
        counts = np.bincount(coordinates[:, axis], minlength=local_component.shape[axis])
        peak_counts.append(int(counts.max()))
        peak_indices.append(int(np.argmax(counts)))
    axis = int(np.argmax(peak_counts))
    index = peak_indices[axis]
    global_index = index + int(crop[axis].start)
    remaining = [candidate for candidate in range(3) if candidate != axis]
    plane_spacing = zooms[remaining][::-1]

    def plane(array: np.ndarray, *, order: int) -> np.ndarray:
        sliced = np.take(array[crop], index, axis=axis).T
        return _resample_plane(sliced, plane_spacing, order=order)

    ct_plane = plane(ct, order=1).astype(np.float32)
    grey = np.clip((ct_plane + 1350.0) / 1500.0, 0.0, 1.0)
    base = np.repeat(grey[..., None], 3, axis=2)
    ref_plane = plane(reference.astype(np.uint8), order=0) > 0
    control_plane = plane(control.astype(np.uint8), order=0) > 0
    treatment_plane = plane(treatment.astype(np.uint8), order=0) > 0
    component_plane = plane(component.astype(np.uint8), order=0) > 0

    def paint(image: np.ndarray, mask: np.ndarray, colour: str, alpha: float = 1.0) -> None:
        rgb = np.asarray(to_rgb(colour), dtype=np.float32)
        image[mask] = (1.0 - alpha) * image[mask] + alpha * rgb

    reference_colour = rt.REFERENCE_PALETTE[rt.REFERENCE][0]
    prediction_colour = PREDICTION_RED
    images: list[np.ndarray] = []

    reference_image = base.copy()
    paint(reference_image, ref_plane, reference_colour, 0.34)
    paint(reference_image, _boundary(ref_plane, 1), reference_colour)
    images.append(reference_image)

    control_image = base.copy()
    paint(control_image, control_plane, prediction_colour, 0.38)
    paint(control_image, _boundary(control_plane, 1), prediction_colour)
    images.append(control_image)

    treatment_image = base.copy()
    paint(treatment_image, treatment_plane, prediction_colour, 0.38)
    paint(treatment_image, _boundary(treatment_plane, 1), prediction_colour)
    images.append(treatment_image)

    axcodes = nib.aff2axcodes(affine)
    code = axcodes[axis]
    orientation = (
        "Sagittal" if code in ("L", "R")
        else "Coronal" if code in ("A", "P")
        else "Axial"
    )
    subtitle = f"{orientation} CT, slice {global_index}"
    dummy = np.zeros((1, 1, 1), dtype=np.int16)
    panels = [
        rt.Panel("Reference annotation", dummy, rt.REFERENCE_PALETTE, subtitle=subtitle),
        rt.Panel("Control prediction", dummy, {1: (prediction_colour, "Prediction")}),
        rt.Panel("+ Soft-clDice prediction", dummy, {1: (prediction_colour, "Prediction")}),
    ]
    component_coordinates = np.argwhere(component_plane)
    component_bbox = [
        float(component_coordinates[:, 1].min()),
        float(component_coordinates[:, 0].min()),
        float(component_coordinates[:, 1].max()),
        float(component_coordinates[:, 0].max()),
    ]
    return images, panels, {
        "axis": axis, "orientation": orientation, "stored_slice": global_index,
        "component_voxels_on_slice": peak_counts[axis],
        "window_hu": [-1350, 150], "isotropic_display_mm": float(np.min(plane_spacing)),
        "component_bbox_pixels": component_bbox,
    }


def _compose_multiscale(
    *,
    whole_images: list[np.ndarray],
    whole_panels: list[rt.Panel],
    whole_box: list[float],
    local_images: list[np.ndarray],
    local_panels: list[rt.Panel],
    local_box: list[float],
    ct_images: list[np.ndarray],
    ct_panels: list[rt.Panel],
    ct_box: list[float],
    pdf_dir: Path,
    png_dir: Path,
    stem: str,
) -> Path:
    """Compose an A->B->C evidence chain with unequal row column counts.

    The generic renderer deliberately uses equal grids. This Discussion plate instead uses
    the same direct three-column comparison at each scale: reference, control prediction and
    Soft-clDice prediction. Green boxes localise one region without introducing additional
    error-class colours or legends.
    """
    figure_width = rt.TEXT_WIDTH_IN
    figure_height = 7.35
    figure = plt.figure(figsize=(figure_width, figure_height), facecolor="white")

    left_in, right_in = 0.10, 0.10
    green = rt.TEAL

    def row_header(y_in: float, text: str) -> None:
        figure.text(
            left_in / figure_width, y_in / figure_height, text,
            ha="left", va="center", fontsize=9.2, fontweight="bold", color=INK,
        )

    def place_row(
        images: list[np.ndarray],
        panels: list[rt.Panel],
        *,
        bottom_in: float,
        image_height_in: float,
        title_width: int,
        width_ratios: list[float] | None = None,
    ) -> list[plt.Axes]:
        columns = len(images)
        gap_in = 0.08
        usable = figure_width - left_in - right_in - gap_in * (columns - 1)
        ratios = np.asarray(width_ratios or [1.0] * columns, dtype=float)
        ratios /= ratios.sum()
        cell_widths = usable * ratios
        axes: list[plt.Axes] = []
        running_left = left_in
        for index, (image, panel) in enumerate(zip(images, panels)):
            cell_width = float(cell_widths[index])
            cell_left = running_left
            running_left += cell_width + gap_in
            image_aspect = image.shape[1] / image.shape[0]
            wanted_width = image_height_in * image_aspect
            if wanted_width <= cell_width:
                width, height = wanted_width, image_height_in
            else:
                width, height = cell_width, cell_width / image_aspect
            image_left = cell_left + (cell_width - width) / 2.0
            image_bottom = bottom_in + (image_height_in - height) / 2.0
            ax = figure.add_axes((
                image_left / figure_width, image_bottom / figure_height,
                width / figure_width, height / figure_height,
            ))
            ax.imshow(image, interpolation="none")
            ax.set_axis_off()
            axes.append(ax)
            figure.text(
                (cell_left + cell_width / 2.0) / figure_width,
                (bottom_in + image_height_in + 0.055) / figure_height,
                textwrap.fill(panel.title, width=title_width),
                ha="center", va="bottom", fontsize=8.15, fontweight="semibold", color=INK,
                linespacing=1.05,
            )
            if panel.subtitle:
                figure.text(
                    (cell_left + cell_width / 2.0) / figure_width,
                    (bottom_in - 0.035) / figure_height,
                    panel.subtitle.replace("\n", " · "),
                    ha="center", va="top", fontsize=7.0, color=MUTED,
                )
        return axes

    def callout_box(ax: plt.Axes, bbox: list[float]) -> None:
        left, top, right, bottom = bbox
        pad = max(3.0, 0.016 * max(ax.images[0].get_array().shape[:2]))
        left, top = left - pad, top - pad
        width, height = right - left + 2 * pad, bottom - top + 2 * pad
        ax.add_patch(Rectangle(
            (left, top), width, height, fill=False,
            edgecolor=green, linewidth=1.55,
        ))
    # A: whole-tree context. The same green box appears on every panel.
    row_header(7.15, "A — Whole-tree context")
    whole_axes = place_row(
        whole_images, whole_panels, bottom_in=5.55, image_height_in=1.32, title_width=28,
    )
    for ax in whole_axes:
        callout_box(ax, whole_box)

    # B: larger local surfaces, each rendered as one solid mask.
    row_header(5.28, "B — Local 3-D view")
    local_axes = place_row(
        local_images, local_panels, bottom_in=3.35, image_height_in=1.62, title_width=29
    )
    for ax in local_axes:
        callout_box(ax, local_box)

    # C: the corresponding masks on CT, with the same reference/prediction colours.
    row_header(3.08, "C — CT-level evidence")
    ct_axes = place_row(
        ct_images, ct_panels, bottom_in=0.50, image_height_in=2.25, title_width=30
    )
    for ax in ct_axes:
        callout_box(ax, ct_box)

    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    destination = pdf_dir / f"{stem}.pdf"
    figure.savefig(destination, dpi=600, facecolor="white")
    figure.savefig(png_dir / f"{stem}.png", dpi=400, facecolor="white")
    plt.close(figure)
    print(f"    include at width=1.00\\linewidth ({figure_width:.2f} in natural)")
    return destination


def main() -> None:
    args = _parse_args()
    apply_theme()
    ranking = gallery.rank_cases(args.treatment, args.comparator, args.cohort, args.metric)
    if args.case is None:
        row = ranking[-1]
        case_id, rule = row["case_id"], "largest paired TLD gain"
    else:
        case_id = str(args.case).zfill(3) if str(args.case).isdigit() else str(args.case)
        row = next((item for item in ranking if item["case_id"] == case_id), None)
        if row is None:
            raise SystemExit(f"Case {case_id!r} is not in the paired cohort.")
        rule = "explicit"

    directories = {
        args.comparator: arms.prediction_dir(args.comparator, args.cohort),
        args.treatment: arms.prediction_dir(args.treatment, args.cohort),
    }
    if any(directory is None for directory in directories.values()):
        raise SystemExit("One or more requested arms has no prediction directory.")

    started = time.time()
    print(f"Loading {arms.display_case(case_id, args.cohort)} ({rule}) ...",
          flush=True)
    case = rt.load_case(case_id, args.cohort, directories)
    reference = case["masks"]["reference"]
    control = case["masks"][args.comparator]
    treatment = case["masks"][args.treatment]
    zooms = np.linalg.norm(case["affine"][:3, :3], axis=0)

    centreline = rt.reference_centreline(reference)
    region_a, component_record = _select_visible_addition(
        reference, control, treatment, centreline, zooms,
        max_radius_mm=args.component_max_radius_mm,
        min_elongation=args.component_min_elongation,
        min_voxels=args.component_min_voxels,
    )
    crop_a = _crop_around(region_a, zooms, args.context_mm)
    print(
        f"  selected component {component_record['component_id']}: "
        f"{component_record['size']} reference-supported added voxels, "
        f"{component_record['recovered_centreline_voxels']} centreline voxels, "
        f"{component_record['long_axis_span_mm']:.1f}-mm long-axis span, "
        f"mean reference radius {component_record['mean_radius_mm']:.2f} mm",
        flush=True,
    )

    _, recovery_counts = rt.recovery_classes(
        control, treatment, reference, basis="centreline", dilate=1, change_dilate=2
    )
    control_metrics = arms.load_per_case(args.comparator, args.cohort)[case_id]
    treatment_metrics = arms.load_per_case(args.treatment, args.cohort)[case_id]
    prediction_palette = {1: (PREDICTION_RED, "Model prediction")}
    panels = [
        rt.Panel(
            "Reference", rt.reference_classes(reference), rt.REFERENCE_PALETTE,
            subtitle=f"{int(reference.sum()):,} voxels",
        ),
        rt.Panel(
            "Control prediction", rt.reference_classes(control), prediction_palette,
            subtitle=(
                f"TLD {100 * control_metrics['td_raw']:.2f}% · "
                f"Dice {100 * control_metrics['dice_raw']:.2f}%"
            ),
        ),
        rt.Panel(
            "+ Soft-clDice prediction", rt.reference_classes(treatment), prediction_palette,
            subtitle=(
                f"TLD {100 * treatment_metrics['td_raw']:.2f}% · "
                f"Dice {100 * treatment_metrics['dice_raw']:.2f}%"
            ),
        ),
    ]
    camera = rt.Camera(
        azimuth=args.azimuth, elevation=args.elevation, px_mm=args.px_mm,
        supersample=args.supersample, smooth_mm=args.smooth_mm,
        taubin_iterations=args.taubin,
    )
    whole_images, whole_stats = rt.render_panels(
        panels, case["affine"], camera=camera,
        annotations={"A": region_a},
    )

    zoom_camera = replace(camera, px_mm=args.zoom_px_mm)
    a_panels = _surface_zoom_panels(
        reference, control, treatment, crop_a
    )
    a_images, a_stats = rt.render_panels(
        a_panels, _cropped_affine(case["affine"], crop_a), camera=zoom_camera,
        annotations={"selected": region_a[crop_a]},
    )

    if args.cohort != "val":
        raise SystemExit("CT confirmation is currently wired for the ATM'22 validation cohort.")
    ct_path = ROOT / "data" / "ATM22" / "imagesTr" / f"ATM_{case_id}_0000.nii.gz"
    ct_image = nib.load(ct_path)
    if not np.allclose(ct_image.affine, case["affine"], atol=1e-4, rtol=1e-5):
        raise SystemExit(f"CT affine does not match masks for ATM {case_id}.")
    ct = np.asanyarray(ct_image.dataobj, dtype=np.float32)
    crop_ct = _crop_around(region_a, zooms, args.ct_context_mm)
    b_images, b_panels, ct_record = _ct_plane_images(
        ct, reference, control, treatment, region_a, crop_ct, zooms, case["affine"]
    )

    destination = _compose_multiscale(
        whole_images=whole_images,
        whole_panels=panels,
        whole_box=whole_stats["annotations"]["A"],
        local_images=a_images,
        local_panels=a_panels,
        local_box=a_stats["annotations"]["selected"],
        ct_images=b_images,
        ct_panels=b_panels,
        ct_box=ct_record["component_bbox_pixels"],
        pdf_dir=args.pdf_output_dir,
        png_dir=args.png_output_dir,
        stem=f"qualitative_plate_{args.cohort}_{case_id}",
    )

    provenance = arms.write_provenance(
        f"qualitative_plate_{args.cohort}_{case_id}.json",
        {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "dissertation/scripts/render_qualitative_plate.py",
            "case_id": case_id,
            "cohort": args.cohort,
            "patient_selection": {
                "rule": rule,
                "rank": row["rank"],
                "cohort_size": len(ranking),
                "paired_tld_difference": row["difference"],
            },
            "region_selection": {
                "definition": (
                    "largest 26-connected reference-supported treatment-only component "
                    "meeting the pre-render thinness, elongation, size and centreline criteria"
                ),
                "component_max_radius_mm": args.component_max_radius_mm,
                "component_min_elongation": args.component_min_elongation,
                "component_min_voxels": args.component_min_voxels,
                "context_mm": args.context_mm,
                "ct_context_mm": args.ct_context_mm,
                "selected_component": component_record,
            },
            "ct_view": ct_record,
            "recovery_counts": recovery_counts,
            "prediction_dirs": {key: value.name for key, value in directories.items()},
            "render": {"whole": whole_stats, "local_3d": a_stats},
            "figure": destination.name,
            "seconds": round(time.time() - started, 1),
        },
    )
    print(f"Wrote {destination}")
    print(f"Wrote {provenance}")


if __name__ == "__main__":
    main()
