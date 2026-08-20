"""Generate real-airway clDice figures from an exported EMA-teacher probability.

The default inputs reproduce the ATM_044 mechanism illustration used in the
dissertation explainer.  First export the *continuous* foreground probabilities
from one EMA-teacher checkpoint with nnU-Net's ``--save_probabilities`` flag.
Then run, from the repository root::

    .venv\Scripts\python.exe dissertation\scripts\generate_cldice_real_airway_patch.py

The morphology is evaluated in 3-D with the exact functions imported from the
Mean-Teacher trainer.  A larger computation crop provides more than the ten-voxel
halo needed by the iterative operation; only its central region is displayed.
The image panels are projections of the resulting 3-D tensors.  They are never
skeletonised as 2-D images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize
import torch

# Running a file from ``dissertation/scripts/`` puts that directory, rather than the
# repository root, on sys.path. Add the root so the canonical trainer implementation
# imports.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nnunet_trainers.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (
    _soft_erode3d,
    _soft_skeleton3d,
)


DEFAULT_PROBABILITIES = (
    ROOT / ".codex_tmp" / "cldice_real_patch" / "d126_final_teacher" / "ATM_044.npz"
)
DEFAULT_HARD_MASK = (
    ROOT / ".codex_tmp" / "cldice_real_patch" / "d126_final_teacher" / "ATM_044.nii.gz"
)
DEFAULT_CT = ROOT / "data" / "ATM22" / "imagesTr" / "ATM_044_0000.nii.gz"
DEFAULT_GT = ROOT / "data" / "ATM22" / "labelsTr" / "ATM_044_0000.nii.gz"
FIGURE_ROOT = ROOT / "dissertation" / "Figures"
DEFAULT_PDF_OUT = FIGURE_ROOT / "pdf" / "analysis" / "teacher_targets"
DEFAULT_PNG_OUT = FIGURE_ROOT / "png" / "analysis" / "teacher_targets"
DEFAULT_APPENDIX_PDF_OUT = FIGURE_ROOT / "pdf" / "appendix"
DEFAULT_APPENDIX_PNG_OUT = FIGURE_ROOT / "png" / "appendix"
DEFAULT_PROVENANCE_OUT = FIGURE_ROOT / "provenance"

# Bounds are half-open and in nibabel's (x, y, z) order.
COMPUTE_BOUNDS = ((205, 310), (265, 318), (360, 460))
DISPLAY_BOUNDS = ((225, 285), (289, 294), (385, 445))
PROFILE_SKELETON_BOUNDS = ((238, 271), (282, 300), (390, 436))
PROFILE_KEEP_BOUNDS = ((245, 267), (285, 298), (395, 429))
CONNECTOR_BOX_XZ = (251, 402, 11, 18)

INK = "#0f172a"
MUTED = "#475569"
BLUE = "#0891b2"
GREEN = "#22c55e"
ORANGE = "#f97316"
RED = "#dc2626"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    parser.add_argument("--hard-mask", type=Path, default=DEFAULT_HARD_MASK)
    parser.add_argument("--ct", type=Path, default=DEFAULT_CT)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    parser.add_argument("--appendix-pdf-output-dir", type=Path, default=DEFAULT_APPENDIX_PDF_OUT)
    parser.add_argument("--appendix-png-output-dir", type=Path, default=DEFAULT_APPENDIX_PNG_OUT)
    parser.add_argument("--provenance-output-dir", type=Path, default=DEFAULT_PROVENANCE_OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _slices(bounds: tuple[tuple[int, int], ...]) -> tuple[slice, ...]:
    return tuple(slice(lo, hi) for lo, hi in bounds)


def _relative_slices(
    inner: tuple[tuple[int, int], ...],
    outer: tuple[tuple[int, int], ...],
) -> tuple[slice, ...]:
    result = []
    for (inner_lo, inner_hi), (outer_lo, outer_hi) in zip(inner, outer):
        if inner_lo < outer_lo or inner_hi > outer_hi:
            raise ValueError(f"Inner bounds {inner} are not contained in {outer}")
        result.append(slice(inner_lo - outer_lo, inner_hi - outer_lo))
    return tuple(result)


def _load_inputs(args: argparse.Namespace) -> dict[str, object]:
    required = (args.probabilities, args.hard_mask, args.ct, args.ground_truth)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        message = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required real-patch inputs are missing:\n"
            f"{message}\nExport probabilities first with --save_probabilities."
        )

    ct_image = nib.load(args.ct)
    gt_image = nib.load(args.ground_truth)
    hard_image = nib.load(args.hard_mask)
    if ct_image.shape != gt_image.shape or ct_image.shape != hard_image.shape:
        raise ValueError(
            f"NIfTI shapes differ: CT={ct_image.shape}, GT={gt_image.shape}, "
            f"hard={hard_image.shape}"
        )

    comp_slices = _slices(COMPUTE_BOUNDS)
    ct = np.asarray(ct_image.dataobj[comp_slices], dtype=np.float32)
    gt = np.asarray(gt_image.dataobj[comp_slices]) > 0
    exported_hard = np.asarray(hard_image.dataobj[comp_slices]) > 0

    # nnU-Net exports (class, z, y, x), whereas nibabel exposes (x, y, z).
    (x0, x1), (y0, y1), (z0, z1) = COMPUTE_BOUNDS
    with np.load(args.probabilities) as archive:
        if "probabilities" not in archive:
            raise KeyError(f"{args.probabilities} has no 'probabilities' array")
        probabilities = archive["probabilities"]
        if probabilities.ndim != 4 or probabilities.shape[0] < 2:
            raise ValueError(
                "Expected nnU-Net probabilities with shape (class, z, y, x); "
                f"got {probabilities.shape}"
            )
        probability = np.array(
            probabilities[1, z0:z1, y0:y1, x0:x1].transpose(2, 1, 0),
            dtype=np.float32,
            copy=True,
        )

    if probability.shape != ct.shape:
        raise ValueError(f"Probability crop {probability.shape} != CT crop {ct.shape}")
    hard = probability > 0.5  # Strict > matches the consistency implementation.
    mismatch = int(np.count_nonzero(hard != exported_hard))

    return {
        "ct": ct,
        "gt": gt,
        "probability": probability,
        "hard": hard,
        "hard_mismatch": mismatch,
        "zooms": tuple(float(value) for value in ct_image.header.get_zooms()[:3]),
    }


def _morphology(
    probability_xyz: np.ndarray,
    hard_xyz: np.ndarray,
    device: str,
) -> dict[str, object]:
    # Trainer tensors are (batch, channel, depth/z, height/y, width/x).
    probability_zyx = np.ascontiguousarray(probability_xyz.transpose(2, 1, 0))
    tensor = torch.from_numpy(probability_zyx)[None, None].to(device)
    hard_tensor = torch.from_numpy(
        np.ascontiguousarray(hard_xyz.transpose(2, 1, 0))
    )[None, None].float().to(device)

    eroded_at: dict[int, torch.Tensor] = {0: tensor}
    current = tensor
    with torch.no_grad():
        for iteration in range(1, 11):
            current = _soft_erode3d(current)
            if iteration in (1, 3, 7, 10):
                eroded_at[iteration] = current
        soft_skeleton = _soft_skeleton3d(tensor, iterations=10)
        hard_skeleton = _soft_skeleton3d(hard_tensor, iterations=10)

    def xyz(value: torch.Tensor) -> np.ndarray:
        return value[0, 0].detach().cpu().numpy().transpose(2, 1, 0)

    return {
        "eroded": {depth: xyz(value) for depth, value in eroded_at.items()},
        "soft_skeleton": xyz(soft_skeleton),
        "hard_skeleton": xyz(hard_skeleton),
    }


def _project(volume_xyz: np.ndarray) -> np.ndarray:
    """Return a z-by-x max projection through the displayed coronal slab."""
    local = volume_xyz[_relative_slices(DISPLAY_BOUNDS, COMPUTE_BOUNDS)]
    return np.max(local, axis=1).T


def _central_ct(volume_xyz: np.ndarray, y: int = 291) -> np.ndarray:
    (x0, x1), _, (z0, z1) = DISPLAY_BOUNDS
    cx0, cy0, cz0 = (bounds[0] for bounds in COMPUTE_BOUNDS)
    return volume_xyz[x0 - cx0:x1 - cx0, y - cy0, z0 - cz0:z1 - cz0].T


def _profile_coordinates(gt_comp: np.ndarray) -> np.ndarray:
    skeleton_region = gt_comp[_relative_slices(PROFILE_SKELETON_BOUNDS, COMPUTE_BOUNDS)]
    skeleton = skeletonize(skeleton_region)
    coordinates = np.argwhere(skeleton)
    coordinates += np.array([bounds[0] for bounds in PROFILE_SKELETON_BOUNDS])
    keep = np.ones(coordinates.shape[0], dtype=bool)
    for axis, (lo, hi) in enumerate(PROFILE_KEEP_BOUNDS):
        keep &= (coordinates[:, axis] >= lo) & (coordinates[:, axis] < hi)
    coordinates = coordinates[keep]
    if coordinates.shape[0] < 10:
        raise RuntimeError("Could not recover the expected annotated connector centreline")
    # This local connector contains one skeleton point per z-plane.  Superior to
    # inferior sorting gives a reproducible anatomical line profile.
    return coordinates[np.argsort(-coordinates[:, 2])]


def _sample(volume: np.ndarray, global_coordinates: np.ndarray) -> np.ndarray:
    local = global_coordinates - np.array([bounds[0] for bounds in COMPUTE_BOUNDS])
    return volume[local[:, 0], local[:, 1], local[:, 2]]


def _distance_mm(coordinates: np.ndarray, zooms: tuple[float, float, float]) -> np.ndarray:
    differences = np.diff(coordinates, axis=0) * np.asarray(zooms)[None, :]
    step = np.linalg.norm(differences, axis=1)
    return np.concatenate(([0.0], np.cumsum(step)))


def _style_image_axis(ax: plt.Axes, panel: str, title: str) -> None:
    ax.set_title(f"{panel}  {title}", fontsize=10.5, weight="bold", color=INK, pad=7)
    ax.set_xlabel("voxel x", fontsize=8.5, color=MUTED)
    ax.set_ylabel("voxel z (superior)", fontsize=8.5, color=MUTED)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def _extent() -> tuple[int, int, int, int]:
    (x0, x1), _, (z0, z1) = DISPLAY_BOUNDS
    return x0, x1, z0, z1


def _add_connector_box(ax: plt.Axes) -> None:
    x, z, width, height = CONNECTOR_BOX_XZ
    ax.add_patch(
        Rectangle((x, z), width, height, fill=False, edgecolor=ORANGE, linewidth=1.6)
    )


def _add_gt_contour(ax: plt.Axes, gt_projection: np.ndarray) -> None:
    ax.contour(
        gt_projection,
        levels=[0.5],
        colors=[GREEN],
        linewidths=0.75,
        origin="lower",
        extent=_extent(),
    )


def _physical_extent(data: dict[str, object]) -> tuple[float, float, float, float]:
    zoom_x, _, zoom_z = data["zooms"]
    (x0, x1), _, (z0, z1) = DISPLAY_BOUNDS
    return 0.0, (x1 - x0) * zoom_x, 0.0, (z1 - z0) * zoom_z


def _add_gt_contour_physical(
    ax: plt.Axes,
    gt_projection: np.ndarray,
    data: dict[str, object],
) -> None:
    ax.contour(
        gt_projection,
        levels=[0.5],
        colors=[GREEN],
        linewidths=0.75,
        origin="lower",
        extent=_physical_extent(data),
    )


def _add_connector_box_physical(ax: plt.Axes, data: dict[str, object]) -> None:
    zoom_x, _, zoom_z = data["zooms"]
    (display_x0, _), _, (display_z0, _) = DISPLAY_BOUNDS
    x, z, width, height = CONNECTOR_BOX_XZ
    ax.add_patch(
        Rectangle(
            ((x - display_x0) * zoom_x, (z - display_z0) * zoom_z),
            width * zoom_x,
            height * zoom_z,
            fill=False,
            edgecolor=ORANGE,
            linewidth=1.35,
        )
    )


def _style_publication_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("left--right distance (mm)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("inferior--superior distance (mm)", fontsize=7.5, color=MUTED)
    ax.tick_params(labelsize=6.8, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def _save_panel(
    fig: plt.Figure,
    pdf_output_dir: Path,
    png_output_dir: Path,
    stem: str,
) -> None:
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    png_output_dir.mkdir(parents=True, exist_ok=True)
    # NO bbox_inches='tight' here, deliberately. Every publication panel declares
    # the same figsize, but tight-cropping trims each one to its own content: a
    # panel carrying a colourbar, or a line plot with a legend, ends up a
    # different shape from a bare image panel. Included side by side at a common
    # \linewidth those become visibly different heights. Constrained layout
    # already handles the internal padding, so saving at the declared figsize
    # gives every panel identical outer dimensions.
    fig.savefig(pdf_output_dir / f"{stem}.pdf", facecolor="white")
    fig.savefig(png_output_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)


def _save(
    fig: plt.Figure,
    pdf_output_dir: Path,
    png_output_dir: Path,
    stem: str,
) -> None:
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    png_output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png_output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _comparison_figure(
    data: dict[str, object],
    morphology: dict[str, object],
    pdf_output_dir: Path,
    png_output_dir: Path,
) -> dict[str, float | int]:
    ct = data["ct"]
    gt = data["gt"]
    probability = data["probability"]
    hard = data["hard"]
    soft_skeleton = morphology["soft_skeleton"]
    hard_skeleton = morphology["hard_skeleton"]

    ct_slice = _central_ct(ct)
    gt_projection = _project(gt.astype(np.float32))
    probability_projection = _project(probability)
    hard_projection = _project(hard.astype(np.float32))
    soft_projection = _project(soft_skeleton)
    hard_skeleton_projection = _project(hard_skeleton)
    deleted_soft_projection = _project(soft_skeleton * (~hard))

    profile_coordinates = _profile_coordinates(gt)
    profile_probability = _sample(probability, profile_coordinates)
    profile_soft_skeleton = _sample(soft_skeleton, profile_coordinates)
    profile_hard_skeleton = _sample(hard_skeleton, profile_coordinates)
    profile_distance = _distance_mm(profile_coordinates, data["zooms"])

    fig, axes = plt.subplots(2, 4, figsize=(16.4, 9.2))
    fig.subplots_adjust(left=0.045, right=0.965, top=0.88, bottom=0.105, wspace=0.28, hspace=0.34)
    fig.suptitle(
        "Real airway patch: what thresholding changes before Mean-Teacher clDice",
        x=0.045,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.045,
        0.925,
        "ATM_044, final fold-0 Dataset126 EMA teacher. Green = manual airway outline (validation only); orange = connector enlarged below.",
        fontsize=9.8,
        color=MUTED,
    )

    ax = axes[0, 0]
    ax.imshow(ct_slice, cmap="gray", vmin=-1000, vmax=-350, origin="lower", extent=_extent())
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "A", "CT at coronal y=291")

    probability_norm = colors.LogNorm(vmin=1e-5, vmax=1.0)
    ax = axes[0, 1]
    probability_image = ax.imshow(
        np.ma.masked_less(probability_projection, 1e-5),
        cmap="magma",
        norm=probability_norm,
        origin="lower",
        extent=_extent(),
    )
    ax.contour(
        probability_projection,
        levels=[0.5],
        colors=[BLUE],
        linewidths=0.8,
        origin="lower",
        extent=_extent(),
    )
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "B", r"Teacher probability $p_T$ (log scale)")
    cbar = fig.colorbar(probability_image, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(r"$p_T$", fontsize=8)

    ax = axes[0, 2]
    ax.imshow(hard_projection, cmap="Blues_r", vmin=0, vmax=1, origin="lower", extent=_extent())
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "C", r"Hard target $H=\mathbf{1}[p_T>0.5]$")

    ax = axes[0, 3]
    below = profile_probability <= 0.5
    ax.semilogy(
        profile_distance,
        np.clip(profile_probability, 1e-7, 1),
        color=ORANGE,
        marker="o",
        markersize=3.2,
        linewidth=1.6,
        label=r"teacher $p_T$",
    )
    ax.axhline(0.5, color=BLUE, linestyle="--", linewidth=1.2, label="hard threshold")
    if np.any(below):
        indices = np.flatnonzero(below)
        ax.axvspan(
            profile_distance[indices[0]],
            profile_distance[indices[-1]],
            color=RED,
            alpha=0.10,
            label="deleted from H",
        )
    ax.set_ylim(1e-7, 1.4)
    ax.set_xlabel("distance along annotated centreline (mm)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("value (log scale)", fontsize=8.5, color=MUTED)
    ax.grid(color="#e2e8f0", linewidth=0.7, which="both")
    ax.legend(loc="lower left", fontsize=7.2, frameon=True)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.set_title("D  Probability through the real gap", fontsize=10.5, weight="bold", color=INK, pad=7)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    skeleton_norm = colors.LogNorm(vmin=1e-5, vmax=1.0)
    ax = axes[1, 0]
    soft_image = ax.imshow(
        np.ma.masked_less(soft_projection, 1e-5),
        cmap="inferno",
        norm=skeleton_norm,
        origin="lower",
        extent=_extent(),
    )
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "E", r"Probability-target skeleton $S(p_T)$")

    ax = axes[1, 1]
    ax.imshow(
        np.ma.masked_less(hard_skeleton_projection, 1e-5),
        cmap="inferno",
        norm=skeleton_norm,
        origin="lower",
        extent=_extent(),
    )
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "F", r"Hard-target skeleton $S(H)$")

    ax = axes[1, 2]
    ax.imshow(
        np.ma.masked_less(deleted_soft_projection, 1e-5),
        cmap="inferno",
        norm=skeleton_norm,
        origin="lower",
        extent=_extent(),
    )
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "G", r"$S(p_T)$ evidence where $H=0$")
    cbar = fig.colorbar(soft_image, ax=axes[1, :3], fraction=0.018, pad=0.025, aspect=35)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("soft-skeleton evidence (log scale)", fontsize=8)

    ax = axes[1, 3]
    zoom_extent = (248, 264, 400, 421)
    x_start = zoom_extent[0] - DISPLAY_BOUNDS[0][0]
    x_stop = zoom_extent[1] - DISPLAY_BOUNDS[0][0]
    z_start = zoom_extent[2] - DISPLAY_BOUNDS[2][0]
    z_stop = zoom_extent[3] - DISPLAY_BOUNDS[2][0]
    ax.imshow(
        ct_slice[z_start:z_stop, x_start:x_stop],
        cmap="gray",
        vmin=-1000,
        vmax=-350,
        origin="lower",
        extent=zoom_extent,
    )
    deleted_zoom = deleted_soft_projection[z_start:z_stop, x_start:x_stop]
    ax.imshow(
        np.ma.masked_less(deleted_zoom, 1e-5),
        cmap="inferno",
        norm=skeleton_norm,
        origin="lower",
        extent=zoom_extent,
        alpha=0.95,
    )
    hard_zoom = hard_skeleton_projection[z_start:z_stop, x_start:x_stop]
    if np.any(hard_zoom > 0.5):
        ax.contour(
            hard_zoom,
            levels=[0.5],
            colors=[BLUE],
            linewidths=1.3,
            origin="lower",
            extent=zoom_extent,
        )
    gt_zoom = gt_projection[z_start:z_stop, x_start:x_stop]
    ax.contour(
        gt_zoom,
        levels=[0.5],
        colors=[GREEN],
        linewidths=0.9,
        origin="lower",
        extent=zoom_extent,
    )
    ax.annotate(
        "faint graded evidence\nretained after threshold would delete it",
        xy=(256, 411),
        xytext=(249, 418.7),
        fontsize=7.4,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.2),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=ORANGE, alpha=0.92),
    )
    _style_image_axis(ax, "H", "Connector zoom")
    ax.text(
        0.02,
        0.02,
        "cyan: S(H)   heat: S(pT) where H=0",
        transform=ax.transAxes,
        fontsize=6.9,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
    )

    fig.text(
        0.5,
        0.037,
        "Exact 3-D operator: 10 erosion updates, then a five-voxel coronal max projection for display. "
        "This held-out inference crop illustrates the two target transforms; it is not a logged historical training batch.",
        ha="center",
        fontsize=8.8,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.5,
        0.014,
        "The probability target retains faint teacher evidence; this figure does not claim that the evidence is anatomically correct or that a separately trained soft-target model recovered the branch.",
        ha="center",
        fontsize=8.1,
        color=RED,
    )
    _save(
        fig,
        pdf_output_dir,
        png_output_dir,
        "atm044_teacher_target_comparison_full",
    )

    return {
        "profile_points": int(profile_coordinates.shape[0]),
        "profile_points_below_0.5": int(np.count_nonzero(profile_probability <= 0.5)),
        "profile_probability_min": float(profile_probability.min()),
        "profile_probability_max": float(profile_probability.max()),
        "profile_soft_skeleton_mass": float(profile_soft_skeleton.sum()),
        "profile_hard_skeleton_mass": float(profile_hard_skeleton.sum()),
        "display_soft_skeleton_mass_where_hard_zero": float(
            (soft_skeleton * (~hard))[_relative_slices(DISPLAY_BOUNDS, COMPUTE_BOUNDS)].sum()
        ),
    }


def _main_text_target_figures(
    data: dict[str, object],
    morphology: dict[str, object],
    pdf_output_dir: Path,
    png_output_dir: Path,
) -> None:
    """Write two final-width 2x2 plates for the Discussion chapter."""
    ct = data["ct"]
    gt = data["gt"]
    probability = data["probability"]
    hard = data["hard"]
    soft_skeleton = morphology["soft_skeleton"]
    hard_skeleton = morphology["hard_skeleton"]

    ct_slice = _central_ct(ct)
    gt_projection = _project(gt.astype(np.float32))
    probability_projection = _project(probability)
    hard_projection = _project(hard.astype(np.float32))
    soft_projection = _project(soft_skeleton)
    hard_skeleton_projection = _project(hard_skeleton)
    deleted_soft_projection = _project(soft_skeleton * (~hard))
    probability_norm = colors.LogNorm(vmin=1e-5, vmax=1.0)
    skeleton_norm = colors.LogNorm(vmin=1e-5, vmax=1.0)

    profile_coordinates = _profile_coordinates(gt)
    profile_probability = _sample(probability, profile_coordinates)
    profile_distance = _distance_mm(profile_coordinates, data["zooms"])

    fig, axes = plt.subplots(2, 2, figsize=(6.72, 6.35))
    fig.subplots_adjust(left=0.09, right=0.93, top=0.97, bottom=0.085, wspace=0.33, hspace=0.34)

    ax = axes[0, 0]
    ax.imshow(ct_slice, cmap="gray", vmin=-1000, vmax=-350, origin="lower", extent=_extent())
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "A", "CT and reference airway")

    ax = axes[0, 1]
    probability_image = ax.imshow(
        np.ma.masked_less(probability_projection, 1e-5),
        cmap="magma",
        norm=probability_norm,
        origin="lower",
        extent=_extent(),
    )
    ax.contour(
        probability_projection,
        levels=[0.5],
        colors=[BLUE],
        linewidths=0.9,
        origin="lower",
        extent=_extent(),
    )
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "B", r"Teacher probability $p_T$")
    probability_cax = fig.add_axes((0.945, 0.56, 0.014, 0.32))
    probability_bar = fig.colorbar(probability_image, cax=probability_cax)
    probability_bar.set_label(r"$p_T$ (log scale)", fontsize=7.5)
    probability_bar.ax.tick_params(labelsize=6.7)

    ax = axes[1, 0]
    ax.imshow(hard_projection, cmap="Blues_r", vmin=0, vmax=1, origin="lower", extent=_extent())
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "C", r"Thresholded target $H=\mathbf{1}[p_T>0.5]$")

    ax = axes[1, 1]
    below = profile_probability <= 0.5
    ax.semilogy(
        profile_distance,
        np.clip(profile_probability, 1e-7, 1),
        color=ORANGE,
        marker="o",
        markersize=2.8,
        linewidth=1.35,
        label=r"teacher $p_T$",
    )
    ax.axhline(0.5, color=BLUE, linestyle="--", linewidth=1.1, label="threshold")
    if np.any(below):
        indices = np.flatnonzero(below)
        ax.axvspan(
            profile_distance[indices[0]],
            profile_distance[indices[-1]],
            color=RED,
            alpha=0.10,
            label="removed from H",
        )
    ax.set_ylim(1e-7, 1.4)
    ax.set_xlabel("distance along reference centreline (mm)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("teacher probability (log)", fontsize=7.5, color=MUTED)
    ax.tick_params(labelsize=6.8, colors=MUTED)
    ax.grid(color="#e2e8f0", linewidth=0.55, which="both")
    ax.legend(loc="lower left", fontsize=6.2, frameon=True)
    ax.set_title("D  Threshold crossing through the connector", fontsize=9.2, weight="bold", color=INK, pad=6)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    _save(fig, pdf_output_dir, png_output_dir, "atm044_target_construction")

    fig, axes = plt.subplots(2, 2, figsize=(6.72, 6.35))
    fig.subplots_adjust(left=0.09, right=0.93, top=0.97, bottom=0.085, wspace=0.33, hspace=0.34)

    ax = axes[0, 0]
    skeleton_image = ax.imshow(
        np.ma.masked_less(soft_projection, 1e-5),
        cmap="inferno",
        norm=skeleton_norm,
        origin="lower",
        extent=_extent(),
    )
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "A", r"Probability-target skeleton $S(p_T)$")

    ax = axes[0, 1]
    ax.imshow(
        np.ma.masked_less(hard_skeleton_projection, 1e-5),
        cmap="inferno",
        norm=skeleton_norm,
        origin="lower",
        extent=_extent(),
    )
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "B", r"Hard-target skeleton $S(H)$")
    skeleton_cax = fig.add_axes((0.945, 0.56, 0.014, 0.32))
    skeleton_bar = fig.colorbar(skeleton_image, cax=skeleton_cax)
    skeleton_bar.set_label("centreline evidence (log)", fontsize=7.5)
    skeleton_bar.ax.tick_params(labelsize=6.7)

    ax = axes[1, 0]
    ax.imshow(
        np.ma.masked_less(deleted_soft_projection, 1e-5),
        cmap="inferno",
        norm=skeleton_norm,
        origin="lower",
        extent=_extent(),
    )
    _add_gt_contour(ax, gt_projection)
    _add_connector_box(ax)
    _style_image_axis(ax, "C", r"$S(p_T)$ evidence where $H=0$")

    ax = axes[1, 1]
    zoom_extent = (248, 264, 400, 421)
    x_start = zoom_extent[0] - DISPLAY_BOUNDS[0][0]
    x_stop = zoom_extent[1] - DISPLAY_BOUNDS[0][0]
    z_start = zoom_extent[2] - DISPLAY_BOUNDS[2][0]
    z_stop = zoom_extent[3] - DISPLAY_BOUNDS[2][0]
    ax.imshow(
        ct_slice[z_start:z_stop, x_start:x_stop],
        cmap="gray",
        vmin=-1000,
        vmax=-350,
        origin="lower",
        extent=zoom_extent,
    )
    deleted_zoom = deleted_soft_projection[z_start:z_stop, x_start:x_stop]
    ax.imshow(
        np.ma.masked_less(deleted_zoom, 1e-5),
        cmap="inferno",
        norm=skeleton_norm,
        origin="lower",
        extent=zoom_extent,
        alpha=0.95,
    )
    hard_zoom = hard_skeleton_projection[z_start:z_stop, x_start:x_stop]
    if np.any(hard_zoom > 0.5):
        ax.contour(
            hard_zoom,
            levels=[0.5],
            colors=[BLUE],
            linewidths=1.3,
            origin="lower",
            extent=zoom_extent,
        )
    gt_zoom = gt_projection[z_start:z_stop, x_start:x_stop]
    ax.contour(
        gt_zoom,
        levels=[0.5],
        colors=[GREEN],
        linewidths=0.9,
        origin="lower",
        extent=zoom_extent,
    )
    ax.annotate(
        "graded trace retained",
        xy=(256, 411),
        xytext=(249.2, 419.2),
        fontsize=6.6,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.0),
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=ORANGE, alpha=0.92),
    )
    _style_image_axis(ax, "D", "Connector enlargement")
    ax.text(
        0.02,
        0.02,
        r"cyan: $S(H)$; heat: $S(p_T)$ where $H=0$",
        transform=ax.transAxes,
        fontsize=6.1,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
    )

    _save(fig, pdf_output_dir, png_output_dir, "atm044_skeleton_consequence")


def _publication_panels(
    data: dict[str, object],
    morphology: dict[str, object],
    analysis_pdf_dir: Path,
    analysis_png_dir: Path,
    cldice_pdf_dir: Path,
    cldice_png_dir: Path,
) -> dict[str, float | int]:
    """Export title-free panels; LaTeX owns panel letters, titles and captions."""
    ct = data["ct"]
    gt = data["gt"]
    probability = data["probability"]
    hard = data["hard"]
    soft_skeleton = morphology["soft_skeleton"]
    hard_skeleton = morphology["hard_skeleton"]
    extent = _physical_extent(data)

    ct_slice = _central_ct(ct)
    gt_projection = _project(gt.astype(np.float32))
    probability_projection = _project(probability)
    hard_projection = _project(hard.astype(np.float32))
    soft_projection = _project(soft_skeleton)
    hard_skeleton_projection = _project(hard_skeleton)
    deleted_soft_projection = _project(soft_skeleton * (~hard))

    def image_panel(
        image: np.ndarray,
        stem: str,
        *,
        cmap: str,
        norm: colors.Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        mask_below: float | None = None,
        colourbar_label: str | None = None,
        connector_box: bool = True,
        gt_contour: bool = True,
        pdf_dir: Path = analysis_pdf_dir,
        png_dir: Path = analysis_png_dir,
    ) -> None:
        fig, ax = plt.subplots(figsize=(3.20, 2.55), layout="constrained")
        display = np.ma.masked_less(image, mask_below) if mask_below is not None else image
        artist = ax.imshow(
            display,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        if gt_contour:
            _add_gt_contour_physical(ax, gt_projection, data)
        if connector_box:
            _add_connector_box_physical(ax, data)
        _style_publication_axis(ax)
        if colourbar_label:
            bar = fig.colorbar(artist, ax=ax, fraction=0.048, pad=0.025)
            bar.set_label(colourbar_label, fontsize=7.2)
            bar.ax.tick_params(labelsize=6.5)
        _save_panel(fig, pdf_dir, png_dir, stem)

    image_panel(ct_slice, "atm044_ct_reference", cmap="gray", vmin=-1000, vmax=-350)
    image_panel(
        probability_projection,
        "atm044_teacher_probability",
        cmap="magma",
        norm=colors.LogNorm(vmin=1e-5, vmax=1),
        mask_below=1e-5,
        colourbar_label=r"teacher probability $p_T$ (log)",
    )
    image_panel(
        hard_projection,
        "atm044_thresholded_target",
        cmap="Blues_r",
        vmin=0,
        vmax=1,
    )

    profile_coordinates = _profile_coordinates(gt)
    profile_probability = _sample(probability, profile_coordinates)
    profile_soft_skeleton = _sample(soft_skeleton, profile_coordinates)
    profile_hard_skeleton = _sample(hard_skeleton, profile_coordinates)
    profile_distance = _distance_mm(profile_coordinates, data["zooms"])
    below = profile_probability <= 0.5
    fig, ax = plt.subplots(figsize=(3.20, 2.55), layout="constrained")
    ax.semilogy(
        profile_distance,
        np.clip(profile_probability, 1e-7, 1),
        color=ORANGE,
        marker="o",
        markersize=2.6,
        linewidth=1.25,
        label=r"$p_T$",
    )
    ax.axhline(0.5, color=BLUE, linestyle="--", linewidth=1.05, label="0.5 threshold")
    if np.any(below):
        indices = np.flatnonzero(below)
        ax.axvspan(
            profile_distance[indices[0]],
            profile_distance[indices[-1]],
            color=RED,
            alpha=0.10,
            label="removed from target",
        )
    ax.set_ylim(1e-7, 1.4)
    ax.set_xlabel("distance along reference centreline (mm)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("teacher probability (log)", fontsize=7.5, color=MUTED)
    ax.tick_params(labelsize=6.8, colors=MUTED)
    ax.grid(color="#e2e8f0", linewidth=0.55, which="both")
    ax.legend(loc="lower left", fontsize=6.2, frameon=True)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    _save_panel(fig, analysis_pdf_dir, analysis_png_dir, "atm044_connector_profile")

    skeleton_norm = colors.LogNorm(vmin=1e-5, vmax=1)
    image_panel(
        soft_projection,
        "atm044_probability_target_skeleton",
        cmap="inferno",
        norm=skeleton_norm,
        mask_below=1e-5,
        colourbar_label="centreline evidence (log)",
    )
    image_panel(
        hard_skeleton_projection,
        "atm044_hard_target_skeleton",
        cmap="inferno",
        norm=skeleton_norm,
        mask_below=1e-5,
    )
    image_panel(
        deleted_soft_projection,
        "atm044_subthreshold_skeleton_evidence",
        cmap="inferno",
        norm=skeleton_norm,
        mask_below=1e-5,
        colourbar_label="retained centreline evidence (log)",
    )

    zoom_bounds = ((248, 264), (289, 294), (400, 421))
    zoom_slices = _relative_slices(zoom_bounds, DISPLAY_BOUNDS)
    zoom_x_mm = (zoom_bounds[0][1] - zoom_bounds[0][0]) * data["zooms"][0]
    zoom_z_mm = (zoom_bounds[2][1] - zoom_bounds[2][0]) * data["zooms"][2]
    zoom_extent = (0, zoom_x_mm, 0, zoom_z_mm)
    ct_zoom = ct_slice[zoom_slices[2], zoom_slices[0]]
    deleted_zoom = deleted_soft_projection[zoom_slices[2], zoom_slices[0]]
    hard_zoom = hard_skeleton_projection[zoom_slices[2], zoom_slices[0]]
    gt_zoom = gt_projection[zoom_slices[2], zoom_slices[0]]
    fig, ax = plt.subplots(figsize=(3.20, 2.55), layout="constrained")
    ax.imshow(ct_zoom, cmap="gray", vmin=-1000, vmax=-350, origin="lower", extent=zoom_extent)
    ax.imshow(
        np.ma.masked_less(deleted_zoom, 1e-5),
        cmap="inferno",
        norm=skeleton_norm,
        origin="lower",
        extent=zoom_extent,
        alpha=0.95,
    )
    if np.any(hard_zoom > 0.5):
        ax.contour(hard_zoom, levels=[0.5], colors=[BLUE], linewidths=1.25,
                   origin="lower", extent=zoom_extent)
    ax.contour(gt_zoom, levels=[0.5], colors=[GREEN], linewidths=0.85,
               origin="lower", extent=zoom_extent)
    ax.set_xlabel("left--right distance (mm)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("inferior--superior distance (mm)", fontsize=7.5, color=MUTED)
    ax.tick_params(labelsize=6.8, colors=MUTED)
    ax.text(
        0.02,
        0.02,
        r"cyan: $S(H)$; heat: $S(p_T)$ where $H=0$",
        transform=ax.transAxes,
        fontsize=6.0,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.1),
    )
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    _save_panel(fig, analysis_pdf_dir, analysis_png_dir, "atm044_connector_zoom")

    # The same title-free panel assets support both the compact Methods figure and
    # a more detailed Appendix sequence assembled by LaTeX.
    for depth in (0, 1, 3, 7, 10):
        image_panel(
            _project(morphology["eroded"][depth]),
            f"soft_erosion_depth_{depth}",
            cmap="viridis",
            vmin=0,
            vmax=1,
            connector_box=False,
            colourbar_label="surviving probability" if depth == 0 else None,
            pdf_dir=cldice_pdf_dir,
            png_dir=cldice_png_dir,
        )
    image_panel(
        soft_projection,
        "soft_skeleton_accumulated",
        cmap="inferno",
        norm=skeleton_norm,
        mask_below=1e-5,
        colourbar_label="centreline evidence (log)",
        connector_box=False,
        pdf_dir=cldice_pdf_dir,
        png_dir=cldice_png_dir,
    )

    return {
        "profile_points": int(profile_coordinates.shape[0]),
        "profile_points_below_0.5": int(np.count_nonzero(profile_probability <= 0.5)),
        "profile_probability_min": float(profile_probability.min()),
        "profile_probability_max": float(profile_probability.max()),
        "profile_soft_skeleton_mass": float(profile_soft_skeleton.sum()),
        "profile_hard_skeleton_mass": float(profile_hard_skeleton.sum()),
        "display_soft_skeleton_mass_where_hard_zero": float(
            (soft_skeleton * (~hard))[_relative_slices(DISPLAY_BOUNDS, COMPUTE_BOUNDS)].sum()
        ),
    }


def _erosion_figure(
    data: dict[str, object],
    morphology: dict[str, object],
    pdf_output_dir: Path,
    png_output_dir: Path,
) -> None:
    gt_projection = _project(data["gt"].astype(np.float32))
    eroded = morphology["eroded"]
    depths = (0, 1, 3, 7, 10)

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.5))
    fig.subplots_adjust(left=0.06, right=0.90, top=0.86, bottom=0.105, wspace=0.24, hspace=0.31)
    fig.suptitle(
        "Soft erosion on the same real 3-D airway probability patch",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.915,
        "Each update replaces a voxel by the minimum over itself and its six face-connected neighbours. The filled tube becomes progressively thinner.",
        fontsize=9.5,
        color=MUTED,
    )

    probability_image = None
    for ax, depth, panel in zip(axes.flat[:5], depths, "ABCDE"):
        projected = _project(eroded[depth])
        probability_image = ax.imshow(
            projected,
            cmap="viridis",
            vmin=0,
            vmax=1,
            origin="lower",
            extent=_extent(),
        )
        _add_gt_contour(ax, gt_projection)
        _style_image_axis(
            ax,
            panel,
            r"Input $p_T$" if depth == 0 else rf"After {depth} erosion update{'s' if depth != 1 else ''}",
        )

    ax = axes.flat[5]
    skeleton_projection = _project(morphology["soft_skeleton"])
    skeleton_image = ax.imshow(
        np.ma.masked_less(skeleton_projection, 1e-5),
        cmap="inferno",
        norm=colors.LogNorm(vmin=1e-5, vmax=1),
        origin="lower",
        extent=_extent(),
    )
    _add_gt_contour(ax, gt_projection)
    _style_image_axis(ax, "F", r"Accumulated skeleton $S(p_T)$")

    probability_cax = fig.add_axes((0.925, 0.53, 0.014, 0.30))
    probability_bar = fig.colorbar(probability_image, cax=probability_cax)
    probability_bar.set_label("surviving probability", fontsize=8)
    probability_bar.ax.tick_params(labelsize=7)
    skeleton_cax = fig.add_axes((0.925, 0.15, 0.014, 0.30))
    skeleton_bar = fig.colorbar(skeleton_image, cax=skeleton_cax)
    skeleton_bar.set_label("centreline evidence (log)", fontsize=8)
    skeleton_bar.ax.tick_params(labelsize=7)

    fig.text(
        0.5,
        0.037,
        "All panels are five-voxel coronal max projections made after the 3-D operation. "
        "Opening and ridge subtraction are what convert these nested eroded maps into panel F.",
        ha="center",
        fontsize=8.8,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.5,
        0.014,
        "These are fixed min/max pooling operations, not learned convolution kernels.",
        ha="center",
        fontsize=8.2,
        color=RED,
    )
    _save(
        fig,
        pdf_output_dir,
        png_output_dir,
        "real_airway_erosion_sequence_full",
    )


def _erosion_main_text_figure(
    data: dict[str, object],
    morphology: dict[str, object],
    pdf_output_dir: Path,
    png_output_dir: Path,
) -> None:
    """Write a four-panel, final-width version for the Methods chapter."""
    gt_projection = _project(data["gt"].astype(np.float32))
    eroded = morphology["eroded"]
    panels = (
        ("A", r"Input probability $p_T$", _project(eroded[0]), "probability"),
        ("B", "After one erosion", _project(eroded[1]), "probability"),
        ("C", "After ten erosions", _project(eroded[10]), "probability"),
        ("D", r"Accumulated skeleton $S(p_T)$", _project(morphology["soft_skeleton"]), "skeleton"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(6.72, 6.25))
    fig.subplots_adjust(left=0.09, right=0.91, top=0.97, bottom=0.08, wspace=0.29, hspace=0.31)
    probability_image = None
    skeleton_image = None
    for ax, (letter, title, projected, kind) in zip(axes.flat, panels):
        if kind == "probability":
            probability_image = ax.imshow(
                projected,
                cmap="viridis",
                vmin=0,
                vmax=1,
                origin="lower",
                extent=_extent(),
            )
        else:
            skeleton_image = ax.imshow(
                np.ma.masked_less(projected, 1e-5),
                cmap="inferno",
                norm=colors.LogNorm(vmin=1e-5, vmax=1),
                origin="lower",
                extent=_extent(),
            )
        _add_gt_contour(ax, gt_projection)
        _style_image_axis(ax, letter, title)

    probability_cax = fig.add_axes((0.925, 0.56, 0.014, 0.32))
    probability_bar = fig.colorbar(probability_image, cax=probability_cax)
    probability_bar.set_label("surviving probability", fontsize=7.5)
    probability_bar.ax.tick_params(labelsize=6.7)
    skeleton_cax = fig.add_axes((0.925, 0.12, 0.014, 0.25))
    skeleton_bar = fig.colorbar(skeleton_image, cax=skeleton_cax)
    skeleton_bar.set_label("centreline evidence (log)", fontsize=7.5)
    skeleton_bar.ax.tick_params(labelsize=6.7)
    _save(fig, pdf_output_dir, png_output_dir, "real_airway_soft_erosion_compact")


def _write_metadata(
    args: argparse.Namespace,
    data: dict[str, object],
    comparison_stats: dict[str, float | int],
) -> None:
    metadata = {
        "case_id": "ATM_044",
        "purpose": "Mechanism illustration; intentionally selected because thresholding breaks an annotated bronchial connector.",
        "provenance": {
            "dataset": "Dataset126_ATM22MT240LungCrop",
            "trainer": "nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring",
            "fold": 0,
            "checkpoint": "checkpoint_final.pth",
            "network_role": "final EMA teacher",
            "inference": "deterministic held-out full-volume sliding-window inference with TTA disabled",
            "important_limitation": "This is not a saved stochastic U-stream training crop and is not the output of a separately trained probability-target model.",
        },
        "inputs": {
            "ct": str(args.ct.resolve()),
            "ground_truth": str(args.ground_truth.resolve()),
            "probabilities": str(args.probabilities.resolve()),
            "hard_mask": str(args.hard_mask.resolve()),
        },
        "array_conventions": {
            "nnunet_npz": "(class, z, y, x); foreground airway is channel 1",
            "nifti_via_nibabel": "(x, y, z)",
        },
        "operator": {
            "hard_target": "H = 1[p_T > 0.5] (strict greater-than)",
            "soft_skeleton_iterations": 10,
            "soft_erosion": "minimum over centre plus six face-connected neighbours",
            "opening_dilation": "3x3x3 maximum pool",
            "implementation": "nnunet_trainers/nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py::_soft_skeleton3d",
        },
        "geometry": {
            "voxel_spacing_xyz_mm": data["zooms"],
            "compute_bounds_xyz_half_open": COMPUTE_BOUNDS,
            "display_bounds_xyz_half_open": DISPLAY_BOUNDS,
            "display_projection": "max over coronal y=289:294 after 3-D computation; CT is y=291",
            "profile_skeleton_bounds_xyz_half_open": PROFILE_SKELETON_BOUNDS,
            "profile_keep_bounds_xyz_half_open": PROFILE_KEEP_BOUNDS,
        },
        "validation": {
            "thresholded_probability_vs_exported_hard_mask_mismatched_voxels_in_compute_crop": data[
                "hard_mismatch"
            ],
            **comparison_stats,
        },
        "interpretation": "S(p_T) is graded centreline evidence. Retaining faint evidence is not proof that it is a real airway.",
    }
    args.provenance_output_dir.mkdir(parents=True, exist_ok=True)
    output = args.provenance_output_dir / "atm044_teacher_target.json"
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    data = _load_inputs(args)
    morphology = _morphology(data["probability"], data["hard"], args.device)
    stats = _publication_panels(
        data,
        morphology,
        args.pdf_output_dir,
        args.png_output_dir,
        FIGURE_ROOT / "pdf" / "methods" / "cldice",
        FIGURE_ROOT / "png" / "methods" / "cldice",
    )
    _write_metadata(args, data, stats)
    print(f"Wrote real-airway clDice figures under {FIGURE_ROOT.resolve()}")


if __name__ == "__main__":
    main()
