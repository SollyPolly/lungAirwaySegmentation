"""Methods figure: one full soft-skeletonisation iteration on real teacher output.

The panel the previous figure lacked is the DILATION: erosion peels the airway,
the 3x3x3 maximum pool regrows the bulk of what it removed, and only what the
opening fails to restore survives as ridge evidence.  Showing erosion depths
alone never makes that visible.

Morphology is imported from the trainer, never reimplemented, so the figure
illustrates the code that produced the reported experiments.  All operations run
in 3-D on a compute crop that carries more than the ten-voxel halo the iteration
needs; only its central region is displayed, as a max projection through a thin
coronal slab.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\generate_soft_skeleton_operator_figure.py
    .venv\\Scripts\\python.exe dissertation\\scripts\\generate_soft_skeleton_operator_figure.py --preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nnunet_trainers.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (  # noqa: E402
    _soft_erode3d,
    _soft_open3d,
    _soft_skeleton3d,
)

CASE = "ATM_044"
PROBABILITIES = ROOT / ".codex_tmp" / "cldice_real_patch" / "d126_final_teacher" / f"{CASE}.npz"
CT_PATH = ROOT / "data" / "ATM22" / "imagesTr" / f"{CASE}_0000.nii.gz"
GT_PATH = ROOT / "data" / "ATM22" / "labelsTr" / f"{CASE}_0000.nii.gz"

PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "methods" / "cldice"
PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "methods" / "cldice"
PREVIEW_OUT = ROOT / "dissertation" / "build" / "figures" / "scout"
PROVENANCE = ROOT / "dissertation" / "Figures" / "provenance"

# Display windows are CENTRES (x, y, z) of a WIN_X x SLAB x WIN_Z box; candidates
# come from scout_soft_skeleton_patch.py.  The in-plane and through-plane voxel
# counts differ on purpose: spacing is 0.82 mm in x against 0.50 mm in z, so equal
# voxel counts would render as a 1.6:1 letterbox once the axes are physical.
WIN_X, WIN_Z, SLAB = 52, 86, 7
HALO = 14  # > the 10 erosion updates, so the compute crop is never edge-limited
CANDIDATES = {
    # Default: a trunk with a side branch that is also thick enough for the peeling to
    # stay visible over several depths.  Thinner distal windows read better at depth 0
    # but are annihilated by the second erosion, which leaves the depth row blank; the
    # thickest windows survive longer but are a bare Y with nothing branching off it.
    "trunk_side_branch": (314, 296, 331),
    "thick_trunk": (218, 296, 451),
    "trunk_branches": (176, 299, 448),
    "wide_y": (240, 291, 416),
    "upper_fan": (320, 291, 288),
    "clean_y": (288, 291, 352),
    "lower_fan": (176, 291, 336),
}
DEFAULT_WINDOW = "trunk_side_branch"

INK = "#1f2933"
MUTED = "#52606d"
GREEN = "#2ca25f"
CYAN = "#5ad2f4"
DARK = "#0d1117"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default=DEFAULT_WINDOW, choices=sorted(CANDIDATES))
    parser.add_argument("--preview", action="store_true",
                        help="Render every candidate window as one contact sheet and stop.")
    return parser.parse_args()


def _load(window: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Return CT, ground truth and teacher probability on the compute crop."""
    xc, yc, zc = window
    x0, y0, z0 = xc - WIN_X // 2, yc - SLAB // 2, zc - WIN_Z // 2
    cx0, cx1 = x0 - HALO, x0 + WIN_X + HALO
    cy0, cy1 = y0 - HALO, y0 + SLAB + HALO
    cz0, cz1 = z0 - HALO, z0 + WIN_Z + HALO

    ct_image = nib.load(CT_PATH)
    gt_image = nib.load(GT_PATH)
    comp = (slice(cx0, cx1), slice(cy0, cy1), slice(cz0, cz1))
    ct = np.asarray(ct_image.dataobj[comp], dtype=np.float32)
    gt = np.asarray(gt_image.dataobj[comp]) > 0

    with np.load(PROBABILITIES) as archive:
        probabilities = archive["probabilities"]  # (class, z, y, x)
        probability = np.array(
            probabilities[1, cz0:cz1, cy0:cy1, cx0:cx1].transpose(2, 1, 0),
            dtype=np.float32,
            copy=True,
        )
    if probability.shape != ct.shape:
        raise ValueError(f"probability {probability.shape} != CT {ct.shape}")
    return {
        "ct": ct,
        "gt": gt,
        "probability": probability,
        "zooms": tuple(float(v) for v in ct_image.header.get_zooms()[:3]),
    }


def _morphology(probability_xyz: np.ndarray) -> dict[str, object]:
    """Erosion, opening and residual at every depth, from the trainer's own code."""
    zyx = np.ascontiguousarray(probability_xyz.transpose(2, 1, 0))
    tensor = torch.from_numpy(zyx)[None, None]

    eroded: dict[int, torch.Tensor] = {0: tensor}
    opened: dict[int, torch.Tensor] = {}
    residual: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        current = tensor
        for depth in range(0, 11):
            if depth > 0:
                current = _soft_erode3d(current)
                eroded[depth] = current
            open_d = _soft_open3d(current)
            opened[depth] = open_d
            residual[depth] = torch.relu(current - open_d)
        skeleton = _soft_skeleton3d(tensor, iterations=10)

        # Guard: the figure must not drift from the trainer's accumulation.
        rebuilt = residual[0]
        for depth in range(1, 11):
            delta = residual[depth]
            rebuilt = rebuilt + torch.relu(delta - rebuilt * delta)
        drift = float((rebuilt - skeleton).abs().max())
        if drift > 1e-6:
            raise RuntimeError(f"accumulation drifted from _soft_skeleton3d by {drift:g}")

    def xyz(value: torch.Tensor) -> np.ndarray:
        return value[0, 0].numpy().transpose(2, 1, 0)

    return {
        "eroded": {d: xyz(v) for d, v in eroded.items()},
        "opened": {d: xyz(v) for d, v in opened.items()},
        "residual": {d: xyz(v) for d, v in residual.items()},
        "skeleton": xyz(skeleton),
        "drift": drift,
    }


def _project(volume_xyz: np.ndarray) -> np.ndarray:
    """Max projection through the displayed coronal slab, returned as (z, x)."""
    inner = volume_xyz[HALO:HALO + WIN_X, HALO:HALO + SLAB, HALO:HALO + WIN_Z]
    return np.max(inner, axis=1).T


def _extent(zooms: tuple[float, float, float]) -> tuple[float, float, float, float]:
    return (0.0, WIN_X * zooms[0], 0.0, WIN_Z * zooms[2])


def _panel(ax, image, extent, *, cmap="viridis", norm=None, vmin=None, vmax=None,
           mask_below=None, title=None):
    shown = np.ma.masked_less(image, mask_below) if mask_below is not None else image
    handle = ax.imshow(shown, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                       origin="lower", extent=extent, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=7.8, color=INK, pad=3.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    return handle


def _contour(ax, image, extent, *, colour, width, dashes=None):
    """Draw one 0.5 iso-line; dashes distinguish the eroded outline from the input."""
    # Matplotlib >= 3.8 dropped ContourSet.collections, so set the style up front.
    linestyle = "solid" if dashes is None else (0, dashes)
    return ax.contour(image, levels=[0.5], colors=[colour], linewidths=width,
                      linestyles=[linestyle], origin="lower", extent=extent)


def _preview() -> None:
    PREVIEW_OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CANDIDATES), 4, figsize=(11.0, 2.55 * len(CANDIDATES)),
                             layout="constrained")
    for row, (name, window) in enumerate(sorted(CANDIDATES.items())):
        data = _load(window)
        morph = _morphology(data["probability"])
        extent = _extent(data["zooms"])
        x0 = _project(morph["eroded"][0])
        _panel(axes[row, 0], x0, extent, vmin=0, vmax=1, title=f"{name}: $X^{{(0)}}$")
        _panel(axes[row, 1], _project(morph["eroded"][1]), extent, vmin=0, vmax=1,
               title="$\\mathcal{E}(X^{(0)})$")
        _panel(axes[row, 2], _project(morph["opened"][0]), extent, vmin=0, vmax=1,
               title="$\\mathcal{O}(X^{(0)})$")
        _panel(axes[row, 3], _project(morph["skeleton"]), extent, cmap="inferno",
               vmin=0, vmax=1, title="$S(X)$")
    path = PREVIEW_OUT / "soft_skeleton_window_preview.png"
    fig.savefig(path, dpi=105)
    print("wrote", path)


def main() -> None:
    args = _parse_args()
    if args.preview:
        _preview()
        return

    window = CANDIDATES[args.window]
    data = _load(window)
    morph = _morphology(data["probability"])
    extent = _extent(data["zooms"])
    zooms = data["zooms"]

    # depth_b is deliberately the stated K: on this patch nothing survives it, and the
    # panel says so rather than pretending the ceiling is a working depth.
    depth_a, depth_b = 3, 10
    survival = [
        float((morph["eroded"][d][HALO:HALO + WIN_X, HALO:HALO + SLAB,
                                  HALO:HALO + WIN_Z] > 0.5).mean())
        for d in range(11)
    ]
    exhausted = next((d for d, v in enumerate(survival) if v == 0.0), None)

    gt_projection = _project(data["gt"].astype(np.float32))
    x0 = _project(morph["eroded"][0])
    e0 = _project(morph["eroded"][1])
    o0 = _project(morph["opened"][0])
    d0 = _project(morph["residual"][0])
    xa = _project(morph["eroded"][depth_a])
    xb = _project(morph["eroded"][depth_b])
    skeleton = _project(morph["skeleton"])

    evidence_norm = colors.LogNorm(vmin=1e-3, vmax=1.0)

    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.9), layout="constrained")

    # --- row 1: one complete iteration ------------------------------------------
    prob = _panel(axes[0, 0], x0, extent, vmin=0, vmax=1, title="$X^{(0)}$: teacher probability")
    _panel(axes[0, 1], e0, extent, vmin=0, vmax=1, title="erosion $\\mathcal{E}(X^{(0)})$")
    _panel(axes[0, 2], o0, extent, vmin=0, vmax=1,
           title="opening $\\mathcal{O}(X^{(0)})=\\mathcal{D}(\\mathcal{E}(X^{(0)}))$")
    res = _panel(axes[0, 3], d0, extent, cmap="inferno", norm=evidence_norm, mask_below=1e-3,
                 title="residual $\\delta^{(0)}$")
    axes[0, 3].set_facecolor(DARK)

    # Contours are what make the dilation legible.  On the erosion panel the solid line
    # is where the map started, so the shrinkage is visible; on the opening panel the
    # dashed line is where erosion left it, so the regrowth back out towards the solid
    # line -- and the distal branch it never reaches -- can be read directly.
    _contour(axes[0, 1], x0, extent, colour="#ffffff", width=1.15)
    _contour(axes[0, 2], x0, extent, colour="#ffffff", width=1.15)
    _contour(axes[0, 2], e0, extent, colour=CYAN, width=0.95, dashes=(2.6, 1.5))
    axes[0, 2].legend(handles=[
        plt.Line2D([], [], color="#ffffff", lw=1.15, label="$X^{(0)}$"),
        plt.Line2D([], [], color=CYAN, lw=0.95, ls=(0, (2.6, 1.5)), label="$\\mathcal{E}(X^{(0)})$"),
    ], loc="lower left", fontsize=5.6, frameon=True, framealpha=0.82, handlelength=1.5,
        borderpad=0.28, labelspacing=0.2)

    # --- row 2: repetition and accumulation -------------------------------------
    _panel(axes[1, 0], xa, extent, vmin=0, vmax=1, title=f"$X^{{({depth_a})}}$")
    _panel(axes[1, 1], xb, extent, vmin=0, vmax=1, title=f"$X^{{({depth_b})}}$")
    # X^(10) is empty for this case: the largest airway radius is 7 voxels, so nothing
    # survives ten erosions.  Draw where the airway was, and say so, rather than
    # printing a blank square that reads as a broken panel.
    _contour(axes[1, 1], x0, extent, colour="#8aa0b5", width=0.8, dashes=(2.2, 1.4))
    if survival[depth_b] == 0.0:
        axes[1, 1].text(0.5, 0.5, f"empty: no support\nsurvives past $k={exhausted - 1}$",
                        transform=axes[1, 1].transAxes, ha="center", va="center",
                        fontsize=6.4, color="#e6edf3",
                        bbox=dict(facecolor="#243447", edgecolor="none", alpha=0.9, pad=2.2))

    axes[1, 2].set_facecolor(DARK)
    depth_colours = {0: "#fde725", 2: "#35b779", 4: "#3b8bd0"}
    for depth in (4, 2, 0):
        layer = _project(morph["residual"][depth])
        axes[1, 2].contourf(layer, levels=[1e-3, 1.0], colors=[depth_colours[depth]],
                            origin="lower", extent=extent, alpha=0.95)
    axes[1, 2].set_title("residuals at $k=0,2,4$", fontsize=7.8, color=INK, pad=3.0)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_xlim(extent[0], extent[1])
    axes[1, 2].set_ylim(extent[2], extent[3])
    for spine in axes[1, 2].spines.values():
        spine.set_color("#cbd5e1")
    axes[1, 2].legend(handles=[
        plt.Line2D([], [], color=depth_colours[d], lw=4.2, label=f"$k={d}$") for d in (0, 2, 4)
    ], loc="lower left", fontsize=5.8, frameon=True, framealpha=0.86, handlelength=1.0,
        borderpad=0.28, labelspacing=0.2)

    _panel(axes[1, 3], skeleton, extent, cmap="inferno", norm=evidence_norm,
           mask_below=1e-3, title="soft skeleton $S(X)$")
    axes[1, 3].set_facecolor(DARK)
    axes[1, 3].contour(gt_projection, levels=[0.5], colors=[GREEN], linewidths=0.55,
                       origin="lower", extent=extent, alpha=0.75)

    # Physical axes on the outer panels only, so the grid stays uncluttered.
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel("inferior--superior (mm)", fontsize=6.8, color=MUTED)
        ax.set_yticks([0, WIN_Z * zooms[2] / 2, WIN_Z * zooms[2]])
        ax.set_yticklabels(["0", f"{WIN_Z * zooms[2] / 2:.0f}", f"{WIN_Z * zooms[2]:.0f}"],
                           fontsize=6.2, color=MUTED)
    for ax in axes[1, :]:
        ax.set_xlabel("left--right (mm)", fontsize=6.8, color=MUTED)
        ax.set_xticks([0, WIN_X * zooms[0] / 2, WIN_X * zooms[0]])
        ax.set_xticklabels(["0", f"{WIN_X * zooms[0] / 2:.0f}", f"{WIN_X * zooms[0]:.0f}"],
                           fontsize=6.2, color=MUTED)

    prob_bar = fig.colorbar(prob, ax=axes[0, :3].tolist(), location="bottom",
                            fraction=0.040, pad=0.030, aspect=44)
    prob_bar.set_label("teacher probability", fontsize=6.8, color=MUTED)
    prob_bar.ax.tick_params(labelsize=6.0, colors=MUTED)
    res_bar = fig.colorbar(res, ax=[axes[0, 3], axes[1, 3]], location="right",
                           fraction=0.030, pad=0.014)
    res_bar.set_label("centreline evidence (log)", fontsize=6.8, color=MUTED)
    res_bar.ax.tick_params(labelsize=6.0, colors=MUTED)

    # Operator arrows between the top-row panels, so the row reads as a process.
    # Placed after the layout has settled, in figure coordinates.
    fig.canvas.draw()
    labels = ("$\\mathcal{E}$", "$\\mathcal{D}$", "$X\\!-\\!\\mathcal{O}$")
    for index, label in enumerate(labels):
        left = axes[0, index].get_position()
        right = axes[0, index + 1].get_position()
        y = (left.y0 + left.y1) / 2
        gap_lo, gap_hi = left.x1, right.x0
        pad = (gap_hi - gap_lo) * 0.18
        arrow = FancyArrowPatch(
            (gap_lo + pad, y), (gap_hi - pad, y),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8.5,
            linewidth=0.9, color=INK, shrinkA=0, shrinkB=0, zorder=5,
        )
        fig.add_artist(arrow)
        fig.text((gap_lo + gap_hi) / 2, y + 0.030, label, ha="center", va="bottom",
                 fontsize=7.2, color=INK, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.0))

    PDF_OUT.mkdir(parents=True, exist_ok=True)
    PNG_OUT.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    stem = "soft_skeleton_iteration"
    fig.savefig(PDF_OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PNG_OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    print("wrote", PDF_OUT / f"{stem}.pdf")

    (PROVENANCE / f"{stem}.json").write_text(json.dumps({
        "case": CASE,
        "probabilities": str(PROBABILITIES.relative_to(ROOT)),
        "window_name": args.window,
        "window_xyz": list(window),
        "display_voxels": [WIN_X, SLAB, WIN_Z],
        "halo_voxels": HALO,
        "iterations": 10,
        "depths_shown": [0, depth_a, depth_b],
        "residual_depths_shown": [0, 2, 4],
        "fraction_above_0p5_by_depth": [round(v, 5) for v in survival],
        "first_depth_with_no_foreground": exhausted,
        "voxel_spacing_mm": list(zooms),
        "morphology_source": "nnunet_trainers/nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py",
        "operators": {
            "erosion": "minimum over centre plus six face-connected neighbours",
            "dilation": "3x3x3 maximum pool",
            "opening": "dilation of erosion",
            "residual": "relu(current - opening(current))",
            "accumulation": "S <- S + relu(delta - S*delta)",
        },
        "accumulation_check_max_abs_drift_from_trainer": morph["drift"],
        "note": "Held-out inference crop, not a logged training batch.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
