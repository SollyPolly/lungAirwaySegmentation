"""Generate implementation-matched dissertation figures for clDice.

The figures deliberately use a 2-D toy airway for legibility, while the labels state
where the production implementation operates in 3-D.  The morphology below mirrors
the repository's clDice pooling semantics on a single slice:

* erosion: minimum over the centre and four face neighbours (the 3-D code adds the
  two depth neighbours, giving a seven-point cross);
* opening: erosion followed by a full 3x3 dilation (3x3x3 in production);
* skeleton: residuals from successive erosions accumulated with a soft OR.

Run from the repository root:

    python scripts/generate_cldice_explainer.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    maximum_filter,
    minimum_filter1d,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "dissertation" / "Figures"

INK = "#0f172a"
MUTED = "#475569"
BLUE = "#2563eb"
TEAL = "#0f766e"
GREEN = "#15803d"
ORANGE = "#c2410c"
PURPLE = "#7e22ce"
RED = "#b91c1c"
PANEL = "#f8fafc"


def _save(fig: plt.Figure, stem: str, category: str = "appendix") -> None:
    pdf_dir = FIGURE_ROOT / "pdf" / category
    png_dir = FIGURE_ROOT / "png" / category
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(png_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_panel(fig: plt.Figure, stem: str, category: str) -> None:
    """Save a title-free panel; LaTeX supplies its letter and subcaption."""
    pdf_dir = FIGURE_ROOT / "pdf" / category / "panels"
    png_dir = FIGURE_ROOT / "png" / category / "panels"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(png_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def _toy_airway(shape: tuple[int, int] = (125, 145)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centreline masks for the trunk, strong branches, and a faint branch."""
    yy, xx = np.indices(shape)
    trunk = (np.abs(xx - 72) <= 1) & (yy >= 10) & (yy <= 101)
    left = (np.abs(yy - (-0.72 * (xx - 72) + 55)) <= 1.1) & (xx >= 27) & (xx <= 73) & (yy <= 91)
    right = (np.abs(yy - (0.68 * (xx - 72) + 55)) <= 1.1) & (xx >= 71) & (xx <= 119) & (yy <= 92)
    left2 = (np.abs(yy - (0.50 * (xx - 48) + 55)) <= 1.0) & (xx >= 25) & (xx <= 49) & (yy <= 72)
    faint = (np.abs(yy - (-0.28 * (xx - 105) + 79)) <= 1.0) & (xx >= 101) & (xx <= 136) & (yy >= 65)
    return trunk, (left | right | left2), faint


def _probability_map(include_faint: bool = True) -> np.ndarray:
    trunk, branches, faint = _toy_airway()
    strong_line = trunk | branches
    strong_distance = distance_transform_edt(~strong_line)
    # A high-confidence tube with a flat-ish core and a graded boundary.
    probability = 0.03 + 0.94 / (1.0 + np.exp((strong_distance - 6.0) / 0.9))
    if include_faint:
        faint_distance = distance_transform_edt(~faint)
        # Its centre stays strictly below 0.5, so thresholding removes it.
        faint_probability = 0.03 + 0.45 / (1.0 + np.exp((faint_distance - 3.5) / 0.8))
        probability = np.maximum(probability, faint_probability)
    # A tiny smooth nuisance ridge illustrates that soft skeleton mass need not be
    # exactly zero everywhere outside the anatomical airway.
    yy, xx = np.indices(probability.shape)
    nuisance = 0.035 * np.exp(-((xx - 19) ** 2 + (yy - 108) ** 2) / 90.0)
    probability = np.maximum(probability, 0.03 + nuisance)
    return np.clip(probability, 0.0, 1.0)


def _soft_erode2d(image: np.ndarray) -> np.ndarray:
    """2-D analogue of the production three-direction 3-D min-pool erosion."""
    along_y = minimum_filter1d(image, size=3, axis=0, mode="nearest")
    along_x = minimum_filter1d(image, size=3, axis=1, mode="nearest")
    return np.minimum(along_y, along_x)


def _soft_open2d(image: np.ndarray) -> np.ndarray:
    return maximum_filter(_soft_erode2d(image), size=3, mode="nearest")


def _soft_skeleton2d(image: np.ndarray, iterations: int = 10) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    current = image.copy()
    delta = np.maximum(current - _soft_open2d(current), 0.0)
    skeleton = delta.copy()
    eroded = [current.copy()]
    residuals = [delta.copy()]
    for _ in range(iterations):
        current = _soft_erode2d(current)
        delta = np.maximum(current - _soft_open2d(current), 0.0)
        skeleton = skeleton + np.maximum(delta - skeleton * delta, 0.0)
        eroded.append(current.copy())
        residuals.append(delta.copy())
    return skeleton, eroded, residuals


def _heat(ax: plt.Axes, image: np.ndarray, title: str, *, cmap="viridis", vmin=0.0, vmax=1.0) -> None:
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11, color=INK, weight="bold", pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(1.0)


def _box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], title: str, body: str,
         *, edge=BLUE, fill="#eff6ff", title_size=11, body_size=9.3) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                           linewidth=1.7, edgecolor=edge, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + 0.025 * w, y + h - 0.23 * h, title, color=INK, fontsize=title_size,
            fontweight="bold", va="center")
    ax.text(x + 0.025 * w, y + h - 0.50 * h, body, color=MUTED, fontsize=body_size,
            va="top", linespacing=1.25)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color=INK,
           dashed=False, label: str | None = None, label_offset=(0.0, 0.0), width=1.8) -> None:
    style = "--" if dashed else "-"
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                            linewidth=width, linestyle=style, color=color)
    ax.add_patch(arrow)
    if label:
        mx = 0.5 * (start[0] + end[0]) + label_offset[0]
        my = 0.5 * (start[1] + end[1]) + label_offset[1]
        ax.text(mx, my, label, color=color, fontsize=8.8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9))


def _curved_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color=INK,
                  label: str | None = None, label_xy: tuple[float, float] | None = None,
                  radius: float = -0.22) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                            linewidth=1.8, color=color,
                            connectionstyle=f"arc3,rad={radius}")
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(*label_xy, label, color=color, fontsize=8.8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92))


def figure_soft_morphology() -> None:
    probability = _probability_map()
    erosion = _soft_erode2d(probability)
    opening = _soft_open2d(probability)
    residual = np.maximum(probability - opening, 0.0)
    skeleton, eroded, residuals = _soft_skeleton2d(probability, 10)

    fig = plt.figure(figsize=(14.5, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=(1.05, 1.05, 0.42))
    fig.suptitle("How soft skeletonisation is built from local pooling", fontsize=18,
                 fontweight="bold", color=INK)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    _heat(axes[0], probability, "1  Airway probability  x")
    _heat(axes[1], erosion, "2  Soft erosion  E(x)")
    _heat(axes[2], opening, "3  Opening  O(x)=dilate(E(x))")
    _heat(axes[3], residual, "4  Ridge residual at depth 0", cmap="magma", vmax=0.15)

    axes2 = [fig.add_subplot(gs[1, i]) for i in range(4)]
    _heat(axes2[0], eroded[0], "Erosion depth 0")
    _heat(axes2[1], eroded[3], "After 3 erosions")
    _heat(axes2[2], eroded[7], "After 7 erosions")
    _heat(axes2[3], skeleton, "Accumulated soft skeleton", cmap="magma")

    text_ax = fig.add_subplot(gs[2, :])
    text_ax.axis("off")
    text_ax.text(0.01, 0.82,
                 "Erosion is a local minimum, implemented as −max-pool(−x): kernels 3×1×1, 1×3×1 and 1×1×3 are merged by a minimum.",
                 fontsize=10.2, color=INK, weight="bold")
    text_ax.text(0.01, 0.46,
                 "Opening asks what survives erosion and regrowth. Subtracting it exposes thin ridge/centreline evidence; repeating on successively eroded maps exposes thicker tubes.",
                 fontsize=9.8, color=MUTED)
    text_ax.text(0.01, 0.10,
                 "Production: 10 between-scale erosions, residuals at depths 0…10. These are fixed min/max pools—not learned convolutions and not a distance cutoff.",
                 fontsize=9.8, color=RED, weight="bold")
    _save(fig, "soft_skeletonisation_full", "appendix")


def figure_single_erosion_step() -> None:
    """Show the exact min-pool/opening/residual arithmetic on a 1-D ridge."""
    x = np.array([0.0, 0.2, 0.8, 0.9, 0.8, 0.2, 0.0])
    eroded = np.array([0.0, 0.0, 0.2, 0.8, 0.2, 0.0, 0.0])
    opened = np.array([0.0, 0.2, 0.8, 0.8, 0.8, 0.2, 0.0])
    residual = np.maximum(x - opened, 0.0)
    arrays = [x, eroded, opened, residual]
    titles = [
        "1  Input probability ridge  x",
        "2  Erosion: local minimum",
        "3  Opening: dilate the erosion",
        "4  Residual: ReLU(x−opening)",
    ]
    colours = [BLUE, TEAL, GREEN, ORANGE]

    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.8), sharey=True)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.78, bottom=0.23, wspace=0.16)
    fig.suptitle("One soft-erosion/opening step, with actual values", fontsize=18,
                 fontweight="bold", color=INK, y=0.96)
    for ax, values, title, colour in zip(axes, arrays, titles, colours):
        ax.bar(np.arange(values.size), values, color=colour, width=0.78)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(values.size))
        ax.set_xticklabels([f"{v:g}" for v in values], fontsize=8.5)
        ax.set_title(title, fontsize=10.5, weight="bold", color=INK, pad=9)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    axes[0].set_ylabel("value", fontsize=9.5, color=MUTED)
    axes[1].annotate("centre output = min(0.8, 0.9, 0.8) = 0.8",
                     xy=(3, 0.8), xytext=(3, 1.0), ha="center", fontsize=8.2, color=TEAL,
                     arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
    axes[3].annotate("only the centre ridge remains: 0.9−0.8=0.1",
                     xy=(3, 0.1), xytext=(3, 0.55), ha="center", fontsize=8.2, color=ORANGE,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    fig.text(0.5, 0.105,
             "In 3-D, this minimum is evaluated along depth, height and width, then the three results are merged by another minimum.",
             ha="center", fontsize=9.8, color=INK, weight="bold")
    fig.text(0.5, 0.055,
             "PyTorch implements the local minimum as −max_pool3d(−x). It behaves like a fixed morphological filter, not a learned convolution.",
             ha="center", fontsize=9.5, color=RED)
    _save(fig, "soft_erosion_one_step", "methods/cldice")


def figure_cldice_directions() -> None:
    target = _probability_map(include_faint=False) > 0.5
    target_skel, _, _ = _soft_skeleton2d(target.astype(float), 10)

    prediction_extra = target.copy()
    yy, xx = np.indices(target.shape)
    extra_line = (np.abs(yy - (-0.55 * (xx - 72) + 85)) <= 1.1) & (xx >= 65) & (xx <= 112)
    prediction_extra |= binary_dilation(extra_line, iterations=4)

    prediction_missing = target.copy()
    prediction_missing[(xx > 102) & (yy > 67)] = False

    pred_extra_skel, _, _ = _soft_skeleton2d(prediction_extra.astype(float), 10)
    pred_missing_skel, _, _ = _soft_skeleton2d(prediction_missing.astype(float), 10)

    def overlay(pred: np.ndarray, pred_skel: np.ndarray) -> np.ndarray:
        rgb = np.ones((*target.shape, 3), dtype=float)
        rgb[target] = np.array([0.80, 0.88, 1.00])
        rgb[pred] = np.minimum(rgb[pred], np.array([0.82, 0.97, 0.88]))
        reference_line = target_skel > 0.03
        predicted_line = pred_skel > 0.03
        rgb[reference_line & ~predicted_line] = np.array([0.10, 0.38, 0.78])
        rgb[predicted_line & ~reference_line] = np.array([0.85, 0.20, 0.13])
        rgb[reference_line & predicted_line] = np.array([0.42, 0.18, 0.62])
        return rgb

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.90, bottom=0.075, hspace=0.24, wspace=0.12)
    fig.suptitle("What the two directions of clDice measure", fontsize=18, fontweight="bold", color=INK)
    _heat(axes[0, 0], target.astype(float), "Reference airway mask", cmap="Blues")
    _heat(axes[0, 1], prediction_extra.astype(float), "Prediction with an extra branch", cmap="Greens")
    axes[0, 2].imshow(overlay(prediction_extra, pred_extra_skel))
    axes[0, 2].set_title("Topology precision falls", fontsize=11, color=INK, weight="bold")
    axes[0, 2].axis("off")
    axes[0, 2].text(0.5, 0.025, "How much predicted centreline lies inside the reference mask?",
                    transform=axes[0, 2].transAxes, ha="center", fontsize=8.8, color=MUTED,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9))

    _heat(axes[1, 0], target.astype(float), "Reference airway mask", cmap="Blues")
    _heat(axes[1, 1], prediction_missing.astype(float), "Prediction missing a branch", cmap="Greens")
    axes[1, 2].imshow(overlay(prediction_missing, pred_missing_skel))
    axes[1, 2].set_title("Topology sensitivity falls", fontsize=11, color=INK, weight="bold")
    axes[1, 2].axis("off")
    axes[1, 2].text(0.5, 0.025, "How much reference centreline is covered by the prediction mask?",
                    transform=axes[1, 2].transAxes, ha="center", fontsize=8.8, color=MUTED,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.9))

    fig.text(0.5, 0.022,
             "clDice is their harmonic mean. Purple = shared centreline; red = predicted-only; blue = reference-only.",
             ha="center", fontsize=10, color=INK, weight="bold")
    _save(fig, "cldice_definition", "background")


def figure_cldice_direction_panels() -> None:
    """Export the two clDice failure modes as independent, title-free panels."""
    target = _probability_map(include_faint=False) > 0.5
    target_skel, _, _ = _soft_skeleton2d(target.astype(float), 10)
    yy, xx = np.indices(target.shape)

    prediction_extra = target.copy()
    extra_line = (np.abs(yy - (-0.55 * (xx - 72) + 85)) <= 1.1) & (xx >= 65) & (xx <= 112)
    prediction_extra |= binary_dilation(extra_line, iterations=4)
    prediction_missing = target.copy()
    prediction_missing[(xx > 102) & (yy > 67)] = False

    cases = (
        ("cldice_topology_precision", prediction_extra),
        ("cldice_topology_sensitivity", prediction_missing),
    )
    for stem, prediction in cases:
        predicted_skeleton, _, _ = _soft_skeleton2d(prediction.astype(float), 10)
        rgb = np.ones((*target.shape, 3), dtype=float)
        rgb[target] = np.array([0.82, 0.90, 1.00])
        rgb[prediction] = np.minimum(rgb[prediction], np.array([0.82, 0.97, 0.88]))
        reference_line = target_skel > 0.03
        predicted_line = predicted_skeleton > 0.03
        rgb[reference_line & ~predicted_line] = np.array([0.10, 0.38, 0.78])
        rgb[predicted_line & ~reference_line] = np.array([0.85, 0.20, 0.13])
        rgb[reference_line & predicted_line] = np.array([0.42, 0.18, 0.62])

        fig, ax = plt.subplots(figsize=(3.20, 2.65), layout="constrained")
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.8)
        _save_panel(fig, stem, "background")


def figure_hard_vs_soft_teacher() -> None:
    teacher = _probability_map(include_faint=True)
    hard = (teacher > 0.5).astype(float)
    soft_skel, _, _ = _soft_skeleton2d(teacher, 10)
    hard_skel, _, _ = _soft_skeleton2d(hard, 10)

    fig = plt.figure(figsize=(14.4, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=(1.0, 0.72))
    fig.suptitle("Mean-Teacher clDice: thresholded teacher target versus probability target",
                 fontsize=18, fontweight="bold", color=INK)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    _heat(axes[0], teacher, "Teacher probability  p_T")
    _heat(axes[1], hard, "Historical target  1[p_T>0.5]", cmap="gray")
    _heat(axes[2], hard_skel, "Soft skeleton of hard target", cmap="magma")
    _heat(axes[3], soft_skel, "Soft skeleton of p_T directly", cmap="magma")

    for ax in (axes[0], axes[3]):
        ax.annotate("sub-threshold branch\npeaks below 0.5", xy=(124, 78), xytext=(93, 112),
                    color=RED, fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=RED, alpha=0.9))

    hard_ax = fig.add_subplot(gs[1, :2])
    hard_ax.axis("off")
    _box(hard_ax, (0.02, 0.12), (0.96, 0.76), "A  Hard-teacher-target soft clDice",
         "Teacher: detached, thresholded at p_T > 0.5, then skeletonised.\n"
         "Student: remains a probability map and is soft-skeletonised.\n"
         "Consequence: a branch at p_T=0.49 is completely absent from the target.",
         edge=ORANGE, fill="#fff7ed", title_size=12, body_size=10.2)
    soft_ax = fig.add_subplot(gs[1, 2:])
    soft_ax.axis("off")
    _box(soft_ax, (0.02, 0.12), (0.96, 0.76), "B  Probability-target soft clDice",
         "Teacher: detached but never thresholded; graded evidence is skeletonised.\n"
         "Student: same differentiable probability path as before.\n"
         "Consequence: p_T=0.49 still contributes continuously, but more weakly than p_T=0.9.",
         edge=PURPLE, fill="#faf5ff", title_size=12, body_size=10.2)

    fig.text(0.5, 0.012,
             "‘Hard’ refers only to the teacher target. Neither training variant is the fully binary, non-differentiable hard-clDice evaluation metric.",
             ha="center", fontsize=9.8, color=RED, weight="bold")
    _save(fig, "synthetic_teacher_target_comparison", "appendix")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlelocation": "center",
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure_cldice_direction_panels()


if __name__ == "__main__":
    main()
