"""Build the Discussion's implementation-to-result mechanism diagram.

The diagram is deliberately a synthesis rather than another performance plot.  It
combines the validation-set thickness census with the implementation-matched
soft-skeleton probe to state the mechanistic hypothesis tested by the calibre and
branch-depth results.  Solid arrows connect definitions or measurements; the final
dashed arrow marks interpretation rather than causal identification.

Run from the repository root::

    .venv\Scripts\python.exe dissertation\scripts\plot_discussion_mechanism.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_theme import INK, MUTED, apply_theme  # noqa: E402


THICKNESS_INPUT = (
    ROOT
    / "data"
    / "skeleton_scale_probe"
    / "results_thickness_val20"
    / "airway_thickness.json"
)
SKELETON_INPUT = (
    ROOT
    / "data"
    / "skeleton_scale_probe"
    / "results"
    / "skeleton_scale_probe.json"
)
PDF_OUTPUT = (
    ROOT
    / "dissertation"
    / "Figures"
    / "pdf"
    / "discussion"
    / "mean_teacher_mechanism.pdf"
)
PNG_OUTPUT = (
    ROOT
    / "dissertation"
    / "Figures"
    / "png"
    / "discussion"
    / "mean_teacher_mechanism.png"
)

BLUE = "#0072B2"
PURPLE = "#7E57C2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PALE_BLUE = "#EFF6FF"
PALE_PURPLE = "#F5F3FF"
PALE_ORANGE = "#FFF7ED"
PALE_GREEN = "#ECFDF5"


def _box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[tuple[str, str]],
    *,
    edge: str,
    fill: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=fill,
    )
    axis.add_patch(patch)
    axis.text(
        x + 0.04 * width,
        y + 0.80 * height,
        title,
        color=INK,
        fontsize=7.7,
        fontweight="bold",
        ha="left",
        va="center",
    )
    top = y + 0.50 * height
    spacing = 0.17 * height
    for index, (text, colour) in enumerate(lines):
        axis.text(
            x + 0.04 * width,
            top - index * spacing,
            text,
            color=colour,
            fontsize=6.8,
            ha="left",
            va="center",
        )


def _arrow(axis, start, end, *, dashed: bool = False, label: str | None = None) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.15,
        linestyle="--" if dashed else "-",
        color=MUTED,
        connectionstyle="arc3,rad=0.0",
    )
    axis.add_patch(arrow)
    if label:
        axis.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.055,
            label,
            fontsize=6.4,
            color=MUTED,
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )


def main() -> None:
    apply_theme()
    thickness = json.loads(THICKNESS_INPUT.read_text())
    skeleton = json.loads(SKELETON_INPUT.read_text())

    thin_volume = 100.0 * float(
        np.mean([row["volume__1-2"] for row in thickness["per_case"]])
    )
    thin_length = 100.0 * float(
        np.mean([row["length__1-2"] for row in thickness["per_case"]])
    )
    one_x = skeleton["aggregate"]["1x"]
    thin_foreground = 100.0 * one_x["fg_share__thick<=2"]
    thin_skeleton = 100.0 * one_x["skel_share__thick<=2"]
    over_representation = thin_skeleton / thin_foreground
    cases = len(skeleton["completed_cases"])
    patches = cases * int(skeleton["patches_per_case"])

    fig, axis = plt.subplots(figsize=(7.0, 2.75))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    width = 0.215
    height = 0.55
    y = 0.29
    xs = [0.015, 0.268, 0.521, 0.774]

    _box(
        axis,
        xs[0],
        y,
        width,
        height,
        "1  Sparse but\nextensive target",
        [
            ("1–2-voxel airways:", MUTED),
            (f"{thin_volume:.2f}% airway volume", BLUE),
            (f"{thin_length:.2f}% centreline length", BLUE),
        ],
        edge=BLUE,
        fill=PALE_BLUE,
    )
    _box(
        axis,
        xs[1],
        y,
        width,
        height,
        "2  Finite soft-skeleton\nedge case",
        [
            ("At ≤2 voxels: erosion(X) = 0", MUTED),
            ("depth-0 residual = X", PURPLE),
            ("object, not ideal centreline", PURPLE),
        ],
        edge=PURPLE,
        fill=PALE_PURPLE,
    )
    _box(
        axis,
        xs[2],
        y,
        width,
        height,
        "3  Effective loss\nallocation",
        [
            (f"{thin_foreground:.2f}% teacher foreground", MUTED),
            (f"{thin_skeleton:.2f}% skeleton mass", ORANGE),
            (f"{over_representation:.1f}× representation", ORANGE),
        ],
        edge=ORANGE,
        fill=PALE_ORANGE,
    )
    _box(
        axis,
        xs[3],
        y,
        width,
        height,
        "4  Mechanistic\nprediction",
        [
            ("Teacher-supported thin branches", MUTED),
            ("greater consistency pressure", GREEN),
            ("local gain; precision may fall", GREEN),
        ],
        edge=GREEN,
        fill=PALE_GREEN,
    )

    _arrow(axis, (xs[0] + width, 0.565), (xs[1], 0.565))
    _arrow(axis, (xs[1] + width, 0.565), (xs[2], 0.565), label="measured on teacher maps")
    _arrow(
        axis,
        (xs[2] + width, 0.565),
        (xs[3], 0.565),
        dashed=True,
        label="interpretation, not proof",
    )

    axis.text(
        0.5,
        0.175,
        "The EMA target can reinforce structure it represents; it cannot independently label an absent branch.",
        color=INK,
        fontsize=7.4,
        fontweight="bold",
        ha="center",
        va="center",
    )
    axis.text(
        0.5,
        0.09,
        f"Operator diagnostic: {patches} training-shaped foreground patches from {cases} final-teacher validation cases.",
        color=MUTED,
        fontsize=6.6,
        ha="center",
        va="center",
    )

    PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PNG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUTPUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(PNG_OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"Created: {PDF_OUTPUT}")
    print(f"Created: {PNG_OUTPUT}")


if __name__ == "__main__":
    main()
