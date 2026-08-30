"""Shared seaborn theme for every statistical figure in the dissertation.

One place decides typography, spines, palette and transparency, so a colour means
the same thing in every chapter and a restyle is a one-file change.

Design decisions, so they are not re-litigated per script:

* ``style="ticks"`` rather than a grid style. The figures are small and dense; a
  background grid competes with the error bands.
* **Hard axes.** Seaborn's defaults leave pale spines. Left and bottom are drawn
  solid in ink at full weight, top and right removed, so the data area is framed
  by two definite lines rather than a faint box.
* **Grey carries the reference, colour carries the treatment.** The control is
  neutral grey; only arms under test take saturated colour, and every supporting
  element (bands, guides, annotations) is grey at low alpha. That keeps the eye on
  the contrast the section is about.
* Error bands are drawn at ``BAND_ALPHA`` with no edge, so overlapping arms stay
  readable where they cross.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Ink and greys. Shared with the TikZ figures so the printed page is coherent.
INK = "#0f172a"
MUTED = "#475569"
GREY = "#94a3b8"
FAINT = "#cbd5e1"

# Scaffolding red: callout boxes, magnification frames, anything pointing AT the data
# rather than being data. Deliberately not one of the Okabe-Ito arm colours below, and
# in particular not the vermillion #D55E00 that means "thresholded target" two figures
# later, so an annotation can never be read as an arm.
ANNOTATION_RED = "#CC0000"

# Okabe-Ito, the standard palette designed to stay distinguishable under
# deuteranopia, protanopia and tritanopia. Used at full saturation: the muting is
# done by alpha on the error bands, not by desaturating the lines, so the arms stay
# separable in print and the bands stay quiet.
#
# Assignment maximises separation between the two thresholded arms, which are the
# pair most at risk of reading alike: vermillion against bluish green rather than
# two warm tones. Control is grey because it is the reference, not a result.
# The two supervised scale references share a hue and differ in lightness, because they
# are the same KIND of thing at two label counts; the objective ablation takes the one
# remaining warm Okabe-Ito tone, and the labelled-only seed the remaining cool one.
ARM_COLOUR: dict[str, str] = {
    "control": "#949494",
    "mt_soft": "#0072B2",
    "mt_soft_5f": "#009E73",
    "mt_hard_f0": "#D55E00",
    "mt_hard_5f": "#009E73",
    "mt_mse": "#E69F00",
    "seed": "#56B4E9",  # analysis-JSON alias
    "seed16": "#56B4E9",
    "ceiling110": "#CC79A7",
    "ceiling260": "#8E4E77",
}
ARM_LABEL: dict[str, str] = {
    "control": "Control (no consistency)",
    "mt_soft": "Mean Teacher, fold 0",
    "mt_soft_5f": "Mean Teacher, 5-fold",
    "mt_hard_f0": "Thresholded target",
    "mt_hard_5f": "Thresholded, 5-fold",
    "mt_mse": "Voxel-MSE consistency",
    "seed": "16-label seed",  # analysis-JSON alias
    "seed16": "16-label seed",
    "ceiling110": "Supervised scale reference",
    "ceiling260": "Supervised scale reference, 260",
}
# Dash patterns are a second channel, so the arms remain separable in greyscale and
# for readers who cannot rely on hue at all. No two arms share one.
ARM_DASH: dict[str, tuple] = {
    "control": (),
    "mt_soft": (),
    "mt_soft_5f": (5.5, 1.4, 1.2, 1.4),
    "mt_hard_f0": (4.0, 1.6),
    "mt_hard_5f": (5.5, 1.4, 1.2, 1.4),
    "seed": (3.0, 1.3),
    "seed16": (3.0, 1.3),
    "ceiling110": (1.4, 1.4),
}
ARM_MARKER: dict[str, str] = {
    "control": "o",
    "mt_soft": "s",
    "mt_soft_5f": "v",
    "mt_hard_f0": "^",
    "mt_hard_5f": "v",
    "seed": "P",
    "seed16": "P",
    "ceiling110": "D",
}

# Five overlapping bootstrap bands turn muddy fast; this is as light as it can go
# and still read as a band on a printed page.
BAND_ALPHA = 0.10
TITLE_PT = 9.0
LABEL_PT = 8.5
TICK_PT = 7.6
LEGEND_PT = 7.0
ANNOTATION_PT = 6.8


def apply_theme() -> None:
    """Install the theme. Call once, before creating any figure."""
    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK,
            "axes.labelsize": LABEL_PT,
            "axes.titlesize": TITLE_PT,
            "axes.titlecolor": INK,
            "text.color": INK,
            # Ticks and their values match the spines. Slate ticks against an ink
            # spine read as two different axes rather than one.
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": TICK_PT,
            "ytick.labelsize": TICK_PT,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": LEGEND_PT,
            "legend.frameon": False,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.4,
            "font.family": "sans-serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def finish(axis: plt.Axes) -> None:
    """Hard left and bottom axes, no top or right.

    Spines, tick marks and tick values are all set here rather than left to the rc
    defaults, so the two axes cannot drift apart when a caller overrides one of them
    with its own ``tick_params``.
    """
    sns.despine(ax=axis, top=True, right=True)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(INK)
        axis.spines[name].set_linewidth(0.9)
    axis.tick_params(axis="both", colors=INK, labelcolor=INK)


def panel_label(axis: plt.Axes, letter: str, x: float = -0.13, y: float = 1.03) -> None:
    """Panel letter outside the data area, above the top-left corner."""
    axis.text(x, y, letter, transform=axis.transAxes, fontsize=LABEL_PT + 1,
              fontweight="bold", va="bottom", ha="left", color=INK, clip_on=False)


def guide_line(axis: plt.Axes, y: float = 0.0) -> None:
    """The zero reference on a difference panel.

    Drawn dark and dashed, above the error bands rather than under them: on a
    difference plot the question is which side of zero a line sits, so the reader has
    to be able to find zero through a stack of translucent bands. Dashed keeps it
    reading as a reference rather than as a sixth arm.
    """
    axis.axhline(y, color=INK, linewidth=1.0, alpha=0.75, zorder=2.5,
                 dashes=(3.5, 2.0))


def broken_axis_marker(axis: plt.Axes, at: float = 0.045, size: float = 0.016) -> None:
    """The double-slash break on the y-axis, for a panel that does not start at zero.

    ``at`` is the height up the left spine, in axes fraction, and ``size`` the half
    length of each slash. A short white segment is laid over the spine first so the
    slashes read as an interruption in the axis rather than as decoration next to it.
    """
    kwargs = dict(transform=axis.transAxes, clip_on=False, zorder=6,
                  solid_capstyle="butt")
    axis.plot([0, 0], [at - size * 1.4, at + size * 1.4], color="white",
              linewidth=2.6, **kwargs)
    for offset in (-size * 0.9, size * 0.9):
        axis.plot([-size * 0.8, size * 0.8],
                  [at + offset - size, at + offset + size],
                  color=INK, linewidth=0.9, **kwargs)


def arm_palette(arms) -> dict[str, str]:
    return {a: ARM_COLOUR.get(a, GREY) for a in arms}


# Continuous ramp over branch depth, for figures that colour the tree by position
# rather than by arm. Built from Okabe-Ito stops in hue order, so it inherits the
# palette's colour-vision safety and lands on the same blue and vermillion the
# class-imbalance figure uses for proximal and distal. Sequential in hue, not in
# lightness alone, because it is read against a white page at small marker sizes.
DEPTH_RAMP = LinearSegmentedColormap.from_list(
    "airway_depth", ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
)
