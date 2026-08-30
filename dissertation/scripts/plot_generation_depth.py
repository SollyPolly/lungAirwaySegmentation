"""Figures and sanity overlays for the branch-depth recovery analysis.

Reads ``generation_depth_analysis.json`` written by ``measure_recall_by_generation.py`` and
regenerates every figure and table from the stored per-case rows, so nothing here needs the
image data. The one exception is ``--overlay-cases``, which re-parses those cases to render
the ground-truth skeleton coloured by depth as a visual check on the labelling.

Four figures, one per chapter role, because the report is capped at twenty:

  ``branch_depth_definition``     Methods: what the depth coordinate means on a real tree, and
                                  how much reference tree sits at each value of it. Panel (a)
                                  is the only output that needs the label images.
  ``generation_depth_recovery``   Results: validation and held-out-test TD by depth per arm,
                                  plus each cohort's paired fold-0 difference against control.
  ``generation_depth_recovery_aeropath``
                                  Appendix: the analogous exploratory AeroPath OOD profile.
  ``depth_calibre_heatmap``       Appendix: how far depth and calibre are the same axis, which
                                  is what stops the depth section reading as a restatement of
                                  the calibre one.

Usage, from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\plot_generation_depth.py
    .venv\\Scripts\\python.exe dissertation\\scripts\\plot_generation_depth.py --overlay-cases ATM_016
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_theme import (  # noqa: E402
    ANNOTATION_RED, ARM_DASH, ARM_LABEL as THEME_ARM_LABEL, ARM_MARKER, BAND_ALPHA,
    DEPTH_RAMP, GREY as THEME_GREY, LEGEND_PT as THEME_LEGEND_PT, apply_theme,
    arm_palette, broken_axis_marker, finish, guide_line, panel_label,
)

DEFAULT_INPUT = (
    ROOT / "data" / "skeleton_scale_probe" / "results_recall_by_generation_soft5f_seed"
    / "generation_depth_analysis.json"
)
DEFAULT_TEST_INPUT = (
    ROOT / "data" / "skeleton_scale_probe" / "results_recall_by_generation_soft5f_seed_test"
    / "generation_depth_analysis.json"
)
DEFAULT_OOD_INPUT = (
    ROOT / "data" / "skeleton_scale_probe" / "results_recall_by_generation_soft5f_aeropath"
    / "generation_depth_analysis.json"
)
FIGURE_ROOT = ROOT / "dissertation" / "Figures"
DEFAULT_PDF_OUT = FIGURE_ROOT / "pdf" / "results"
DEFAULT_PNG_OUT = FIGURE_ROOT / "png" / "results"
METHODS_PDF_OUT = FIGURE_ROOT / "pdf" / "methods" / "depth"
METHODS_PNG_OUT = FIGURE_ROOT / "png" / "methods" / "depth"
APPENDIX_PDF_OUT = FIGURE_ROOT / "pdf" / "appendix"
APPENDIX_PNG_OUT = FIGURE_ROOT / "png" / "appendix"
# The 260 training cases, for the Methods census bars. 20 cases is a thin basis for a
# distribution the reader is asked to weigh later differences against.
DEFAULT_CENSUS_INPUT = (
    ROOT / "data" / "skeleton_scale_probe" / "results_recall_by_generation"
    / "generation_depth_census_train260.json"
)

# Shared with generate_hu_imbalance_histogram.py so a colour means the same thing.
INK = "#0f172a"
MUTED = "#475569"
GREY = "#94a3b8"

# One colour per arm, held constant across both panels.
ARM_STYLE: dict[str, tuple[str, str, str]] = {
    "control": ("#64748b", "-", "o"),
    "mt_soft": ("#0891b2", "-", "s"),
    "mt_soft_5f": ("#009E73", "-.", "v"),
    "mt_hard_f0": ("#f97316", "--", "^"),
    "mt_hard_5f": ("#b45309", "--", "v"),
    "ceiling110": ("#7c3aed", ":", "D"),
}
ARM_LABEL: dict[str, str] = {
    "control": "Control (no consistency)",
    "mt_soft": "Mean Teacher, fold 0",
    "mt_soft_5f": "Mean Teacher, 5-fold",
    "mt_hard_f0": "Thresholded target",
    "mt_hard_5f": "Thresholded, 5-fold",
    "ceiling110": "Supervised, 110 labels",
}

LABEL_PT = 8.5
TICK_PT = 7.0
LEGEND_PT = 6.8
ANNOTATION_PT = 6.8

# Only like-for-like fold-0 comparisons belong in the difference panel. The
# five-fold ensemble remains in the absolute panel because there is no five-fold
# no-consistency ensemble against which to attribute its difference.
CONTROLLED_DELTA_ARMS = ("mt_soft",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Validation-cohort generation-depth JSON.",
    )
    parser.add_argument(
        "--test-input", type=Path, default=DEFAULT_TEST_INPUT,
        help="Held-out-test generation-depth JSON used for panels (c) and (d).",
    )
    parser.add_argument(
        "--ood-input", type=Path, default=DEFAULT_OOD_INPUT,
        help="Optional AeroPath OOD generation-depth JSON for the Appendix figure.",
    )
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Truncate the depth axis (default: the deepest bin present in every case, plus a tail bin).",
    )
    parser.add_argument(
        "--overlay-cases",
        nargs="*",
        default=[],
        help="Re-parse these cases and render the GT skeleton coloured by depth.",
    )
    parser.add_argument("--ground-truth-dir", type=Path, default=ROOT / "data" / "ATM22" / "labelsTr")
    parser.add_argument(
        "--definition-case",
        default="ATM_016",
        help="Case drawn in panel (a) of the Methods branch-depth figure.",
    )
    parser.add_argument("--methods-pdf-dir", type=Path, default=METHODS_PDF_OUT)
    parser.add_argument("--methods-png-dir", type=Path, default=METHODS_PNG_OUT)
    parser.add_argument(
        "--census-input",
        type=Path,
        default=DEFAULT_CENSUS_INPUT,
        help="Cohort census for the Methods bars, from measure_depth_census_cohort.py. "
             "Falls back to the evaluation cohort if the file is absent.",
    )
    parser.add_argument("--appendix-pdf-dir", type=Path, default=APPENDIX_PDF_OUT)
    parser.add_argument("--appendix-png-dir", type=Path, default=APPENDIX_PNG_OUT)
    return parser.parse_args()


def _panel_label(axis: plt.Axes, letter: str, x: float = -0.13) -> None:
    panel_label(axis, letter, x=x)


def _style(axis: plt.Axes) -> None:
    """Hard left/bottom axes from the shared theme."""
    finish(axis)


def _save(fig: plt.Figure, pdf_dir: Path, png_dir: Path, stem: str) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir / f"{stem}.pdf", facecolor="white")
    fig.savefig(png_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  wrote {pdf_dir / (stem + '.pdf')}")


def _per_case_series(rows: list[dict], arm: str, depth: int, key: str) -> np.ndarray:
    return np.asarray(
        [
            r[key]
            for r in rows
            if r["arm"] == arm and r["generation"] == depth and np.isfinite(r[key])
        ],
        dtype=np.float64,
    )


def _complete_depth(data: dict) -> int:
    """Deepest depth present in every case; beyond it, arms are not fully paired."""
    cases = data["cases"]
    gt_rows = data["per_case_generation_gt"]
    max_depth = max(r["generation"] for r in gt_rows)
    complete = 0
    for depth in range(max_depth + 1):
        present = {r["case_id"] for r in gt_rows if r["generation"] == depth and r["gt_centreline_voxels"] > 0}
        if len(present) == len(cases):
            complete = depth
        else:
            break
    return complete


def _long_frame(rows: list[dict], arms: list[str], cut: int) -> pd.DataFrame:
    """Per-case TD by depth, with everything at or beyond ``cut`` pooled into one bin.

    Pooled, not truncated. Two reasons, and the Results text depends on both. Depth 9 is
    present in only 16 of the 20 cases and depth 13 in 3, so plotting those depths
    individually changes cohort part-way along the axis; and truncating instead would
    throw away the deep tree that this section is entirely about. The tail is aggregated
    as summed numerator over summed denominator per case, which is the ratio-of-sums that
    defines TLD at every other depth, so the pooled point is the same quantity as its
    neighbours rather than a mean of ratios.
    """
    totals: dict[tuple[str, str, int], list[int]] = {}
    for row in rows:
        if row["arm"] not in arms:
            continue
        key = (row["arm"], row["case_id"], min(row["generation"], cut))
        entry = totals.setdefault(key, [0, 0])
        entry[0] += int(row["detected_centreline_voxels"])
        entry[1] += int(row["gt_centreline_voxels"])
    return pd.DataFrame(
        [
            {"arm": arm, "case_id": case, "generation": depth, "td": found / total}
            for (arm, case, depth), (found, total) in totals.items()
            if total > 0
        ]
    )


def _paired_frame(rows: list[dict], baseline: str, arms: list[str],
                  cut: int) -> pd.DataFrame:
    """Per-case paired differences against the baseline arm.

    Differences are formed per case BEFORE any aggregation, so the interval seaborn
    bootstraps is the paired interval the text reports, not the difference of two
    independent cohort means.
    """
    frame = _long_frame(rows, arms + [baseline], cut)
    wide = frame.pivot_table(index=["case_id", "generation"], columns="arm", values="td")
    records = []
    for arm in arms:
        if arm == baseline or arm not in wide.columns:
            continue
        delta = (wide[arm] - wide[baseline]).dropna().reset_index(name="delta")
        delta["arm"] = arm
        records.append(delta)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _draw_arms(axis, frame, value, arms, palette):
    """One seaborn lineplot per arm, so dashes and markers stay per-arm."""
    for arm in arms:
        subset = frame[frame["arm"] == arm]
        if subset.empty:
            continue
        line = sns.lineplot(
            data=subset, x="generation", y=value, ax=axis,
            color=palette[arm], marker=ARM_MARKER.get(arm, "o"),
            markeredgecolor="white", markeredgewidth=0.4,
            errorbar=("ci", 95), n_boot=2000, seed=20260819,
            err_kws={"alpha": BAND_ALPHA, "linewidth": 0},
            label=THEME_ARM_LABEL.get(arm, arm), legend=False,
        )
        dash = ARM_DASH.get(arm, ())
        if dash:
            line.lines[-1].set_dashes(dash)


def _depth_share(data: dict, cut: int) -> dict[int, float]:
    """Percent of reference centreline length held at each depth, over all cases.

    Pooled at ``cut`` to match the tail bin the curves use, so the share printed above a
    point is the share of the tree that point actually covers.
    """
    rows = data["per_case_generation_gt"]
    total = sum(r["gt_centreline_voxels"] for r in rows)
    share: dict[int, float] = {}
    for depth in range(cut + 1):
        held = sum(
            r["gt_centreline_voxels"] for r in rows
            if (r["generation"] >= cut if depth == cut else r["generation"] == depth)
        )
        share[depth] = 100.0 * held / total if total else 0.0
    return share


def _depth_labels(depths: list[int], cut: int) -> list[str]:
    """Tick labels, with the pooled tail bin marked so it cannot read as one depth."""
    return [f"{d}+" if d == cut else str(d) for d in depths]


def _recovery_row(
    axes: tuple[plt.Axes, plt.Axes],
    data: dict,
    cut: int,
    letters: tuple[str, str],
    cohort_label: str,
    *,
    show_legends: bool,
    show_xlabels: bool,
    percentage_scale: bool = False,
) -> None:
    """Draw one cohort as absolute recovery and a matched fold-0 difference."""
    rows = data["per_case_arm_generation"]
    baseline = data["baseline_arm"]
    arms = list(data["arms"])
    depths = list(range(cut + 1))
    palette = arm_palette(arms)

    absolute = _long_frame(rows, arms, cut)
    paired = _paired_frame(
        rows,
        baseline,
        [arm for arm in arms if arm in CONTROLLED_DELTA_ARMS],
        cut,
    )
    if percentage_scale:
        absolute["td"] *= 100.0
        paired["delta"] *= 100.0

    axis = axes[0]
    _draw_arms(axis, absolute, "td", arms, palette)
    axis.set_xlabel("Branch depth from trachea" if show_xlabels else "")
    axis.set_ylabel("Tree length detected (%)" if percentage_scale else "Tree length detected")
    axis.set_xticks(depths)
    axis.set_xticklabels(_depth_labels(depths, cut))
    # Shared x-axes hide upper-row values by default. Show the depth values on every
    # panel; ``show_xlabels`` still keeps the full axis title on the bottom row only.
    axis.tick_params(axis="x", labelbottom=True)
    if show_legends:
        axis.legend(loc="lower left", fontsize=THEME_LEGEND_PT, frameon=False)
    axis.set_title(cohort_label, loc="left", fontweight="bold")
    # The panel is zoomed on 0.86-1.00, so mark the axis as broken.
    broken_axis_marker(axis)
    _panel_label(axis, letters[0])
    _style(axis)

    axis = axes[1]
    guide_line(axis, 0.0)
    treatment = [arm for arm in arms if arm in CONTROLLED_DELTA_ARMS]
    _draw_arms(axis, paired, "delta", treatment, palette)
    axis.set_xlabel("Branch depth from trachea" if show_xlabels else "")
    axis.set_ylabel(
        "$\\Delta$ tree length detected vs control\n(percentage points)"
        if percentage_scale
        else r"$\Delta$ tree length detected vs control"
    )
    axis.set_xticks(depths)
    axis.set_xticklabels(_depth_labels(depths, cut))
    axis.tick_params(axis="x", labelbottom=True)
    if show_legends:
        axis.legend(loc="upper left", fontsize=THEME_LEGEND_PT, frameon=False)
    _panel_label(axis, letters[1])
    _style(axis)

    share = _depth_share(data, cut)
    top = axis.secondary_xaxis("top")
    top.set_xticks(depths)
    # Whole-percent rounding makes these exhaustive bins appear to sum to 101%.
    # One decimal preserves the exact 100% partition while remaining compact.
    top.set_xticklabels([f"{share[d]:.1f}" for d in depths], fontsize=6.4, color=MUTED)
    top.set_xlabel("reference centreline share (%)", fontsize=7.2, color=MUTED)
    top.tick_params(length=0)
    top.spines["top"].set_visible(False)


def _figure_recovery(
    validation: dict,
    held_out_test: dict,
    args: argparse.Namespace,
    cut: int,
) -> None:
    """Results figure: matched validation/test rows on common column scales."""
    if validation["baseline_arm"] != held_out_test["baseline_arm"]:
        raise ValueError("Validation and TEST JSONs use different baseline arms")
    if set(validation["arms"]) != set(held_out_test["arms"]):
        raise ValueError("Validation and TEST JSONs must contain the same arms")

    # Shared axes make changes between cohorts visually comparable rather than an
    # artefact of independently chosen plot limits.
    fig, axes = plt.subplots(
        2, 2, figsize=(6.9, 5.35), sharex="col", sharey="col", layout="constrained"
    )
    _recovery_row(
        (axes[0, 0], axes[0, 1]), validation, cut, ("a", "b"),
        f"Validation ($n={len(validation['cases'])}$)",
        show_legends=True, show_xlabels=False, percentage_scale=True,
    )
    _recovery_row(
        (axes[1, 0], axes[1, 1]), held_out_test, cut, ("c", "d"),
        f"Held-out test ($n={len(held_out_test['cases'])}$)",
        show_legends=False, show_xlabels=True, percentage_scale=True,
    )
    _save(fig, args.pdf_output_dir, args.png_output_dir, "generation_depth_recovery")


def _figure_recovery_ood(data: dict, args: argparse.Namespace) -> None:
    """Exploratory AeroPath counterpart, kept separate from the main 2x2 figure."""
    cut = _complete_depth(data)
    if cut < 0:
        raise ValueError("AeroPath has no branch depth represented in every case")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained")
    _recovery_row(
        (axes[0], axes[1]), data, cut, ("a", "b"),
        f"AeroPath OOD ($n={len(data['cases'])}$)",
        show_legends=True, show_xlabels=True,
    )
    _save(
        fig,
        args.appendix_pdf_dir,
        args.appendix_png_dir,
        "generation_depth_recovery_aeropath",
    )


def _case_skeleton(case_id: str, args: argparse.Namespace):
    """Reference skeleton voxels of one case with their branch depth.

    The only place in this script that touches image data. Kept separate so both the
    Methods explanatory panel and the sanity overlays read a case the same way, and so
    a machine without the ATM'22 labels can still rebuild every JSON-driven figure.
    """
    import nibabel as nib

    from generate_hu_imbalance_histogram import branch_generation_labels
    from lung_airway_segmentation.metrics.topology import (
        _largest_connected_component, parse_reference_skeleton_branches,
    )

    # The array is read in its STORED orientation, not resampled to a canonical one.
    # Skeletonisation is not flip-invariant -- its tie-breaks are directional -- so
    # reorienting the voxels before parsing changes the branch count (288 to 283 on
    # ATM_016) and this panel would stop describing the same tree as the census beside
    # it. Orientation is a display concern, and the affine is carried out for the
    # drawing code to resolve it there.
    image = nib.load(args.ground_truth_dir / f"{case_id}_0000.nii.gz")
    spacing = tuple(float(z) for z in image.header.get_zooms()[:3])
    component = _largest_connected_component(np.asanyarray(image.dataobj) > 0)
    voxel_generation, branch_generation, _ = branch_generation_labels(component)
    skeleton = parse_reference_skeleton_branches(component) > 0
    return (np.argwhere(skeleton), voxel_generation[skeleton], spacing, image.affine,
            branch_generation, int(branch_generation.max()))


_OPPOSITE = {"L": "R", "R": "L", "A": "P", "P": "A", "S": "I", "I": "S"}
# Per display plane, the anatomical direction that must increase to the RIGHT and the
# one that must increase UPWARD on the page. Radiological convention: the patient is
# viewed from the front on a coronal and from below on an axial, so patient-left runs
# to the viewer's right; a sagittal is read with anterior on the left.
_PLANE_TARGETS = {
    "coronal": ("L", "S"),
    "sagittal": ("P", "S"),
    "axial": ("L", "A"),
}


def _display_axes(affine, plane: str):
    """Which array axis carries each screen direction for this plane, and its sign.

    Read off the affine rather than assumed, because an axis index means nothing on its
    own: ATM'22 stores LPS, so axis 2 increases SUPERIORLY and a viewer that assumes the
    third axis runs head-down draws the tree upside down, trachea below the carina. A
    sign of -1 means the stored axis runs opposite to the screen direction wanted, and
    the coordinate is negated so no per-plane axis inversion is needed downstream.
    """
    import nibabel as nib

    lookup: dict[str, tuple[int, float]] = {}
    for index, code in enumerate(nib.aff2axcodes(affine)):
        lookup[code] = (index, 1.0)
        lookup[_OPPOSITE[code]] = (index, -1.0)
    return tuple(lookup[target] for target in _PLANE_TARGETS[plane])


def _draw_depth_tree(axis, coordinates, depths, spacing, affine, maximum,
                     plane: str = "coronal", size: float = 0.7):
    """One projection of the skeleton, every voxel tinted by its branch depth."""
    (horizontal, h_sign), (vertical, v_sign) = _display_axes(affine, plane)
    scatter = axis.scatter(
        coordinates[:, horizontal] * spacing[horizontal] * h_sign,
        coordinates[:, vertical] * spacing[vertical] * v_sign,
        c=depths, s=size, cmap=DEPTH_RAMP, vmin=0, vmax=maximum, linewidths=0,
    )
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return scatter


def _census_series(payload: dict):
    """Per-depth share of centreline length, branches per case, and cohort completeness."""
    rows = payload["per_case_generation_gt"]
    cases = payload.get("cases") or sorted({r["case_id"] for r in rows})
    depths = list(range(max(r["generation"] for r in rows) + 1))
    total = sum(r["gt_centreline_voxels"] for r in rows)

    share = [
        100 * sum(r["gt_centreline_voxels"] for r in rows if r["generation"] == d) / total
        for d in depths
    ]
    branches = [
        sum(r["gt_branch_count"] for r in rows if r["generation"] == d) / len(cases)
        for d in depths
    ]
    # How many cases actually reach each depth. Airway trees do not all end at the same
    # generation, so a deep bar is a pooled statistic over a shrinking subset; the count
    # is what lets a reader see that rather than read every bar as equally supported.
    reaching = [
        len({r["case_id"] for r in rows
             if r["generation"] == d and r["gt_centreline_voxels"] > 0})
        for d in depths
    ]
    complete = [count == len(cases) for count in reaching]
    return depths, share, branches, complete, reaching, len(cases)


def _draw_depth_census(axis, data: dict, cohort: dict | None = None) -> list[int]:
    """How much reference tree each depth holds, and how many branches carry it.

    Bars take their colour from the depth ramp rather than one flat hue, so this
    panel's x coordinate is also the colour key for the tree beside it. Every bar is
    filled, the deep ones included: that tree really is there, and hollowing those bars
    broke the ramp off exactly where the interesting depths begin.

    ``cohort`` is the distribution the bars describe -- the 260 training cases, which is
    a far steadier estimate than 20 -- while ``data`` stays the evaluation cohort. The
    two are not interchangeable and the panel keeps them apart: the evaluation cohort's
    own share is marked on each bar, so the reader can see whether the 20 cases are
    typical, and the red rule marks where THAT cohort stops being complete, because that
    is what decides where the recovery curves in Results end. Recomputing the rule on the
    training pool would silently change what it means.
    """
    depths, share, branches, complete, reaching, case_count = _census_series(cohort or data)

    top = max(1, depths[-1])
    for depth, height in zip(depths, share):
        axis.bar(depth, height, width=0.72, zorder=2, color=DEPTH_RAMP(depth / top))
    axis.set_ylabel("Share of reference tree length (%)", fontsize=LABEL_PT)
    axis.set_xticks(depths)

    # Numbers-at-risk row, as under a survival curve. Case support falls off gradually
    # rather than at a boundary -- 245 of 260 reach depth 8 but only 46 reach depth 12 --
    # so a binary "present in every case" mark would assert a cliff that is not there and
    # would say nothing about how thin the deepest bars have become.
    blended = axis.get_xaxis_transform()
    for depth, count in zip(depths, reaching):
        axis.text(depth, -0.135, str(count), transform=blended, fontsize=ANNOTATION_PT - 0.4,
                  color=MUTED, ha="center", va="top", clip_on=False)
    axis.text(-0.012, -0.135, "cases", transform=axis.transAxes,
              fontsize=ANNOTATION_PT - 0.4, color=MUTED, ha="right", va="top", clip_on=False)
    axis.set_xlabel("Branch depth from trachea", fontsize=LABEL_PT, labelpad=22)

    twin = axis.twinx()
    twin.plot(depths, branches, color=MUTED, linewidth=1.0, marker="o", markersize=2.4,
              zorder=3)
    twin.set_ylabel("Mean branches per case", fontsize=LABEL_PT, color=MUTED)
    twin.tick_params(labelsize=TICK_PT, colors=MUTED, length=3)
    twin.spines["top"].set_visible(False)
    # The two overlaid series get a legend rather than floating notes. Two quantities on
    # one panel, one per axis, is exactly the layout a reader mis-assigns, and the bars
    # cannot be colour matched to their own axis because their colour already carries
    # depth. The bars themselves are not in it: the left axis label already names them,
    # and a third entry would say nothing the axis does not.
    keys = [
        plt.Line2D([], [], color=MUTED, marker="o", markersize=2.4, linewidth=1.0,
                   label="mean branches per case"),
    ]

    evaluation = data if cohort is not None else None
    if evaluation is not None:
        eval_depths, eval_share, _, complete, _, _ = _census_series(evaluation)
        # A tick across each bar rather than a second bar series: the question is whether
        # the evaluation cohort sits where the training pool does, which is a comparison
        # of levels, and a level is what a rule across the bar reads as.
        for depth, height in zip(eval_depths, eval_share):
            axis.hlines(height, depth - 0.36, depth + 0.36, color=INK, linewidth=1.1,
                        zorder=4)
        depths = eval_depths if len(eval_depths) > len(depths) else depths
        keys.append(
            plt.Line2D([], [], color=INK, marker="_", markersize=7, markeredgewidth=1.1,
                       linestyle="none", label="validation-20 share")
        )

    # On the twin so it draws over the bars, and upper left because the distribution is
    # unimodal with its peak mid-axis, leaving the shallow end quiet.
    twin.legend(handles=keys, loc="upper left", fontsize=LEGEND_PT, frameon=False,
                handlelength=1.5, handletextpad=0.5, borderpad=0.1, labelspacing=0.35)

    # The pooled tail is a RANGE, so it is drawn as one. The first incomplete depth is the
    # reason for pooling, but the pooled bin begins one depth earlier -- the last complete
    # depth heads it -- so a single rule at the incompleteness boundary would fall inside
    # the group it is supposed to delimit, with the bin's own first member on the far side.
    first_incomplete = next((d for d, ok in zip(depths, complete) if not ok), None)
    if first_incomplete is not None:
        pooled_from = first_incomplete - 1
        axis.axvspan(pooled_from - 0.5, depths[-1] + 0.5, color=ANNOTATION_RED, alpha=0.07,
                     zorder=0, linewidth=0)
        axis.axvline(pooled_from - 0.5, color=ANNOTATION_RED, linewidth=0.9, zorder=5,
                     dashes=(2.6, 1.8))
        # No figure number in the annotation: baking a LaTeX cross-reference into a PDF
        # makes the image lie the first time the numbering shifts. The caption carries it.
        axis.text(pooled_from - 0.25, axis.get_ylim()[1] * 0.97,
                  f"pooled as ${pooled_from}+$\nin the recovery\nanalysis",
                  fontsize=ANNOTATION_PT, color=ANNOTATION_RED, ha="left", va="top")

    return depths


def _figure_depth_definition(data: dict, args: argparse.Namespace) -> None:
    """Methods figure: what branch depth means, and where the reference tree sits on it.

    Two questions the depth-stratified results depend on and neither answers. Panel (a)
    fixes the coordinate anatomically; panel (b) says how much tree is at each value of
    it, so a difference at one depth can be weighed against the tree that exists there.
    """
    label_path = args.ground_truth_dir / f"{args.definition_case}_0000.nii.gz"
    with_tree = label_path.exists()
    if not with_tree:
        print(f"  {label_path} not found; building the census panel alone")

    if with_tree:
        fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), layout="constrained",
                                 width_ratios=[1.0, 1.9])
        coordinates, depths, spacing, affine, branch_generation, maximum = _case_skeleton(
            args.definition_case, args
        )
        scatter = _draw_depth_tree(axes[0], coordinates, depths, spacing, affine, maximum)
        # The colour key rides inside the panel rather than being attached to it.
        # The skeleton is tall and narrow, so an attached colourbar is pushed to the
        # bottom of a mostly empty axes box and reads as belonging to neither panel.
        # Clear the trachea tip: the axis is inverted, so growing the lower limit adds
        # empty room under the tree for the bar to sit in.
        bottom, topmost = axes[0].get_ylim()
        axes[0].set_ylim(bottom + 0.12 * (bottom - topmost), topmost)
        cax = axes[0].inset_axes([0.06, 0.015, 0.88, 0.030])
        bar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
        bar.set_ticks(list(range(0, maximum + 1, 2)))
        bar.set_label("Branch depth from trachea", fontsize=ANNOTATION_PT, color=MUTED)
        bar.ax.tick_params(labelsize=TICK_PT, colors=MUTED, length=2)
        bar.outline.set_visible(False)
        axes[0].set_title(
            f"{args.definition_case}, coronal ({branch_generation.size} branches)",
            fontsize=ANNOTATION_PT, color=MUTED,
        )
        _panel_label(axes[0], "a", x=-0.07)
        census_axis = axes[1]
        _panel_label(census_axis, "b", x=-0.10)
    else:
        fig, census_axis = plt.subplots(figsize=(4.4, 3.0), layout="constrained")

    cohort = None
    if args.census_input and args.census_input.exists():
        cohort = json.loads(args.census_input.read_text())
        print(f"  census bars from {len(cohort['cases'])} cases ({args.census_input.name})")
    else:
        print(f"  {args.census_input} not found; census bars fall back to the 20 eval cases")
    _draw_depth_census(census_axis, data, cohort)
    _style(census_axis)
    _save(fig, args.methods_pdf_dir, args.methods_png_dir, "branch_depth_definition")


def _figure_depth_calibre(data: dict, args: argparse.Namespace) -> None:
    """Appendix figure: how far branch depth and airway calibre are the same axis.

    Supporting material rather than a result. It says why the calibre and depth
    sections agree without either being a restatement of the other, which the
    Discussion needs once and the Results do not need at all.
    """
    joint = data.get("joint_depth_calibre")
    if joint is None:
        print("  no joint_depth_calibre block in the input; skipping the heatmap")
        return
    depths = list(range(max(r["generation"] for r in data["per_case_generation_gt"]) + 1))

    fig, axis = plt.subplots(figsize=(4.8, 3.0), layout="constrained")
    matrix = np.asarray(joint["centreline_by_generation"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        fractions = 100 * matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    cmap = LinearSegmentedColormap.from_list("depth_calibre", ["#ffffff", "#0072B2", INK])
    image = axis.imshow(fractions.T, aspect="auto", origin="lower", cmap=cmap,
                        vmin=0, vmax=100)
    axis.set_yticks(range(len(joint["band_names"])))
    axis.set_yticklabels(joint["band_names"], fontsize=TICK_PT)
    axis.set_xticks(depths)
    axis.set_xlabel("Branch depth from trachea", fontsize=LABEL_PT)
    axis.set_ylabel("Operational thickness (voxels)", fontsize=LABEL_PT)
    bar = fig.colorbar(image, ax=axis, pad=0.02)
    bar.set_label("% of that depth's tree length", fontsize=ANNOTATION_PT, color=MUTED)
    bar.ax.tick_params(labelsize=TICK_PT, colors=MUTED, length=2)
    bar.outline.set_visible(False)
    # Number inside the maths so the sign sets as a minus rather than a hyphen.
    axis.text(0.98, 0.06, f"Spearman $\\rho = {joint['spearman_rho']:.2f}$",
              transform=axis.transAxes, fontsize=ANNOTATION_PT, ha="right", color=INK)
    _style(axis)
    _save(fig, args.appendix_pdf_dir, args.appendix_png_dir, "depth_calibre_heatmap")


def _figure_overlay(case_id: str, args: argparse.Namespace) -> None:
    """Sanity check: render the reference skeleton coloured by branch depth."""
    coordinates, depths, spacing, affine, branch_generation, maximum = _case_skeleton(
        case_id, args
    )
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.9), layout="constrained")
    for axis, plane in zip(axes, _PLANE_TARGETS):
        scatter = _draw_depth_tree(axis, coordinates, depths, spacing, affine, maximum,
                                   plane, size=0.6)
        axis.set_title(plane.capitalize(), fontsize=LABEL_PT, color=INK)
    bar = fig.colorbar(scatter, ax=axes, pad=0.02, fraction=0.03)
    bar.set_label("Branch depth", fontsize=ANNOTATION_PT, color=MUTED)
    bar.ax.tick_params(labelsize=TICK_PT, colors=MUTED, length=2)
    fig.suptitle(
        f"{case_id}: reference skeleton by branch depth "
        f"({branch_generation.size} branches, depth 0-{maximum})",
        fontsize=LABEL_PT, color=INK,
    )
    _save(fig, args.pdf_output_dir, args.png_output_dir, f"generation_depth_overlay_{case_id}")

    counts = np.bincount(branch_generation[branch_generation >= 0], minlength=maximum + 1)
    print(f"  {case_id}: branches per depth 0..3 = {counts[:4].tolist()} "
          f"(expect 1 trachea, 2 main, 4-6 lobar)")


def main() -> None:
    args = _parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    apply_theme()
    if not args.test_input.is_file():
        raise FileNotFoundError(
            f"Held-out TEST analysis not found: {args.test_input}\n"
            "Run measure_recall_by_generation.py for TEST before regenerating the figure."
        )
    validation = json.loads(args.input.read_text())
    held_out_test = json.loads(args.test_input.read_text())
    validation_complete = _complete_depth(validation)
    test_complete = _complete_depth(held_out_test)
    cut = args.max_depth if args.max_depth is not None else validation_complete
    if test_complete < cut:
        raise ValueError(
            f"TEST is fully paired only through depth {test_complete}, but the validation "
            f"figure requires depth {cut}. Review the fixed tail-bin support before plotting."
        )
    print(
        f"Depths present in all cases: validation 0-{validation_complete}; "
        f"TEST 0-{test_complete}; plotting both 0-{cut}"
    )

    _figure_depth_definition(validation, args)
    _figure_recovery(validation, held_out_test, args, cut)
    _figure_depth_calibre(validation, args)
    if args.ood_input.is_file():
        ood = json.loads(args.ood_input.read_text())
        print(
            f"AeroPath depths present in all {len(ood['cases'])} cases: "
            f"0-{_complete_depth(ood)}"
        )
        _figure_recovery_ood(ood, args)
    else:
        print(f"  no AeroPath depth JSON at {args.ood_input}; skipping OOD figure")
    for case_id in args.overlay_cases:
        _figure_overlay(case_id, args)


if __name__ == "__main__":
    main()
