"""Calibre-stratified paired differences in tree length and precision for Results.

The two panels show every patient-level treatment-minus-control difference, together
with the mean and a paired patient-bootstrap confidence interval in each calibre band.
Tree-length detection is stratified by reference calibre; precision is stratified by
the prediction's own calibre because false positives have no reference calibre. A
magnified inset makes the smaller tree-length differences above two voxels legible.

Reads ``recall_by_calibre.json`` written by ``measure_recall_by_calibre.py``; nothing
here touches image data.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\plot_calibre_delta.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_theme import (  # noqa: E402
    ANNOTATION_PT,
    ANNOTATION_RED,
    ARM_COLOUR,
    INK,
    LEGEND_PT,
    MUTED,
    apply_theme,
    finish,
    guide_line,
)

DEFAULT_INPUT = (
    ROOT / "data" / "skeleton_scale_probe" / "results_recall_by_calibre_soft5f_seed"
    / "recall_by_calibre.json"
)
DEFAULT_PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "results"
DEFAULT_PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "results"

TREATMENT = "mt_soft"
INSET_YLIM = (-2.6, 3.6)
INSET_LABEL_MARGIN = 0.4
CI_LINEWIDTH = 0.8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    parser.add_argument(
        "--no-inset",
        action="store_true",
        help="Draw panel (a) without the magnified inset and write the "
        "'_noinset' variant so the default figure is left alone.",
    )
    return parser.parse_args()


def _frames(
    data: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    baseline = data["baseline_arm"]
    bands = [group[0] for group in data["class_groups"]]
    by_arm: dict[str, dict[str, dict]] = {}
    for row in data["per_case_arm"]:
        by_arm.setdefault(row["arm"], {})[row["case_id"]] = row

    control, treatment = by_arm[baseline], by_arm[TREATMENT]
    cases = sorted(set(control) & set(treatment))

    tld_deltas, precision_deltas, shares = [], [], []
    for band in bands:
        tld_key = f"td__{band}"
        precision_key = f"precision__{band}"
        share_key = f"gt_length_share__{band}"
        for case in cases:
            treatment_value = treatment[case].get(tld_key)
            control_value = control[case].get(tld_key)
            if (
                treatment_value is None
                or control_value is None
                or not np.isfinite(treatment_value)
                or not np.isfinite(control_value)
            ):
                continue
            tld_deltas.append(
                {
                    "band": band,
                    "case_id": case,
                    "delta": 100.0 * (treatment_value - control_value),
                }
            )
            treatment_precision = treatment[case].get(precision_key)
            control_precision = control[case].get(precision_key)
            if (
                treatment_precision is not None
                and control_precision is not None
                and np.isfinite(treatment_precision)
                and np.isfinite(control_precision)
            ):
                precision_deltas.append(
                    {
                        "band": band,
                        "case_id": case,
                        "delta": 100.0 * (treatment_precision - control_precision),
                    }
                )
        band_shares = [
            control[case][share_key]
            for case in cases
            if control[case].get(share_key) is not None
        ]
        shares.append(
            {
                "band": band,
                "share": 100.0 * float(np.mean(band_shares)),
            }
        )
    return (
        pd.DataFrame(tld_deltas),
        pd.DataFrame(precision_deltas),
        pd.DataFrame(shares),
        bands,
    )


def _draw_bands(
    axis,
    delta: pd.DataFrame,
    order: list[str],
    colour: str,
    point_size: float,
    marker_size: float,
) -> None:
    """Draw patient differences and their mean bootstrap interval on one axis."""
    guide_line(axis, 0.0)
    sns.stripplot(
        data=delta,
        x="band",
        y="delta",
        order=order,
        ax=axis,
        color=colour,
        alpha=0.5,
        size=point_size,
        jitter=0.17,
        linewidth=0,
        zorder=2,
    )
    sns.pointplot(
        data=delta,
        x="band",
        y="delta",
        order=order,
        ax=axis,
        color=colour,
        errorbar=("ci", 95),
        n_boot=2000,
        seed=20260819,
        marker="s",
        markersize=marker_size,
        linestyle="none",
        err_kws={"linewidth": CI_LINEWIDTH, "color": INK},
        capsize=0.16,
        zorder=4,
    )


def _magnified_inset(
    axis,
    delta: pd.DataFrame,
    order: list[str],
    colour: str,
    n_bands: int,
):
    """Redraw bands of three voxels and above on a three-times-finer axis."""
    axis.set_xlim(-0.5, n_bands - 0.5 + INSET_LABEL_MARGIN)
    span = n_bands + INSET_LABEL_MARGIN
    inset = axis.inset_axes([1.0 / span, 0.400, (n_bands - 1.0) / span, 0.575])
    _draw_bands(
        inset,
        delta[delta["band"].isin(order)],
        order,
        colour,
        point_size=2.4,
        marker_size=4.0,
    )

    inset.set_ylim(*INSET_YLIM)
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.set_xticks([])
    inset.set_yticks([-2.0, 0.0, 2.0])
    inset.yaxis.tick_right()
    inset.tick_params(axis="y", labelsize=6.2, length=2.2, colors=INK, pad=1.5)
    inset.set_facecolor("white")
    inset.set_zorder(5)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(ANNOTATION_RED)
        spine.set_linewidth(0.9)
    inset.text(
        0.988,
        0.955,
        "magnified ×3, same cases",
        transform=inset.transAxes,
        fontsize=ANNOTATION_PT - 0.6,
        va="top",
        ha="right",
        color=MUTED,
    )

    indicator = axis.indicate_inset(
        bounds=(
            0.5,
            INSET_YLIM[0],
            n_bands - 1.0,
            INSET_YLIM[1] - INSET_YLIM[0],
        ),
        inset_ax=inset,
        edgecolor=ANNOTATION_RED,
        facecolor="none",
        linewidth=0.9,
        zorder=6,
        alpha=1.0,
    )
    indicator.rectangle.set_linestyle((0, (2.6, 1.8)))
    for connector in indicator.connectors:
        connector.set_color(ANNOTATION_RED)
        connector.set_linewidth(0.7)
        connector.set_alpha(0.85)
    return inset


def main() -> None:
    args = _parse_args()
    apply_theme()
    data = json.loads(args.input.read_text())
    tld_delta, precision_delta, share, bands = _frames(data)
    if precision_delta.empty:
        raise SystemExit(
            "Input has no prediction-calibre precision. Re-run "
            "measure_recall_by_calibre.py with --with-precision."
        )
    present = [band for band in bands if (tld_delta["band"] == band).any()]

    fig, (tld_axis, precision_axis) = plt.subplots(
        2, 1, figsize=(5.5, 5.15), sharex=True, layout="constrained"
    )
    colour = ARM_COLOUR[TREATMENT]
    # Fix the horizontal jitter so regenerating the figure is pixel-stable.
    np.random.seed(20260819)
    _draw_bands(tld_axis, tld_delta, present, colour, point_size=2.7, marker_size=4.8)
    _draw_bands(
        precision_axis, precision_delta, present, colour,
        point_size=2.7, marker_size=4.8,
    )

    tld_axis.set_title(
        "(a) Tree-length detection by reference calibre", loc="left", fontweight="semibold"
    )
    precision_axis.set_title(
        "(b) Voxel precision by prediction calibre", loc="left", fontweight="semibold"
    )
    tld_axis.set_xlabel("")
    tld_axis.set_ylabel("$\\Delta$ tree length detected\n(percentage points)")
    precision_axis.set_xlabel("Calibre band (operational thickness, voxels)")
    precision_axis.set_ylabel("$\\Delta$ voxel precision\n(percentage points)")
    finish(tld_axis)
    finish(precision_axis)

    share_map = {row["band"]: row["share"] for _, row in share.iterrows()}
    top = tld_axis.secondary_xaxis("top")
    top.set_xticks(range(len(present)))
    top.set_xticklabels(
        [f"{share_map[band]:.0f}" for band in present],
        fontsize=6.6,
        color=MUTED,
    )
    top.set_xlabel(
        "share of reference centreline length (%)",
        fontsize=7.4,
        color=MUTED,
    )
    top.tick_params(length=0)
    top.spines["top"].set_visible(False)

    if not args.no_inset:
        _magnified_inset(tld_axis, tld_delta, present[1:], colour, len(present))

    handles = [
        plt.Line2D(
            [],
            [],
            color=colour,
            marker="s",
            linestyle="none",
            markersize=5.0,
            label="+ Soft-clDice, band mean (95% CI)",
        ),
        plt.Line2D(
            [],
            [],
            color=colour,
            marker="o",
            linestyle="none",
            markersize=3.0,
            alpha=0.45,
            label="individual cases",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        fontsize=LEGEND_PT,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.6,
    )

    args.pdf_output_dir.mkdir(parents=True, exist_ok=True)
    args.png_output_dir.mkdir(parents=True, exist_ok=True)
    stem = "calibre_delta_td_noinset" if args.no_inset else "calibre_delta_td"
    fig.savefig(args.pdf_output_dir / f"{stem}.pdf", facecolor="white")
    fig.savefig(args.png_output_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("  wrote", args.pdf_output_dir / f"{stem}.pdf")


if __name__ == "__main__":
    main()
