"""Two arm-level figures the Results chapter still argues in prose alone.

``objective_comparison``
    Why a centreline consistency objective rather than a voxelwise one. Both objective
    arms are drawn as PAIRED DIFFERENCES against the same no-consistency comparator, one
    panel per metric, with every case shown. Drawing the two arms head to head would
    compare two training runs directly and lose the matched-comparator logic the rest of
    the chapter rests on; drawing each against the shared comparator keeps it.

``supervision_scale``
    Where the semi-supervised arms sit against models trained on more labelled cases. The
    supervised rungs are a SCALE REFERENCE and nothing else: they differ from the treated
    arms in label count, in fold membership and in training protocol at once, so the gap
    between them is not a label-efficiency measurement and the figure says so on its face.

``objective_and_scale``
    Both, stacked, for when one slot has to carry both arguments.

The unmatched voxel-MSE weight
------------------------------
The soft-clDice arm runs at $w_{\\max}=0.10$. Its voxel-MSE counterpart at the same weight
is not scored yet, so the registry falls back to the $w_{\\max}=0.30$ run. That
substitution is detected, not assumed: the weight is read back from the directory that
resolved, the arm is labelled with the weight it actually carries, and an unmissable note
is printed under the panels. When the matched arm lands the registry picks it up, the note
disappears on its own, and nothing in this file needs editing.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\plot_arm_comparison.py
    .venv\\Scripts\\python.exe dissertation\\scripts\\plot_arm_comparison.py \\
        --figure scale --cohort test
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import figure_arms as arms  # noqa: E402
from figure_theme import (  # noqa: E402
    ANNOTATION_PT, ARM_COLOUR, FAINT, GREY, INK, LABEL_PT, MUTED, apply_theme, finish,
)

DEFAULT_PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "appendix"
DEFAULT_PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "appendix"

# A4 text block is 171.8 mm. Panels are authored at final width so an 8 pt label is 8 pt
# on the page rather than whatever LaTeX's scale factor makes of it.
TEXT_WIDTH_IN = 6.76

PANEL_LETTERS = "abcdefgh"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=arms.COHORTS, default="val")
    parser.add_argument(
        "--figure",
        nargs="+",
        choices=("objective", "scale", "combined", "all"),
        default=["all"],
    )
    parser.add_argument("--comparator", default="control")
    parser.add_argument(
        "--objective-arms",
        nargs="+",
        default=["mse", "soft_f0"],
        help="Consistency objectives to contrast against the comparator.",
    )
    parser.add_argument(
        "--scale-arms",
        nargs="+",
        default=["seed", "control", "soft_f0", "scale110", "scale260"],
        help="Arms for the supervision-scale panel, in the order they should appear. "
        "Anything not scored yet is dropped, and the drop is printed.",
    )
    parser.add_argument("--metrics", nargs="+", default=list(arms.PRIMARY_METRICS))
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    return parser.parse_args()


def _colour(arm: str) -> str:
    theme_key = arms.ARMS[arm].theme_key
    return ARM_COLOUR.get(theme_key, GREY) if theme_key else GREY


def _arm_label(arm: str, cohort: str) -> str:
    """The arm's name, carrying the weight the resolved directory actually holds.

    The weight is printed on the tick rather than left to the caption, because the two
    consistency objectives are not currently matched on it and a reader comparing the two
    columns has to be able to see that without leaving the figure.
    """
    if arm == "mse":
        weight = arms.mse_weight(cohort)
        return "Voxel MSE" + (rf" ($w_{{\max}}{{=}}{weight}$)" if weight else "")
    if arm == "soft_f0":
        return r"Soft-clDice ($w_{\max}{=}0.10$)"
    return arms.ARMS[arm].short


def _letter(axis, index: int) -> None:
    """Panel letter inside the data area.

    These panels are barely an inch wide and the leftmost carries a two-line y label, so
    the usual position outside the top-left corner would collide with the title on every
    column and with the label on the first.
    """
    axis.text(0.015, 0.985, PANEL_LETTERS[index], transform=axis.transAxes,
              fontsize=LABEL_PT, fontweight="bold", va="top", ha="left", color=INK)


def _jitter(n: int, width: float = 0.16, seed: int = 11) -> np.ndarray:
    """Deterministic spread across a category, so re-running does not reshuffle points."""
    return np.linspace(-width, width, n) if n < 4 else (
        np.random.default_rng(seed).uniform(-width, width, n)
    )


def _drop_unscored(arm_keys: list[str], cohort: str) -> list[str]:
    kept = []
    for arm in arm_keys:
        if arms.available(arm, cohort):
            kept.append(arm)
        else:
            print(f"  skipping {arm!r}: not scored on the {cohort} cohort yet")
    return kept


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def _categorical_axis(axis, arm_keys, cohort, *, show_names: bool) -> None:
    """Arms run down the y-axis, not across the x.

    Arm names are five to eight words of English. As x tick labels on a panel an inch
    wide they either overlap or have to be rotated to the point of illegibility; on the
    y-axis they read horizontally at full length, and only the leftmost panel of a row
    has to carry them at all.
    """
    axis.set_yticks(range(len(arm_keys)))
    axis.set_yticklabels([_arm_label(a, cohort) for a in arm_keys] if show_names
                         else [""] * len(arm_keys))
    axis.set_ylim(len(arm_keys) - 0.4, -0.6)  # first arm at the top
    finish(axis)


def _difference_panel(axis, arm_keys, comparator, cohort, metric, *, show_names) -> list[dict]:
    """Paired difference against the shared comparator, every case drawn."""
    axis.axvline(0.0, color=INK, linewidth=1.0, alpha=0.75, zorder=2.5, dashes=(3.5, 2.0))
    records = []
    for position, arm in enumerate(arm_keys):
        summary = arms.paired_summary(arm, comparator, cohort, metric)
        if summary is None:
            continue
        colour = _colour(arm)
        differences = np.asarray(summary["differences"])
        axis.scatter(
            differences, position + _jitter(len(differences)),
            s=7.0, facecolor=colour, edgecolor="none", alpha=0.45, zorder=2,
        )
        axis.errorbar(
            summary["mean"], position,
            xerr=[[summary["mean"] - summary["ci_low"]],
                  [summary["ci_high"] - summary["mean"]]],
            fmt="s", markersize=4.4, color=colour, ecolor=colour,
            elinewidth=1.2, capsize=2.6, capthick=1.2, zorder=4,
        )
        # Anchored to the panel's right edge, not to the mean. On a metric the arm
        # barely moves, the mean sits on the zero guide and a centred label lands on it.
        axis.annotate(
            f"{summary['wins']}/{summary['n']}",
            xy=(0.985, position + 0.34), xycoords=("axes fraction", "data"),
            ha="right", va="center", fontsize=ANNOTATION_PT, color=MUTED,
        )
        records.append(summary)
    _categorical_axis(axis, arm_keys, cohort, show_names=show_names)
    return records


def _level_panel(axis, arm_keys, cohort, metric, *, show_names) -> list[dict]:
    """Absolute per-case scores, one row per arm."""
    records = []
    for position, arm in enumerate(arm_keys):
        rows = arms.load_per_case(arm, cohort)
        if rows is None:
            continue
        values = np.asarray([r[metric] for r in rows.values() if r.get(metric) is not None])
        colour = _colour(arm)
        axis.scatter(
            values, position + _jitter(len(values)),
            s=7.0, facecolor=colour, edgecolor="none", alpha=0.40, zorder=2,
        )
        mean = float(values.mean())
        half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
        axis.errorbar(
            mean, position, xerr=half, fmt="s", markersize=4.4, color=colour,
            ecolor=colour, elinewidth=1.2, capsize=2.6, capthick=1.2, zorder=4,
        )
        records.append({"arm": arm, "metric": metric, "n": len(values),
                        "mean": mean, "ci_half_width": half})
    _categorical_axis(axis, arm_keys, cohort, show_names=show_names)
    return records


def _scale_shading(axis, arm_keys) -> None:
    """Set the supervised rungs apart from the arms that share one label budget.

    They are not a continuation of the same series: the rungs differ from the treated
    arms in label count, fold membership and training protocol at once. A pale band
    behind them says so before the caption has to.
    """
    positions = [i for i, arm in enumerate(arm_keys) if arm.startswith("scale")]
    if not positions:
        return
    axis.axhspan(min(positions) - 0.5, max(positions) + 0.5, color=FAINT, alpha=0.35,
                 zorder=0, linewidth=0)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _save(figure, args, stem: str) -> Path:
    args.pdf_output_dir.mkdir(parents=True, exist_ok=True)
    args.png_output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.pdf_output_dir / f"{stem}.pdf"
    figure.savefig(destination, facecolor="white")
    figure.savefig(args.png_output_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(figure)
    print(f"  wrote {destination}")
    return destination


def _substitution_note(cohort: str) -> str | None:
    """The sentence the caption must carry while the weights are not matched."""
    if arms.mse_is_matched(cohort):
        return None
    weight = arms.mse_weight(cohort) or "an unrecorded weight"
    return (f"The two objectives are NOT weight-matched: the voxel-MSE arm shown runs at "
            f"$w_{{\\max}}={weight}$ against $w_{{\\max}}=0.10$ for soft-clDice, so it "
            f"bounds rather than isolates the effect of the objective's geometry.")


# Captions are emitted for the LaTeX figure environment rather than drawn into the
# artwork, which is the convention the rest of the document's figures follow. They are
# printed and stored in the provenance record so the wording and the plot cannot drift.
def _caption_objective(cohort: str) -> str:
    caption = (
        "paired per-case difference from the same no-consistency comparator, on the "
        f"{arms.COHORT_SHORT[cohort].lower()} cohort. Points are cases, squares are means "
        "with 95\\% intervals, and the fraction at the right of each row is the number of cases "
        "favouring that arm."
    )
    note = _substitution_note(cohort)
    return caption + (" " + note if note else "")


def _caption_scale(shaded: list[str]) -> str:
    caption = "Per-case scores with cohort means and 95\\% intervals."
    if shaded:
        caption += (
            " The shaded rows are supervised SCALE REFERENCES. They differ from the "
            "semi-supervised arms in labelled-case count, fold membership and training "
            "protocol at once, so the distance to them is not a label-efficiency "
            "measurement and no equivalence is claimed."
        )
    return caption


def figure_objective(args) -> dict:
    cohort = args.cohort
    arm_keys = _drop_unscored(args.objective_arms, cohort)
    if not arm_keys:
        print("  nothing to draw: no objective arm is scored on this cohort")
        return {}

    figure, axes = plt.subplots(
        1, len(args.metrics), figsize=(TEXT_WIDTH_IN, 1.95), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55] + [1.0] * (len(args.metrics) - 1)},
    )
    axes = np.atleast_1d(axes)
    records = []
    for index, (axis, metric) in enumerate(zip(axes, args.metrics)):
        records += _difference_panel(axis, arm_keys, args.comparator, cohort, metric,
                                     show_names=index == 0)
        axis.set_title(arms.METRIC_LABEL.get(metric, metric))
        _letter(axis, index)
    axes[0].set_xlabel("Difference from the no-consistency comparator", loc="left")

    caption = _caption_objective(cohort)
    _save(figure, args, f"objective_comparison_{cohort}")
    return {"objective": records, "weight_matched": arms.mse_is_matched(cohort),
            "mse_weight": arms.mse_weight(cohort), "suggested_caption": caption}


def figure_scale(args) -> dict:
    cohort = args.cohort
    arm_keys = _drop_unscored(args.scale_arms, cohort)
    if not arm_keys:
        print("  nothing to draw: no arm on the scale axis is scored")
        return {}

    figure, axes = plt.subplots(
        1, len(args.metrics), figsize=(TEXT_WIDTH_IN, 2.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55] + [1.0] * (len(args.metrics) - 1)},
    )
    axes = np.atleast_1d(axes)
    records = []
    for index, (axis, metric) in enumerate(zip(axes, args.metrics)):
        _scale_shading(axis, arm_keys)
        records += _level_panel(axis, arm_keys, cohort, metric, show_names=index == 0)
        axis.set_title(arms.METRIC_LABEL.get(metric, metric))
        _letter(axis, index)
    axes[0].set_xlabel(f"Score on the {arms.COHORT_SHORT[cohort].lower()} cohort",
                       loc="left")

    shaded = [a for a in arm_keys if a.startswith("scale")]
    caption = _caption_scale(shaded)
    _save(figure, args, f"supervision_scale_{cohort}")
    return {"scale": records, "shaded_arms": shaded, "suggested_caption": caption}


def figure_combined(args) -> dict:
    cohort = args.cohort
    level_arms = _drop_unscored(args.scale_arms, cohort)
    delta_arms = _drop_unscored(args.objective_arms, cohort)
    if not level_arms or not delta_arms:
        print("  nothing to draw: the combined figure needs both rows")
        return {}

    figure, axes = plt.subplots(
        2, len(args.metrics), figsize=(TEXT_WIDTH_IN, 3.9), constrained_layout=True,
        gridspec_kw={
            "width_ratios": [1.55] + [1.0] * (len(args.metrics) - 1),
            "height_ratios": [len(level_arms), len(delta_arms) + 0.6],
        },
    )
    axes = np.atleast_2d(axes)
    records = {"scale": [], "objective": []}
    for index, metric in enumerate(args.metrics):
        top, bottom = axes[0, index], axes[1, index]
        _scale_shading(top, level_arms)
        records["scale"] += _level_panel(top, level_arms, cohort, metric,
                                         show_names=index == 0)
        top.set_title(arms.METRIC_LABEL.get(metric, metric))
        _letter(top, index)

        records["objective"] += _difference_panel(
            bottom, delta_arms, args.comparator, cohort, metric, show_names=index == 0
        )
        _letter(bottom, len(args.metrics) + index)

    axes[0, 0].set_xlabel(f"Score on the {arms.COHORT_SHORT[cohort].lower()} cohort",
                          loc="left")
    axes[1, 0].set_xlabel("Difference from the no-consistency comparator", loc="left")

    caption = (
        "Top row: where each arm sits. " + _caption_scale(
            [a for a in level_arms if a.startswith("scale")]
        )
        + " Bottom row: " + _caption_objective(cohort)
    )
    _save(figure, args, f"objective_and_scale_{cohort}")
    records["suggested_caption"] = caption
    return records


def main() -> None:
    args = _parse_args()
    apply_theme()
    print(arms.describe_availability())

    wanted = set(args.figure)
    if "all" in wanted:
        wanted = {"objective", "scale", "combined"}

    provenance: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "dissertation/scripts/plot_arm_comparison.py",
        "cohort": args.cohort,
        "comparator": args.comparator,
        "metrics": args.metrics,
        "source_directories": {
            arm: arms.source_directory(arm, args.cohort)
            for arm in dict.fromkeys(args.objective_arms + args.scale_arms + [args.comparator])
        },
        "figures": {},
    }

    if "objective" in wanted:
        print("\nobjective comparison:")
        provenance["figures"]["objective_comparison"] = figure_objective(args)
    if "scale" in wanted:
        print("\nsupervision scale:")
        provenance["figures"]["supervision_scale"] = figure_scale(args)
    if "combined" in wanted:
        print("\ncombined:")
        provenance["figures"]["objective_and_scale"] = figure_combined(args)

    destination = arms.write_provenance(f"arm_comparison_{args.cohort}.json", provenance)
    print(f"\nWrote {destination}")

    # The caption is written here rather than drawn into the artwork, following the rest
    # of the document's figures. Printing it is what stops the wording in the chapter and
    # the substitution actually plotted from drifting apart.
    for name, record in provenance["figures"].items():
        caption = record.get("suggested_caption") if isinstance(record, dict) else None
        if caption:
            print(f"\ncaption text for {name}:\n  {caption}")


if __name__ == "__main__":
    main()
