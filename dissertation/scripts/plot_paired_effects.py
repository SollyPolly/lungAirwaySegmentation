"""Patient-level treatment effect across all three evaluation cohorts.

The chapter currently supports "the gain is diffuse rather than carried by a few cases"
with win counts, and "it survives a change of cohort" with three separate tables of means.
Both claims are about the DISTRIBUTION of per-patient differences, which a mean cannot
show and a win count can only hint at.

Two figures, from the same numbers:

``paired_effect_summary``
    One panel per metric; within a panel the three cohorts stand side by side, each with
    every patient's difference, the cohort mean and its 95\\% interval. Putting the
    cohorts adjacent on one axis is what makes this a generalisation figure rather than
    three unrelated ones: the reader compares effect sizes directly, including on
    AeroPath, which is a different scanner and a different annotation protocol.

``paired_effect_cases``
    One panel per cohort for a single metric, patients ordered by their own difference and
    labelled. This is the panel to point at when the question is whether one or two
    patients carry the result.

Both are strictly paired: a patient enters only if both arms scored it, and the difference
is always treatment minus comparator on the same patient. AeroPath is never pooled with
the two ATM'22 cohorts.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\plot_paired_effects.py
    .venv\\Scripts\\python.exe dissertation\\scripts\\plot_paired_effects.py \\
        --treatment soft_5f --figure summary
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
    ANNOTATION_PT, GREY, INK, LABEL_PT, MUTED, TICK_PT, apply_theme, finish, guide_line,
)

DEFAULT_PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "appendix"
DEFAULT_PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "appendix"
TEXT_WIDTH_IN = 6.76
PANEL_LETTERS = "abcdefgh"

# One tone per cohort. Deliberately a lightness ramp of ONE hue rather than three hues:
# the cohorts are the same contrast measured three times, not three treatments, and giving
# them arm-like colours would invite the reader to read them as arms.
COHORT_TONE = {"val": "#2C6E9B", "test": "#5C9BC4", "ood": "#9CC3DC"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", default="soft_f0")
    parser.add_argument("--comparator", default="control")
    parser.add_argument("--cohorts", nargs="+", default=list(arms.COHORTS))
    parser.add_argument("--metrics", nargs="+", default=list(arms.PRIMARY_METRICS))
    parser.add_argument(
        "--case-metric",
        default="td_raw",
        help="Metric the per-patient panel uses; only one fits on a labelled axis.",
    )
    parser.add_argument(
        "--figure", nargs="+", choices=("summary", "cases", "all"), default=["all"]
    )
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    return parser.parse_args()


def _jitter(n: int, width: float = 0.15, seed: int = 7) -> np.ndarray:
    return np.linspace(-width, width, n) if n < 4 else (
        np.random.default_rng(seed).uniform(-width, width, n)
    )


def _usable_cohorts(args) -> list[str]:
    kept = []
    for cohort in args.cohorts:
        if arms.available(args.treatment, cohort) and arms.available(args.comparator, cohort):
            kept.append(cohort)
        else:
            print(f"  skipping the {cohort!r} cohort: one of the two arms is not scored")
    return kept


def _letter(axis, index: int) -> None:
    axis.text(0.015, 0.985, PANEL_LETTERS[index], transform=axis.transAxes,
              fontsize=LABEL_PT, fontweight="bold", va="top", ha="left", color=INK)


# --------------------------------------------------------------------------
# Figure 1: effect size per metric, cohorts side by side
# --------------------------------------------------------------------------
def figure_summary(args, cohorts: list[str]) -> dict:
    figure, axes = plt.subplots(
        1, len(args.metrics), figsize=(TEXT_WIDTH_IN, 2.45), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    records = []

    for index, (axis, metric) in enumerate(zip(axes, args.metrics)):
        guide_line(axis, 0.0)
        for position, cohort in enumerate(cohorts):
            summary = arms.paired_summary(args.treatment, args.comparator, cohort, metric)
            if summary is None:
                continue
            colour = COHORT_TONE.get(cohort, GREY)
            differences = np.asarray(summary["differences"])
            axis.scatter(
                position + _jitter(len(differences)), differences,
                s=7.0, facecolor=colour, edgecolor="none", alpha=0.45, zorder=2,
            )
            axis.errorbar(
                position, summary["mean"],
                yerr=[[summary["mean"] - summary["ci_low"]],
                      [summary["ci_high"] - summary["mean"]]],
                fmt="s", markersize=4.4, color=colour, ecolor=colour,
                elinewidth=1.2, capsize=2.6, capthick=1.2, zorder=4,
            )
            axis.annotate(
                f"{summary['wins']}/{summary['n']}",
                xy=(position, 0.02), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=ANNOTATION_PT, color=MUTED,
            )
            records.append(summary)

        axis.set_xticks(range(len(cohorts)))
        axis.set_xticklabels([arms.COHORT_SHORT[c] for c in cohorts], rotation=20,
                             ha="right", rotation_mode="anchor")
        axis.set_xlim(-0.6, len(cohorts) - 0.4)
        axis.set_title(arms.METRIC_LABEL.get(metric, metric))
        _letter(axis, index)
        finish(axis)
        # Clear space under the data for the win counts, which sit on the floor of the
        # panel rather than beside a mean that may be sitting on the zero guide.
        low, high = axis.get_ylim()
        axis.set_ylim(low - 0.12 * (high - low), high)

    axes[0].set_ylabel(f"{arms.ARMS[args.treatment].short}\nminus {arms.ARMS[args.comparator].short}")
    stem = f"paired_effect_summary_{args.treatment}"
    _save(figure, args, stem)
    return {"records": records, "stem": stem}


# --------------------------------------------------------------------------
# Figure 2: one patient per row, so a single dominant case would be visible
# --------------------------------------------------------------------------
def figure_cases(args, cohorts: list[str]) -> dict:
    metric = args.case_metric
    summaries = [
        arms.paired_summary(args.treatment, args.comparator, cohort, metric)
        for cohort in cohorts
    ]
    pairs = [(c, s) for c, s in zip(cohorts, summaries) if s is not None]
    if not pairs:
        print("  nothing to draw: no cohort has both arms scored")
        return {}

    counts = [s["n"] for _, s in pairs]
    # The three cohorts share ONE difference axis, which is the comparison the figure is
    # for. They do not share a row pitch: the cohorts hold different numbers of patients,
    # and forcing a common pitch would leave a quarter of the twenty-case panels blank
    # for a nicety no claim depends on.
    figure, axes = plt.subplots(
        1, len(pairs), figsize=(TEXT_WIDTH_IN, 0.135 * max(counts) + 1.15),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    span = max(
        max(abs(d) for d in s["differences"]) for _, s in pairs
    ) * 1.12

    for index, (axis, (cohort, summary)) in enumerate(zip(axes, pairs)):
        order = np.argsort(summary["differences"])[::-1]  # largest gain at the top
        differences = np.asarray(summary["differences"])[order]
        case_ids = [summary["case_ids"][i] for i in order]
        colour = COHORT_TONE.get(cohort, GREY)
        positions = np.arange(len(differences))

        axis.axvline(0.0, color=INK, linewidth=1.0, alpha=0.75, dashes=(3.5, 2.0), zorder=1)
        axis.axvspan(summary["ci_low"], summary["ci_high"], color=colour, alpha=0.12,
                     linewidth=0, zorder=0)
        axis.axvline(summary["mean"], color=colour, linewidth=1.1, zorder=2)
        axis.hlines(positions, 0.0, differences, color=colour, linewidth=0.9, alpha=0.55,
                    zorder=3)
        axis.scatter(differences, positions, s=11.0, facecolor=colour, edgecolor="none",
                     zorder=4)

        axis.set_yticks(positions)
        axis.set_yticklabels([arms.display_case(c, cohort) for c in case_ids],
                             fontsize=TICK_PT - 1.4)
        # Every panel spans the same difference range, so the three cohorts are directly
        # comparable rather than each self-scaled.
        axis.set_xlim(-span, span)
        axis.set_ylim(len(differences) - 0.4, -0.8)
        axis.set_title(
            f"{arms.COHORT_SHORT[cohort]} ($n={summary['n']}$, "
            f"{summary['wins']} improved)"
        )
        _letter(axis, index)
        finish(axis)

    label = arms.METRIC_LABEL.get(metric, metric)
    for axis in axes:
        axis.set_xlabel(f"Difference in {label.lower()}")
    stem = f"paired_effect_cases_{args.treatment}_{metric}"
    _save(figure, args, stem)
    return {"metric": metric, "stem": stem,
            "records": [s for _, s in pairs]}


def _save(figure, args, stem: str) -> Path:
    args.pdf_output_dir.mkdir(parents=True, exist_ok=True)
    args.png_output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.pdf_output_dir / f"{stem}.pdf"
    figure.savefig(destination, facecolor="white")
    figure.savefig(args.png_output_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(figure)
    print(f"  wrote {destination}")
    return destination


def main() -> None:
    args = _parse_args()
    apply_theme()
    print(arms.describe_availability())

    wanted = set(args.figure)
    if "all" in wanted:
        wanted = {"summary", "cases"}

    print(f"\n{arms.ARMS[args.treatment].label} minus {arms.ARMS[args.comparator].label}:")
    cohorts = _usable_cohorts(args)
    if not cohorts:
        raise SystemExit("No cohort has both arms scored; nothing to draw.")

    provenance: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "dissertation/scripts/plot_paired_effects.py",
        "treatment": args.treatment,
        "comparator": args.comparator,
        "cohorts": cohorts,
        "source_directories": {
            cohort: {
                "treatment": arms.source_directory(args.treatment, cohort),
                "comparator": arms.source_directory(args.comparator, cohort),
            }
            for cohort in cohorts
        },
        "figures": {},
    }

    if "summary" in wanted:
        provenance["figures"]["paired_effect_summary"] = figure_summary(args, cohorts)
    if "cases" in wanted:
        provenance["figures"]["paired_effect_cases"] = figure_cases(args, cohorts)

    for cohort in cohorts:
        line = [f"  {arms.COHORT_SHORT[cohort]:>14}"]
        for metric in args.metrics:
            summary = arms.paired_summary(args.treatment, args.comparator, cohort, metric)
            if summary is None:
                continue
            line.append(f"{arms.METRIC_SHORT.get(metric, metric)} {summary['mean']:+.4f} "
                        f"({summary['wins']}/{summary['n']})")
        print("   ".join(line))

    destination = arms.write_provenance(
        f"paired_effects_{args.treatment}.json", provenance
    )
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
