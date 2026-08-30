"""Training diagnostics, but only the two questions worth a figure slot.

An ordinary loss curve says nothing a reader of this document needs, and the report has a
hard cap of twenty figures and tables. This script therefore draws only what the paired
replicate protocol was run to answer:

1. **How much of the reported difference could be training-run variation?** Replicates of
   one arm differ only in seed, so the spread between their trajectories is a direct
   picture of run-to-run noise, and the treatment-minus-comparator difference has to be
   read against it. This is the panel that either supports or undermines the chapter's
   central claim, which is why it is the one diagnostic that earns a slot.

2. **What does the consistency weight actually do to the optimiser?** The ramp is a
   scheduled quantity; the ratio of the weighted consistency gradient to the supervised
   one, and the cosine between them, are what that schedule produces. Together they show
   whether the consistency term is pulling with the supervised gradient or against it, and
   at what point in training it starts to matter.

It refuses to draw a seed-variability panel from one run per arm. A single trajectory
drawn as though it described a distribution is exactly the overclaim the paired protocol
exists to prevent, so with fewer replicates than ``--require-replicates`` the script says
what is missing and stops. Pass ``--allow-single`` to draw the trajectories anyway, which
relabels the figure as descriptive.

Input
-----
Parsed diagnostics CSVs, as written by ``scripts/parse_mt_diagnostics.py``. Each training
log becomes one CSV, and the arm and replicate are read from the file name:

    <arm>_rep<N>.csv        one replicate of one arm
    mt240_<arm>.csv         a historical single run, replicate unset

To produce them, from the repository root, for each log scp'd down from the HPC::

    .venv\\Scripts\\python.exe scripts\\parse_mt_diagnostics.py <log.txt> \\
        --out-stem softcldice_rep1

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\plot_training_diagnostics.py
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import figure_arms as arms  # noqa: E402
from figure_theme import (  # noqa: E402
    ANNOTATION_PT, ARM_COLOUR, GREY, INK, LABEL_PT, MUTED,
    apply_theme, finish, guide_line,
)

DEFAULT_INPUT_GLOB = "runs/mt_diagnostics/*.csv"
DEFAULT_PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "appendix"
DEFAULT_PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "appendix"
TEXT_WIDTH_IN = 6.76
PANEL_LETTERS = "abcdefgh"

# Arm names as they appear in a diagnostics file name, mapped onto the theme's colours so
# a trajectory and a score panel agree about which arm is which.
ARM_PATTERNS = (
    ("control", "control", "control"),
    ("softcldice", "soft-clDice", "mt_soft"),
    ("plainmse", "voxel MSE", "mt_mse"),
    ("mse", "voxel MSE", "mt_mse"),
    ("hard", "thresholded target", "mt_hard_f0"),
)

# Column, axis label, and whether a higher value is better. Defaults are the three that
# answer the two questions above; everything else the parser writes is available by name.
PANELS = {
    "pseudo_dice": ("Validation pseudo-Dice", "nnU-Net's own running validation score"),
    "grad_weighted_ratio": (
        "Weighted consistency / supervised gradient",
        "How large the consistency gradient is once the ramp has scaled it",
    ),
    "grad_cosine": (
        "Cosine between the two gradients",
        "Positive means the consistency term pulls with the supervised one",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--metrics", nargs="+", default=list(PANELS))
    parser.add_argument(
        "--require-replicates", type=int, default=2,
        help="Minimum replicates in at least one arm before a variability panel is drawn.",
    )
    parser.add_argument(
        "--allow-single", action="store_true",
        help="Draw the trajectories with fewer replicates than that, as description only.",
    )
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="Centred rolling mean over this many epochs. Epoch-level diagnostics are "
        "noisy enough to hide the trend; 1 disables it and the caption should say which.",
    )
    parser.add_argument(
        "--clip-percentile", type=float, default=99.0,
        help="Percentile the y-axis is clipped to on a heavy-tailed panel. The data is "
        "never clipped; the panel reports how many epochs fall outside the axis.",
    )
    parser.add_argument(
        "--final-epochs", type=int, default=25,
        help="Epochs averaged at the end of training for the printed spread.",
    )
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    return parser.parse_args()


def _identify(stem: str) -> tuple[str, str, int | None]:
    """``(arm key, arm label, replicate)`` from a diagnostics file name."""
    lowered = stem.lower()
    replicate = None
    match = re.search(r"rep(?:licate)?[_-]?(\d+)", lowered)
    if match:
        replicate = int(match.group(1))
    for token, label, theme_key in ARM_PATTERNS:
        if token in lowered:
            return theme_key, label, replicate
    return "unknown", stem, replicate


def load_runs(pattern: str) -> list[dict]:
    paths = sorted(ROOT.glob(pattern))
    runs = []
    for path in paths:
        frame = pd.read_csv(path)
        if "epoch" not in frame.columns:
            print(f"  skipping {path.name}: no epoch column")
            continue
        theme_key, label, replicate = _identify(path.stem)
        runs.append({
            "path": path,
            "stem": path.stem,
            "arm": theme_key,
            "arm_label": label,
            "replicate": replicate,
            "frame": frame.sort_values("epoch"),
        })
    return runs


def _series(frame: pd.DataFrame, column: str, smooth: int) -> tuple[np.ndarray, np.ndarray] | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().sum() < 3:
        return None
    if smooth > 1:
        values = values.rolling(smooth, center=True, min_periods=1).mean()
    epochs = frame["epoch"].to_numpy(dtype=float)
    keep = values.notna().to_numpy()
    return epochs[keep], values.to_numpy(dtype=float)[keep]


def _robust_limits(stacked: np.ndarray, percentile: float) -> tuple[float, float, int] | None:
    """Limits that survive one bad epoch, and how many points they exclude.

    The weighted gradient ratio is a ratio, so a single step where the supervised
    gradient nearly vanishes sends it three orders of magnitude above everything else and
    flattens the whole series onto the axis. Clipping the AXIS rather than the data keeps
    every point in the file and lets the panel report how many it could not show.
    """
    finite = stacked[np.isfinite(stacked)]
    if finite.size < 10:
        return None
    # An interquartile fence, not a percentile of the series itself. Smoothing spreads one
    # bad epoch over the whole window, so a p99 cut is computed from partly contaminated
    # values and lets the spike back in; quartiles are unmoved by any number of outliers
    # short of a quarter of the run.
    first, third = np.percentile(finite, [25.0, 75.0])
    spread = float(third - first)
    if spread <= 0:
        spread = float(np.percentile(finite, percentile) - first) or 1.0
    low, high = float(first) - 3.0 * spread, float(third) + 6.0 * spread
    # Never extend the axis past the data: the ratio is non-negative, and a fence
    # reaching below zero would advertise room the quantity cannot occupy.
    low = max(low, float(finite.min()) - 0.04 * spread)
    if float(finite.max()) <= high and float(finite.min()) >= low:
        return None                       # nothing pathological; leave the axis alone
    excluded = int((finite > high).sum() + (finite < low).sum())
    return low, high, excluded


def _replicate_counts(runs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run["arm"]] = counts.get(run["arm"], 0) + 1
    return counts


def _final_window(frame: pd.DataFrame, column: str, epochs: int) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.tail(epochs).mean()) if len(values) else None


def print_spread(runs: list[dict], column: str, window: int) -> list[dict]:
    """Run-to-run spread at the end of training, which is the number the claim needs."""
    print(f"\nMean {column} over the last {window} epochs:")
    records = []
    by_arm: dict[str, list[tuple[str, float]]] = {}
    for run in runs:
        value = _final_window(run["frame"], column, window)
        if value is None:
            continue
        by_arm.setdefault(run["arm"], []).append((run["stem"], value))
        records.append({"stem": run["stem"], "arm": run["arm"],
                        "replicate": run["replicate"], column: value})
    for arm, entries in by_arm.items():
        values = [v for _, v in entries]
        label = next(r["arm_label"] for r in runs if r["arm"] == arm)
        spread = (max(values) - min(values)) if len(values) > 1 else float("nan")
        print(f"  {label:<20} n={len(values)}  mean {np.mean(values):.4f}"
              + (f"  range {spread:.4f}" if len(values) > 1 else "  (single run)"))
        for stem, value in entries:
            print(f"      {stem:<34} {value:.4f}")
    return records


def figure_diagnostics(runs: list[dict], args, sufficient: bool) -> Path | None:
    metrics = [m for m in args.metrics
               if any(_series(r["frame"], m, args.smooth) for r in runs)]
    if not metrics:
        print("  none of the requested columns are present in any diagnostics file")
        return None

    figure, axes = plt.subplots(
        1, len(metrics), figsize=(TEXT_WIDTH_IN, 2.35), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for index, (axis, column) in enumerate(zip(axes, metrics)):
        drawn = []
        for run in runs:
            series = _series(run["frame"], column, args.smooth)
            if series is None:
                continue
            epochs, values = series
            colour = ARM_COLOUR.get(run["arm"], GREY)
            axis.plot(epochs, values, color=colour, linewidth=1.0, alpha=0.75,
                      solid_capstyle="round")
            drawn.append(values)
        limits = _robust_limits(np.concatenate(drawn), args.clip_percentile) if drawn else None
        if limits is not None:
            low, high, excluded = limits
            axis.set_ylim(low, high)
            if excluded:
                # Cased in white: the excluded points are by definition drawn as
                # lines running off the top of this panel, so the note has to sit
                # over them and stay readable.
                axis.annotate(
                    f"{excluded} epoch{'s' if excluded > 1 else ''} above the axis",
                    xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                    fontsize=ANNOTATION_PT, color=MUTED, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9),
                )
        if column == "grad_cosine":
            guide_line(axis, 0.0)
        if column == "grad_weighted_ratio":
            # The ramp is the scheduled input; the ratio is what the optimiser saw. Drawn
            # together so the lag between the two is visible rather than asserted.
            for run in runs:
                ramp = _series(run["frame"], "grad_weight", 1)
                if ramp is None:
                    continue
                axis.plot(ramp[0], ramp[1], color=MUTED, linewidth=0.8, alpha=0.5,
                          dashes=(2.4, 1.8))
                break
            axis.annotate("consistency weight ramp", xy=(0.03, 0.06),
                          xycoords="axes fraction", ha="left", va="bottom",
                          fontsize=ANNOTATION_PT, color=MUTED, zorder=6,
                          bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                                    alpha=0.9))
        axis.set_title(PANELS.get(column, (column, ""))[0])
        axis.set_xlabel("Epoch")
        axis.text(0.015, 0.985, PANEL_LETTERS[index], transform=axis.transAxes,
                  fontsize=LABEL_PT, fontweight="bold", va="top", ha="left", color=INK)
        finish(axis)

    seen: dict[str, str] = {}
    for run in runs:
        seen.setdefault(run["arm"], run["arm_label"])
    handles = [
        plt.Line2D([], [], color=ARM_COLOUR.get(arm, GREY), linewidth=1.4, label=label)
        for arm, label in seen.items()
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                  frameon=False)

    stem = "training_diagnostics" if sufficient else "training_diagnostics_single_run"
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

    runs = load_runs(args.input_glob)
    if not runs:
        raise SystemExit(
            f"No diagnostics CSVs matched {args.input_glob!r}.\n"
            "Parse a training log first, from the repository root:\n"
            "  .venv\\Scripts\\python.exe scripts\\parse_mt_diagnostics.py "
            "<training_log.txt> --out-stem softcldice_rep1"
        )

    print(f"Found {len(runs)} diagnostics file(s):")
    for run in runs:
        replicate = f"replicate {run['replicate']}" if run["replicate"] else "no replicate id"
        print(f"  {run['stem']:<34} {run['arm_label']:<20} {replicate:<16} "
              f"{len(run['frame'])} epochs")

    counts = _replicate_counts(runs)
    sufficient = max(counts.values()) >= args.require_replicates
    if not sufficient and not args.allow_single:
        print(
            f"\nNot drawing the figure. The seed-variability panel needs at least "
            f"{args.require_replicates} replicates of one arm, and the most any arm has "
            f"is {max(counts.values())}:"
        )
        for arm, count in counts.items():
            label = next(r["arm_label"] for r in runs if r["arm"] == arm)
            print(f"  {label:<20} {count}")
        print("\nA single trajectory cannot show run-to-run variation, and drawing it as "
              "though it could is the overclaim the paired protocol exists to prevent.\n"
              "Re-run with --allow-single to draw the trajectories as description only.")
        return

    print("\ndrawing:")
    destination = figure_diagnostics(runs, args, sufficient)

    records = {}
    for column in args.metrics:
        records[column] = print_spread(runs, column, args.final_epochs)

    provenance = arms.write_provenance(
        "training_diagnostics.json",
        {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "dissertation/scripts/plot_training_diagnostics.py",
            "input_glob": args.input_glob,
            "smooth_epochs": args.smooth,
            "final_epoch_window": args.final_epochs,
            "replicates_per_arm": counts,
            "replicate_requirement_met": sufficient,
            "runs": [
                {"stem": r["stem"], "arm": r["arm"], "replicate": r["replicate"],
                 "path": str(r["path"].relative_to(ROOT)), "epochs": len(r["frame"])}
                for r in runs
            ],
            "final_window": records,
            "figure": destination.name if destination else None,
        },
    )
    print(f"\nWrote {provenance}")


if __name__ == "__main__":
    main()
