"""Qualitative 3D renders of what the consistency term actually changed, per patient.

Three figure products, all built from the same masks and the same camera:

``qualitative_comparison_<case>``
    Reference, the no-consistency comparator, the Mean Teacher, and a fourth panel
    colouring the reference tree by which arm found it. This is the figure that shows a
    reader what a $+0.02$ tree-length difference looks like on an airway.

``error_classes_<arm>_<case>``
    One arm, its tree coloured true positive / false positive / false negative. Read on
    its own it says where a model's errors sit; read beside the same case for another arm
    it says which of them moved.

``distal_recovery_<case>``
    Only the fourth panel of the comparison figure, at full width, for when the recovery
    itself is the point being made rather than the three-way comparison.

Choosing the cases
------------------
A qualitative figure is worthless if the reader suspects the case was chosen to flatter
the result, so the choice is made by a stated rule and never by eye. Cases are ranked by
the paired difference in tree length detected, treatment minus comparator, on the cohort
being drawn. ``--select`` then names positions in that ranking rather than case
identifiers:

    median   the case at the middle of the ranking -- the representative one
    best     the largest gain
    worst    the smallest, which on the validation cohort is still a gain
    quartile the upper and lower quartile cases

The full ranking is printed and written to the provenance record, so an examiner can see
every case that was not chosen and where the chosen one sat. ``--cases`` overrides the
rule with explicit identifiers; the provenance then records that the selection was manual.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\render_case_gallery.py
    .venv\\Scripts\\python.exe dissertation\\scripts\\render_case_gallery.py \\
        --select median,best,worst --figure comparison
    .venv\\Scripts\\python.exe dissertation\\scripts\\render_case_gallery.py \\
        --cohort test --cases 002 --figure error-classes --arms control soft_f0
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import figure_arms as arms  # noqa: E402
import render_tree as rt  # noqa: E402
from figure_theme import apply_theme  # noqa: E402

DEFAULT_PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "appendix"
DEFAULT_PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "appendix"

SELECTION_RULES = ("median", "best", "worst", "quartile", "all")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=arms.COHORTS, default="val")
    parser.add_argument("--treatment", default="soft_f0", help="Arm key from figure_arms.")
    parser.add_argument("--comparator", default="control")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="Arms for the error-class figure. Defaults to comparator then treatment.",
    )
    parser.add_argument(
        "--figure",
        nargs="+",
        choices=("comparison", "error-classes", "recovery", "all"),
        default=["all"],
    )
    parser.add_argument(
        "--select",
        default="median",
        help="Comma-separated positions in the tree-length ranking: "
        + ", ".join(SELECTION_RULES),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Explicit case identifiers, overriding --select.",
    )
    parser.add_argument("--metric", default="td_raw", help="Metric the ranking uses.")
    parser.add_argument(
        "--recovery-basis",
        choices=("centreline", "voxel"),
        default="centreline",
        help="What the change panel classifies. The centreline basis is what tree "
        "length detected counts; the voxel basis is dominated by one-voxel wall "
        "disagreement on large airways and understates the branch recovery.",
    )
    parser.add_argument(
        "--centreline-dilate",
        type=int,
        default=1,
        help="Dilation applied to the classified centreline FOR DISPLAY ONLY; the "
        "counts printed on the panel are taken before it.",
    )
    parser.add_argument(
        "--change-dilate",
        type=int,
        default=2,
        help="Dilation for the recovered and lost classes only, so a hundred changed "
        "voxels in five thousand are visible at print size. Display only.",
    )
    parser.add_argument("--azimuth", type=float, default=-30.0)
    parser.add_argument("--elevation", type=float, default=10.0)
    parser.add_argument(
        "--px-mm",
        type=float,
        default=0.32,
        help="Millimetres per pixel of the final render, before supersampling.",
    )
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument(
        "--smooth-mm",
        type=float,
        default=0.25,
        help="Signed-distance smoothing before the isosurface. Capped by connectivity: "
        "past about 0.35 mm it pinches thin branches off, and the run prints the check.",
    )
    parser.add_argument("--taubin", type=int, default=16, help="Mesh smoothing iterations.")
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Case selection
# --------------------------------------------------------------------------
def rank_cases(treatment: str, comparator: str, cohort: str, metric: str) -> list[dict]:
    """Every case, ordered by the paired difference on ``metric``, worst first."""
    summary = arms.paired_summary(treatment, comparator, cohort, metric)
    if summary is None:
        raise SystemExit(
            f"Cannot rank cases: {treatment} or {comparator} is not scored on {cohort}.\n\n"
            + arms.describe_availability()
        )
    rows = [
        {
            "case_id": case_id,
            "difference": difference,
            "treatment": treated,
            "comparator": referenced,
        }
        for case_id, difference, treated, referenced in zip(
            summary["case_ids"],
            summary["differences"],
            summary["treatment"],
            summary["reference_values"],
        )
    ]
    rows.sort(key=lambda row: row["difference"])
    for position, row in enumerate(rows):
        row["rank"] = position + 1
    return rows


def select_cases(ranking: list[dict], rules: list[str]) -> list[tuple[str, str]]:
    """``(case_id, rule)`` pairs, in ranking order, with duplicates collapsed."""
    n = len(ranking)
    positions: dict[str, str] = {}
    for rule in rules:
        if rule == "all":
            for row in ranking:
                positions.setdefault(row["case_id"], "all")
        elif rule == "median":
            positions.setdefault(ranking[n // 2]["case_id"], "median")
        elif rule == "best":
            positions.setdefault(ranking[-1]["case_id"], "best")
        elif rule == "worst":
            positions.setdefault(ranking[0]["case_id"], "worst")
        elif rule == "quartile":
            positions.setdefault(ranking[n // 4]["case_id"], "lower quartile")
            positions.setdefault(ranking[(3 * n) // 4]["case_id"], "upper quartile")
        else:
            raise SystemExit(f"Unknown selection rule {rule!r}; choose from {SELECTION_RULES}")
    order = {row["case_id"]: row["rank"] for row in ranking}
    return sorted(positions.items(), key=lambda item: order[item[0]])


def print_ranking(ranking: list[dict], chosen: dict[str, str], cohort: str, metric: str) -> None:
    label = arms.METRIC_SHORT.get(metric, metric)
    print(f"\nCases ranked by paired {label} difference on the {arms.COHORT_SHORT[cohort]} "
          f"cohort (n={len(ranking)}):")
    print(f"  {'rank':>4}  {'case':<12} {'comparator':>11} {'treatment':>10} "
          f"{'difference':>11}   selection")
    for row in ranking:
        mark = chosen.get(row["case_id"], "")
        print(f"  {row['rank']:>4}  {arms.display_case(row['case_id'], cohort):<12} "
              f"{row['comparator']:>11.4f} {row['treatment']:>10.4f} "
              f"{row['difference']:>+11.4f}   {mark}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _voxel_note(count: int) -> str:
    return f"{count:,} voxels"


def _metric_note(rows: dict[str, dict] | None, case_id: str, keys=("td_raw", "dice_raw")) -> str:
    if rows is None or case_id not in rows:
        return ""
    parts = []
    for key in keys:
        value = rows[case_id].get(key)
        if value is not None:
            parts.append(f"{arms.METRIC_SHORT.get(key, key)} {value:.3f}")
    return "   ".join(parts)


def build_comparison(
    case: dict,
    cohort: str,
    treatment: str,
    comparator: str,
    *,
    basis: str,
    dilate: int,
    change_dilate: int,
) -> tuple[list, dict]:
    reference = case["masks"]["reference"]
    control = case["masks"][comparator]
    treated = case["masks"][treatment]
    case_id = case["case_id"]

    comparator_rows = arms.load_per_case(comparator, cohort)
    treatment_rows = arms.load_per_case(treatment, cohort)
    recovery, counts = rt.recovery_classes(
        control, treated, reference, basis=basis, dilate=dilate, change_dilate=change_dilate
    )
    recovered = counts[rt.RECOVERY_PALETTE[rt.RECOVERED][1]]
    lost = counts[rt.RECOVERY_PALETTE[rt.LOST][1]]
    # On the centreline basis these two counts divide by the skeleton length to give the
    # paired tree-length difference exactly, so the panel states the difference it implies
    # rather than leaving the reader to trust that the render matches the table. Kept short:
    # in a four-panel row a panel is 1.69 in wide.
    change = (
        "\n" + f"{counts['implied_delta']:+.4f} tree length"
        if basis == "centreline"
        else ""
    )

    panels = [
        rt.Panel("Reference", rt.reference_classes(reference), rt.REFERENCE_PALETTE,
                 subtitle=_voxel_note(int(reference.sum()))),
        rt.Panel(arms.ARMS[comparator].short, rt.error_classes(control, reference),
                 rt.ERROR_PALETTE, subtitle=_metric_note(comparator_rows, case_id)),
        rt.Panel(arms.ARMS[treatment].short, rt.error_classes(treated, reference),
                 rt.ERROR_PALETTE, subtitle=_metric_note(treatment_rows, case_id)),
        rt.Panel("Centreline change" if basis == "centreline" else "Change",
                 recovery, rt.RECOVERY_PALETTE,
                 subtitle=f"{recovered:,} recovered, {lost:,} lost{change}"),
    ]
    return panels, counts


def build_error_classes(case: dict, cohort: str, arm_keys: list[str]) -> list:
    reference = case["masks"]["reference"]
    panels = []
    for key in arm_keys:
        rows = arms.load_per_case(key, cohort)
        panels.append(
            rt.Panel(
                arms.ARMS[key].short,
                rt.error_classes(case["masks"][key], reference),
                rt.ERROR_PALETTE,
                subtitle=_metric_note(rows, case["case_id"],
                                      keys=("td_raw", "bd_raw", "prec_raw")),
            )
        )
    return panels


def render_case(
    case_id: str,
    rule: str,
    args: argparse.Namespace,
    wanted: set[str],
    ranking: list[dict],
) -> dict:
    cohort = args.cohort
    position = next(row["rank"] for row in ranking if row["case_id"] == case_id)
    standing = (f"selected as the {rule} case; rank {position} of {len(ranking)} "
                f"by paired tree-length difference"
                if rule != "chosen by hand"
                else f"chosen by hand; rank {position} of {len(ranking)} "
                     f"by paired tree-length difference")
    arm_keys = args.arms or [args.comparator, args.treatment]
    needed = sorted(set(arm_keys) | {args.comparator, args.treatment})
    directories = {}
    for key in needed:
        directory = arms.prediction_dir(key, cohort)
        if directory is None:
            raise SystemExit(f"Arm {key!r} has no predictions for cohort {cohort!r}.")
        directories[key] = directory

    started = time.time()
    print(f"\nLoading case {arms.display_case(case_id, cohort)} "
          f"({', '.join(needed)}) ...", flush=True)
    case = rt.load_case(case_id, cohort, directories)
    camera = rt.Camera(
        azimuth=args.azimuth,
        elevation=args.elevation,
        px_mm=args.px_mm,
        supersample=args.supersample,
        smooth_mm=args.smooth_mm,
        taubin_iterations=args.taubin,
    )

    record: dict = {
        "case_id": case_id,
        "cohort": cohort,
        "selection_rule": rule,
        "reference": str(rt.reference_path(case_id, cohort).relative_to(ROOT)),
        "prediction_dirs": {k: v.name for k, v in directories.items()},
        "figures": {},
    }

    if {"comparison", "recovery"} & wanted:
        panels, counts = build_comparison(
            case, cohort, args.treatment, args.comparator,
            basis=args.recovery_basis, dilate=args.centreline_dilate,
            change_dilate=args.change_dilate,
        )
        record["recovery_counts"] = counts

    if "comparison" in wanted:
        print("  rendering the four-panel comparison ...", flush=True)
        images, stats = rt.render_panels(panels, case["affine"], camera=camera)
        # Two palettes in one figure. Listed rather than merged by class id, because the
        # error and change palettes number their classes independently.
        legend = list(rt.ERROR_PALETTE.values()) + [
            rt.RECOVERY_PALETTE[k] for k in (rt.RECOVERED, rt.LOST, rt.MISSED_BOTH)
        ]
        destination = rt.compose(
            images, panels,
            pdf_dir=args.pdf_output_dir, png_dir=args.png_output_dir,
            stem=f"qualitative_comparison_{cohort}_{case_id}",
            legend=legend, legend_columns=3,
            caption=f"{arms.display_case(case_id, cohort)}, {standing}",
        )
        record["figures"]["comparison"] = destination.name
        record["render"] = stats["frame"]
        print(f"  wrote {destination}", flush=True)

    if "recovery" in wanted:
        print("  rendering the recovery panel ...", flush=True)
        single = [panels[3]]
        images, stats = rt.render_panels(single, case["affine"], camera=camera)
        destination = rt.compose(
            images, single,
            pdf_dir=args.pdf_output_dir, png_dir=args.png_output_dir,
            stem=f"distal_recovery_{cohort}_{case_id}",
            legend=list(rt.RECOVERY_PALETTE.values())[:4], legend_columns=2,
            caption=f"{arms.display_case(case_id, cohort)}, {standing}",
        )
        record["figures"]["recovery"] = destination.name
        print(f"  wrote {destination}", flush=True)

    if "error-classes" in wanted:
        print("  rendering the error-class panels ...", flush=True)
        panels = build_error_classes(case, cohort, arm_keys)
        images, stats = rt.render_panels(panels, case["affine"], camera=camera)
        stem = f"error_classes_{cohort}_{'_'.join(arm_keys)}_{case_id}"
        destination = rt.compose(
            images, panels,
            pdf_dir=args.pdf_output_dir, png_dir=args.png_output_dir,
            stem=stem,
            legend=rt.ERROR_PALETTE,
            caption=f"{arms.display_case(case_id, cohort)}, native output, {standing}",
        )
        record["figures"]["error_classes"] = destination.name
        record["render"] = stats["frame"]
        print(f"  wrote {destination}", flush=True)

    record["seconds"] = round(time.time() - started, 1)
    return record


def main() -> None:
    args = _parse_args()
    apply_theme()
    print(arms.describe_availability())

    wanted = set(args.figure)
    if "all" in wanted:
        wanted = {"comparison", "error-classes", "recovery"}

    ranking = rank_cases(args.treatment, args.comparator, args.cohort, args.metric)
    if args.cases:
        known = {row["case_id"] for row in ranking}
        selection = []
        for raw in args.cases:
            case_id = str(raw).zfill(3) if str(raw).isdigit() else str(raw)
            if case_id not in known:
                raise SystemExit(
                    f"Case {case_id} is not in the {args.cohort} cohort. "
                    f"Available: {', '.join(sorted(known))}"
                )
            selection.append((case_id, "chosen by hand"))
    else:
        selection = select_cases(ranking, [r.strip() for r in args.select.split(",")])

    print_ranking(ranking, dict(selection), args.cohort, args.metric)

    records = [render_case(case_id, rule, args, wanted, ranking)
               for case_id, rule in selection]

    destination = arms.write_provenance(
        f"case_gallery_{args.cohort}.json",
        {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "dissertation/scripts/render_case_gallery.py",
            "cohort": args.cohort,
            "treatment": args.treatment,
            "comparator": args.comparator,
            "selection": {
                "rule": "explicit --cases" if args.cases else args.select,
                "ranked_by": f"paired {args.metric} difference, treatment minus comparator",
                "ranking": ranking,
                "chosen": [{"case_id": c, "rule": r} for c, r in selection],
            },
            "cases": records,
        },
    )
    print(f"\nWrote {destination}")
    print(f"Figures in {args.pdf_output_dir}")


if __name__ == "__main__":
    main()
