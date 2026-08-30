"""When the trachea-seeded component filter helps, and when it deletes real airway.

The chapter reports every metric twice, native and after a connected-component filter, and
says in prose that the filter can remove a detached but genuine subtree. That is a claim
about geometry, and a reader cannot check it against a pair of numbers.

This draws it. For each selected case the raw prediction is rendered once, with every voxel
coloured by what the filter did to it and whether it was right to:

    grey    kept
    teal    removed, and it was a false positive -- the filter did its job
    red     removed, and it was annotated airway -- the filter deleted real tree

The filter is not read from a stored mask. It is applied here by the same function the
scorer calls, ``keep_component_containing_trachea``, at the same connectivity and with the
prediction's own affine, so the picture is of the operation the LCC columns actually report
and not of a re-implementation that might differ at the margin.

Choosing the cases
------------------
By the effect the filter had, from scores already on disk, never by eye:

    helps   the largest gain in precision from filtering
    hurts   the largest loss in tree length from filtering
    least   the case the filter changed least, as a null reference

The full ranking is printed and stored, so the two cases drawn can be located in it.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\render_postprocessing_cases.py
    .venv\\Scripts\\python.exe dissertation\\scripts\\render_postprocessing_cases.py \\
        --arm soft_5f --select helps,hurts,least
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import figure_arms as arms  # noqa: E402
import render_tree as rt  # noqa: E402
from figure_theme import apply_theme  # noqa: E402

from lung_airway_segmentation.inference.postprocess import (  # noqa: E402
    keep_component_containing_trachea,
)

DEFAULT_PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "appendix"
DEFAULT_PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "appendix"

SELECTION_RULES = ("helps", "hurts", "least")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=arms.COHORTS, default="val")
    parser.add_argument("--arm", default="soft_f0", help="Arm key from figure_arms.")
    parser.add_argument(
        "--select",
        default="helps,hurts",
        help="Comma-separated rules: " + ", ".join(SELECTION_RULES),
    )
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Explicit case identifiers, overriding --select.")
    parser.add_argument(
        "--connectivity", type=int, choices=(6, 18, 26), default=6,
        help="Must match the connectivity the scorer used for the LCC columns.",
    )
    parser.add_argument("--azimuth", type=float, default=-30.0)
    parser.add_argument("--elevation", type=float, default=10.0)
    parser.add_argument("--px-mm", type=float, default=0.32)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--smooth-mm", type=float, default=0.25)
    parser.add_argument("--taubin", type=int, default=16)
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    return parser.parse_args()


def rank_cases(arm: str, cohort: str) -> list[dict]:
    """Every case with what the filter did to it, from the stored scores."""
    rows = arms.load_per_case(arm, cohort)
    if rows is None:
        raise SystemExit(
            f"Arm {arm!r} is not scored on the {cohort} cohort.\n\n"
            + arms.describe_availability()
        )
    ranking = []
    for case_id, row in rows.items():
        ranking.append({
            "case_id": case_id,
            "delta_precision": row["prec_lcc"] - row["prec_raw"],
            "delta_td": row["td_lcc"] - row["td_raw"],
            "delta_dice": row["dice_lcc"] - row["dice_raw"],
            "retained_fraction": row.get("lcc_retained_fraction"),
        })
    ranking.sort(key=lambda r: r["delta_td"])
    return ranking


def select_cases(ranking: list[dict], rules: list[str]) -> list[tuple[str, str]]:
    chosen: dict[str, str] = {}
    for rule in rules:
        if rule == "helps":
            best = max(ranking, key=lambda r: r["delta_precision"])
            chosen.setdefault(best["case_id"], "largest precision gain")
        elif rule == "hurts":
            worst = min(ranking, key=lambda r: r["delta_td"])
            chosen.setdefault(worst["case_id"], "largest tree-length loss")
        elif rule == "least":
            quiet = max(ranking, key=lambda r: r["retained_fraction"] or 0.0)
            chosen.setdefault(quiet["case_id"], "least changed")
        else:
            raise SystemExit(f"Unknown rule {rule!r}; choose from {SELECTION_RULES}")
    return list(chosen.items())


def print_ranking(ranking: list[dict], chosen: dict[str, str], cohort: str) -> None:
    print(f"\nEffect of the component filter on the {arms.COHORT_SHORT[cohort]} cohort "
          f"(LCC minus native), ordered by tree-length change:")
    print(f"  {'case':<12} {'retained':>9} {'d TLD':>9} {'d Prec.':>9} {'d Dice':>9}"
          f"   selection")
    for row in ranking:
        retained = row["retained_fraction"]
        print(f"  {arms.display_case(row['case_id'], cohort):<12} "
              f"{(f'{retained:.4f}' if retained is not None else '--'):>9} "
              f"{row['delta_td']:>+9.4f} {row['delta_precision']:>+9.4f} "
              f"{row['delta_dice']:>+9.4f}   {chosen.get(row['case_id'], '')}")


def build_panel(
    case: dict, row: dict, cohort: str, connectivity: int
) -> tuple[rt.Panel, dict]:
    reference = case["masks"]["reference"]
    raw = case["masks"]["prediction"]
    filtered = keep_component_containing_trachea(
        raw.astype(np.uint8), connectivity=connectivity, affine=case["affine"]
    ) > 0

    labels = rt.postprocess_classes(raw, filtered, reference)
    counts = rt.class_counts(labels, rt.POSTPROCESS_PALETTE)
    removed_airway = counts["Removed, true airway"]
    removed_spurious = counts["Removed, false positive"]
    retained = float(filtered.sum()) / max(int(raw.sum()), 1)

    # Drawn by matplotlib, not typeset by LaTeX, so a per cent sign needs no escape.
    # Kept to about thirty characters a line: a panel is 2.2 in wide.
    subtitle = "\n".join([
        f"{retained * 100:.1f}% of voxels kept",
        f"{removed_spurious:,} spurious removed",
        f"{removed_airway:,} annotated airway removed",
        f"TLD {row['delta_td']:+.4f}, prec. {row['delta_precision']:+.4f}",
    ])
    title = arms.display_case(case["case_id"], cohort)
    return rt.Panel(title, labels, rt.POSTPROCESS_PALETTE, subtitle=subtitle), {
        **counts, "retained_fraction": retained,
    }


def main() -> None:
    args = _parse_args()
    apply_theme()
    print(arms.describe_availability())

    cohort, arm = args.cohort, args.arm
    directory = arms.prediction_dir(arm, cohort)
    if directory is None:
        raise SystemExit(f"Arm {arm!r} has no predictions for cohort {cohort!r}.")

    ranking = rank_cases(arm, cohort)
    if args.cases:
        known = {r["case_id"] for r in ranking}
        selection = []
        for raw_id in args.cases:
            case_id = str(raw_id).zfill(3) if str(raw_id).isdigit() else str(raw_id)
            if case_id not in known:
                raise SystemExit(f"Case {case_id} is not in the {cohort} cohort.")
            selection.append((case_id, "chosen by hand"))
    else:
        selection = select_cases(ranking, [r.strip() for r in args.select.split(",")])
    print_ranking(ranking, dict(selection), cohort)

    by_case = {r["case_id"]: r for r in ranking}
    camera = rt.Camera(
        azimuth=args.azimuth, elevation=args.elevation, px_mm=args.px_mm,
        supersample=args.supersample, smooth_mm=args.smooth_mm,
        taubin_iterations=args.taubin,
    )

    # Each case is rendered in its own frame: two different patients have no common
    # anatomy to align, so a shared frame would only shrink both to fit the larger chest.
    images, panels, records = [], [], []
    for case_id, rule in selection:
        print(f"\nLoading case {arms.display_case(case_id, cohort)} ({rule}) ...",
              flush=True)
        case = rt.load_case(case_id, cohort, {"prediction": directory})
        panel, counts = build_panel(case, by_case[case_id], cohort, args.connectivity)
        rendered, stats = rt.render_panels([panel], case["affine"], camera=camera)
        images.append(rendered[0])
        panels.append(panel)
        records.append({
            "case_id": case_id, "selection_rule": rule, "voxel_counts": counts,
            "score_deltas": by_case[case_id], "render": stats["frame"],
        })
        print(f"  {counts}", flush=True)

    stem = f"postprocessing_cases_{cohort}_{arm}"
    destination = rt.compose(
        images, panels,
        pdf_dir=args.pdf_output_dir, png_dir=args.png_output_dir, stem=stem,
        legend=list(rt.POSTPROCESS_PALETTE.values()), legend_columns=3,
        caption=f"{arms.ARMS[arm].label}, native output, "
                f"{args.connectivity}-connectivity trachea-seeded filter",
    )
    print(f"\nwrote {destination}")

    provenance = arms.write_provenance(
        f"postprocessing_cases_{cohort}_{arm}.json",
        {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "dissertation/scripts/render_postprocessing_cases.py",
            "cohort": cohort,
            "arm": arm,
            "prediction_dir": directory.name,
            "connectivity": args.connectivity,
            "selection": {
                "rule": "explicit --cases" if args.cases else args.select,
                "ranking": ranking,
                "chosen": [{"case_id": c, "rule": r} for c, r in selection],
            },
            "cases": records,
        },
    )
    print(f"Wrote {provenance}")


if __name__ == "__main__":
    main()
