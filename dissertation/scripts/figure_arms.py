"""One registry of scored arms, so no two figures can disagree about what an arm is.

Every result figure in the Results chapter reads the same per-case score files that
``statistics/paired_significance_tests.py`` reads: the ``*topology*.json`` written beside
each prediction directory by ``scripts/evaluate_nnunet_predictions.py``. That file holds a
``table_per_case`` list, one row per case keyed by ``case_id``, so every comparison built
on it is PAIRED by patient rather than by cohort mean.

Two design decisions worth stating once:

* **Arms are named by what they are, not by dataset number.** ``Dataset126_val_mt240_...``
  is an implementation detail and is confined to this file. A figure asks for ``soft_f0``
  on the ``val`` cohort and gets whichever directory currently holds it.
* **A directory is a list of candidates, not a string.** An arm that has not been scored
  yet still has an entry here, listing the name its output will take. ``resolve`` returns
  the first candidate that exists, so a figure that substitutes an unmatched arm today
  picks up the matched one the moment it lands, with no edit. ``available`` is the guard a
  script uses to decide which of the two it drew, and every script that substitutes must
  say so on the figure itself.

Nothing here invents a number. An arm that is not on disk returns ``None``, and the
calling script is expected to draw the figure without it and state what is missing.

Run it directly to see what is currently scored::

    .venv\\Scripts\\python.exe dissertation\\scripts\\figure_arms.py
"""

from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRED_ROOT = ROOT / "data" / "nnunet" / "predict_out"
PROVENANCE_ROOT = ROOT / "dissertation" / "Figures" / "provenance"

# Cohort keys used throughout. "ood" is AeroPath, a different scanner and a different
# annotation protocol, so it is never pooled with the two ATM'22 cohorts.
COHORTS = ("val", "test", "ood")
COHORT_LABEL = {
    "val": "External validation",
    "test": "Held-out test",
    "ood": "AeroPath (out of distribution)",
}
COHORT_SHORT = {"val": "Validation", "test": "Held-out test", "ood": "AeroPath"}

# Metric keys as the scorer writes them. Raw is primary everywhere in the chapter; the
# ``_lcc`` twin is the trachea-seeded connected-component sensitivity.
METRIC_LABEL = {
    "dice_raw": "Dice",
    "td_raw": "Tree length detected",
    "bd_raw": "Branch detection",
    "prec_raw": "Precision",
    "cldice_raw": "clDice",
    "dice_lcc": "Dice (LCC)",
    "td_lcc": "Tree length detected (LCC)",
    "bd_lcc": "Branch detection (LCC)",
    "prec_lcc": "Precision (LCC)",
    "cldice_lcc": "clDice (LCC)",
}
METRIC_SHORT = {
    "dice_raw": "Dice",
    "td_raw": "TLD",
    "bd_raw": "BD",
    "prec_raw": "Prec.",
    "cldice_raw": "clDice",
    "dice_lcc": "Dice",
    "td_lcc": "TLD",
    "bd_lcc": "BD",
    "prec_lcc": "Prec.",
    "cldice_lcc": "clDice",
}
# The order the chapter reports them in: topology first, because that is the axis the
# treatment is claimed to move, then the two overlap metrics it must not damage.
PRIMARY_METRICS = ("td_raw", "bd_raw", "dice_raw", "prec_raw")


@dataclass(frozen=True)
class Arm:
    """One trained configuration, and where its scores live per cohort."""

    key: str
    label: str                       # full name, for a legend or an axis
    short: str                       # tick-label length
    dirs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    theme_key: str | None = None     # colour/dash lookup in figure_theme, when it has one
    note: str = ""                   # caveat a figure must print if it draws this arm

    def resolve(self, cohort: str) -> Path | None:
        """First candidate directory for ``cohort`` that exists on disk."""
        for pattern in self.dirs.get(cohort, ()):  # ordered: matched arm first
            for hit in sorted(PRED_ROOT.glob(pattern)):
                if hit.is_dir():
                    return hit
        return None


# The registry. Candidate lists are ordered best-first: where an arm has a matched and an
# unmatched variant, the matched one is listed first so it wins automatically once scored.
ARMS: dict[str, Arm] = {
    "control": Arm(
        key="control",
        label="No consistency (comparator)",
        short="No consistency",
        theme_key="control",
        dirs={
            "val": ("Dataset126_val_mt240_control_final_teacher",),
            "test": ("Dataset126_test_mt240_control_final_teacher",),
            "ood": ("Dataset126_aeropath_mt240_control_final_teacher",),
        },
    ),
    "soft_f0": Arm(
        key="soft_f0",
        label=r"Soft-clDice consistency ($w_{\max}{=}0.10$)",
        short="Soft-clDice",
        theme_key="mt_soft",
        dirs={
            "val": ("Dataset126_val_mt240_softcldice_final_teacher",),
            "test": ("Dataset126_test_mt240_softcldice_final_teacher",),
            "ood": ("Dataset126_aeropath_mt240_softcldice_final_teacher",),
        },
    ),
    "soft_5f": Arm(
        key="soft_5f",
        label="Soft-clDice, five-fold ensemble",
        short="Soft-clDice 5f",
        theme_key="mt_soft_5f",
        dirs={
            "val": ("Dataset126_val_mt240_softcldice_5fold_final_teacher",),
            "test": ("Dataset126_test_mt240_softcldice_5fold_final_teacher",),
            "ood": ("Dataset126_aeropath_mt240_softcldice_5fold_final_teacher",),
        },
    ),
    "soft_w003": Arm(
        key="soft_w003",
        label=r"Soft-clDice consistency ($w_{\max}{=}0.03$)",
        short=r"Soft-clDice $w{=}0.03$",
        dirs={"val": ("Dataset126_val_mt240_softcldice_w003_final_teacher",)},
    ),
    "soft_w030": Arm(
        key="soft_w030",
        label=r"Soft-clDice consistency ($w_{\max}{=}0.30$)",
        short=r"Soft-clDice $w{=}0.30$",
        dirs={"val": ("Dataset126_val_mt240_softcldice_w030_final_teacher",)},
    ),
    "hard_f0": Arm(
        key="hard_f0",
        label="Thresholded target, fold 0",
        short="Thresholded",
        theme_key="mt_hard_f0",
        dirs={
            "val": ("Dataset126_val_mt240_lungcrop_final_teacher",),
            "test": ("Dataset126_test_mt240_hard_f0_final_teacher",),
            "ood": ("Dataset126_aeropath_mt240_hard_f0_final_teacher",),
        },
    ),
    "hard_5f": Arm(
        key="hard_5f",
        label="Thresholded target, five-fold ensemble",
        short="Thresholded 5f",
        theme_key="mt_hard_5f",
        dirs={
            "val": ("Dataset126_val_mt240_5f_lungcrop_final_teacher",),
            "test": ("Dataset126_test_mt240_hard_5f_final_teacher",),
            "ood": ("Dataset126_aeropath_mt240_5f_final_teacher",),
        },
    ),
    # The objective ablation. The matched weight is listed first and does not exist yet;
    # until it does, resolve() returns the 0.30 run, and every figure drawing it must
    # print that the weights are not matched.
    "mse": Arm(
        key="mse",
        label="Voxel-MSE consistency",
        short="Voxel MSE",
        theme_key="mt_mse",
        dirs={
            "val": (
                "Dataset126_val_mt240_plainmse_w010_final_teacher",
                "Dataset126_val_mt240_plainmse_final_teacher",
                "Dataset126_val_mt240_plainmse_w030_final_teacher",
            ),
            "test": (
                "Dataset126_test_mt240_plainmse_w010_final_teacher",
                "Dataset126_test_mt240_plainmse_w030_final_teacher",
            ),
            "ood": (
                "Dataset126_aeropath_mt240_plainmse_w010_final_teacher",
                "Dataset126_aeropath_mt240_plainmse_w030_final_teacher",
            ),
        },
        note="weight not matched to the soft-clDice arm",
    ),
    "seed": Arm(
        key="seed",
        label="16-label seed",
        short="16-label seed",
        theme_key="seed16",
        dirs={
            "val": ("Dataset123_val_seed_lungcrop",),
            "test": ("Dataset123_test_seed_f0",),
            "ood": ("Dataset123_aeropath_seed_f0",),
        },
    ),
    "scale110": Arm(
        key="scale110",
        label="110-case pool supervised scale reference",
        short="110-case pool",
        theme_key="ceiling110",
        dirs={
            "val": ("Dataset111_val_l110_nods_nomirror_f0",),
            "test": ("Dataset111_test_l110_nods_nomirror_f0",),
            "ood": ("Dataset111_aeropath_l110_nods_nomirror_f0",),
        },
        note="scale reference, not a matched label-efficiency point",
    ),
    # From scratch, all 260 labels. Not yet scored; the glob is deliberately loose
    # because the prediction directory has not been named yet.
    "scale260": Arm(
        key="scale260",
        label="260-case pool supervised scale reference",
        short="260-case pool",
        theme_key="ceiling260",
        dirs={
            "val": ("Dataset127_val_sup260*", "Dataset127_val*"),
            "test": ("Dataset127_test_sup260*", "Dataset127_test*"),
            "ood": ("Dataset127_aeropath_sup260*", "Dataset127_aeropath*"),
        },
        note="scale reference, not a matched label-efficiency point",
    ),
}

# The weight actually carried by whichever voxel-MSE directory resolved, recovered from
# the directory name rather than assumed, so a caption cannot state the wrong one.
_MSE_WEIGHTS = {"w010": "0.10", "w030": "0.30", "w003": "0.03"}


def mse_weight(cohort: str = "val") -> str | None:
    resolved = ARMS["mse"].resolve(cohort)
    if resolved is None:
        return None
    for token, weight in _MSE_WEIGHTS.items():
        if token in resolved.name:
            return weight
    return None


def mse_is_matched(cohort: str = "val", target: str = "0.10") -> bool:
    """True when the resolved voxel-MSE arm carries the soft-clDice weight."""
    return mse_weight(cohort) == target


def _score_file(directory: Path) -> Path | None:
    """The cohort-wide score file in a prediction directory.

    Some directories also hold single-case probes and connectivity sweeps written during
    debugging. Those are excluded by name rather than by date, so re-running a probe
    cannot silently change which file a figure reads.
    """
    hits = [
        Path(path)
        for path in glob.glob(str(directory / "*topology*.json"))
        if "case016" not in Path(path).name
        and "lcc_sensitivity" not in Path(path).name
        and "_part" not in Path(path).name
    ]
    return sorted(hits)[0] if hits else None


def load_per_case(arm: str, cohort: str) -> dict[str, dict] | None:
    """``{case_id: metric row}`` for one arm on one cohort, or None if unscored."""
    directory = ARMS[arm].resolve(cohort)
    if directory is None:
        return None
    score_file = _score_file(directory)
    if score_file is None:
        return None
    rows = json.loads(score_file.read_text()).get("table_per_case")
    if not rows:
        return None
    return {str(row["case_id"]): row for row in rows}


def available(arm: str, cohort: str) -> bool:
    return load_per_case(arm, cohort) is not None


def source_directory(arm: str, cohort: str) -> str | None:
    directory = ARMS[arm].resolve(cohort)
    return directory.name if directory is not None else None


def prediction_dir(arm: str, cohort: str) -> Path | None:
    """Where the masks themselves live, for the renders rather than the plots."""
    return ARMS[arm].resolve(cohort)


def paired_values(
    arm: str, reference: str, cohort: str, metric: str
) -> tuple[list[str], list[float], list[float]] | None:
    """Case ids and the two aligned value lists, intersected on case id.

    Returns None when either arm is unscored, so a caller can skip a panel rather than
    draw an unpaired one.
    """
    treatment_rows = load_per_case(arm, cohort)
    reference_rows = load_per_case(reference, cohort)
    if treatment_rows is None or reference_rows is None:
        return None
    kept, treated, referenced = [], [], []
    for case_id in sorted(set(treatment_rows) & set(reference_rows)):
        a = treatment_rows[case_id].get(metric)
        b = reference_rows[case_id].get(metric)
        if a is None or b is None:
            continue
        kept.append(case_id)
        treated.append(float(a))
        referenced.append(float(b))
    return (kept, treated, referenced) if kept else None


def paired_summary(arm: str, reference: str, cohort: str, metric: str) -> dict | None:
    """Mean paired difference with a t interval, and the win count.

    The interval is the Student one on the per-case differences, matching
    ``statistics/paired_significance_tests.py``, so a figure and the table of tests cannot
    quote different intervals for the same contrast.
    """
    paired = paired_values(arm, reference, cohort, metric)
    if paired is None:
        return None
    case_ids, treated, referenced = paired
    differences = [a - b for a, b in zip(treated, referenced)]
    n = len(differences)
    mean = sum(differences) / n
    if n > 1:
        from scipy import stats  # local: keeps the module importable without scipy

        sd = math.sqrt(sum((d - mean) ** 2 for d in differences) / (n - 1))
        se = sd / math.sqrt(n)
        half = float(stats.t.ppf(0.975, n - 1)) * se
        p_wilcoxon = _wilcoxon(differences)
    else:
        sd = se = half = float("nan")
        p_wilcoxon = float("nan")
    return {
        "arm": arm,
        "reference": reference,
        "cohort": cohort,
        "metric": metric,
        "case_ids": case_ids,
        "differences": differences,
        "treatment": treated,
        "reference_values": referenced,
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "p_wilcoxon": p_wilcoxon,
        "wins": sum(1 for d in differences if d > 0),
    }


def _wilcoxon(differences: list[float]) -> float:
    """Signed-rank p, or 1.0 where it is undefined (every difference zero)."""
    from scipy import stats

    try:
        return float(stats.wilcoxon(differences).pvalue)
    except ValueError:
        return 1.0


def cohort_mean(arm: str, cohort: str, metric: str) -> float | None:
    rows = load_per_case(arm, cohort)
    if rows is None:
        return None
    values = [row[metric] for row in rows.values() if row.get(metric) is not None]
    return sum(values) / len(values) if values else None


def display_case(case_id: str, cohort: str) -> str:
    """Case label for an axis. AeroPath ids are staged into the ATM 900 block."""
    text = str(case_id).lstrip("0") or "0"
    if cohort == "ood":
        number = int(text)
        return f"AeroPath {number - 900}" if number > 900 else f"AeroPath {number}"
    return f"ATM {str(case_id).zfill(3)}"


def write_provenance(name: str, payload: dict) -> Path:
    """Record what a figure was built from, beside the other figure provenance."""
    PROVENANCE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = PROVENANCE_ROOT / name
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def describe_availability() -> str:
    """A table of what is scored, printed by every script before it draws anything."""
    width = max(len(a.short) for a in ARMS.values()) + 2
    lines = [f"{'arm':<{width}}" + "".join(f"{COHORT_SHORT[c]:>16}" for c in COHORTS)]
    for arm in ARMS.values():
        cells = []
        for cohort in COHORTS:
            directory = arm.resolve(cohort)
            cells.append("scored" if directory and _score_file(directory) else "--")
        lines.append(f"{arm.short:<{width}}" + "".join(f"{cell:>16}" for cell in cells))
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_availability())
    weight = mse_weight("val")
    state = "matched" if mse_is_matched("val") else "NOT matched to soft-clDice"
    print(f"\nvoxel-MSE arm resolved to w_max={weight} ({state})")
