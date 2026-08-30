"""Paired significance tests for every comparison the dissertation actually claims.

Examiners are entitled to see the test statistics, not a prose assertion that a difference
"fell within" something. This script is the single source of those numbers: it emits a LaTeX
table body, a set of ``\\newcommand`` macros for inline prose, and a JSON provenance record,
all from the per-case score files. Prose therefore cannot drift from the data -- re-run the
script and the document updates.

Every cohort is scored per case, so every comparison here is PAIRED by ``case_id``:
the paired t-statistic is t = mean(d) / (sd(d)/sqrt(n)) on the per-case differences d,
with df = n-1, which is a one-sample t-test of d against zero.

Two guards against overclaiming, both requested by the supervisor's own reading of the
earlier draft:

* **Multiple comparisons.** Each contrast tests five metrics, so a Bonferroni-adjusted
  p-value (p x 5, capped at 1) is reported alongside the raw one. Claims are made on the
  adjusted column.
* **Normality.** The paired t assumes roughly normal differences, which is questionable for
  bounded metrics on n=20 with outliers. A Wilcoxon signed-rank p-value is reported beside
  it; where the two disagree the distribution-free result is the honest one to quote.

Run from the repo root:
    .venv\\Scripts\\python.exe dissertation/scripts/statistics/paired_significance_tests.py
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

from scipy import stats

PRED_ROOT = Path("data/nnunet/predict_out")
OUT_TABLES = Path("dissertation/Tables")
OUT_PROVENANCE = Path("dissertation/Figures/provenance")

METRICS = [
    ("dice_raw", "Dice", "Dice"),
    ("td_raw", "TLD", "TLD"),
    ("bd_raw", "BD", "BD"),
    ("prec_raw", "Prec", "Prec."),
    ("cldice_raw", "clDice", "clDice"),
]
N_METRICS = len(METRICS)          # Bonferroni family size, per contrast
ALPHA = 0.05

# Directory names per arm, per cohort.
VAL = {
    "ceiling": "Dataset111_val_l110_nods_nomirror_f0",
    "seed": "Dataset123_val_seed_lungcrop",
    "control": "Dataset126_val_mt240_control_final_teacher",
    "hard_f0": "Dataset126_val_mt240_lungcrop_final_teacher",
    "hard_5f": "Dataset126_val_mt240_5f_lungcrop_final_teacher",
    "soft_f0": "Dataset126_val_mt240_softcldice_final_teacher",
    "soft_b2": "Dataset126_val_mt240_softcldice_asym_final_teacher",
}
TEST = {
    "ceiling": "Dataset111_test_l110_nods_nomirror_f0",
    "seed": "Dataset123_test_seed_f0",
    "control": "Dataset126_test_mt240_control_final_teacher",
    "hard_f0": "Dataset126_test_mt240_hard_f0_final_teacher",
    "hard_5f": "Dataset126_test_mt240_hard_5f_final_teacher",
    "soft_f0": "Dataset126_test_mt240_softcldice_final_teacher",
    "soft_b2": "Dataset126_test_mt240_softcldice_asym_final_teacher",
}
OOD = {
    "ceiling": "Dataset111_aeropath_l110_nods_nomirror_f0",
    "seed": "Dataset123_aeropath_seed_f0",
    "control": "Dataset126_aeropath_mt240_control_final_teacher",
    "hard_f0": "Dataset126_aeropath_mt240_hard_f0_final_teacher",
    "hard_5f": "Dataset126_aeropath_mt240_5f_final_teacher",
    "soft_f0": "Dataset126_aeropath_mt240_softcldice_final_teacher",
    "soft_b2": "Dataset126_aeropath_mt240_softcldice_asym_final_teacher",
}

LABELS = {
    "ceiling": "110-label supervised",
    "seed": "16-label seed",
    "control": "no-consistency control",
    "hard_f0": "thresholded target",
    "hard_5f": "thresholded target, 5 folds",
    "soft_f0": "probability target",
    "soft_b2": r"probability target, $\beta{=}2$",
}

# (cohort, arm, reference, macro prefix) -- exactly the comparisons the text claims.
CONTRASTS = [
    ("development", VAL, "soft_f0", "control", "ValSoftCtrl"),
    ("development", VAL, "hard_5f", "control", "ValHardCtrl"),
    ("development", VAL, "hard_5f", "ceiling", "ValHardCeil"),
    ("development", VAL, "soft_b2", "soft_f0", "ValBetaSoft"),
    ("sealed test", TEST, "soft_f0", "control", "TestSoftCtrl"),
    ("sealed test", TEST, "hard_5f", "control", "TestHardCtrl"),
    ("sealed test", TEST, "hard_5f", "ceiling", "TestHardCeil"),
    ("sealed test", TEST, "soft_b2", "control", "TestBetaCtrl"),
    ("AeroPath", OOD, "soft_f0", "control", "OodSoftCtrl"),
    ("AeroPath", OOD, "hard_5f", "ceiling", "OodHardCeil"),
]


def load_cases(directory: str) -> dict | None:
    hits = [f for f in glob.glob(str(PRED_ROOT / directory / "*topology*.json"))
            if "case016" not in f and "lcc_sensitivity" not in f]
    if not hits:
        return None
    with open(hits[0]) as handle:
        payload = json.load(handle)
    rows = payload.get("table_per_case")
    return {r["case_id"]: r for r in rows} if rows else None


def paired_test(arm: dict, ref: dict, key: str) -> dict | None:
    """Full paired comparison on one metric. Returns None if unpairable."""
    ids = sorted(set(arm) & set(ref))
    x = [arm[i][key] for i in ids if arm[i].get(key) is not None and ref[i].get(key) is not None]
    y = [ref[i][key] for i in ids if arm[i].get(key) is not None and ref[i].get(key) is not None]
    if len(x) < 3:
        return None

    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)

    t_stat, p_t = stats.ttest_rel(x, y)
    # Wilcoxon is undefined when every difference is zero; report it as a null instead.
    try:
        p_w = float(stats.wilcoxon(x, y).pvalue)
    except ValueError:
        p_w = 1.0

    crit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    return {
        "metric": key,
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - crit * se,
        "ci_high": mean + crit * se,
        "t": float(t_stat),
        "df": n - 1,
        "p_raw": float(p_t),
        "p_bonferroni": min(1.0, float(p_t) * N_METRICS),
        "p_wilcoxon": p_w,
        "cohens_dz": mean / sd if sd else float("nan"),
        "wins": sum(1 for d in diffs if d > 0),
    }


def stars(p_adjusted: float) -> str:
    if p_adjusted < 0.001:
        return "***"
    if p_adjusted < 0.01:
        return "**"
    if p_adjusted < 0.05:
        return "*"
    return ""


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_PROVENANCE.mkdir(parents=True, exist_ok=True)

    results, macros, latex_rows, missing = [], [], [], []

    for cohort, arms, arm_key, ref_key, prefix in CONTRASTS:
        arm = load_cases(arms[arm_key])
        ref = load_cases(arms[ref_key])
        if arm is None or ref is None:
            missing.append(f"{cohort}: {arm_key} vs {ref_key}")
            continue

        header = f"{LABELS[arm_key]} vs {LABELS[ref_key]}"
        print(f"\n=== {cohort.upper()}: {header} ===")
        print(f"  {'metric':8s} {'diff':>9s} {'95% CI':>20s} {'t':>7s} {'p':>9s} "
              f"{'p(Bonf)':>9s} {'Wilcox':>9s} {'dz':>6s} {'wins':>7s}")

        # Ten columns: metric, diff, CI, t, df, p, p(Bonf), Wilcoxon, dz, wins.
        latex_rows.append(rf"    \multicolumn{{10}}{{l}}{{\itshape {cohort}: {header}}} \\")

        for key, short, tex_name in METRICS:
            row = paired_test(arm, ref, key)
            if row is None:
                continue
            row.update(cohort=cohort, arm=arm_key, reference=ref_key, label=header)
            results.append(row)

            print(f"  {short:8s} {row['mean']:+9.4f} "
                  f"[{row['ci_low']:+7.4f},{row['ci_high']:+7.4f}] "
                  f"{row['t']:+7.2f} {row['p_raw']:9.2e} {row['p_bonferroni']:9.2e} "
                  f"{row['p_wilcoxon']:9.2e} {row['cohens_dz']:+6.2f} "
                  f"{row['wins']:3d}/{row['n']:<3d}")

            latex_rows.append(
                rf"    \quad {tex_name} & ${row['mean']:+.4f}$ & "
                rf"$[{row['ci_low']:+.4f},\,{row['ci_high']:+.4f}]$ & "
                rf"${row['t']:+.2f}$ & {row['df']} & {fmt_p(row['p_raw'])} & "
                rf"{fmt_p(row['p_bonferroni'])}{stars(row['p_bonferroni'])} & "
                rf"{fmt_p(row['p_wilcoxon'])} & ${row['cohens_dz']:+.2f}$ & "
                rf"{row['wins']}/{row['n']} \\"
            )

            macros.append((f"{prefix}{short}Diff", f"{row['mean']:+.4f}"))
            macros.append((f"{prefix}{short}T", f"{row['t']:+.2f}"))
            macros.append((f"{prefix}{short}P", fmt_p_plain(row['p_bonferroni'])))
            macros.append((f"{prefix}{short}Wins", f"{row['wins']}/{row['n']}"))
        latex_rows.append(r"    \addlinespace")

    write_outputs(results, macros, latex_rows)
    if missing:
        print("\nNOT SCORED YET (skipped, no numbers invented):")
        for m in missing:
            print("  -", m)


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return r"$<\!10^{-4}$"
    return f"${p:.4f}$"


def fmt_p_plain(p: float) -> str:
    return r"<10^{-4}" if p < 1e-4 else f"{p:.4f}"


def write_outputs(results, macros, latex_rows) -> None:
    # The WHOLE table environment is emitted, not just the rows: \input inside a tabular
    # alignment is a TeX error ("Missing \cr inserted"), so the caption and label live here.
    body = OUT_TABLES / "paired_tests_table.tex"
    body.write_text(
        "% GENERATED by dissertation/scripts/statistics/paired_significance_tests.py.\n"
        "% Do not edit: re-run the script so the table and the data stay in agreement.\n"
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\scriptsize\n"
        "  \\setlength{\\tabcolsep}{3pt}\n"
        "  \\caption[Paired significance tests for every claimed comparison]{Paired\n"
        "  significance tests for every comparison claimed in this dissertation, on the native\n"
        "  (RAW) mask. $\\bar d$ is the mean per-case difference, arm minus reference. Stars mark\n"
        "  the Bonferroni-adjusted $p$: ${*}\\,p<0.05$, ${**}\\,p<0.01$, ${***}\\,p<0.001$.\n"
        "  Generated by \\texttt{dissertation/scripts/statistics/paired\\_significance\\_tests.py}.}\n"
        "  \\label{tab:paired-tests}\n"
        "  \\begin{tabular}{lrcrrrrrrr}\n"
        "    \\toprule\n"
        "    Metric & $\\bar d$ & 95\\% CI & $t$ & df & $p$ & $p_{\\text{adj}}$ & Wilcoxon"
        " & $d_z$ & wins \\\\\n"
        "    \\midrule\n"
        + "\n".join(latex_rows) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )
    print(f"\nwrote {body}  ({len(latex_rows)} data rows)")

    numbers = OUT_TABLES / "statistics_numbers.tex"
    seen, lines = set(), []
    for name, value in macros:
        if name in seen:
            continue
        seen.add(name)
        lines.append(rf"\newcommand{{\stat{name}}}{{{value}}}")
    numbers.write_text(
        "% GENERATED by dissertation/scripts/statistics/paired_significance_tests.py.\n"
        "% Do not edit. Use e.g. \\statTestSoftCtrlTLDDiff in prose so the number in the\n"
        "% text can never drift from the number in the table.\n"
        + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {numbers}  ({len(lines)} macros)")

    prov = OUT_PROVENANCE / "paired_tests.json"
    prov.write_text(json.dumps({
        "test": "paired two-sided t-test on per-case differences (scipy.stats.ttest_rel)",
        "correction": f"Bonferroni across {N_METRICS} metrics per contrast",
        "supporting_test": "Wilcoxon signed-rank (scipy.stats.wilcoxon)",
        "alpha": ALPHA,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"wrote {prov}  ({len(results)} comparisons)")


if __name__ == "__main__":
    main()
