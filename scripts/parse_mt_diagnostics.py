"""Parse a Mean-Teacher nnU-Net training log into a tidy CSV and a diagnostic figure.

The SoftCLDice diagnostic trainer prints four tagged lines per epoch
(``[MTGradient]``, ``[MTSoftCLDice]``, ``[MTTeacherEvidence]`` and
``[MTHardCounterfactual]``) alongside nnU-Net's own ``train_loss`` / ``val_loss`` /
``Pseudo dice`` / ``Epoch time`` lines.  Everything needed to compare the clDice
consistency against voxel MSE, and the soft target against the hard one, is
already in that log -- no re-prediction and no GPU time required.

Usage, from the repository root::

    python scripts/parse_mt_diagnostics.py <training_log.txt> --out-stem mt240_softcldice

It runs equally well on the HPC (the ``ctfm`` env has matplotlib) or locally after
``scp``-ing the log down; the log is plain text and small.

Ordering note: the trainer announces ``Epoch N`` and then prints that epoch's
diagnostics *afterwards*, so a record is only complete when the next ``Epoch``
header appears.  The parser therefore flushes on the header, not on the metrics.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

EPOCH_RE = re.compile(r"Epoch (\d+)\s*$")
TAG_RE = re.compile(r"\[(MTGradient|MTSoftCLDice|MTTeacherEvidence|MTHardCounterfactual)\]\s*(.*)$")
PSEUDO_RE = re.compile(r"Pseudo dice \[([^\]]*)\]")
SCALAR_RES = {
    "train_loss": re.compile(r"train_loss\s+(-?[\d.eE+]+)"),
    "val_loss": re.compile(r"val_loss\s+(-?[\d.eE+]+)"),
    "epoch_time_s": re.compile(r"Epoch time:\s*([\d.]+)\s*s"),
}

# Keys carry characters that are awkward in a column name (``p>0.1``,
# ``subthr_p>=0.05_prob_mass``).  Normalise rather than drop them.
def _clean_key(tag: str, key: str) -> str:
    key = key.replace(">=", "_ge_").replace(">", "_gt_").replace(".", "p")
    key = re.sub(r"[^0-9A-Za-z_]", "_", key)
    prefix = {
        "MTGradient": "grad",
        "MTSoftCLDice": "cl",
        "MTTeacherEvidence": "ev",
        "MTHardCounterfactual": "cf",
    }[tag]
    return f"{prefix}_{key}"


def _parse_pairs(tag: str, rest: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for token in rest.split():
        if "=" not in token:
            continue
        # rsplit: the LAST '=' separates key from value, so keys containing
        # '>=' survive intact.
        raw_key, raw_value = token.rsplit("=", 1)
        key = _clean_key(tag, raw_key)
        try:
            out[key] = float(raw_value)
        except ValueError:
            out[key] = raw_value
    return out


def parse_log(path: Path) -> list[dict[str, float | str]]:
    records: list[dict[str, float | str]] = []
    current: dict[str, float | str] = {}
    epoch: int | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = EPOCH_RE.search(line)
        if header:
            if epoch is not None and len(current) > 1:
                records.append(current)
            epoch = int(header.group(1))
            current = {"epoch": epoch}
            continue
        if epoch is None:
            continue

        tagged = TAG_RE.search(line)
        if tagged:
            current.update(_parse_pairs(tagged.group(1), tagged.group(2)))
            continue

        for name, pattern in SCALAR_RES.items():
            found = pattern.search(line)
            if found:
                current[name] = float(found.group(1))

        pseudo = PSEUDO_RE.search(line)
        if pseudo:
            # nnU-Net may render this as ``[np.float32(0.9329)]``, so require a
            # decimal point with digits on both sides rather than any '.' run.
            values = re.findall(r"\d*\.\d+", pseudo.group(1))
            if values:
                current["pseudo_dice"] = float(values[0])

    if epoch is not None and len(current) > 1:
        records.append(current)
    return records


def write_csv(records: list[dict], path: Path) -> list[str]:
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    columns.sort(key=lambda c: (c != "epoch", c))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return columns


def _series(records: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for record in records:
        if key in record and isinstance(record[key], float):
            xs.append(record["epoch"])
            ys.append(record[key])
    return xs, ys


def make_figure(records: list[dict], pdf_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fixed figsize and NO bbox_inches='tight': every panel this project emits
    # should have identical outer dimensions so that \includegraphics at a common
    # width yields a common height.
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    fig.subplots_adjust(left=.085, right=.985, top=.94, bottom=.09, wspace=.28, hspace=.42)

    def plot(ax, keys, labels, title, ylabel, logy=False):
        drawn = False
        for key, label in zip(keys, labels):
            xs, ys = _series(records, key)
            if xs:
                # Markers matter: the hard counterfactual is recorded once per
                # epoch-block, not every epoch, so its series is sparse and would
                # be invisible as a bare line.
                ax.plot(xs, ys, lw=1.3, marker="o", ms=2.2, label=label)
                drawn = True
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7.5)
        if logy:
            ax.set_yscale("log")
        if drawn:
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.text(.5, .5, "no data in log", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#888")
        ax.grid(alpha=.25, lw=.5)

    # (a) the soft-vs-hard target question, measured on the SAME patch.
    plot(axes[0][0],
         ["cl_soft_loss", "cf_same_patch_soft_loss", "cf_hard_loss"],
         [r"soft $\mathcal{L}_{cons}$ (all patches)",
          r"soft $\mathcal{L}$ (counterfactual patch)",
          r"hard $\mathcal{L}$ (counterfactual patch)"],
         "(a) soft vs hard teacher target", "loss")

    # (b) the clDice-vs-MSE scale argument, same patches, log axis.
    plot(axes[0][1],
         ["cl_soft_loss", "cl_prob_mse"],
         [r"soft clDice $\mathcal{L}_{cons}$", "voxel MSE on the same patches"],
         "(b) clDice vs voxel MSE magnitude", "loss (log)", logy=True)

    # (c) how much evidence a 0.5 threshold would discard.
    plot(axes[1][0],
         ["ev_subthr_prob_mass", "ev_subthr_skel_mass", "ev_soft_only_patches"],
         ["sub-threshold probability mass",
          "sub-threshold skeleton mass",
          "patches with soft but no hard evidence"],
         "(c) evidence below the 0.5 threshold", "share")

    # (d) the honest gradient-level contribution, plus its alignment.
    plot(axes[1][1],
         ["grad_weighted_fraction", "grad_cosine"],
         ["weighted consistency gradient fraction",
          r"cosine(sup, cons) gradients"],
         "(d) gradient contribution and alignment", "value")
    axes[1][1].axhline(0.0, color="#999", lw=.7, ls="--")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="nnU-Net training_log_*.txt")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mt_diagnostics"))
    parser.add_argument("--out-stem", default="mt_diagnostics")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    records = parse_log(args.log)
    if not records:
        raise SystemExit(f"No epoch records parsed from {args.log}")

    csv_path = args.out_dir / f"{args.out_stem}.csv"
    columns = write_csv(records, csv_path)
    print(f"parsed {len(records)} epochs -> {csv_path}")
    print(f"last epoch: {records[-1]['epoch']}")
    diagnostic_cols = [c for c in columns if c.split("_")[0] in {"grad", "cl", "ev", "cf"}]
    print(f"{len(diagnostic_cols)} diagnostic columns")

    if not args.no_figure:
        make_figure(
            records,
            args.out_dir / f"{args.out_stem}.pdf",
            args.out_dir / f"{args.out_stem}.png",
        )
        print(f"figure -> {args.out_dir / (args.out_stem + '.png')}")


if __name__ == "__main__":
    main()
