"""Plot training history artifacts for one baseline run directory.

Usage with an explicit run directory:

    python -m scripts.plot_run_history --run-dir runs/baseline_unet/20260410_205441

If you prefer copy-paste editing, you can also set ``DEFAULT_RUN_DIR`` below
and run the script without any flags.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Optional convenience override. Paste a run directory here if you do not want
# to pass --run-dir every time.
DEFAULT_RUN_DIR = None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the plotting CLI parser."""
    parser = argparse.ArgumentParser(
        description="Plot training curves for one run directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing history.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional plot output directory. Defaults to <run-dir>/plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    return parser


def load_json(path: Path) -> dict:
    """Load a JSON file and require a top-level object."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return data


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """Resolve the requested run directory from CLI or file-level default."""
    run_dir = args.run_dir if args.run_dir is not None else DEFAULT_RUN_DIR
    if run_dir is None:
        raise ValueError(
            "No run directory was provided. Pass --run-dir or set DEFAULT_RUN_DIR in scripts/plot_run_history.py."
        )
    return Path(run_dir).resolve()


def extract_series(history_entries: list[dict], metric_name: str) -> tuple[list[int], list[float]]:
    """Extract one metric series from the history entries."""
    epochs = []
    values = []
    for entry in history_entries:
        if metric_name in entry and entry[metric_name] is not None:
            epochs.append(int(entry["epoch"]))
            values.append(float(entry[metric_name]))
    return epochs, values


def build_training_curve_figure(history_entries: list[dict], run_name: str):
    """Build the main loss, dice, and learning-rate figure for one run."""
    train_loss_epochs, train_loss_values = extract_series(history_entries, "train_loss")
    val_loss_epochs, val_loss_values = extract_series(history_entries, "val_loss")
    train_dice_epochs, train_dice_values = extract_series(history_entries, "train_dice")
    val_dice_epochs, val_dice_values = extract_series(history_entries, "val_dice")
    lr_epochs, lr_values = extract_series(history_entries, "learning_rate")

    figure, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    figure.suptitle(f"Training Curves: {run_name}", fontsize=14)

    axes[0].plot(train_loss_epochs, train_loss_values, label="Train loss", color="#1f77b4", linewidth=2.0)
    if val_loss_epochs:
        axes[0].plot(val_loss_epochs, val_loss_values, label="Validation loss", color="#ff7f0e", marker="o")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(train_dice_epochs, train_dice_values, label="Train Dice", color="#2ca02c", linewidth=2.0)
    if val_dice_epochs:
        axes[1].plot(val_dice_epochs, val_dice_values, label="Validation Dice", color="#d62728", marker="o")
    axes[1].set_ylabel("Dice")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(lr_epochs, lr_values, label="Learning rate", color="#9467bd", linewidth=2.0)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    figure.tight_layout()
    return figure


def save_training_summary(summary: dict, output_path: Path) -> None:
    """Write a small text summary alongside the plots."""
    lines = [
        f"Run: {summary['run_name']}",
        f"Run directory: {summary['run_dir']}",
        f"Epochs recorded: {summary['num_epochs']}",
        f"Best validation Dice: {summary['best_val_dice']}",
        f"Best epoch: {summary['best_epoch']}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_argument_parser().parse_args()
    run_dir = resolve_run_dir(args)

    history_payload = load_json(run_dir / "history.json")
    history_entries = history_payload.get("history", [])
    if not history_entries:
        raise ValueError(f"No history entries were found in {run_dir / 'history.json'}.")

    best_payload = history_payload.get("best", {})
    output_dir = args.output_dir.resolve() if args.output_dir is not None else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = run_dir.parent.name
    figure = build_training_curve_figure(history_entries, run_name=run_name)
    figure.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "num_epochs": len(history_entries),
        "best_val_dice": best_payload.get("val_dice"),
        "best_epoch": best_payload.get("epoch"),
    }
    save_training_summary(summary, output_dir / "summary.txt")

    print(f"Saved plots to {output_dir}")
    if summary["best_val_dice"] is not None:
        print(f"Best validation Dice: {float(summary['best_val_dice']):.4f} at epoch {summary['best_epoch']}")
    else:
        print("No validation Dice was recorded in this run history.")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
