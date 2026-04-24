"""Plot cross-run comparisons for the patch and full-volume baseline histories.

Usage with the defaults for the current baseline runs:

    python -m plot_scripts.plot_baseline_comparison

Usage with explicit run directories:

    python -m plot_scripts.plot_baseline_comparison ^
        --patch-run-dir runs/baseline_unet_patches/20260417_232126 ^
        --full-volume-run-dir runs/baseline_full_volume_hpc_50ep/20260418_023030
"""

from __future__ import annotations

import argparse
import csv
import json
from matplotlib.colors import to_rgba
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from lung_airway_segmentation.io.case_layout import resolve_case_paths
from lung_airway_segmentation.io.nifti import load_canonical_image
from lung_airway_segmentation.training.config import resolve_project_path


DEFAULT_PATCH_RUN_DIR = Path("runs/baseline_unet_patches/20260417_232126")
DEFAULT_FULL_VOLUME_RUN_DIR = Path("runs/baseline_full_volume_hpc_50ep/20260418_023030")
DEFAULT_OUTPUT_DIR = Path("runs/comparisons/baseline_patch_vs_full_volume")

PATCH_COLOR = "#1f77b4"
FULL_VOLUME_COLOR = "#ff7f0e"
GROUND_TRUTH_COLOR = "#2ca02c"
WARMUP_SHADE_COLOR = "#808080"

FIGURE_DPI = 240
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12
TITLE_SIZE = 15
ANNOTATION_FONT_SIZE = 11
DEFAULT_QUAL_MAX_CASES = 1


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Plot comparison figures for patch and full-volume baseline runs.",
    )
    parser.add_argument(
        "--patch-run-dir",
        type=Path,
        default=DEFAULT_PATCH_RUN_DIR,
        help="Patch-based run directory containing history.json.",
    )
    parser.add_argument(
        "--full-volume-run-dir",
        type=Path,
        default=DEFAULT_FULL_VOLUME_RUN_DIR,
        help="Full-volume run directory containing history.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where comparison plots and summary text will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively after saving.",
    )
    parser.add_argument(
        "--include-title",
        action="store_true",
        help="Add on-figure titles. Useful for standalone viewing, but usually unnecessary in LaTeX.",
    )
    parser.add_argument(
        "--qual-case-id",
        default=None,
        help="Optional explicit case ID for the qualitative comparison panel. Defaults to the most informative case.",
    )
    parser.add_argument(
        "--qual-max-cases",
        type=int,
        default=DEFAULT_QUAL_MAX_CASES,
        help="How many qualitative cases to include. Defaults to 1 for a cleaner figure.",
    )
    return parser


def load_json(path: Path) -> dict:
    """Load a JSON file and require a top-level object."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return payload


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load one CSV file into a list of dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def extract_series(history_entries: list[dict], metric_name: str) -> tuple[list[int], list[float]]:
    """Extract one metric series from a run history."""
    epochs: list[int] = []
    values: list[float] = []
    for entry in history_entries:
        if metric_name in entry and entry[metric_name] is not None:
            epochs.append(int(entry["epoch"]))
            values.append(float(entry[metric_name]))
    return epochs, values


def load_history_entries(run_dir: Path) -> tuple[dict, list[dict]]:
    """Load and validate the history payload for one run directory."""
    history_path = run_dir / "history.json"
    payload = load_json(history_path)
    history_entries = payload.get("history", [])
    if not history_entries:
        raise ValueError(f"No history entries were found in {history_path}.")
    return payload, history_entries


def load_resolved_config(run_dir: Path) -> dict:
    """Load the resolved training configuration for one run."""
    return load_json(run_dir / "resolved_config.json")


def resolve_data_root(run_dir: Path, resolved_config: dict) -> Path:
    """Resolve the local raw dataset root for a saved run."""
    run_metadata_path = run_dir / "run_metadata.json"
    if run_metadata_path.is_file():
        run_metadata = load_json(run_metadata_path)
        data_root = run_metadata.get("data_root")
        if data_root:
            candidate = Path(data_root)
            if candidate.exists():
                return candidate.resolve()

    configured_root = Path(resolved_config["data"]["raw_data_root"])
    candidate = resolve_project_path(configured_root).resolve()
    if candidate.exists():
        return candidate

    if configured_root.is_absolute() and configured_root.exists():
        return configured_root.resolve()

    raise FileNotFoundError(
        "Could not resolve a local dataset root from the saved run metadata or resolved config."
    )


def load_per_case_metrics(run_dir: Path) -> list[dict[str, object]]:
    """Load the optional per-case evaluation table for one run."""
    metrics_path = run_dir / "evaluation" / "per_case_metrics.csv"
    if not metrics_path.is_file():
        return []

    rows = load_csv_rows(metrics_path)
    numeric_float_fields = {
        "threshold",
        "dice",
        "iou",
        "precision",
        "recall",
        "voxel_count_ratio",
    }
    numeric_int_fields = {
        "checkpoint_epoch",
        "predicted_voxels",
        "target_voxels",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    }

    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        parsed_row: dict[str, object] = {}
        for key, value in row.items():
            if key in numeric_float_fields:
                parsed_row[key] = None if value in ("", "None") else float(value)
            elif key in numeric_int_fields:
                parsed_row[key] = int(value)
            else:
                parsed_row[key] = value
        parsed_rows.append(parsed_row)
    return parsed_rows


def load_evaluation_summary(run_dir: Path) -> dict | None:
    """Load the optional evaluation summary for one run."""
    summary_path = run_dir / "evaluation" / "summary.json"
    if not summary_path.is_file():
        return None
    return load_json(summary_path)


def case_sort_key(case_id: str) -> tuple[bool, int | str]:
    """Sort case identifiers numerically when possible."""
    return (not case_id.isdigit(), int(case_id) if case_id.isdigit() else case_id)


def shared_warmup_epochs(patch_config: dict, full_volume_config: dict) -> int | None:
    """Return the shared warmup epoch count when both runs use the same schedule."""
    patch_warmup = patch_config["training"].get("scheduler", {}).get("warmup_epochs")
    full_warmup = full_volume_config["training"].get("scheduler", {}).get("warmup_epochs")
    if patch_warmup is None or full_warmup is None:
        return None
    patch_warmup = int(patch_warmup)
    full_warmup = int(full_warmup)
    return patch_warmup if patch_warmup == full_warmup else None


def style_axis(axis, legend_loc: str = "best") -> None:
    """Apply a readable default style for report figures."""
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=LEGEND_FONT_SIZE, loc=legend_loc)


def add_warmup_context(axis, warmup_epochs: int | None) -> None:
    """Shade the scheduler warmup region when available."""
    if warmup_epochs is None or warmup_epochs <= 0:
        return
    axis.axvspan(1, warmup_epochs, color=WARMUP_SHADE_COLOR, alpha=0.08, linewidth=0)
    axis.axvline(warmup_epochs, color=WARMUP_SHADE_COLOR, linestyle=":", linewidth=1.0, alpha=0.45)


def annotate_best_validation(axis, best_payload: dict, label_prefix: str, color: str, text_offset: tuple[int, int]) -> None:
    """Annotate the best validation checkpoint on a Dice curve."""
    best_epoch = best_payload.get("epoch")
    best_val_dice = best_payload.get("val_dice")
    if best_epoch is None or best_val_dice is None:
        return

    best_epoch = int(best_epoch)
    best_val_dice = float(best_val_dice)
    axis.scatter([best_epoch], [best_val_dice], color=color, s=36, zorder=5)
    axis.annotate(
        f"{label_prefix} best\nE{best_epoch}: {best_val_dice:.4f}",
        xy=(best_epoch, best_val_dice),
        xytext=text_offset,
        textcoords="offset points",
        fontsize=ANNOTATION_FONT_SIZE,
        color=color,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": color},
    )


def build_validation_gap_series(history_entries: list[dict], train_metric_name: str, val_metric_name: str) -> tuple[list[int], list[float]]:
    """Compute train-minus-validation metric gaps at validation checkpoints."""
    epochs: list[int] = []
    gaps: list[float] = []
    for entry in history_entries:
        if entry.get(train_metric_name) is None or entry.get(val_metric_name) is None:
            continue
        epochs.append(int(entry["epoch"]))
        gaps.append(float(entry[train_metric_name]) - float(entry[val_metric_name]))
    return epochs, gaps


def build_dice_comparison_figure(
    patch_history: list[dict],
    full_volume_history: list[dict],
    patch_best: dict,
    full_volume_best: dict,
    include_title: bool,
    warmup_epochs: int | None,
) -> plt.Figure:
    """Build the patch-vs-full-volume Dice comparison figure."""
    patch_train_epochs, patch_train_values = extract_series(patch_history, "train_dice")
    patch_val_epochs, patch_val_values = extract_series(patch_history, "val_dice")
    full_train_epochs, full_train_values = extract_series(full_volume_history, "train_dice")
    full_val_epochs, full_val_values = extract_series(full_volume_history, "val_dice")

    figure, axis = plt.subplots(figsize=(10, 6.5))
    add_warmup_context(axis, warmup_epochs)
    axis.plot(
        patch_train_epochs,
        patch_train_values,
        color=PATCH_COLOR,
        linewidth=2.0,
        label="Patch train Dice",
    )
    if patch_val_epochs:
        axis.plot(
            patch_val_epochs,
            patch_val_values,
            color=PATCH_COLOR,
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markersize=6,
            label="Patch val Dice",
        )
    axis.plot(
        full_train_epochs,
        full_train_values,
        color=FULL_VOLUME_COLOR,
        linewidth=2.0,
        label="Full-vol train Dice",
    )
    if full_val_epochs:
        axis.plot(
            full_val_epochs,
            full_val_values,
            color=FULL_VOLUME_COLOR,
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markersize=6,
            label="Full-vol val Dice",
        )

    annotate_best_validation(axis, patch_best, "Patch", PATCH_COLOR, text_offset=(-70, 14))
    annotate_best_validation(axis, full_volume_best, "Full-vol", FULL_VOLUME_COLOR, text_offset=(-110, -8))

    if include_title:
        axis.set_title("Patch vs Full-Volume Dice", fontsize=TITLE_SIZE)
    axis.set_xlabel("Epoch", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("Dice", fontsize=AXIS_LABEL_SIZE)
    style_axis(axis, legend_loc="upper left")
    figure.tight_layout()
    return figure


def build_training_loss_figure(
    patch_history: list[dict],
    full_volume_history: list[dict],
    include_title: bool,
    warmup_epochs: int | None,
) -> plt.Figure:
    """Build the training-loss comparison figure."""
    patch_epochs, patch_values = extract_series(patch_history, "train_loss")
    full_epochs, full_values = extract_series(full_volume_history, "train_loss")

    figure, axis = plt.subplots(figsize=(10, 6.5))
    add_warmup_context(axis, warmup_epochs)
    axis.plot(
        patch_epochs,
        patch_values,
        color=PATCH_COLOR,
        linewidth=2.0,
        label="Patch train loss",
    )
    axis.plot(
        full_epochs,
        full_values,
        color=FULL_VOLUME_COLOR,
        linewidth=2.0,
        label="Full-vol train loss",
    )

    if include_title:
        axis.set_title("Patch vs Full-Volume Training Loss", fontsize=TITLE_SIZE)
    axis.set_xlabel("Epoch", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("Loss", fontsize=AXIS_LABEL_SIZE)
    style_axis(axis, legend_loc="upper right")
    figure.tight_layout()
    return figure


def build_validation_loss_figure(
    patch_history: list[dict],
    full_volume_history: list[dict],
    include_title: bool,
    warmup_epochs: int | None,
) -> plt.Figure:
    """Build the validation-loss comparison figure."""
    patch_epochs, patch_values = extract_series(patch_history, "val_loss")
    full_epochs, full_values = extract_series(full_volume_history, "val_loss")

    figure, axis = plt.subplots(figsize=(10, 6.5))
    add_warmup_context(axis, warmup_epochs)
    axis.plot(
        patch_epochs,
        patch_values,
        color=PATCH_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=6,
        label="Patch val loss",
    )
    axis.plot(
        full_epochs,
        full_values,
        color=FULL_VOLUME_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=6,
        label="Full-vol val loss",
    )

    if include_title:
        axis.set_title("Patch vs Full-Volume Validation Loss", fontsize=TITLE_SIZE)
    axis.set_xlabel("Epoch", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("Validation loss", fontsize=AXIS_LABEL_SIZE)
    style_axis(axis, legend_loc="upper right")
    figure.tight_layout()
    return figure


def build_learning_rate_figure(
    patch_history: list[dict],
    full_volume_history: list[dict],
    include_title: bool,
    warmup_epochs: int | None,
) -> plt.Figure:
    """Build the learning-rate comparison figure."""
    patch_epochs, patch_values = extract_series(patch_history, "learning_rate")
    full_epochs, full_values = extract_series(full_volume_history, "learning_rate")

    figure, axis = plt.subplots(figsize=(10, 6.5))
    add_warmup_context(axis, warmup_epochs)
    axis.plot(
        patch_epochs,
        patch_values,
        color=PATCH_COLOR,
        linewidth=2.0,
        label="Patch LR",
    )
    axis.plot(
        full_epochs,
        full_values,
        color=FULL_VOLUME_COLOR,
        linewidth=2.0,
        linestyle="--",
        label="Full-vol LR",
    )

    if include_title:
        axis.set_title("Patch vs Full-Volume Learning Rate", fontsize=TITLE_SIZE)
    axis.set_xlabel("Epoch", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("Learning rate", fontsize=AXIS_LABEL_SIZE)
    style_axis(axis, legend_loc="upper right")
    figure.tight_layout()
    return figure


def build_dice_gap_figure(
    patch_history: list[dict],
    full_volume_history: list[dict],
    include_title: bool,
    warmup_epochs: int | None,
) -> plt.Figure:
    """Build the train-minus-validation Dice gap comparison figure."""
    patch_epochs, patch_values = build_validation_gap_series(patch_history, "train_dice", "val_dice")
    full_epochs, full_values = build_validation_gap_series(full_volume_history, "train_dice", "val_dice")

    figure, axis = plt.subplots(figsize=(10, 6.5))
    add_warmup_context(axis, warmup_epochs)
    axis.plot(
        patch_epochs,
        patch_values,
        color=PATCH_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=6,
        label="Patch Dice gap",
    )
    axis.plot(
        full_epochs,
        full_values,
        color=FULL_VOLUME_COLOR,
        linewidth=2.0,
        marker="o",
        markersize=6,
        label="Full-vol Dice gap",
    )

    if include_title:
        axis.set_title("Patch vs Full-Volume Dice Generalisation Gap", fontsize=TITLE_SIZE)
    axis.set_xlabel("Epoch", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("Train Dice - val Dice", fontsize=AXIS_LABEL_SIZE)
    style_axis(axis, legend_loc="upper left")
    figure.tight_layout()
    return figure


def build_per_case_dice_figure(
    patch_case_metrics: list[dict[str, object]],
    full_case_metrics: list[dict[str, object]],
    include_title: bool,
) -> plt.Figure:
    """Build the per-case Dice comparison from saved best-checkpoint predictions."""
    patch_by_case = {str(row["case_id"]): row for row in patch_case_metrics}
    full_by_case = {str(row["case_id"]): row for row in full_case_metrics}
    common_case_ids = sorted(set(patch_by_case) & set(full_by_case), key=case_sort_key)
    if not common_case_ids:
        raise ValueError("No common per-case evaluation rows were found across the two runs.")

    patch_values = [float(patch_by_case[case_id]["dice"]) for case_id in common_case_ids]
    full_values = [float(full_by_case[case_id]["dice"]) for case_id in common_case_ids]

    x_positions = np.arange(len(common_case_ids), dtype=np.float64)
    bar_width = 0.36

    figure, axis = plt.subplots(figsize=(10, 6.5))
    patch_bars = axis.bar(
        x_positions - bar_width / 2.0,
        patch_values,
        width=bar_width,
        color=PATCH_COLOR,
        alpha=0.9,
        label="Patch per-case Dice",
    )
    full_bars = axis.bar(
        x_positions + bar_width / 2.0,
        full_values,
        width=bar_width,
        color=FULL_VOLUME_COLOR,
        alpha=0.9,
        label="Full-vol per-case Dice",
    )

    axis.axhline(np.mean(patch_values), color=PATCH_COLOR, linestyle="--", linewidth=1.5, alpha=0.8, label="Patch mean")
    axis.axhline(
        np.mean(full_values),
        color=FULL_VOLUME_COLOR,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Full-vol mean",
    )

    for bars in (patch_bars, full_bars):
        for bar in bars:
            height = float(bar.get_height())
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.0015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    if include_title:
        axis.set_title("Per-Case Dice from Saved Best-Checkpoint Predictions", fontsize=TITLE_SIZE)
    axis.set_xlabel("Validation case", fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel("Dice", fontsize=AXIS_LABEL_SIZE)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(common_case_ids)
    style_axis(axis, legend_loc="upper left")
    figure.tight_layout()
    return figure


def load_cropped_target_mask(case_id: str, crop_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]], data_root: Path) -> np.ndarray:
    """Load and crop the reference airway mask for one case."""
    case_paths = resolve_case_paths(case_id, data_root=data_root)
    airway_path = case_paths["airway"]
    if airway_path is None:
        raise ValueError(f"Case {case_id} does not have a reference airway mask.")

    target_full = np.asarray(load_canonical_image(airway_path).dataobj) > 0
    z_bounds, y_bounds, x_bounds = crop_box
    return target_full[
        z_bounds[0]:z_bounds[1],
        y_bounds[0]:y_bounds[1],
        x_bounds[0]:x_bounds[1],
    ]


def build_rgba_overlay(mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    """Build an explicit RGBA overlay so mask fills render with fixed colors."""
    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    red, green, blue, _ = to_rgba(color)
    rgba[..., 0] = red
    rgba[..., 1] = green
    rgba[..., 2] = blue
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return rgba


def rotate_display_slice_180(array: np.ndarray) -> np.ndarray:
    """Rotate one 2D display slice for easier qualitative viewing.

    This is a display-only transform used in the qualitative comparison panel.
    It does not affect saved predictions, metrics, or any upstream preprocessing.
    """
    return np.rot90(array, 2)


def select_qualitative_case_ids(
    common_case_ids: list[str],
    patch_by_case: dict[str, dict[str, object]],
    full_by_case: dict[str, dict[str, object]],
    requested_case_id: str | None,
    max_cases: int,
) -> list[str]:
    """Choose which evaluated cases to show in the qualitative panel."""
    if max_cases <= 0:
        raise ValueError("--qual-max-cases must be at least 1.")

    if requested_case_id is not None:
        requested_case_id = str(requested_case_id)
        if requested_case_id not in common_case_ids:
            raise ValueError(f"Requested qualitative case {requested_case_id} was not found in the common evaluated cases.")
        return [requested_case_id]

    ranked_case_ids = sorted(
        common_case_ids,
        key=lambda case_id: (
            float(patch_by_case[case_id]["dice"]) - float(full_by_case[case_id]["dice"]),
            float(patch_by_case[case_id]["dice"]),
        ),
        reverse=True,
    )
    return ranked_case_ids[: min(max_cases, len(ranked_case_ids))]


def build_qualitative_comparison_figure(
    patch_run_dir: Path,
    full_volume_run_dir: Path,
    patch_case_metrics: list[dict[str, object]],
    full_case_metrics: list[dict[str, object]],
    data_root: Path,
    include_title: bool,
    requested_case_id: str | None,
    max_cases: int,
) -> tuple[plt.Figure, list[str]]:
    """Build a multi-case qualitative comparison panel from saved best-checkpoint predictions."""
    patch_by_case = {str(row["case_id"]): row for row in patch_case_metrics}
    full_by_case = {str(row["case_id"]): row for row in full_case_metrics}
    common_case_ids = sorted(set(patch_by_case) & set(full_by_case), key=case_sort_key)
    if not common_case_ids:
        raise ValueError("No common per-case cases are available for qualitative comparison.")

    case_ids = select_qualitative_case_ids(
        common_case_ids=common_case_ids,
        patch_by_case=patch_by_case,
        full_by_case=full_by_case,
        requested_case_id=requested_case_id,
        max_cases=max_cases,
    )
    row_records: list[dict[str, object]] = []

    for case_id in case_ids:
        patch_case_dir = patch_run_dir / "predictions" / case_id
        full_case_dir = full_volume_run_dir / "predictions" / case_id

        patch_metadata = load_json(patch_case_dir / "prediction_metadata.json")
        crop_box = tuple(tuple(int(value) for value in bounds) for bounds in patch_metadata["crop_box"])

        ct_cropped = np.asarray(nib.load(str(patch_case_dir / "ct_cropped.nii.gz")).dataobj, dtype=np.float32)
        patch_pred = np.asarray(nib.load(str(patch_case_dir / "airway_pred_cropped.nii.gz")).dataobj) > 0
        full_pred = np.asarray(nib.load(str(full_case_dir / "airway_pred_cropped.nii.gz")).dataobj) > 0
        target_cropped = load_cropped_target_mask(case_id, crop_box=crop_box, data_root=data_root)

        slice_index = int(np.argmax(target_cropped.sum(axis=(1, 2))))
        ct_slice = ct_cropped[slice_index]
        target_slice = target_cropped[slice_index]
        patch_slice = patch_pred[slice_index]
        full_slice = full_pred[slice_index]

        intensity_bounds = np.percentile(ct_slice, [2.0, 98.0])
        vmin = float(intensity_bounds[0])
        vmax = float(intensity_bounds[1])
        if np.isclose(vmin, vmax):
            vmin = None
            vmax = None

        row_records.append(
            {
                "case_id": case_id,
                "slice_index": slice_index,
                "ct_slice": ct_slice,
                "target_slice": target_slice,
                "patch_slice": patch_slice,
                "full_slice": full_slice,
                "vmin": vmin,
                "vmax": vmax,
                "patch_dice": float(patch_by_case[case_id]["dice"]),
                "full_dice": float(full_by_case[case_id]["dice"]),
            }
        )

    num_rows = len(row_records)
    num_cols = 3
    figure_height = 3.65 if num_rows == 1 else 3.3 * num_rows
    figure, axes = plt.subplots(num_rows, num_cols, figsize=(10.8, figure_height), squeeze=False)
    column_titles = ["CT + ground truth", "Patch prediction", "Full-vol prediction"]
    for column_index, title in enumerate(column_titles):
        axes[0, column_index].set_title(title, fontsize=AXIS_LABEL_SIZE)

    for row_index, record in enumerate(row_records):
        ct_slice = rotate_display_slice_180(np.asarray(record["ct_slice"], dtype=np.float32))
        target_slice = rotate_display_slice_180(np.asarray(record["target_slice"], dtype=bool))
        patch_slice = rotate_display_slice_180(np.asarray(record["patch_slice"], dtype=bool))
        full_slice = rotate_display_slice_180(np.asarray(record["full_slice"], dtype=bool))
        vmin = record["vmin"]
        vmax = record["vmax"]

        ct_target_axis, patch_axis, full_axis = axes[row_index]
        row_axes = (ct_target_axis, patch_axis, full_axis)
        for axis in row_axes:
            axis.imshow(ct_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")

        ct_target_axis.imshow(
            build_rgba_overlay(target_slice, GROUND_TRUTH_COLOR, alpha=0.34),
            interpolation="none",
        )
        if np.any(target_slice):
            ct_target_axis.contour(
                target_slice.astype(np.float32),
                levels=[0.5],
                colors=[GROUND_TRUTH_COLOR],
                linewidths=0.9,
            )

        patch_axis.imshow(
            build_rgba_overlay(patch_slice, PATCH_COLOR, alpha=0.30),
            interpolation="none",
        )
        full_axis.imshow(
            build_rgba_overlay(full_slice, FULL_VOLUME_COLOR, alpha=0.30),
            interpolation="none",
        )

        if np.any(patch_slice):
            patch_axis.contour(
                patch_slice.astype(np.float32),
                levels=[0.5],
                colors=[PATCH_COLOR],
                linewidths=0.9,
            )
        if np.any(full_slice):
            full_axis.contour(
                full_slice.astype(np.float32),
                levels=[0.5],
                colors=[FULL_VOLUME_COLOR],
                linewidths=0.9,
            )
        if np.any(target_slice):
            for axis in (patch_axis, full_axis):
                axis.contour(
                    target_slice.astype(np.float32),
                    levels=[0.5],
                    colors=["white"],
                    linewidths=0.75,
                )

    if include_title:
        figure.suptitle("Representative Validation Slices", fontsize=TITLE_SIZE)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    else:
        figure.tight_layout()
    return figure, case_ids


def build_run_summary(
    run_label: str,
    run_dir: Path,
    history_payload: dict,
    history_entries: list[dict],
    evaluation_summary: dict | None,
) -> list[str]:
    """Build compact summary lines for one run."""
    best_payload = history_payload.get("best", {})
    train_loss_epochs, train_loss_values = extract_series(history_entries, "train_loss")
    train_dice_epochs, train_dice_values = extract_series(history_entries, "train_dice")
    val_dice_epochs, val_dice_values = extract_series(history_entries, "val_dice")

    lines = [
        f"{run_label}:",
        f"  run_dir: {run_dir}",
        f"  epochs_recorded: {len(history_entries)}",
        f"  best_val_dice_history: {best_payload.get('val_dice')}",
        f"  best_epoch_history: {best_payload.get('epoch')}",
    ]
    if train_loss_epochs:
        lines.append(f"  final_train_loss: {train_loss_values[-1]}")
    if train_dice_epochs:
        lines.append(f"  final_train_dice: {train_dice_values[-1]}")
    if val_dice_epochs:
        lines.append(f"  final_val_dice_history: {val_dice_values[-1]}")
    if evaluation_summary is not None:
        lines.append(f"  saved_prediction_case_count: {evaluation_summary.get('num_cases')}")
        lines.append(f"  saved_prediction_dice_mean: {evaluation_summary.get('dice_mean')}")
        lines.append(f"  saved_prediction_precision_mean: {evaluation_summary.get('precision_mean')}")
        lines.append(f"  saved_prediction_recall_mean: {evaluation_summary.get('recall_mean')}")
    return lines


def save_summary_text(
    output_path: Path,
    patch_run_dir: Path,
    full_volume_run_dir: Path,
    patch_payload: dict,
    patch_history: list[dict],
    patch_evaluation_summary: dict | None,
    full_volume_payload: dict,
    full_volume_history: list[dict],
    full_volume_evaluation_summary: dict | None,
) -> None:
    """Write a text summary alongside the saved figures."""
    lines = [
        "Patch vs full-volume baseline comparison",
        "",
    ]
    lines.extend(
        build_run_summary(
            run_label="Patch",
            run_dir=patch_run_dir,
            history_payload=patch_payload,
            history_entries=patch_history,
            evaluation_summary=patch_evaluation_summary,
        )
    )
    lines.append("")
    lines.extend(
        build_run_summary(
            run_label="Full-volume",
            run_dir=full_volume_run_dir,
            history_payload=full_volume_payload,
            history_entries=full_volume_history,
            evaluation_summary=full_volume_evaluation_summary,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_argument_parser().parse_args()

    patch_run_dir = args.patch_run_dir.resolve()
    full_volume_run_dir = args.full_volume_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_payload, patch_history = load_history_entries(patch_run_dir)
    full_volume_payload, full_volume_history = load_history_entries(full_volume_run_dir)
    patch_config = load_resolved_config(patch_run_dir)
    full_volume_config = load_resolved_config(full_volume_run_dir)
    warmup_epochs = shared_warmup_epochs(patch_config, full_volume_config)

    patch_case_metrics = load_per_case_metrics(patch_run_dir)
    full_case_metrics = load_per_case_metrics(full_volume_run_dir)
    patch_evaluation_summary = load_evaluation_summary(patch_run_dir)
    full_volume_evaluation_summary = load_evaluation_summary(full_volume_run_dir)

    figure_outputs: list[tuple[str, plt.Figure]] = []

    figure_outputs.append(
        (
            "dice_comparison.png",
            build_dice_comparison_figure(
                patch_history=patch_history,
                full_volume_history=full_volume_history,
                patch_best=patch_payload.get("best", {}),
                full_volume_best=full_volume_payload.get("best", {}),
                include_title=args.include_title,
                warmup_epochs=warmup_epochs,
            ),
        )
    )
    figure_outputs.append(
        (
            "training_loss_comparison.png",
            build_training_loss_figure(
                patch_history=patch_history,
                full_volume_history=full_volume_history,
                include_title=args.include_title,
                warmup_epochs=warmup_epochs,
            ),
        )
    )
    figure_outputs.append(
        (
            "validation_loss_comparison.png",
            build_validation_loss_figure(
                patch_history=patch_history,
                full_volume_history=full_volume_history,
                include_title=args.include_title,
                warmup_epochs=warmup_epochs,
            ),
        )
    )
    figure_outputs.append(
        (
            "learning_rate_comparison.png",
            build_learning_rate_figure(
                patch_history=patch_history,
                full_volume_history=full_volume_history,
                include_title=args.include_title,
                warmup_epochs=warmup_epochs,
            ),
        )
    )
    figure_outputs.append(
        (
            "dice_gap_comparison.png",
            build_dice_gap_figure(
                patch_history=patch_history,
                full_volume_history=full_volume_history,
                include_title=args.include_title,
                warmup_epochs=warmup_epochs,
            ),
        )
    )

    if patch_case_metrics and full_case_metrics:
        figure_outputs.append(
            (
                "per_case_dice_comparison.png",
                build_per_case_dice_figure(
                    patch_case_metrics=patch_case_metrics,
                    full_case_metrics=full_case_metrics,
                    include_title=args.include_title,
                ),
            )
        )
        data_root = resolve_data_root(patch_run_dir, patch_config)
        qualitative_figure, qualitative_case_ids = build_qualitative_comparison_figure(
            patch_run_dir=patch_run_dir,
            full_volume_run_dir=full_volume_run_dir,
            patch_case_metrics=patch_case_metrics,
            full_case_metrics=full_case_metrics,
            data_root=data_root,
            include_title=args.include_title,
            requested_case_id=args.qual_case_id,
            max_cases=args.qual_max_cases,
        )
        figure_outputs.append(
            (
                "qualitative_case_comparison.png",
                qualitative_figure,
            )
        )
    else:
        qualitative_case_ids = []

    for filename, figure in figure_outputs:
        output_path = output_dir / filename
        figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Saved {filename} to {output_path}")

    if qualitative_case_ids:
        print(f"Qualitative cases: {', '.join(qualitative_case_ids)}")

    save_summary_text(
        output_path=output_dir / "summary.txt",
        patch_run_dir=patch_run_dir,
        full_volume_run_dir=full_volume_run_dir,
        patch_payload=patch_payload,
        patch_history=patch_history,
        patch_evaluation_summary=patch_evaluation_summary,
        full_volume_payload=full_volume_payload,
        full_volume_history=full_volume_history,
        full_volume_evaluation_summary=full_volume_evaluation_summary,
    )
    print(f"Saved summary to {output_dir / 'summary.txt'}")

    if args.show:
        plt.show()
    else:
        for _, figure in figure_outputs:
            plt.close(figure)


if __name__ == "__main__":
    main()
