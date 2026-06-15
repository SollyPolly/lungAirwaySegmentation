"""Run indexing utilities for comparing experiments across the runs directory."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from lung_airway_segmentation.settings import RUNS_ROOT


RUN_INDEX_FILENAME = "run_index.csv"
RUN_INDEX_COLUMNS = [
    "run_dir",
    "run_name",
    "created_at",
    "study_name",
    "run_label",
    "experiment_name",
    "description",
    "model_name",
    "training_regime",
    "device",
    "seed",
    "epochs_configured",
    "epochs_completed",
    "batch_size",
    "effective_batch_size",
    "patch_size",
    "optimizer_name",
    "scheduler_name",
    "learning_rate",
    "weight_decay",
    "positive_class_weight",
    "pretrained_enabled",
    "freeze_encoder",
    "best_epoch",
    "best_val_dice",
    "best_topology_epoch",
    "best_val_cldice",
    "evaluation_name",
    "evaluation_dir",
    "eval_case_ids",
    "eval_prediction_filename",
    "eval_checkpoint_epoch",
    "eval_threshold",
    "eval_num_cases",
    "eval_dice_mean",
    "eval_iou_mean",
    "eval_precision_mean",
    "eval_recall_mean",
    "eval_tree_length_detected_mean",
    "eval_branch_detected_mean",
    "eval_cldice_mean",
    "eval_topology_precision_mean",
    "eval_topology_sensitivity_mean",
    "predictions_saved",
    "status",
]


def load_json_if_exists(path: Path) -> dict | None:
    """Load one JSON file if it exists and contains an object."""
    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return data


def load_json_list_if_exists(path: Path) -> list | None:
    """Load one JSON file if it exists and contains a list."""
    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")
    return data


def list_evaluation_dirs(run_dir: Path) -> list[Path]:
    """List saved evaluation variants that contain a summary artifact."""
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("evaluation")
        and (path / "summary.json").is_file()
    )


def detect_predictions_saved(run_dir: Path) -> bool:
    """Return whether the run directory already contains saved case predictions."""
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.is_dir():
        return False

    return any((case_dir / "prediction_metadata.json").is_file() for case_dir in predictions_dir.iterdir())


def determine_run_status(
    *,
    history: dict | None,
    evaluation_summary: dict | None,
    predictions_saved: bool,
    run_dir: Path,
) -> str:
    """Summarize the run lifecycle from available artifacts."""
    if evaluation_summary is not None:
        return "evaluated"
    if predictions_saved:
        return "predicted"
    if history is not None and (run_dir / "last_model.pt").is_file():
        return "trained"
    return "initialized"


def infer_training_regime(training_config: dict, run_metadata: dict) -> str:
    """Infer the training regime from either resolved config or legacy metadata."""
    regime = training_config.get("training_regime")
    if regime:
        return str(regime)

    searchable_text = " ".join(
        str(value).lower()
        for value in (
            run_metadata.get("experiment_name", ""),
            run_metadata.get("description", ""),
        )
    )
    if "full_volume" in searchable_text or "full volume" in searchable_text:
        return "full_volume"
    if "patch" in searchable_text:
        return "patch"
    return ""


def build_run_index_row(
    run_dir: Path,
    runs_root: Path = RUNS_ROOT,
    evaluation_dir: Path | None = None,
) -> dict[str, object]:
    """Build one flattened index row for a run and optional evaluation variant."""
    run_metadata = load_json_if_exists(run_dir / "run_metadata.json")
    if run_metadata is None:
        raise FileNotFoundError(f"Run metadata is missing for run directory: {run_dir}")

    resolved_config = load_json_if_exists(run_dir / "resolved_config.json") or {}
    history = load_json_if_exists(run_dir / "history.json")
    if evaluation_dir is None:
        default_evaluation_dir = run_dir / "evaluation"
        evaluation_dir = default_evaluation_dir if (default_evaluation_dir / "summary.json").is_file() else None
    evaluation_summary = (
        load_json_if_exists(evaluation_dir / "summary.json")
        if evaluation_dir is not None
        else None
    )
    per_case_metrics = (
        load_json_list_if_exists(evaluation_dir / "per_case_metrics.json")
        if evaluation_dir is not None
        else None
    ) or []

    training_config = resolved_config.get("training", {})
    model_config = resolved_config.get("model", {})
    optimizer_config = training_config.get("optimizer", {})
    loss_config = training_config.get("loss", {})
    sampling_config = training_config.get("sampling", {})
    pretrained_config = model_config.get("pretrained", {}) if isinstance(model_config.get("pretrained"), dict) else {}

    history_rows = history.get("history", []) if history is not None else []
    best = history.get("best", {}) if history is not None else {}
    # best_topology block exists only for runs from 2026-06-15 onward (three-
    # checkpoint topology-aware selection); empty for older runs.
    best_topology = history.get("best_topology", {}) if history is not None else {}
    predictions_saved = detect_predictions_saved(run_dir)
    patch_size = sampling_config.get("patch_size")
    first_case_metrics = per_case_metrics[0] if per_case_metrics else {}
    prediction_mask_path = Path(first_case_metrics["prediction_mask_path"]) if first_case_metrics.get("prediction_mask_path") else None
    evaluation_name = evaluation_dir.name if evaluation_dir is not None else ""

    try:
        relative_run_dir = run_dir.relative_to(runs_root)
    except ValueError:
        relative_run_dir = run_dir

    row = {
        "run_dir": str(relative_run_dir),
        "run_name": run_metadata.get("run_name", run_dir.name),
        "created_at": run_metadata.get("created_at", ""),
        "study_name": run_metadata.get("study_name") or training_config.get("study_name", ""),
        "run_label": run_metadata.get("run_label") or training_config.get("run_label", ""),
        "experiment_name": run_metadata.get("experiment_name", run_dir.parent.name),
        "description": run_metadata.get("description", ""),
        "model_name": run_metadata.get(
            "model_name",
            model_config.get("model_name", run_metadata.get("model", {}).get("name", "")),
        ),
        "training_regime": infer_training_regime(training_config, run_metadata) or run_metadata.get("data_pipeline", ""),
        "device": run_metadata.get("device", ""),
        "seed": training_config.get("seed", run_metadata.get("seed", "")),
        "epochs_configured": training_config.get("epochs", run_metadata.get("num_epochs", "")),
        "epochs_completed": len(history_rows),
        "batch_size": training_config.get("batch_size", run_metadata.get("batch_size", "")),
        "effective_batch_size": run_metadata.get("effective_batch_size", ""),
        "patch_size": "x".join(str(value) for value in patch_size) if isinstance(patch_size, list) else "",
        "optimizer_name": run_metadata.get("optimizer_name", optimizer_config.get("name", "")),
        "scheduler_name": run_metadata.get("scheduler_name", training_config.get("scheduler", {}).get("name", "")),
        "learning_rate": optimizer_config.get("lr", run_metadata.get("learning_rate", "")),
        "weight_decay": optimizer_config.get("weight_decay", ""),
        # Historical supervised runs predate this config field and used a
        # hardcoded positive-class weight of 10.
        "positive_class_weight": loss_config.get("positive_class_weight", 10.0),
        "pretrained_enabled": pretrained_config.get("enabled", ""),
        "freeze_encoder": pretrained_config.get("freeze_encoder", ""),
        "best_epoch": best.get("epoch", ""),
        "best_val_dice": best.get("val_dice", ""),
        "best_topology_epoch": best_topology.get("epoch", ""),
        "best_val_cldice": best_topology.get("val_cldice", ""),
        "evaluation_name": evaluation_name,
        "evaluation_dir": str(evaluation_dir.relative_to(run_dir)) if evaluation_dir is not None else "",
        "eval_case_ids": ",".join(str(case["case_id"]) for case in per_case_metrics if "case_id" in case),
        "eval_prediction_filename": prediction_mask_path.name if prediction_mask_path is not None else "",
        "eval_checkpoint_epoch": first_case_metrics.get("checkpoint_epoch", ""),
        "eval_threshold": first_case_metrics.get("threshold", ""),
        "eval_num_cases": evaluation_summary.get("num_cases", "") if evaluation_summary is not None else "",
        "eval_dice_mean": evaluation_summary.get("dice_mean", "") if evaluation_summary is not None else "",
        "eval_iou_mean": evaluation_summary.get("iou_mean", "") if evaluation_summary is not None else "",
        "eval_precision_mean": evaluation_summary.get("precision_mean", "") if evaluation_summary is not None else "",
        "eval_recall_mean": evaluation_summary.get("recall_mean", "") if evaluation_summary is not None else "",
        "eval_tree_length_detected_mean": evaluation_summary.get("tree_length_detected_mean", "") if evaluation_summary is not None else "",
        "eval_branch_detected_mean": evaluation_summary.get("branch_detected_mean", "") if evaluation_summary is not None else "",
        "eval_cldice_mean": evaluation_summary.get("cldice_mean", "") if evaluation_summary is not None else "",
        "eval_topology_precision_mean": evaluation_summary.get("topology_precision_mean", "") if evaluation_summary is not None else "",
        "eval_topology_sensitivity_mean": evaluation_summary.get("topology_sensitivity_mean", "") if evaluation_summary is not None else "",
        "predictions_saved": predictions_saved,
        "status": determine_run_status(
            history=history,
            evaluation_summary=evaluation_summary,
            predictions_saved=predictions_saved,
            run_dir=run_dir,
        ),
    }
    return row


def collect_run_index_rows(runs_root: Path = RUNS_ROOT) -> list[dict[str, object]]:
    """Scan the runs root and return one row per evaluation variant or unevaluated run."""
    if not runs_root.exists():
        return []

    metadata_paths = sorted(runs_root.rglob("run_metadata.json"))
    rows = []
    for metadata_path in metadata_paths:
        run_dir = metadata_path.parent
        evaluation_dirs = list_evaluation_dirs(run_dir)
        if evaluation_dirs:
            rows.extend(
                build_run_index_row(run_dir, runs_root=runs_root, evaluation_dir=evaluation_dir)
                for evaluation_dir in evaluation_dirs
            )
        else:
            rows.append(build_run_index_row(run_dir, runs_root=runs_root))
    rows.sort(
        key=lambda row: (
            str(row["created_at"]),
            str(row["run_dir"]),
            str(row["evaluation_name"]),
        ),
        reverse=True,
    )
    return rows


def write_run_index_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """Write the aggregated run index to one CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    with temporary_output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary_output_path, output_path)
        return output_path
    except PermissionError:
        fallback_output_path = output_path.with_name(f"{output_path.stem}.pending{output_path.suffix}")
        os.replace(temporary_output_path, fallback_output_path)
        return fallback_output_path


def refresh_run_index(
    runs_root: Path = RUNS_ROOT,
    output_path: Path | None = None,
) -> Path:
    """Regenerate the shared run index CSV from all run directories."""
    rows = collect_run_index_rows(runs_root)
    resolved_output_path = output_path or (runs_root / RUN_INDEX_FILENAME)
    return write_run_index_csv(rows, resolved_output_path)
