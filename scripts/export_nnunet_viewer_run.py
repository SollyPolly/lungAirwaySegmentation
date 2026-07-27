"""Export nnU-Net hard-mask predictions into the viewer-compatible runs layout.

nnU-Net keeps its native workspace under data/nnunet. The Marimo viewer expects
per-case prediction folders under runs/<study>/<run>/predictions*/ with a
prediction_metadata.json gate file. This script creates that lightweight derived
artifact without moving or modifying the nnU-Net workspace.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from lung_airway_segmentation.config import load_yaml_config, resolve_project_path
from lung_airway_segmentation.datasets.splits import cases_for_split, create_split_from_config
from lung_airway_segmentation.io.atm22_layout import list_case_ids

DEFAULT_PRED_DIR = Path("data/nnunet/predict_out/Dataset111_val")
DEFAULT_OUT_RUN = Path(
    "runs/nnunet-track-a/"
    "2026-07-03__dataset111-3d-fullres-5fold-final__nnunetv2"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Flat nnU-Net prediction output folder containing ATM_XXX.nii.gz.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/atm22.yaml"),
        help="ATM'22 data YAML.",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/nnunet/atm22_split_l20.yaml"),
        help="Canonical ATM'22 split YAML.",
    )
    parser.add_argument(
        "--report-split",
        choices=("val", "test", "train"),
        default="val",
        help="Split represented by --pred-dir.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case IDs; overrides --split-config.",
    )
    parser.add_argument(
        "--out-run-dir",
        type=Path,
        default=DEFAULT_OUT_RUN,
        help="Viewer-compatible run folder to create under runs/.",
    )
    parser.add_argument(
        "--prediction-set",
        default=None,
        help="Prediction-set folder name. Default: predictions_<report-split>.",
    )
    parser.add_argument(
        "--score-json",
        type=Path,
        default=None,
        help="Optional score JSON to copy into the exported run root. Default: "
        "<pred-dir>/nnunet111_<report-split>_topology.json when present.",
    )
    parser.add_argument("--dataset-id", type=int, default=111)
    parser.add_argument("--dataset-name", default="ATM22")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--checkpoint", default="checkpoint_final")
    parser.add_argument(
        "--folds",
        default="0,1,2,3,4",
        help="Comma-separated folds used for prediction.",
    )
    parser.add_argument(
        "--study-name",
        default="nnunet-track-a",
        help="runs/<study-name>/ bucket + study label in the metadata.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Short run label. Default: dataset<id>-<config>-5fold-final.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment label. Default: nnunetv2-dataset<id>-<config>.",
    )
    parser.add_argument(
        "--description",
        default=(
            "Stock nnU-Net v2 Track-A control exported for mask_visualisation. "
            "Predictions are hard native-argmax masks from the 5-fold ensemble; "
            "the native nnU-Net workspace remains under data/nnunet."
        ),
        help="Free-text description recorded in run_metadata.json.",
    )
    parser.add_argument(
        "--checkpoint-model",
        default="5fold_ensemble_native_argmax",
        help="checkpoint_model tag recorded in run_metadata.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in an existing exported run.",
    )
    args = parser.parse_args()
    if args.run_label is None:
        args.run_label = f"dataset{args.dataset_id}-{args.configuration}-5fold-final"
    if args.experiment_name is None:
        args.experiment_name = f"nnunetv2-dataset{args.dataset_id}-{args.configuration}"
    return args


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_cases(
    split: dict[str, list[str]],
    report_split: str,
    score_json: Path | None,
    cases_override: str | None,
) -> list[str]:
    if cases_override:
        return [value.strip().zfill(3) for value in cases_override.split(",") if value.strip()]
    if score_json is not None and score_json.is_file():
        score = load_json(score_json)
        cases = [str(case_id).zfill(3) for case_id in score.get("report_cases", [])]
        if cases:
            return cases

    cases = cases_for_split(split, report_split)
    if not cases:
        raise ValueError(f"No {report_split} cases found in the split configuration.")
    return [str(case_id).zfill(3) for case_id in cases]


def write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} already exists; pass --overwrite to replace it.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_run_metadata(args: argparse.Namespace, split: dict[str, list[str]], cases: list[str]) -> dict:
    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    run_name = args.out_run_dir.name
    return {
        "study_name": args.study_name,
        "run_label": args.run_label,
        "experiment_name": args.experiment_name,
        "description": args.description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "run_dir": str(args.out_run_dir),
        "config_files": {
            "data": str(args.data_config),
            "split": str(args.split_config),
            "nnunet_prediction_dir": str(args.pred_dir),
        },
        "data_root": "data/ATM22",
        "data_pipeline": "nnunetv2_native_export",
        "model_name": "nnunetv2",
        "checkpoint_model": args.checkpoint_model,
        "nnunet": {
            "dataset_id": args.dataset_id,
            "dataset_name": args.dataset_name,
            "configuration": args.configuration,
            "checkpoint": args.checkpoint,
            "folds": folds,
            "source_pred_dir": str(args.pred_dir),
        },
        "splits": {
            "train_count": len(split["labelled_train"]) + len(split["unlabelled_train"]),
            "val_count": len(split["val"]),
            "test_count": len(split["test"]),
            "train_case_ids": cases_for_split(split, "train"),
            "val_case_ids": split["val"],
            "test_case_ids": split["test"],
            "exported_case_ids": cases,
        },
    }


def build_resolved_config(args: argparse.Namespace) -> dict:
    return {
        "data": {
            "dataset_name": "atm22",
            "batch_root": "data/ATM22",
            "preprocessing": {"hu_window": [-1024, 2048]},
        },
        "model": {
            "model_name": "nnunetv2",
            "dataset_id": args.dataset_id,
            "dataset_name": args.dataset_name,
            "configuration": args.configuration,
            "folds": [int(value.strip()) for value in args.folds.split(",") if value.strip()],
            "checkpoint": args.checkpoint,
        },
        "training": {
            "study_name": args.study_name,
            "run_label": args.run_label,
            "experiment_name": args.experiment_name,
            "validation": {"threshold": None},
        },
    }


def main() -> None:
    args = parse_args()
    prediction_set = args.prediction_set or f"predictions_{args.report_split}"
    score_json = args.score_json
    if score_json is None:
        candidate = args.pred_dir / f"nnunet{args.dataset_id}_{args.report_split}_topology.json"
        score_json = candidate if candidate.is_file() else None

    data_config = load_yaml_config(args.data_config)
    batch_root = resolve_project_path(data_config["batch_root"])
    split = create_split_from_config(
        list_case_ids(batch_root),
        load_yaml_config(args.split_config),
    )
    cases = resolve_cases(split, args.report_split, score_json, args.cases)

    args.out_run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_run_dir / "run_metadata.json",
        build_run_metadata(args, split, cases),
        overwrite=args.overwrite,
    )
    write_json(
        args.out_run_dir / "resolved_config.json",
        build_resolved_config(args),
        overwrite=args.overwrite,
    )

    if score_json is not None and score_json.is_file():
        copy_file(score_json, args.out_run_dir / score_json.name, overwrite=args.overwrite)

    for case_id in cases:
        src = args.pred_dir / f"ATM_{case_id}.nii.gz"
        case_dir = args.out_run_dir / prediction_set / case_id
        copy_file(src, case_dir / "airway_pred_full.nii.gz", overwrite=args.overwrite)
        write_json(
            case_dir / "prediction_metadata.json",
            {
                "case_id": case_id,
                "study_name": args.study_name,
                "run_label": args.run_label,
                "experiment_name": args.experiment_name,
                "source_prediction_path": str(src),
                "source_prediction_dir": str(args.pred_dir),
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": None,
                "threshold": None,
                "operating_point": "native_argmax",
                "folds": [int(value.strip()) for value in args.folds.split(",") if value.strip()],
                "largest_component_saved": False,
                "note": "nnU-Net hard mask copied as airway_pred_full.nii.gz for mask_visualisation.",
            },
            overwrite=args.overwrite,
        )

    print(
        f"Exported {len(cases)} nnU-Net prediction(s) to "
        f"{args.out_run_dir / prediction_set}"
    )


if __name__ == "__main__":
    main()
