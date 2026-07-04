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


DEFAULT_SPLIT_RUN = Path(
    "runs/atm-l110-supervised/"
    "2026-06-26__06-12-47__cldice-w1-cbdice-w2-p96-l110__baseline_unet"
)
DEFAULT_PRED_DIR = Path("data/nnunet/predict_out/Dataset111_val")
DEFAULT_OUT_RUN = Path(
    "runs/nnunet-track-a/"
    "2026-07-03__dataset111-3d-fullres-5fold-final__nnunetv2"
)

SPLIT_KEYS = {
    "val": "val_case_ids",
    "test": "test_case_ids",
    "train": "train_case_ids",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Flat nnU-Net prediction output folder containing ATM_XXX.nii.gz.",
    )
    parser.add_argument(
        "--split-run-dir",
        type=Path,
        default=DEFAULT_SPLIT_RUN,
        help="Run dir whose run_metadata.json provides the held-out split IDs.",
    )
    parser.add_argument(
        "--report-split",
        choices=tuple(SPLIT_KEYS),
        default="val",
        help="Split represented by --pred-dir.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite files in an existing exported run.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_cases(split_run_dir: Path, report_split: str, score_json: Path | None) -> list[str]:
    if score_json is not None and score_json.is_file():
        score = load_json(score_json)
        cases = [str(case_id).zfill(3) for case_id in score.get("report_cases", [])]
        if cases:
            return cases

    metadata = load_json(split_run_dir / "run_metadata.json")
    cases = metadata.get("splits", {}).get(SPLIT_KEYS[report_split], [])
    if not cases:
        raise ValueError(f"No {report_split} cases found in {split_run_dir / 'run_metadata.json'}")
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


def build_run_metadata(args: argparse.Namespace, source_metadata: dict, cases: list[str]) -> dict:
    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    run_name = args.out_run_dir.name
    return {
        "study_name": "nnunet-track-a",
        "run_label": f"dataset{args.dataset_id}-{args.configuration}-5fold-final",
        "experiment_name": f"nnunetv2-dataset{args.dataset_id}-{args.configuration}",
        "description": (
            "Stock nnU-Net v2 Track-A control exported for mask_visualisation. "
            "Predictions are hard native-argmax masks from the 5-fold ensemble; "
            "the native nnU-Net workspace remains under data/nnunet."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "run_dir": str(args.out_run_dir),
        "config_files": {
            "source_split_run": str(args.split_run_dir),
            "nnunet_prediction_dir": str(args.pred_dir),
        },
        "data_root": "data/ATM22",
        "data_pipeline": "nnunetv2_native_export",
        "model_name": "nnunetv2",
        "checkpoint_model": "5fold_ensemble_native_argmax",
        "nnunet": {
            "dataset_id": args.dataset_id,
            "dataset_name": args.dataset_name,
            "configuration": args.configuration,
            "checkpoint": args.checkpoint,
            "folds": folds,
            "source_pred_dir": str(args.pred_dir),
        },
        "splits": {
            "train_count": source_metadata.get("splits", {}).get("train_count"),
            "val_count": source_metadata.get("splits", {}).get("val_count"),
            "test_count": source_metadata.get("splits", {}).get("test_count"),
            "train_case_ids": source_metadata.get("splits", {}).get("train_case_ids", []),
            "val_case_ids": source_metadata.get("splits", {}).get("val_case_ids", []),
            "test_case_ids": source_metadata.get("splits", {}).get("test_case_ids", []),
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
            "study_name": "nnunet-track-a",
            "run_label": f"dataset{args.dataset_id}-{args.configuration}-5fold-final",
            "experiment_name": f"nnunetv2-dataset{args.dataset_id}-{args.configuration}",
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

    source_metadata = load_json(args.split_run_dir / "run_metadata.json")
    cases = resolve_cases(args.split_run_dir, args.report_split, score_json)

    args.out_run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_run_dir / "run_metadata.json",
        build_run_metadata(args, source_metadata, cases),
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
                "study_name": "nnunet-track-a",
                "run_label": f"dataset{args.dataset_id}-{args.configuration}-5fold-final",
                "experiment_name": f"nnunetv2-dataset{args.dataset_id}-{args.configuration}",
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
