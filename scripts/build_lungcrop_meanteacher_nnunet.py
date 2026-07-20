"""Build the paired lung-ROI nnU-Net seed and Mean-Teacher datasets.

Recommended experiment:

* Dataset123_ATM22L20LungCrop: the same 20 real-GT cases as the @20 seed.
* Dataset124_ATM22MTLungCrop: those 20 cases plus 90 label-withheld cases whose
  segmentation files contain only nnU-Net's ignore value (2).

The NIfTIs remain on their original grid. CT voxels outside an affine-aware
lung bounding box (margin 8, superior/trachea extension 120 by default) are
set to zero. nnU-Net's own non-zero crop therefore supplies the actual ROI and
retains the metadata needed to restore predictions to the full grid.

The fold-0 validation cases are pinned to the original Dataset120 seed split.
Dataset124 trains on the same 16 GT cases plus all 90 unlabelled cases and
validates only on the same four real-GT cases.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lung_airway_segmentation.datasets.splits import create_semisupervised_split
from lung_airway_segmentation.io.atm22_layout import list_case_ids, resolve_lung_mask_path
from lung_airway_segmentation.io.nnunet_export import _place, nnunet_dataset_json
from lung_airway_segmentation.io.nnunet_lungcrop import (
    bbox_from_json,
    write_ignore_target,
    write_lung_roi_ct,
    write_roi_ground_truth,
)
from lung_airway_segmentation.training.config import load_yaml_config, resolve_project_path

IGNORE_INDEX = 2
MT_LABELS = {"background": 0, "airway": 1, "ignore": IGNORE_INDEX}
DEFAULT_FOLD0_VAL = ("008", "050", "135", "158")


def _padded(case_id: str) -> str:
    value = str(case_id)
    if value.upper().startswith("ATM_"):
        value = value[4:]
    if not value.isdigit():
        raise ValueError(f"Invalid ATM case id: {case_id!r}")
    return f"{int(value):03d}"


def _case_key(case_id: str) -> str:
    return f"ATM_{_padded(case_id)}"


def _ct_path(batch_root: Path, case_id: str) -> Path:
    path = batch_root / "imagesTr" / f"{_case_key(case_id)}_0000.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"CT not found: {path}")
    return path


def _gt_path_for_labelled_case(batch_root: Path, case_id: str) -> Path:
    key = _case_key(case_id)
    candidates = (batch_root / "labelsTr" / f"{key}.nii.gz", batch_root / "labelsTr" / f"{key}_0000.nii.gz")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Real-GT case {key} has no airway label.")
    return path


def _require_fresh_dataset(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to merge into non-empty dataset directory: {path}. "
            "Use fresh dataset IDs/names or move the old directory aside."
        )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dataset_metadata(margin: int, superior_margin: int) -> dict:
    return {
        "method": "full_grid_zero_outside_lung_bbox",
        "margin_voxels": int(margin),
        "superior_margin_voxels": int(superior_margin),
        "affine_aware_superior_axis": True,
        "prediction_restore": "automatic via nnU-Net nonzero-crop metadata",
    }


def assemble(args) -> tuple[Path, Path]:
    data_config = load_yaml_config(args.data_config)
    training_config = load_yaml_config(args.training_config)
    batch_root = resolve_project_path(data_config["batch_root"])
    counts = training_config["labelled_split"]
    split = create_semisupervised_split(
        list_case_ids(batch_root),
        test_count=int(counts["test_count"]),
        val_count=int(counts["val_count"]),
        labelled_count=int(counts["labelled_count"]),
        seed=int(training_config.get("seed", 15)),
    )
    labelled = [_padded(case) for case in split["labelled_train"]]
    unlabelled = [_padded(case) for case in split["unlabelled_train"]]
    fold0_val = [_padded(case.strip()) for case in args.fold0_val.split(",") if case.strip()]
    if len(fold0_val) != 4 or not set(fold0_val).issubset(labelled):
        raise ValueError(
            f"Fold-0 validation must be four members of the labelled-20 pool; got {fold0_val}, "
            f"labelled pool={labelled}."
        )
    labelled_train = sorted(set(labelled) - set(fold0_val))
    if len(labelled) != 20 or len(labelled_train) != 16 or len(unlabelled) != 90:
        raise ValueError(
            "This controlled experiment expects 20 labelled cases (16 train/4 val) and 90 unlabelled; "
            f"got {len(labelled)}, {len(labelled_train)}/{len(fold0_val)}, and {len(unlabelled)}."
        )

    raw_root = Path(args.nnunet_raw)
    seed_dir = raw_root / f"Dataset{args.seed_dataset_id:03d}_{args.seed_dataset_name}"
    mt_dir = raw_root / f"Dataset{args.mt_dataset_id:03d}_{args.mt_dataset_name}"
    if seed_dir == mt_dir:
        raise ValueError("Seed and MT dataset directories must differ.")
    _require_fresh_dataset(seed_dir)
    _require_fresh_dataset(mt_dir)
    for dataset_dir in (seed_dir, mt_dir):
        (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    seed_records: dict[str, dict] = {}
    mt_records: dict[str, dict] = {}
    provenance: dict[str, str] = {}

    # These are the only cases for which this builder resolves or reads GT.
    for case_id in labelled:
        key = _case_key(case_id)
        ct_path = _ct_path(batch_root, case_id)
        lung_path = resolve_lung_mask_path(case_id, batch_root=batch_root, lung_root=args.lung_root)
        if not lung_path.is_file():
            raise FileNotFoundError(f"Precomputed lung mask not found: {lung_path}")
        seed_ct = seed_dir / "imagesTr" / f"{key}_0000.nii.gz"
        seed_gt = seed_dir / "labelsTr" / f"{key}.nii.gz"
        roi_record = write_lung_roi_ct(
            ct_path,
            lung_path,
            seed_ct,
            margin_voxels=args.margin_voxels,
            superior_margin_voxels=args.superior_margin_voxels,
        )
        gt_record = write_roi_ground_truth(
            _gt_path_for_labelled_case(batch_root, case_id),
            ct_path,
            bbox_from_json(roi_record["bbox"]),
            seed_gt,
        )
        record = {**roi_record, **gt_record, "provenance": "gt"}
        seed_records[key] = record

        # Hardlink/copy the exact seed inputs into Dataset124. This guarantees
        # that warm-start and MT see identical voxels for the labelled 20.
        _place(seed_ct, mt_dir / "imagesTr" / seed_ct.name, args.reuse_mode)
        _place(seed_gt, mt_dir / "labelsTr" / seed_gt.name, args.reuse_mode)
        mt_records[key] = dict(record)
        provenance[key] = "gt"
        print(f"GT     {key}: ROI {roi_record['roi_shape']} ({roi_record['roi_fraction']:.1%})", flush=True)

    # Leak guard: this loop constructs paths only to CT and lung masks. It never
    # calls a resolver that searches labelsTr and never opens withheld GT.
    for case_id in unlabelled:
        key = _case_key(case_id)
        ct_path = _ct_path(batch_root, case_id)
        lung_path = resolve_lung_mask_path(case_id, batch_root=batch_root, lung_root=args.lung_root)
        if not lung_path.is_file():
            raise FileNotFoundError(f"Precomputed lung mask not found: {lung_path}")
        mt_ct = mt_dir / "imagesTr" / f"{key}_0000.nii.gz"
        mt_target = mt_dir / "labelsTr" / f"{key}.nii.gz"
        roi_record = write_lung_roi_ct(
            ct_path,
            lung_path,
            mt_ct,
            margin_voxels=args.margin_voxels,
            superior_margin_voxels=args.superior_margin_voxels,
        )
        ignore_record = write_ignore_target(ct_path, mt_target, IGNORE_INDEX)
        mt_records[key] = {**roi_record, **ignore_record, "provenance": "ignore"}
        provenance[key] = "ignore"
        print(f"IGNORE {key}: ROI {roi_record['roi_shape']} ({roi_record['roi_fraction']:.1%})", flush=True)

    seed_split = [{
        "train": [_case_key(case) for case in labelled_train],
        "val": [_case_key(case) for case in fold0_val],
    }]
    mt_split = [{
        "train": [_case_key(case) for case in labelled_train + unlabelled],
        "val": [_case_key(case) for case in fold0_val],
    }]
    crop_metadata = _dataset_metadata(args.margin_voxels, args.superior_margin_voxels)

    seed_json = nnunet_dataset_json(len(labelled))
    seed_json["lung_roi"] = crop_metadata
    _write_json(seed_dir / "dataset.json", seed_json)
    _write_json(seed_dir / "splits_final.json", seed_split)
    _write_json(seed_dir / "lung_crop_manifest.json", {
        "dataset_role": "supervised_seed",
        "lung_roi": crop_metadata,
        "folds": {"0": seed_split[0]},
        "excluded_external_val": sorted(split["val"]),
        "excluded_sealed_test": sorted(split["test"]),
        "cases": seed_records,
    })

    mt_json = nnunet_dataset_json(len(labelled) + len(unlabelled), labels=MT_LABELS)
    mt_json["lung_roi"] = crop_metadata
    mt_json["semi_supervised"] = {
        "version": 1,
        "ignore_index": IGNORE_INDEX,
        "case_provenance": provenance,
        "folds": {"0": mt_split[0]},
        "supervised_loss_scope": "gt_only",
        "consistency_loss_scope": "unlabelled_only",
    }
    _write_json(mt_dir / "dataset.json", mt_json)
    _write_json(mt_dir / "splits_final.json", mt_split)
    _write_json(mt_dir / "label_provenance.json", {
        "method": "mean_teacher_ignore_label_plus_explicit_two_stream",
        "ignore_index": IGNORE_INDEX,
        "num_gt": len(labelled),
        "num_ignore": len(unlabelled),
        "labels": provenance,
    })
    _write_json(mt_dir / "lung_crop_manifest.json", {
        "dataset_role": "mean_teacher",
        "lung_roi": crop_metadata,
        "folds": {"0": mt_split[0]},
        "excluded_external_val": sorted(split["val"]),
        "excluded_sealed_test": sorted(split["test"]),
        "withheld_gt_read": False,
        "cases": mt_records,
    })

    print(f"\nBuilt seed: {seed_dir} (20 GT; fold0 16 train/4 GT val)")
    print(f"Built MT:   {mt_dir} (20 GT + 90 ignore; fold0 16 GT + 90 unlabelled train/4 GT val)")
    print("Copy each generated splits_final.json into its nnUNet_preprocessed dataset directory after preprocessing.")
    return seed_dir, mt_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/atm22.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training/supervised_atm.yaml"))
    parser.add_argument("--nnunet-raw", type=Path, default=os.environ.get("nnUNet_raw"))
    parser.add_argument("--lung-root", type=Path, default=None, help="Defaults to <batch_root>/lungTr.")
    parser.add_argument("--seed-dataset-id", type=int, default=123)
    parser.add_argument("--seed-dataset-name", default="ATM22L20LungCrop")
    parser.add_argument("--mt-dataset-id", type=int, default=124)
    parser.add_argument("--mt-dataset-name", default="ATM22MTLungCrop")
    parser.add_argument("--fold0-val", default=",".join(DEFAULT_FOLD0_VAL))
    parser.add_argument("--margin-voxels", type=int, default=8)
    parser.add_argument("--superior-margin-voxels", type=int, default=120)
    parser.add_argument("--reuse-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    if args.nnunet_raw is None:
        raise SystemExit("Set --nnunet-raw or the nnUNet_raw environment variable.")
    if args.margin_voxels < 0 or args.superior_margin_voxels < 0:
        raise SystemExit("ROI margins must be non-negative.")
    assemble(args)


if __name__ == "__main__":
    main()
