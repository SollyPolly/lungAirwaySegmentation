"""Build the expanded lung-crop Mean-Teacher dataset from Dataset124.

``Dataset126_ATM22MT240LungCrop`` is a strict scale-up of Dataset124:

* the exact 20 GT + 90 ignore-labelled, lung-cropped Dataset124 files are
  reused byte-for-byte;
* the explicit 150 TrainBatch2 cases are appended as lung-cropped CTs with
  all-ignore targets;
* the original 20 labelled, 20 external-validation, 20 sealed-test, and four
  fold-0 internal-validation memberships remain frozen.

The resulting raw dataset contains 20 GT + 240 ignore cases. Fold 0 trains on
16 GT + 240 unlabelled cases and validates on four real-GT cases. The builder
never resolves or opens airway labels for any of the 240 unlabelled cases.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.config import load_yaml_config, resolve_project_path
from lung_airway_segmentation.datasets.splits import create_split_from_config
from lung_airway_segmentation.io.atm22_layout import list_case_ids, resolve_lung_mask_path
from lung_airway_segmentation.io.nnunet_export import _place, nnunet_dataset_json
from lung_airway_segmentation.io.nnunet_lungcrop import (
    parse_case_intensity_overrides,
    resolve_lung_roi,
    write_ignore_target,
    write_lung_roi_ct,
)
from scripts.build_lungcrop_meanteacher_nnunet import (
    IGNORE_INDEX,
    MT_LABELS,
    _case_key,
    _ct_path,
    _dataset_metadata,
    _padded,
    _require_fresh_dataset,
    _write_json,
)

EXPECTED_GT = 20
EXPECTED_OLD_UNLABELLED = 90
EXPECTED_ADDED_UNLABELLED = 150
EXPECTED_EXPANDED_UNLABELLED = 240
EXPECTED_EXTERNAL_VAL = 20
EXPECTED_TEST = 20
EXPECTED_FOLD0_VAL = {"008", "050", "135", "158"}
EXPECTED_UINT16_OVERRIDES = 10


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _normalised_keys(values) -> set[str]:
    return {_case_key(value) for value in values}


def _preflight_added_inputs(
    batch_root: Path,
    added: set[str],
    lung_root: Path | None,
    intensity_overrides: dict[str, dict],
    *,
    margin_voxels: int,
    superior_margin_voxels: int,
) -> dict[str, dict]:
    """Validate all new CTs/masks and their dtype contract before writing output."""
    inputs: dict[str, dict] = {}
    actual_uint16: set[str] = set()
    for case_id in sorted(added):
        ct_path = _ct_path(batch_root, case_id)
        lung_path = resolve_lung_mask_path(
            case_id,
            batch_root=batch_root,
            lung_root=lung_root,
        )
        if not lung_path.is_file():
            raise FileNotFoundError(f"Precomputed Batch-2 lung mask not found: {lung_path}")

        ct_image = nib.load(str(ct_path))
        lung_image = nib.load(str(lung_path))
        bounds, _ = resolve_lung_roi(
            ct_image,
            lung_image,
            margin_voxels=margin_voxels,
            superior_margin_voxels=superior_margin_voxels,
        )
        storage_dtype = np.dtype(ct_image.get_data_dtype())
        is_uint16 = storage_dtype.kind == "u" and storage_dtype.itemsize == 2
        if is_uint16:
            actual_uint16.add(case_id)
        if case_id in intensity_overrides:
            proxy = ct_image.dataobj
            proxy_slope = getattr(proxy, "slope", 1.0)
            proxy_intercept = getattr(proxy, "inter", 0.0)
            proxy_slope = 1.0 if proxy_slope is None else float(proxy_slope)
            proxy_intercept = 0.0 if proxy_intercept is None else float(proxy_intercept)
            if not np.isclose(proxy_slope, 1.0) or not np.isclose(proxy_intercept, 0.0):
                raise ValueError(
                    f"ATM_{case_id} has both a custom intensity override and non-identity "
                    f"NIfTI scaling ({proxy_slope:g}, {proxy_intercept:g})."
                )
        inputs[case_id] = {
            "ct_path": ct_path,
            "lung_path": lung_path,
            "storage_dtype": str(storage_dtype),
            "bbox": [[int(axis.start), int(axis.stop)] for axis in bounds],
        }

    declared = set(intensity_overrides)
    if actual_uint16 != declared:
        unexpected = sorted(actual_uint16 - declared)
        stale = sorted(declared - actual_uint16)
        raise ValueError(
            "The frozen uint16 intensity manifest does not match Batch-2 CT storage "
            f"dtypes: undeclared_uint16={unexpected}, declared_but_not_uint16={stale}."
        )
    return inputs


def _source_contract(source_dir: Path) -> tuple[dict, dict[str, str], dict, dict]:
    dataset_json = _read_json(source_dir / "dataset.json")
    contract = dataset_json.get("semi_supervised")
    if not isinstance(contract, dict):
        raise ValueError(f"{source_dir} has no dataset.json semi_supervised contract.")

    raw_provenance = contract.get("case_provenance")
    if not isinstance(raw_provenance, dict):
        raise ValueError("Dataset124 semi_supervised contract has no case_provenance mapping.")
    provenance = {_case_key(key): str(value).lower() for key, value in raw_provenance.items()}
    if set(provenance.values()) != {"gt", "ignore"}:
        raise ValueError("Dataset124 provenance must contain only 'gt' and 'ignore'.")
    if list(provenance.values()).count("gt") != EXPECTED_GT:
        raise ValueError(f"Dataset124 must contain {EXPECTED_GT} GT cases.")
    if list(provenance.values()).count("ignore") != EXPECTED_OLD_UNLABELLED:
        raise ValueError(f"Dataset124 must contain {EXPECTED_OLD_UNLABELLED} ignore cases.")

    split_payload = _read_json(source_dir / "splits_final.json")
    if not isinstance(split_payload, list) or len(split_payload) != 1:
        raise ValueError("Dataset124 splits_final.json must contain exactly fold 0.")
    fold0 = split_payload[0]
    contract_fold0 = contract.get("folds", {}).get("0")
    if contract_fold0 != fold0:
        raise ValueError("Dataset124 splits_final.json does not match its embedded contract.")

    manifest = _read_json(source_dir / "lung_crop_manifest.json")
    cases = manifest.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(provenance):
        raise ValueError("Dataset124 crop manifest does not cover exactly its provenance cases.")
    return dataset_json, provenance, fold0, manifest


def assemble(args) -> Path:
    data_config = load_yaml_config(args.data_config)
    split_config = load_yaml_config(args.split_config)
    batch_root = resolve_project_path(data_config["batch_root"])
    split = create_split_from_config(list_case_ids(batch_root), split_config)

    labelled = {_padded(case) for case in split["labelled_train"]}
    unlabelled = {_padded(case) for case in split["unlabelled_train"]}
    external_val = {_padded(case) for case in split["val"]}
    sealed_test = {_padded(case) for case in split["test"]}
    fold0_val = {_padded(case) for case in split_config.get("internal_fold0_val", [])}
    added = {_padded(case) for case in split_config.get("added_unlabelled_case_ids", [])}

    counts = (len(labelled), len(unlabelled), len(external_val), len(sealed_test))
    expected = (EXPECTED_GT, EXPECTED_EXPANDED_UNLABELLED, EXPECTED_EXTERNAL_VAL, EXPECTED_TEST)
    if counts != expected:
        raise ValueError(
            "Expanded split must contain 20 labelled/240 unlabelled/20 external-val/20 test; "
            f"got {counts}."
        )
    if fold0_val != EXPECTED_FOLD0_VAL or not fold0_val.issubset(labelled):
        raise ValueError(
            f"Fold-0 validation must be {sorted(EXPECTED_FOLD0_VAL)}, got {sorted(fold0_val)}."
        )
    if len(added) != EXPECTED_ADDED_UNLABELLED or not added.issubset(unlabelled):
        raise ValueError(
            f"added_unlabelled_case_ids must contain {EXPECTED_ADDED_UNLABELLED} unlabelled cases."
        )
    intensity_overrides = parse_case_intensity_overrides(
        split_config,
        allowed_case_ids=added,
    )
    if len(intensity_overrides) != EXPECTED_UINT16_OVERRIDES:
        raise ValueError(
            f"Expected {EXPECTED_UINT16_OVERRIDES} frozen uint16 intensity overrides, "
            f"got {len(intensity_overrides)}."
        )

    raw_root = Path(args.nnunet_raw)
    source_dir = (
        Path(args.source_dataset_dir)
        if args.source_dataset_dir is not None
        else raw_root / "Dataset124_ATM22MTLungCrop"
    )
    source_json, source_provenance, source_fold0, source_manifest = _source_contract(source_dir)
    source_gt = {_padded(key) for key, value in source_provenance.items() if value == "gt"}
    source_ignore = {_padded(key) for key, value in source_provenance.items() if value == "ignore"}
    if source_gt != labelled:
        raise ValueError("Expanded labelled-20 membership differs from Dataset124.")
    if source_ignore != (unlabelled - added):
        raise ValueError("Expanded legacy unlabelled-90 membership differs from Dataset124.")
    if added & set(source_gt | source_ignore | external_val | sealed_test):
        raise ValueError("Batch-2 additions overlap a frozen Batch-1 role.")

    expected_source_train = _normalised_keys((labelled - fold0_val) | source_ignore)
    expected_source_val = _normalised_keys(fold0_val)
    if set(source_fold0.get("train", [])) != expected_source_train:
        raise ValueError("Dataset124 fold-0 training membership is not the frozen 16 GT + 90.")
    if set(source_fold0.get("val", [])) != expected_source_val:
        raise ValueError("Dataset124 fold-0 validation membership is not the frozen four GT.")
    if {_padded(case) for case in source_manifest.get("excluded_external_val", [])} != external_val:
        raise ValueError("Dataset124 external-validation membership differs from the expanded manifest.")
    if {_padded(case) for case in source_manifest.get("excluded_sealed_test", [])} != sealed_test:
        raise ValueError("Dataset124 sealed-test membership differs from the expanded manifest.")

    crop_metadata = source_json.get("lung_roi")
    if not isinstance(crop_metadata, dict):
        raise ValueError("Dataset124 has no lung_roi metadata.")
    margin = int(crop_metadata.get("margin_voxels", -1))
    superior_margin = int(crop_metadata.get("superior_margin_voxels", -1))
    if crop_metadata != _dataset_metadata(margin, superior_margin):
        raise ValueError("Dataset124 lung_roi metadata is incomplete or unsupported.")

    source_cases = source_manifest["cases"]
    for key in sorted(source_provenance):
        image_path = source_dir / "imagesTr" / f"{key}_0000.nii.gz"
        target_path = source_dir / "labelsTr" / f"{key}.nii.gz"
        if not image_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"Dataset124 source pair is incomplete for {key}.")
    added_inputs = _preflight_added_inputs(
        batch_root,
        added,
        args.lung_root,
        intensity_overrides,
        margin_voxels=margin,
        superior_margin_voxels=superior_margin,
    )
    intensity_contract = {
        _case_key(case_id): dict(intensity_overrides[case_id])
        for case_id in sorted(intensity_overrides)
    }

    output_dir = raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    _require_fresh_dataset(output_dir)
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    output_provenance: dict[str, str] = {}
    case_records: dict[str, dict] = {}
    for key in sorted(source_provenance):
        image_path = source_dir / "imagesTr" / f"{key}_0000.nii.gz"
        target_path = source_dir / "labelsTr" / f"{key}.nii.gz"
        _place(image_path, images_dir / image_path.name, args.reuse_mode)
        _place(target_path, labels_dir / target_path.name, args.reuse_mode)
        output_provenance[key] = source_provenance[key]
        case_records[key] = {**source_cases[key], "reused_from": str(source_dir)}

    # Leak guard: only CT and lung-mask paths are constructed for the 150 new
    # cases. Their available labelsTr files are intentionally never resolved.
    for case_id in sorted(added):
        key = _case_key(case_id)
        ct_path = added_inputs[case_id]["ct_path"]
        lung_path = added_inputs[case_id]["lung_path"]
        intensity_transform = intensity_overrides.get(case_id)
        roi_record = write_lung_roi_ct(
            ct_path,
            lung_path,
            images_dir / f"{key}_0000.nii.gz",
            margin_voxels=margin,
            superior_margin_voxels=superior_margin,
            intensity_scale=(
                intensity_transform["scale"] if intensity_transform is not None else None
            ),
            intensity_offset=(
                intensity_transform["offset"] if intensity_transform is not None else 0.0
            ),
        )
        if intensity_transform is not None:
            roi_record["intensity_transform"] = {
                **intensity_transform,
                **roi_record["intensity_transform"],
            }
        if roi_record["bbox"] != added_inputs[case_id]["bbox"]:
            raise RuntimeError(f"ATM_{case_id} lung ROI changed after successful preflight.")
        ignore_record = write_ignore_target(
            ct_path,
            labels_dir / f"{key}.nii.gz",
            IGNORE_INDEX,
        )
        output_provenance[key] = "ignore"
        case_records[key] = {
            **roi_record,
            **ignore_record,
            "provenance": "ignore",
            "source_batch": "TrainBatch2.rar",
        }
        transform_note = " [uint16 -> HU]" if intensity_transform is not None else ""
        print(
            f"ADD    {key}: ROI {roi_record['roi_shape']} "
            f"({roi_record['roi_fraction']:.1%}){transform_note}",
            flush=True,
        )

    labelled_train = labelled - fold0_val
    fold0 = {
        "train": sorted(_normalised_keys(labelled_train | unlabelled)),
        "val": sorted(_normalised_keys(fold0_val)),
    }
    contract = {
        "version": 3,
        "ignore_index": IGNORE_INDEX,
        "case_provenance": output_provenance,
        "folds": {"0": fold0},
        "supervised_loss_scope": "gt_only",
        "consistency_loss_scope": "unlabelled_only",
        "source_dataset": source_dir.name,
        "added_unlabelled": sorted(_normalised_keys(added)),
        "ct_intensity_overrides": intensity_contract,
    }
    dataset_json = nnunet_dataset_json(
        EXPECTED_GT + EXPECTED_EXPANDED_UNLABELLED,
        labels=MT_LABELS,
    )
    dataset_json["lung_roi"] = crop_metadata
    dataset_json["semi_supervised"] = contract
    _write_json(output_dir / "dataset.json", dataset_json)
    _write_json(output_dir / "splits_final.json", [fold0])
    _write_json(
        output_dir / "label_provenance.json",
        {
            "method": "expanded_mean_teacher_from_dataset124_plus_batch2",
            "ignore_index": IGNORE_INDEX,
            "num_gt": EXPECTED_GT,
            "num_ignore": EXPECTED_EXPANDED_UNLABELLED,
            "source_dataset": source_dir.name,
            "added_unlabelled": sorted(_normalised_keys(added)),
            "labels": output_provenance,
        },
    )
    _write_json(
        output_dir / "lung_crop_manifest.json",
        {
            "dataset_role": "expanded_mean_teacher",
            "source_dataset": source_dir.name,
            "lung_roi": crop_metadata,
            "folds": {"0": fold0},
            "excluded_external_val": sorted(_normalised_keys(external_val)),
            "excluded_sealed_test": sorted(_normalised_keys(sealed_test)),
            "withheld_gt_read": False,
            "ct_intensity_overrides": intensity_contract,
            "cases": case_records,
        },
    )

    print(
        f"\nBuilt {output_dir}: 20 GT + 240 ignore; "
        "fold 0 = 16 GT + 240 unlabelled train / 4 GT val."
    )
    print("External VAL20 and sealed TEST20 are excluded from Dataset126.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/atm22.yaml"))
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/nnunet/atm22_split_l20_u240.yaml"),
    )
    parser.add_argument("--nnunet-raw", type=Path, default=os.environ.get("nnUNet_raw"))
    parser.add_argument("--source-dataset-dir", type=Path, default=None)
    parser.add_argument("--lung-root", type=Path, default=None)
    parser.add_argument("--dataset-id", type=int, default=126)
    parser.add_argument("--dataset-name", default="ATM22MT240LungCrop")
    parser.add_argument("--reuse-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    if args.nnunet_raw is None:
        raise SystemExit("Set --nnunet-raw or export nnUNet_raw.")
    assemble(args)


if __name__ == "__main__":
    main()
