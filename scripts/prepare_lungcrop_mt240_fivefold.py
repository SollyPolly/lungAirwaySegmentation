"""Install leakage-safe five-fold splits into existing Dataset123/126 metadata.

This does not rebuild or rewrite any NIfTI/preprocessed arrays. It upgrades the
raw and preprocessed ``splits_final.json`` files so that:

* the same frozen 20 real-GT cases form five disjoint four-case validation sets;
* each Dataset123 seed fold trains on the complementary 16 GT cases;
* each Dataset126 MT fold trains on those same 16 GT plus all 240 unlabelled
  cases;
* no ignore-labelled case can enter validation;
* historical fold 0 remains byte-for-membership compatible.

Run this once on HPC before launching folds 1--4. The operation is idempotent
and refuses any pre-existing split that does not match historical fold 0.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from lung_airway_segmentation.config import load_yaml_config
from scripts.build_lungcrop_meanteacher_nnunet import _case_key, _padded


SEED_DATASET = "Dataset123_ATM22L20LungCrop"
MT_DATASET = "Dataset126_ATM22MT240LungCrop"
EXPECTED_LABELLED = 20
EXPECTED_UNLABELLED = 240
EXPECTED_FOLDS = 5
EXPECTED_VAL_PER_FOLD = 4


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _normalised_keys(values) -> set[str]:
    return {_case_key(value) for value in values}


def build_fivefold_splits(split_config: Path) -> tuple[list[dict], list[dict]]:
    config = load_yaml_config(split_config)
    labelled = _normalised_keys(config["splits"]["labelled_train"])
    unlabelled = _normalised_keys(config["splits"]["unlabelled_train"])
    raw_val_folds = config.get("internal_fivefold_val")
    if len(labelled) != EXPECTED_LABELLED or len(unlabelled) != EXPECTED_UNLABELLED:
        raise ValueError(
            "Five-fold MT240 expects exactly 20 labelled and 240 unlabelled cases; "
            f"got {len(labelled)} and {len(unlabelled)}."
        )
    if not isinstance(raw_val_folds, list) or len(raw_val_folds) != EXPECTED_FOLDS:
        raise ValueError("internal_fivefold_val must contain exactly five folds.")

    val_folds = [_normalised_keys(fold) for fold in raw_val_folds]
    if any(len(fold) != EXPECTED_VAL_PER_FOLD for fold in val_folds):
        raise ValueError("Every internal validation fold must contain exactly four cases.")
    flattened = [case for fold in val_folds for case in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != labelled:
        raise ValueError(
            "The five validation folds must be disjoint and cover the labelled-20 exactly."
        )
    legacy_fold0 = _normalised_keys(config.get("internal_fold0_val", []))
    if val_folds[0] != legacy_fold0:
        raise ValueError("Five-fold fold 0 differs from the frozen historical fold 0.")
    if labelled & unlabelled:
        raise ValueError("Labelled and unlabelled memberships overlap.")

    seed_folds: list[dict] = []
    mt_folds: list[dict] = []
    for val in val_folds:
        seed_train = labelled - val
        seed_folds.append({"train": sorted(seed_train), "val": sorted(val)})
        mt_folds.append({"train": sorted(seed_train | unlabelled), "val": sorted(val)})
    return seed_folds, mt_folds


def _assert_existing_fold0(path: Path, expected_fold0: dict) -> None:
    existing = _read_json(path)
    if not isinstance(existing, list) or not existing:
        raise ValueError(f"{path} must contain at least fold 0.")
    for field in ("train", "val"):
        if set(existing[0].get(field, [])) != set(expected_fold0[field]):
            raise ValueError(
                f"{path} historical fold 0 {field} membership differs from the frozen contract."
            )


def _upgrade_dataset_directory(
    dataset_dir: Path,
    folds: list[dict],
    *,
    require_contract: bool,
) -> None:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    split_path = dataset_dir / "splits_final.json"
    _assert_existing_fold0(split_path, folds[0])

    dataset_path = dataset_dir / "dataset.json"
    dataset_json = _read_json(dataset_path)
    if require_contract:
        contract = dataset_json.get("semi_supervised")
        if not isinstance(contract, dict):
            raise ValueError(f"{dataset_path} has no semi_supervised contract.")
        provenance = {
            _case_key(key): str(value).lower()
            for key, value in contract.get("case_provenance", {}).items()
        }
        if list(provenance.values()).count("gt") != EXPECTED_LABELLED:
            raise ValueError(f"{dataset_path} does not contain exactly 20 GT cases.")
        if list(provenance.values()).count("ignore") != EXPECTED_UNLABELLED:
            raise ValueError(f"{dataset_path} does not contain exactly 240 ignore cases.")
        for fold in folds:
            if any(provenance.get(key) != "gt" for key in fold["val"]):
                raise ValueError("An MT validation fold contains a non-GT case.")
            train_gt = sum(provenance.get(key) == "gt" for key in fold["train"])
            train_ignore = sum(provenance.get(key) == "ignore" for key in fold["train"])
            if (train_gt, train_ignore) != (16, EXPECTED_UNLABELLED):
                raise ValueError(
                    "An MT training fold is not exactly 16 GT + 240 unlabelled cases."
                )
        contract["folds"] = {str(index): fold for index, fold in enumerate(folds)}
        dataset_json["semi_supervised"] = contract
        _write_json_atomic(dataset_path, dataset_json)

    manifest_path = dataset_dir / "lung_crop_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        manifest["folds"] = {str(index): fold for index, fold in enumerate(folds)}
        _write_json_atomic(manifest_path, manifest)

    _write_json_atomic(split_path, folds)
    print(f"Updated {dataset_dir}: {len(folds)} folds", flush=True)


def install_fivefold(
    nnunet_raw: Path,
    nnunet_preprocessed: Path,
    split_config: Path,
) -> None:
    seed_folds, mt_folds = build_fivefold_splits(split_config)
    targets = (
        (nnunet_raw / SEED_DATASET, seed_folds, False),
        (nnunet_preprocessed / SEED_DATASET, seed_folds, False),
        (nnunet_raw / MT_DATASET, mt_folds, True),
        (nnunet_preprocessed / MT_DATASET, mt_folds, True),
    )
    for dataset_dir, folds, require_contract in targets:
        _upgrade_dataset_directory(
            dataset_dir,
            folds,
            require_contract=require_contract,
        )

    print("\nFive-fold contract installed:", flush=True)
    for index, fold in enumerate(seed_folds):
        val_ids = ", ".join(_padded(case) for case in fold["val"])
        print(
            f"  fold {index}: Dataset123 16 GT train / val [{val_ids}]; "
            "Dataset126 adds all 240 unlabelled to train",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/nnunet/atm22_split_l20_u240.yaml"),
    )
    parser.add_argument("--nnunet-raw", type=Path, default=os.environ.get("nnUNet_raw"))
    parser.add_argument(
        "--nnunet-preprocessed",
        type=Path,
        default=os.environ.get("nnUNet_preprocessed"),
    )
    args = parser.parse_args()
    if args.nnunet_raw is None or args.nnunet_preprocessed is None:
        raise SystemExit(
            "Set nnUNet_raw and nnUNet_preprocessed, or pass both directory arguments."
        )
    install_fivefold(
        Path(args.nnunet_raw),
        Path(args.nnunet_preprocessed),
        args.split_config,
    )


if __name__ == "__main__":
    main()
