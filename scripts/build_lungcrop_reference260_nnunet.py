"""Build the unconstrained best-effort supervised reference on 260 labelled cases.

``Dataset132_ATM22REF260LungCrop`` answers a different question from
``Dataset127_ATM22SUP260LungCrop``. Dataset127 is a *matched* ceiling: it is
handicapped exactly as MT240 is -- imported Dataset123 plans, the
``NoDeepSupervision_NoMirroring`` trainer, and folds that rotate only the 20
real-GT cases -- so ``SUP260 fold k`` minus ``MT240 fold k`` isolates the label
effect and nothing else.

Dataset132 drops every one of those handicaps. It answers "what would a
competent practitioner get if they were handed these lung-ROI inputs and all the
labels, and simply ran nnU-Net?":

* nnU-Net plans this dataset itself, rather than inheriting Dataset123's plans;
* the stock ``nnUNetTrainer`` trains it, with deep supervision and mirroring on;
* the split is an ordinary five-fold cross-validation over all 260 cases
  (208 train / 52 val), not a rotation of a 20-case labelled pool.

Because four things differ at once, this arm is **not** subtractable from MT240
fold-for-fold. It is a reference level, reported on the external VAL20 and the
sealed TEST20, and it must be described as such.

Images and labels are hardlinked from Dataset127, so the voxels are identical
and the expensive ROI gating is not repeated. Dataset127's own membership
already excludes the external validation and sealed test cases; that exclusion
is re-checked here rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lung_airway_segmentation.config import load_yaml_config
from lung_airway_segmentation.io.nnunet_export import _place, nnunet_dataset_json
from scripts.build_lungcrop_meanteacher_nnunet import (
    _case_key,
    _require_fresh_dataset,
    _write_json,
)

SOURCE_DATASET = "Dataset127_ATM22SUP260LungCrop"
EXPECTED_CASES = 260
EXPECTED_FOLDS = 5
EXPECTED_TRAIN = 208
EXPECTED_VAL = 52


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _source_contract(source_dir: Path) -> tuple[dict, list[str]]:
    """Read Dataset127's ceiling contract and return its 260 case keys."""
    dataset_json = _read_json(source_dir / "dataset.json")
    contract = dataset_json.get("supervised_ceiling")
    if not isinstance(contract, dict):
        raise ValueError(
            f"{source_dir} has no dataset.json supervised_ceiling contract; this builder "
            "must derive from the oracle build, not from an arbitrary dataset."
        )
    provenance = contract.get("case_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{source_dir} supervised_ceiling contract has no case_provenance.")
    keys = sorted(_case_key(key) for key in provenance)
    if len(keys) != EXPECTED_CASES:
        raise ValueError(f"{source_dir} holds {len(keys)} cases, expected {EXPECTED_CASES}.")
    unexpected = set(provenance.values()) - {"gt", "oracle_gt"}
    if unexpected:
        raise ValueError(
            f"{source_dir} provenance must be entirely real annotation ('gt' or 'oracle_gt'), "
            f"found {sorted(unexpected)}."
        )
    return dataset_json, keys


def _crossval_splits(keys: list[str]) -> list[dict]:
    """nnU-Net's own five-fold split, generated here so concurrent folds cannot race.

    Five array subjobs starting together would otherwise each find no
    ``splits_final.json`` and race to write one. Calling nnU-Net's generator
    rather than reimplementing it keeps the split identical to what a plain
    ``nnUNetv2_train`` would have produced for itself.
    """
    from nnunetv2.utilities.crossval_split import generate_crossval_split

    splits = generate_crossval_split(keys, seed=12345, n_splits=EXPECTED_FOLDS)
    if len(splits) != EXPECTED_FOLDS:
        raise ValueError(f"Expected {EXPECTED_FOLDS} folds, got {len(splits)}.")
    resolved = []
    for index, fold in enumerate(splits):
        train, val = list(fold["train"]), list(fold["val"])
        if len(train) != EXPECTED_TRAIN or len(val) != EXPECTED_VAL:
            raise ValueError(
                f"Fold {index} must be {EXPECTED_TRAIN} train / {EXPECTED_VAL} val, "
                f"got {len(train)}/{len(val)}."
            )
        if set(train) & set(val):
            raise ValueError(f"Fold {index} train and val overlap.")
        resolved.append({"train": train, "val": val})
    return resolved


def assemble(args) -> Path:
    raw_root = Path(args.nnunet_raw)
    source_dir = raw_root / SOURCE_DATASET
    source_json, keys = _source_contract(source_dir)

    split_config = load_yaml_config(args.split_config)
    external_val = {_case_key(case) for case in split_config["splits"]["val"]}
    sealed_test = {_case_key(case) for case in split_config["splits"]["test"]}
    leaked = sorted(set(keys) & (external_val | sealed_test))
    if leaked:
        raise ValueError(
            f"{SOURCE_DATASET} membership overlaps the external validation or sealed "
            f"test: {leaked}"
        )

    crop_metadata = source_json.get("lung_roi")
    if not isinstance(crop_metadata, dict):
        raise ValueError(f"{source_dir} has no lung_roi metadata.")

    output_dir = raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    _require_fresh_dataset(output_dir)
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        image_path = source_dir / "imagesTr" / f"{key}_0000.nii.gz"
        label_path = source_dir / "labelsTr" / f"{key}.nii.gz"
        for path in (image_path, label_path):
            if not path.is_file():
                raise FileNotFoundError(f"{source_dir} is missing {path.name} for {key}.")
        _place(image_path, images_dir / image_path.name, args.reuse_mode)
        _place(label_path, labels_dir / label_path.name, args.reuse_mode)

    splits = _crossval_splits(keys)

    contract = {
        "version": 1,
        "role": "unconstrained_best_effort_supervised_reference",
        "derived_from": SOURCE_DATASET,
        "num_labelled": len(keys),
        "images_and_labels": f"{args.reuse_mode} of {SOURCE_DATASET}; identical voxels",
        "plans": "planned on this dataset by nnUNetv2_plan_and_preprocess, NOT imported",
        "splits": (
            f"nnU-Net generate_crossval_split(seed=12345, n_splits={EXPECTED_FOLDS}); "
            f"{EXPECTED_TRAIN} train / {EXPECTED_VAL} val; NOT paired with Dataset126"
        ),
        "trainer": "stock nnUNetTrainer, deep supervision and mirroring enabled",
        "comparable_to_mt240": False,
        "excluded_external_val": sorted(external_val),
        "excluded_sealed_test": sorted(sealed_test),
    }

    dataset_json = nnunet_dataset_json(len(keys))
    dataset_json["lung_roi"] = crop_metadata
    dataset_json["reference_arm"] = contract
    _write_json(output_dir / "dataset.json", dataset_json)
    _write_json(output_dir / "splits_final.json", splits)
    _write_json(
        output_dir / "label_provenance.json",
        {
            "method": "real_annotation_for_every_case",
            "num_labelled": len(keys),
            "source_dataset": SOURCE_DATASET,
            "labels": {key: "real_gt" for key in keys},
        },
    )

    print(
        f"\nBuilt {output_dir}: {len(keys)} labelled lung-ROI cases, five ordinary folds of "
        f"{EXPECTED_TRAIN} train / {EXPECTED_VAL} val."
    )
    print("External VAL20 and sealed TEST20 are excluded.")
    print("This is a reference level, not a paired ceiling; it is not subtractable from MT240.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/nnunet/atm22_split_l20_u240.yaml"),
    )
    parser.add_argument("--nnunet-raw", type=Path, default=os.environ.get("nnUNet_raw"))
    parser.add_argument("--dataset-id", type=int, default=132)
    parser.add_argument("--dataset-name", default="ATM22REF260LungCrop")
    parser.add_argument("--reuse-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    if args.nnunet_raw is None:
        raise SystemExit("Set --nnunet-raw or export nnUNet_raw.")
    assemble(args)


if __name__ == "__main__":
    main()
