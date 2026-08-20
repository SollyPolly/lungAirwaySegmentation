"""Build the fully supervised 260-case lung-crop dataset that matches MT240.

``Dataset127_ATM22SUP260LungCrop`` is the label ceiling for the Mean-Teacher
experiment: it contains the *same 260 scans* as
``Dataset126_ATM22MT240LungCrop`` and the *same five frozen folds*, but every
one of the 240 provenance-unlabelled cases carries its real ATM'22 airway
annotation instead of an all-ignore target.

* images are reused byte-for-byte from Dataset126, so the only difference
  between the two datasets is the content of ``labelsTr``;
* the 20 real-GT targets are reused byte-for-byte as well;
* the 240 remaining targets are written from ``data/ATM22/labelsTr`` and gated
  with each case's own Dataset126 lung ROI, so target and image agree voxel for
  voxel;
* the five folds are rebuilt with the same code that installed Dataset126's
  folds and are then asserted equal to them, fold for fold.

Each fold therefore trains on 256 labelled cases (the 16 GT that its MT
counterpart trains on, plus all 240) and validates on the same four real-GT
cases. ``SUP260 fold k`` minus ``MT240 fold k`` reads directly as the part of
the label gap that consistency on the 240 does not recover.

Unlike every other builder in this directory this one deliberately *does* read
the withheld ground truth -- that is the whole point of an oracle ceiling. It
is the only script permitted to do so, and it refuses to touch the external
validation or sealed test cases.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.config import load_yaml_config, resolve_project_path
from lung_airway_segmentation.io.nnunet_export import _place, nnunet_dataset_json
from lung_airway_segmentation.io.nnunet_lungcrop import (
    assert_same_nifti_grid,
    bbox_from_json,
    write_roi_ground_truth,
)
from scripts.build_lungcrop_meanteacher_nnunet import (
    _case_key,
    _require_fresh_dataset,
    _write_json,
)
from scripts.prepare_lungcrop_mt240_fivefold import build_fivefold_splits

SOURCE_DATASET = "Dataset126_ATM22MT240LungCrop"
EXPECTED_GT = 20
EXPECTED_ORACLE = 240
EXPECTED_TRAIN = 256
EXPECTED_VAL = 4
EXPECTED_FOLDS = 5


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _source_contract(source_dir: Path) -> tuple[dict, dict[str, str], list[dict], dict]:
    """Read and validate Dataset126's provenance, folds and crop manifest."""
    dataset_json = _read_json(source_dir / "dataset.json")
    contract = dataset_json.get("semi_supervised")
    if not isinstance(contract, dict):
        raise ValueError(f"{source_dir} has no dataset.json semi_supervised contract.")

    raw_provenance = contract.get("case_provenance")
    if not isinstance(raw_provenance, dict):
        raise ValueError(f"{source_dir} semi_supervised contract has no case_provenance.")
    provenance = {_case_key(key): str(value).lower() for key, value in raw_provenance.items()}
    if set(provenance.values()) != {"gt", "ignore"}:
        raise ValueError(f"{source_dir} provenance must contain only 'gt' and 'ignore'.")
    if list(provenance.values()).count("gt") != EXPECTED_GT:
        raise ValueError(f"{source_dir} must contain exactly {EXPECTED_GT} GT cases.")
    if list(provenance.values()).count("ignore") != EXPECTED_ORACLE:
        raise ValueError(f"{source_dir} must contain exactly {EXPECTED_ORACLE} ignore cases.")

    folds = _read_json(source_dir / "splits_final.json")
    if not isinstance(folds, list) or len(folds) != EXPECTED_FOLDS:
        raise ValueError(
            f"{source_dir}/splits_final.json must hold five folds; run "
            "scripts.prepare_lungcrop_mt240_fivefold first."
        )

    manifest = _read_json(source_dir / "lung_crop_manifest.json")
    cases = manifest.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(provenance):
        raise ValueError(
            f"{source_dir} crop manifest does not cover exactly its provenance cases."
        )
    return dataset_json, provenance, folds, manifest


def _assert_folds_match_source(folds: list[dict], source_folds: list[dict]) -> None:
    """The ceiling must be paired with MT240 fold for fold, not merely similar."""
    for index, (fold, source_fold) in enumerate(zip(folds, source_folds)):
        for field in ("train", "val"):
            if set(fold[field]) != set(source_fold[field]):
                raise ValueError(
                    f"Fold {index} {field} membership differs from {SOURCE_DATASET}; "
                    "the supervised ceiling would not be paired with MT240."
                )
        if len(fold["train"]) != EXPECTED_TRAIN or len(fold["val"]) != EXPECTED_VAL:
            raise ValueError(
                f"Fold {index} must be {EXPECTED_TRAIN} train / {EXPECTED_VAL} val, got "
                f"{len(fold['train'])}/{len(fold['val'])}."
            )


def _gt_path(batch_root: Path, key: str) -> Path:
    candidates = (
        batch_root / "labelsTr" / f"{key}.nii.gz",
        batch_root / "labelsTr" / f"{key}_0000.nii.gz",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Oracle case {key} has no airway label under {batch_root}.")
    return path


def preflight_oracle_targets(
    batch_root: Path,
    source_dir: Path,
    source_cases: dict,
    oracle_keys: list[str],
    *,
    max_lost_fraction: float,
) -> dict[str, dict]:
    """Read every withheld label and check it survives its own lung ROI.

    This runs before a single file is written. A label whose airway leaves the
    ROI would be an unlearnable target, since the CT is exactly zero there, so
    the build stops rather than quietly training against voxels the network can
    never see.
    """
    reports: dict[str, dict] = {}
    offenders: list[tuple[str, float, int]] = []
    for key in oracle_keys:
        record = source_cases.get(key)
        if not isinstance(record, dict) or "bbox" not in record:
            raise ValueError(f"{source_dir} crop manifest has no ROI bounds for {key}.")
        image_path = source_dir / "imagesTr" / f"{key}_0000.nii.gz"
        if not image_path.is_file():
            raise FileNotFoundError(f"{source_dir} image is missing for {key}: {image_path}")

        gt_path = _gt_path(batch_root, key)
        gt_image = nib.load(str(gt_path))
        assert_same_nifti_grid(
            nib.load(str(image_path)),
            gt_image,
            reference_name=f"{SOURCE_DATASET} image {key}",
            candidate_name=f"ATM airway GT {key}",
        )
        bounds = bbox_from_json(record["bbox"])
        gt = np.asanyarray(gt_image.dataobj) > 0
        total = int(gt.sum())
        if total == 0:
            raise ValueError(f"Airway GT for {key} is empty: {gt_path}")
        retained = int(gt[bounds].sum())
        lost = total - retained
        fraction = lost / total
        reports[key] = {
            "gt": str(gt_path),
            "foreground_voxels": total,
            "retained_foreground_voxels": retained,
            "lost_foreground_voxels": lost,
            "lost_foreground_fraction": fraction,
        }
        if fraction > max_lost_fraction:
            offenders.append((key, fraction, lost))
        print(
            f"CHECK  {key}: {retained:,}/{total:,} airway voxels inside the lung ROI "
            f"({fraction:.4%} lost)",
            flush=True,
        )

    if offenders:
        offenders.sort(key=lambda item: item[1], reverse=True)
        worst = ", ".join(
            f"{key} {fraction:.3%} ({lost:,} vox)" for key, fraction, lost in offenders[:10]
        )
        raise ValueError(
            f"{len(offenders)} oracle case(s) lose more airway to the lung ROI than the "
            f"{max_lost_fraction:.3%} tolerance; worst: {worst}. The ROI is frozen by "
            "Dataset126 and must not be widened here, because that would change the images "
            "and break the pairing with MT240. Re-run with --max-lost-foreground-fraction "
            "above the worst value once the loss is judged acceptable; the per-case numbers "
            "are then recorded in lung_crop_manifest.json."
        )
    return reports


def assemble(args) -> Path:
    data_config = load_yaml_config(args.data_config)
    batch_root = resolve_project_path(data_config["batch_root"])

    raw_root = Path(args.nnunet_raw)
    source_dir = (
        Path(args.source_dataset_dir)
        if args.source_dataset_dir is not None
        else raw_root / SOURCE_DATASET
    )
    source_json, provenance, source_folds, source_manifest = _source_contract(source_dir)

    _, folds = build_fivefold_splits(args.split_config)
    _assert_folds_match_source(folds, source_folds)

    split_config = load_yaml_config(args.split_config)
    external_val = {_case_key(case) for case in split_config["splits"]["val"]}
    sealed_test = {_case_key(case) for case in split_config["splits"]["test"]}
    if set(provenance) & (external_val | sealed_test):
        raise ValueError("Dataset126 membership overlaps the external validation or sealed test.")

    crop_metadata = source_json.get("lung_roi")
    if not isinstance(crop_metadata, dict):
        raise ValueError(f"{source_dir} has no lung_roi metadata.")

    source_cases = source_manifest["cases"]
    gt_keys = sorted(key for key, value in provenance.items() if value == "gt")
    oracle_keys = sorted(key for key, value in provenance.items() if value == "ignore")

    reports = preflight_oracle_targets(
        batch_root,
        source_dir,
        source_cases,
        oracle_keys,
        max_lost_fraction=args.max_lost_foreground_fraction,
    )

    output_dir = raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    _require_fresh_dataset(output_dir)
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    output_provenance: dict[str, str] = {}
    case_records: dict[str, dict] = {}

    for key in sorted(provenance):
        image_path = source_dir / "imagesTr" / f"{key}_0000.nii.gz"
        if not image_path.is_file():
            raise FileNotFoundError(f"{source_dir} image is missing for {key}: {image_path}")
        _place(image_path, images_dir / image_path.name, args.reuse_mode)

    # The labelled 20 keep the exact targets both the seed and MT240 trained on.
    for key in gt_keys:
        source_target = source_dir / "labelsTr" / f"{key}.nii.gz"
        if not source_target.is_file():
            raise FileNotFoundError(f"{source_dir} real-GT target is missing: {source_target}")
        _place(source_target, labels_dir / source_target.name, args.reuse_mode)
        output_provenance[key] = "gt"
        case_records[key] = {
            **source_cases[key],
            "provenance": "gt",
            "target_source": str(source_target),
            "reused_from": str(source_dir),
        }

    # The 240 exchange their all-ignore target for the real annotation. This is
    # the only place in the repository that resolves withheld ATM'22 GT.
    for key in oracle_keys:
        record = source_cases[key]
        image_path = source_dir / "imagesTr" / f"{key}_0000.nii.gz"
        gt_record = write_roi_ground_truth(
            Path(reports[key]["gt"]),
            image_path,
            bbox_from_json(record["bbox"]),
            labels_dir / f"{key}.nii.gz",
            fail_on_foreground_loss=False,
        )
        if gt_record["retained_foreground_voxels"] != reports[key]["retained_foreground_voxels"]:
            raise RuntimeError(f"{key} airway retention changed after a successful preflight.")
        output_provenance[key] = "oracle_gt"
        case_records[key] = {
            **record,
            **gt_record,
            "lost_foreground_fraction": reports[key]["lost_foreground_fraction"],
            "provenance": "oracle_gt",
            "reused_image_from": str(source_dir),
        }
        print(
            f"ORACLE {key}: {gt_record['retained_foreground_voxels']:,} airway voxels written "
            f"({reports[key]['lost_foreground_fraction']:.4%} lost to the ROI)",
            flush=True,
        )

    total_foreground = sum(reports[key]["foreground_voxels"] for key in oracle_keys)
    total_lost = sum(reports[key]["lost_foreground_voxels"] for key in oracle_keys)
    contract = {
        "version": 1,
        "role": "supervised_label_ceiling_for_mean_teacher",
        "case_provenance": output_provenance,
        "folds": {str(index): fold for index, fold in enumerate(folds)},
        "paired_with": source_dir.name,
        "num_gt": len(gt_keys),
        "num_oracle_gt": len(oracle_keys),
        "oracle_gt_source": str(batch_root / "labelsTr"),
        "oracle_roi_foreground_voxels": total_foreground,
        "oracle_roi_lost_foreground_voxels": total_lost,
        "max_lost_foreground_fraction": float(args.max_lost_foreground_fraction),
    }

    dataset_json = nnunet_dataset_json(len(output_provenance))
    dataset_json["lung_roi"] = crop_metadata
    dataset_json["supervised_ceiling"] = contract
    _write_json(output_dir / "dataset.json", dataset_json)
    _write_json(output_dir / "splits_final.json", folds)
    _write_json(
        output_dir / "label_provenance.json",
        {
            "method": "oracle_labels_for_the_mt240_unlabelled_pool",
            "num_gt": len(gt_keys),
            "num_oracle_gt": len(oracle_keys),
            "source_dataset": source_dir.name,
            "labels": output_provenance,
        },
    )
    _write_json(
        output_dir / "lung_crop_manifest.json",
        {
            "dataset_role": "supervised_ceiling_260",
            "source_dataset": source_dir.name,
            "lung_roi": crop_metadata,
            "folds": {str(index): fold for index, fold in enumerate(folds)},
            "excluded_external_val": sorted(external_val),
            "excluded_sealed_test": sorted(sealed_test),
            "withheld_gt_read": True,
            "cases": case_records,
        },
    )

    print(
        f"\nBuilt {output_dir}: {len(gt_keys)} GT + {len(oracle_keys)} oracle-GT cases; "
        f"each fold = {EXPECTED_TRAIN} labelled train / {EXPECTED_VAL} GT val, paired "
        f"fold-for-fold with {source_dir.name}."
    )
    print(
        f"Oracle ROI retention: {total_foreground - total_lost:,}/{total_foreground:,} airway "
        f"voxels ({total_lost / total_foreground:.4%} lost)."
    )
    print("External VAL20 and sealed TEST20 are excluded.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/atm22.yaml"))
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/nnunet/atm22_split_l20_u240.yaml"),
    )
    parser.add_argument("--nnunet-raw", type=Path, default=os.environ.get("nnUNet_raw"))
    parser.add_argument("--source-dataset-dir", type=Path, default=None)
    parser.add_argument("--dataset-id", type=int, default=127)
    parser.add_argument("--dataset-name", default="ATM22SUP260LungCrop")
    parser.add_argument("--reuse-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--max-lost-foreground-fraction",
        type=float,
        default=0.0,
        help=(
            "Per-case share of airway GT the frozen lung ROI may discard. The default "
            "refuses any loss; raise it only after reading the preflight report."
        ),
    )
    args = parser.parse_args()
    if args.nnunet_raw is None:
        raise SystemExit("Set --nnunet-raw or export nnUNet_raw.")
    if not 0.0 <= args.max_lost_foreground_fraction < 1.0:
        raise SystemExit("--max-lost-foreground-fraction must lie in [0, 1).")
    assemble(args)


if __name__ == "__main__":
    main()
