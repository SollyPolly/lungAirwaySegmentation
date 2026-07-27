"""Build the matched lung-crop offline pseudo-label nnU-Net dataset.

This is the offline counterpart to Dataset124's corrected two-stream
Mean-Teacher experiment:

* images are reused byte-for-byte from ``Dataset124_ATM22MTLungCrop``;
* the same 16 real-GT cases and all 90 label-withheld cases are used for train;
* the same four real-GT cases are used for fold-0 validation;
* the 90 training targets come only from Dataset123 seed predictions.

The default keeps the seed's native argmax masks. That is the cleanest
comparison with online MT, whose teacher targets are not LCC-cleaned. An
explicit ``lcc6`` option is retained for a separately named historical-policy
variant, but must not silently replace the raw-target primary experiment.

Stage 1 emits the exact 90 cropped CTs for Dataset123 prediction. Stage 2
assembles ``Dataset125_ATM22SSLLungCrop`` after those predictions exist.
Withheld on-disk GT is never resolved or read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.config import load_yaml_config, resolve_project_path
from lung_airway_segmentation.datasets.splits import create_semisupervised_split
from lung_airway_segmentation.inference.postprocess import keep_component_containing_trachea
from lung_airway_segmentation.io.atm22_layout import list_case_ids, resolve_lung_mask_path
from lung_airway_segmentation.io.nnunet_export import _place, nnunet_dataset_json
from lung_airway_segmentation.io.nnunet_lungcrop import (
    assert_same_nifti_grid,
    bbox_from_json,
    write_lung_roi_ct,
)

EXPECTED_GT = 20
EXPECTED_PSEUDO = 90
EXPECTED_TRAIN_GT = 16
EXPECTED_VAL_GT = 4
EXPECTED_VAL_CASES = {"ATM_008", "ATM_050", "ATM_135", "ATM_158"}


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_key(value: str) -> str:
    name = Path(str(value)).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    suffix = name[4:] if name.upper().startswith("ATM_") else name
    if not suffix.isdigit():
        raise ValueError(f"Invalid ATM case identifier: {value!r}")
    return f"ATM_{int(suffix):03d}"


def _require_fresh_dataset(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to merge into non-empty dataset directory: {path}. "
            "Use a fresh dataset ID/name or move the incomplete directory aside."
        )


def _source_contract(source_dir: Path) -> tuple[dict, dict[str, str], dict]:
    dataset_json = _read_json(source_dir / "dataset.json")
    contract = dataset_json.get("semi_supervised")
    if not isinstance(contract, dict):
        raise ValueError(f"{source_dir} has no Dataset124 semi_supervised contract.")

    raw_provenance = contract.get("case_provenance")
    if not isinstance(raw_provenance, dict):
        raise ValueError("Dataset124 semi_supervised contract has no case_provenance mapping.")
    provenance: dict[str, str] = {}
    for raw_key, raw_value in raw_provenance.items():
        key = _normalise_key(raw_key)
        value = str(raw_value).lower()
        if value not in {"gt", "ignore"}:
            raise ValueError(f"Unexpected Dataset124 provenance for {key}: {raw_value!r}")
        provenance[key] = value

    folds = contract.get("folds")
    fold0 = folds.get("0") if isinstance(folds, dict) else None
    if not isinstance(fold0, dict):
        raise ValueError("Dataset124 contract has no pinned fold 0.")
    fold0 = {
        "train": [_normalise_key(key) for key in fold0.get("train", [])],
        "val": [_normalise_key(key) for key in fold0.get("val", [])],
    }

    gt = {key for key, value in provenance.items() if value == "gt"}
    withheld = {key for key, value in provenance.items() if value == "ignore"}
    train = set(fold0["train"])
    val = set(fold0["val"])
    if len(gt) != EXPECTED_GT or len(withheld) != EXPECTED_PSEUDO:
        raise ValueError(
            f"Expected Dataset124 provenance to contain {EXPECTED_GT} GT and "
            f"{EXPECTED_PSEUDO} ignore cases; got {len(gt)} and {len(withheld)}."
        )
    if val != EXPECTED_VAL_CASES or not val.issubset(gt):
        raise ValueError(
            f"Fold-0 validation must be the pinned real-GT cases "
            f"{sorted(EXPECTED_VAL_CASES)}, got {sorted(val)}."
        )
    if len(fold0["train"]) != len(train) or len(fold0["val"]) != len(val):
        raise ValueError("Dataset124 fold 0 contains duplicate case identifiers.")
    if train != (gt - val) | withheld:
        raise ValueError(
            "Dataset124 fold 0 is not the expected 16-GT + 90-unlabelled training pool."
        )
    if len(train & gt) != EXPECTED_TRAIN_GT:
        raise ValueError(f"Expected {EXPECTED_TRAIN_GT} GT train cases.")
    return dataset_json, provenance, fold0


def _source_image(source_dir: Path, case_key: str) -> Path:
    path = source_dir / "imagesTr" / f"{case_key}_0000.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset124 image is missing: {path}")
    return path


def emit_predict_input(source_dir: Path, out_dir: Path, mode: str) -> list[str]:
    """Place the exact Dataset124 unlabelled images into a prediction folder."""
    _, provenance, _ = _source_contract(source_dir)
    pseudo_cases = sorted(key for key, value in provenance.items() if value == "ignore")
    for key in pseudo_cases:
        _place(_source_image(source_dir, key), out_dir / f"{key}_0000.nii.gz", mode)
    print(
        f"Placed {len(pseudo_cases)} exact Dataset124 lung-crop CTs -> {out_dir} "
        f"(mode={mode})."
    )
    return pseudo_cases


def emit_predict_input_from_atm(
    data_config_path: Path,
    training_config_path: Path,
    out_dir: Path,
    *,
    lung_root: Path | None = None,
    margin_voxels: int = 8,
    superior_margin_voxels: int = 120,
) -> list[str]:
    """Rebuild the same 90 lung-crop inputs directly from local ATM data.

    This is the local-prediction fallback when Dataset124 raw files have not
    been downloaded. It uses the same split and ``write_lung_roi_ct`` function
    as the Dataset123/124 builder and never resolves or reads withheld GT.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Prediction input directory is not empty: {out_dir}. "
            "Use a fresh directory to prevent stale-case mixing."
        )
    data_config = load_yaml_config(data_config_path)
    training_config = load_yaml_config(training_config_path)
    batch_root = resolve_project_path(data_config["batch_root"])
    counts = training_config["labelled_split"]
    split = create_semisupervised_split(
        list_case_ids(batch_root),
        test_count=int(counts["test_count"]),
        val_count=int(counts["val_count"]),
        labelled_count=int(counts["labelled_count"]),
        seed=int(training_config.get("seed", 15)),
    )
    unlabelled = sorted(_normalise_key(case_id) for case_id in split["unlabelled_train"])
    if len(unlabelled) != EXPECTED_PSEUDO:
        raise ValueError(f"Expected {EXPECTED_PSEUDO} unlabelled cases, got {len(unlabelled)}.")

    for key in unlabelled:
        case_id = key[4:]
        image_path = batch_root / "imagesTr" / f"{key}_0000.nii.gz"
        if not image_path.is_file():
            raise FileNotFoundError(f"ATM CT is missing: {image_path}")
        lung_path = resolve_lung_mask_path(
            case_id,
            batch_root=batch_root,
            lung_root=lung_root,
        )
        if not lung_path.is_file():
            raise FileNotFoundError(f"Precomputed lung mask is missing: {lung_path}")
        record = write_lung_roi_ct(
            image_path,
            lung_path,
            out_dir / f"{key}_0000.nii.gz",
            margin_voxels=margin_voxels,
            superior_margin_voxels=superior_margin_voxels,
        )
        print(
            f"INPUT  {key}: ROI {record['roi_shape']} ({record['roi_fraction']:.1%})",
            flush=True,
        )
    print(f"Built {len(unlabelled)} local m{margin_voxels}/s{superior_margin_voxels} inputs -> {out_dir}")
    return unlabelled


def _save_mask_like(
    mask: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    destination: Path,
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(mask.astype(np.uint8, copy=False), reference.affine, header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qcode))
    if sform is not None:
        output.set_sform(sform, int(scode))
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(destination))


def _validated_pseudo_mask(
    pseudo_path: Path,
    image_path: Path,
    roi_record: dict,
    *,
    postprocessing: str,
) -> tuple[np.ndarray, dict]:
    if not pseudo_path.is_file():
        raise FileNotFoundError(f"Missing seed pseudo-label: {pseudo_path}")
    image = nib.load(str(image_path))
    pseudo_image = nib.load(str(pseudo_path))
    assert_same_nifti_grid(
        image,
        pseudo_image,
        reference_name="Dataset124 cropped CT",
        candidate_name="Dataset123 pseudo-label",
    )
    values = np.asanyarray(pseudo_image.dataobj)
    unique = np.unique(values)
    if not np.isfinite(unique).all() or not np.isin(unique, (0, 1)).all():
        raise ValueError(f"Pseudo-label must be binary 0/1: {pseudo_path}; values={unique.tolist()}")
    mask = values > 0

    bounds = bbox_from_json(roi_record["bbox"])
    inside = np.zeros(mask.shape, dtype=bool)
    inside[bounds] = True
    outside_foreground = int(np.count_nonzero(mask & ~inside))
    if outside_foreground:
        raise ValueError(
            f"Pseudo-label {pseudo_path} has {outside_foreground} foreground voxels outside "
            "the Dataset124 lung ROI. Check that prediction used the emitted cropped inputs."
        )

    before = int(mask.sum())
    if postprocessing == "lcc6":
        mask = np.asarray(
            keep_component_containing_trachea(mask, affine=pseudo_image.affine, connectivity=6) > 0,
            dtype=bool,
        )
    after = int(mask.sum())
    return mask.astype(np.uint8), {
        "source": str(pseudo_path),
        "foreground_voxels_raw": before,
        "foreground_voxels_written": after,
        "retained_fraction": float(after / max(before, 1)),
        "postprocessing": postprocessing,
        "outside_roi_foreground": outside_foreground,
    }


def assemble(args) -> Path:
    source_dir = Path(args.source_dataset_dir)
    source_dataset_json, source_provenance, fold0 = _source_contract(source_dir)
    source_manifest = _read_json(source_dir / "lung_crop_manifest.json")
    source_cases = source_manifest.get("cases")
    if not isinstance(source_cases, dict):
        raise ValueError("Dataset124 lung_crop_manifest.json has no cases mapping.")

    seed_checkpoint = Path(args.seed_checkpoint)
    if not seed_checkpoint.is_file():
        raise FileNotFoundError(f"Dataset123 seed checkpoint is missing: {seed_checkpoint}")
    seed_checkpoint_sha256 = _sha256(seed_checkpoint)
    pseudo_dir = Path(args.pseudo_dir)
    if not pseudo_dir.is_dir():
        raise FileNotFoundError(f"Pseudo-label directory is missing: {pseudo_dir}")

    output_dir = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    _require_fresh_dataset(output_dir)
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    output_provenance: dict[str, str] = {}
    case_records: dict[str, dict] = {}
    for key in sorted(source_provenance):
        image_path = _source_image(source_dir, key)
        _place(image_path, images_dir / f"{key}_0000.nii.gz", args.reuse_mode)
        roi_record = source_cases.get(key)
        if not isinstance(roi_record, dict) or "bbox" not in roi_record:
            raise ValueError(f"Dataset124 crop manifest has no ROI bounds for {key}.")

        if source_provenance[key] == "gt":
            gt_path = source_dir / "labelsTr" / f"{key}.nii.gz"
            if not gt_path.is_file():
                raise FileNotFoundError(f"Dataset124 real-GT target is missing: {gt_path}")
            _place(gt_path, labels_dir / gt_path.name, args.reuse_mode)
            output_provenance[key] = "gt"
            case_records[key] = {"provenance": "gt", "source": str(gt_path)}
            continue

        # Never resolve or read Dataset124's all-ignore label (and never consult
        # the original ATM withheld-GT directory). The only target source here
        # is the explicitly supplied Dataset123 prediction directory.
        pseudo_path = pseudo_dir / f"{key}.nii.gz"
        mask, pseudo_record = _validated_pseudo_mask(
            pseudo_path,
            image_path,
            roi_record,
            postprocessing=args.pseudo_postprocessing,
        )
        reference = nib.load(str(image_path))
        _save_mask_like(mask, reference, labels_dir / f"{key}.nii.gz")
        output_provenance[key] = "pseudo"
        case_records[key] = {"provenance": "pseudo", **pseudo_record}
        print(
            f"PSEUDO {key}: {pseudo_record['foreground_voxels_raw']:,} -> "
            f"{pseudo_record['foreground_voxels_written']:,} vox "
            f"({args.pseudo_postprocessing})",
            flush=True,
        )

    output_fold = {
        "train": list(fold0["train"]),
        "val": list(fold0["val"]),
    }
    contract = {
        "version": 1,
        "case_provenance": output_provenance,
        "folds": {"0": output_fold},
        "batch_sampling": "one_gt_one_pseudo",
        "validation_scope": "gt_only",
        "gt_loss": "nnunet_dice_plus_ce",
        "pseudo_loss": "nnunet_dice_plus_ce_equal_stream_weight",
        "pseudo_source": {
            "dataset": "Dataset123_ATM22L20LungCrop",
            "trainer": args.seed_trainer,
            "fold": 0,
            "checkpoint": str(seed_checkpoint),
            "checkpoint_sha256": seed_checkpoint_sha256,
            "prediction_dir": str(pseudo_dir),
            "tta": False,
            "target_postprocessing": args.pseudo_postprocessing,
        },
    }
    output_dataset_json = nnunet_dataset_json(len(output_provenance))
    output_dataset_json["lung_roi"] = source_dataset_json.get("lung_roi", source_manifest.get("lung_roi"))
    output_dataset_json["offline_self_training"] = contract
    _write_json(output_dir / "dataset.json", output_dataset_json)
    _write_json(output_dir / "splits_final.json", [output_fold])
    _write_json(
        output_dir / "label_provenance.json",
        {
            "method": "fixed_seed_pseudo_labels_plus_explicit_two_stream",
            "num_gt": sum(value == "gt" for value in output_provenance.values()),
            "num_pseudo": sum(value == "pseudo" for value in output_provenance.values()),
            "labels": output_provenance,
            **contract["pseudo_source"],
        },
    )
    _write_json(
        output_dir / "lung_crop_manifest.json",
        {
            "dataset_role": "offline_pseudo_label",
            "source_dataset": source_dir.name,
            "lung_roi": output_dataset_json["lung_roi"],
            "folds": {"0": output_fold},
            "excluded_external_val": source_manifest.get("excluded_external_val"),
            "excluded_sealed_test": source_manifest.get("excluded_sealed_test"),
            "withheld_gt_read": False,
            "cases": case_records,
        },
    )

    n_gt = sum(value == "gt" for value in output_provenance.values())
    n_pseudo = sum(value == "pseudo" for value in output_provenance.values())
    print(
        f"\nBuilt {output_dir}: {n_gt} GT + {n_pseudo} fixed pseudo-labels; "
        f"fold 0 = {EXPECTED_TRAIN_GT} GT + {EXPECTED_PSEUDO} pseudo train / "
        f"{EXPECTED_VAL_GT} GT val."
    )
    print("Copy splits_final.json into the preprocessed Dataset125 directory after preprocessing.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument(
        "--emit-predict-input",
        type=Path,
        help="Stage 1: place the 90 exact Dataset124 cropped CTs here and exit.",
    )
    stage.add_argument(
        "--emit-predict-input-from-atm",
        type=Path,
        help="Local fallback: rebuild the same 90 crops from configured ATM CT/lung files.",
    )
    stage.add_argument("--assemble", action="store_true", help="Stage 2: assemble Dataset125.")
    parser.add_argument("--nnunet-raw", type=Path, default=os.environ.get("nnUNet_raw"))
    parser.add_argument("--source-dataset-dir", type=Path, default=None)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/atm22.yaml"))
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/nnunet/atm22_split_l20.yaml"),
    )
    parser.add_argument("--lung-root", type=Path, default=None)
    parser.add_argument("--margin-voxels", type=int, default=8)
    parser.add_argument("--superior-margin-voxels", type=int, default=120)
    parser.add_argument("--pseudo-dir", type=Path)
    parser.add_argument("--seed-checkpoint", type=Path)
    parser.add_argument(
        "--seed-trainer",
        default="nnUNetTrainer_NoDeepSupervision_NoMirroring",
    )
    parser.add_argument("--dataset-id", type=int, default=125)
    parser.add_argument("--dataset-name", default="ATM22SSLLungCrop")
    parser.add_argument(
        "--pseudo-postprocessing",
        choices=("raw", "lcc6"),
        default="raw",
        help="Primary matched arm uses raw. Use lcc6 only as an explicitly named variant.",
    )
    parser.add_argument("--reuse-mode", choices=("symlink", "hardlink", "copy"), default="hardlink")
    args = parser.parse_args()

    if args.emit_predict_input_from_atm is not None:
        emit_predict_input_from_atm(
            args.data_config,
            args.training_config,
            args.emit_predict_input_from_atm,
            lung_root=args.lung_root,
            margin_voxels=args.margin_voxels,
            superior_margin_voxels=args.superior_margin_voxels,
        )
        return

    if args.nnunet_raw is None:
        raise SystemExit("Set --nnunet-raw or export nnUNet_raw.")
    if args.source_dataset_dir is None:
        args.source_dataset_dir = Path(args.nnunet_raw) / "Dataset124_ATM22MTLungCrop"

    if args.emit_predict_input is not None:
        emit_predict_input(Path(args.source_dataset_dir), args.emit_predict_input, args.reuse_mode)
        return
    if args.pseudo_dir is None or args.seed_checkpoint is None:
        raise SystemExit("--assemble requires both --pseudo-dir and --seed-checkpoint.")
    assemble(args)


if __name__ == "__main__":
    main()
