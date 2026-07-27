import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import yaml

from lung_airway_segmentation.io.nnunet_export import nnunet_dataset_json
from scripts.build_lungcrop_expanded_meanteacher_nnunet import assemble
from scripts.build_lungcrop_meanteacher_nnunet import MT_LABELS, _dataset_metadata


ROOT = Path(__file__).resolve().parents[1]
SPLIT_CONFIG = ROOT / "configs" / "nnunet" / "atm22_split_l20_u240.yaml"


def _save(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _key(case_id: str) -> str:
    return f"ATM_{case_id}"


def test_expanded_builder_reuses_dataset124_and_never_needs_batch2_gt(tmp_path: Path):
    config = yaml.safe_load(SPLIT_CONFIG.read_text(encoding="utf-8"))
    labelled = set(config["splits"]["labelled_train"])
    unlabelled = set(config["splits"]["unlabelled_train"])
    added = set(config["added_unlabelled_case_ids"])
    old_unlabelled = unlabelled - added
    fold_val = set(config["internal_fold0_val"])
    all_ids = set().union(*(set(values) for values in config["splits"].values()))

    batch_root = tmp_path / "ATM22"
    image = np.ones((4, 5, 6), dtype=np.int16)
    lung = np.zeros_like(image, dtype=np.uint8)
    lung[1:3, 1:4, 1:5] = 1
    for case_id in all_ids:
        _save(batch_root / "imagesTr" / f"{_key(case_id)}_0000.nii.gz", image)
    # No labelsTr is created at all. New unlabelled GT therefore cannot leak.
    for case_id in added:
        _save(batch_root / "lungTr" / f"{_key(case_id)}_lung.nii.gz", lung)

    raw_root = tmp_path / "raw"
    source_dir = raw_root / "Dataset124_ATM22MTLungCrop"
    provenance = {
        **{_key(case_id): "gt" for case_id in sorted(labelled)},
        **{_key(case_id): "ignore" for case_id in sorted(old_unlabelled)},
    }
    source_fold = {
        "train": sorted(_key(case_id) for case_id in ((labelled - fold_val) | old_unlabelled)),
        "val": sorted(_key(case_id) for case_id in fold_val),
    }
    crop_metadata = _dataset_metadata(8, 120)
    dataset_json = nnunet_dataset_json(110, labels=MT_LABELS)
    dataset_json["lung_roi"] = crop_metadata
    dataset_json["semi_supervised"] = {
        "version": 1,
        "ignore_index": 2,
        "case_provenance": provenance,
        "folds": {"0": source_fold},
    }
    source_dir.mkdir(parents=True)
    (source_dir / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    (source_dir / "splits_final.json").write_text(
        json.dumps([source_fold]), encoding="utf-8"
    )
    (source_dir / "lung_crop_manifest.json").write_text(
        json.dumps(
            {
                "excluded_external_val": config["splits"]["val"],
                "excluded_sealed_test": config["splits"]["test"],
                "cases": {
                    key: {"provenance": value}
                    for key, value in provenance.items()
                },
            }
        ),
        encoding="utf-8",
    )
    for key, value in provenance.items():
        _save(source_dir / "imagesTr" / f"{key}_0000.nii.gz", image)
        target = np.zeros_like(image, dtype=np.uint8)
        if value == "ignore":
            target.fill(2)
        _save(source_dir / "labelsTr" / f"{key}.nii.gz", target)

    data_config = tmp_path / "data.yaml"
    data_config.write_text(f"batch_root: '{batch_root.as_posix()}'\n", encoding="utf-8")
    output = assemble(
        SimpleNamespace(
            data_config=data_config,
            split_config=SPLIT_CONFIG,
            nnunet_raw=raw_root,
            source_dataset_dir=source_dir,
            lung_root=None,
            dataset_id=126,
            dataset_name="ATM22MT240LungCrop",
            reuse_mode="hardlink",
        )
    )

    assert len(list((output / "imagesTr").glob("*.nii.gz"))) == 260
    assert len(list((output / "labelsTr").glob("*.nii.gz"))) == 260
    result_json = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    result_provenance = result_json["semi_supervised"]["case_provenance"]
    assert list(result_provenance.values()).count("gt") == 20
    assert list(result_provenance.values()).count("ignore") == 240
    fold = json.loads((output / "splits_final.json").read_text(encoding="utf-8"))[0]
    assert len(fold["train"]) == 256
    assert set(fold["val"]) == {_key(case_id) for case_id in fold_val}

    first_added = _key(sorted(added)[0])
    added_target = nib.load(str(output / "labelsTr" / f"{first_added}.nii.gz"))
    assert np.unique(np.asarray(added_target.dataobj)).tolist() == [2]
