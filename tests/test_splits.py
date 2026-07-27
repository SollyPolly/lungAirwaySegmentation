"""Tests for the ATM'22 split shared by nnU-Net, MT, and SSL builders."""

from pathlib import Path

import pytest
import yaml

from lung_airway_segmentation.datasets.splits import (
    create_semisupervised_split,
    create_split_from_config,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "nnunet"


def _load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_semisupervised_split_is_deterministic_and_disjoint():
    case_ids = [str(index) for index in range(150)]

    first = create_semisupervised_split(
        case_ids,
        test_count=20,
        val_count=20,
        labelled_count=20,
        seed=15,
    )
    second = create_semisupervised_split(
        case_ids,
        test_count=20,
        val_count=20,
        labelled_count=20,
        seed=15,
    )

    assert first == second
    assert len(first["test"]) == 20
    assert len(first["val"]) == 20
    assert len(first["labelled_train"]) == 20
    assert len(first["unlabelled_train"]) == 90
    groups = [set(values) for values in first.values()]
    assert set.union(*groups) == set(case_ids)
    assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i + 1, 4))


def test_semisupervised_split_rejects_impossible_counts():
    with pytest.raises(ValueError, match="exceeds"):
        create_semisupervised_split(
            ["001", "002"],
            test_count=1,
            val_count=1,
            labelled_count=1,
        )


def test_legacy_frozen_split_ignores_only_declared_batch2_cases():
    config = _load_config("atm22_split_l20.yaml")
    legacy_ids = [
        case_id
        for values in config["splits"].values()
        for case_id in values
    ]
    merged_ids = legacy_ids + config["allowed_additional_case_ids"]

    split = create_split_from_config(merged_ids, config)

    assert {key: len(values) for key, values in split.items()} == {
        "labelled_train": 20,
        "unlabelled_train": 90,
        "val": 20,
        "test": 20,
    }
    assert split["labelled_train"] == sorted(config["splits"]["labelled_train"])
    assert split["val"] == sorted(config["splits"]["val"])
    assert split["test"] == sorted(config["splits"]["test"])


def test_expanded_split_changes_only_unlabelled_membership():
    legacy_config = _load_config("atm22_split_l20.yaml")
    expanded_config = _load_config("atm22_split_l20_u240.yaml")
    inventory = [
        case_id
        for values in expanded_config["splits"].values()
        for case_id in values
    ]

    expanded = create_split_from_config(inventory, expanded_config)

    for key in ("labelled_train", "val", "test"):
        assert expanded[key] == sorted(legacy_config["splits"][key])
    assert set(expanded["unlabelled_train"]) == (
        set(legacy_config["splits"]["unlabelled_train"])
        | set(expanded_config["added_unlabelled_case_ids"])
    )
    assert {key: len(values) for key, values in expanded.items()} == {
        "labelled_train": 20,
        "unlabelled_train": 240,
        "val": 20,
        "test": 20,
    }


def test_frozen_split_rejects_undeclared_inventory_change():
    config = {
        "splits": {
            "labelled_train": ["001"],
            "unlabelled_train": ["002"],
            "val": ["003"],
            "test": ["004"],
        }
    }
    with pytest.raises(ValueError, match="not declared"):
        create_split_from_config(["001", "002", "003", "004", "005"], config)


def test_frozen_split_rejects_overlap_and_missing_cases():
    overlapping = {
        "splits": {
            "labelled_train": ["001"],
            "unlabelled_train": ["001"],
            "val": ["003"],
            "test": ["004"],
        }
    }
    with pytest.raises(ValueError, match="overlap"):
        create_split_from_config(["001", "003", "004"], overlapping)

    missing = {
        "splits": {
            "labelled_train": ["001"],
            "unlabelled_train": ["002"],
            "val": ["003"],
            "test": ["004"],
        }
    }
    with pytest.raises(ValueError, match="missing"):
        create_split_from_config(["001", "002", "003"], missing)


def test_count_split_requires_and_enforces_inventory_size():
    config = {
        "expected_total_cases": 4,
        "seed": 15,
        "labelled_split": {
            "test_count": 1,
            "val_count": 1,
            "labelled_count": 1,
        },
    }
    with pytest.raises(ValueError, match="expected 4"):
        create_split_from_config(["001", "002", "003", "004", "005"], config)

    without_size = dict(config)
    without_size.pop("expected_total_cases")
    with pytest.raises(ValueError, match="expected_total_cases"):
        create_split_from_config(["001", "002", "003", "004"], without_size)
