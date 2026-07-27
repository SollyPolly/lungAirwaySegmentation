"""Tests for the ATM'22 split shared by nnU-Net, MT, and SSL builders."""

import pytest

from lung_airway_segmentation.datasets.splits import create_semisupervised_split


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
