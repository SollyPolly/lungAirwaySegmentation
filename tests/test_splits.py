"""Tests for deterministic dataset split generation.

This module verifies that the split helper preserves all case IDs, creates
non-overlapping train/validation/test groups, and remains reproducible for a
fixed random seed.
"""

import pytest

from lung_airway_segmentation.datasets.splits import create_train_val_test_split


def test_create_train_val_test_split_raises_when_fractions_do_not_sum_to_one():
    case_ids = ["1", "2", "3", "4"]

    with pytest.raises(ValueError, match="sum to 1.0"):
        create_train_val_test_split(
            case_ids,
            train_split=0.6,
            val_split=0.2,
            test_split=0.3,
        )


def test_create_train_val_test_split_preserves_all_case_ids_without_overlap():
    case_ids = [str(i) for i in range(10)]

    train_ids, val_ids, test_ids = create_train_val_test_split(
        case_ids,
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        seed=15,
    )

    combined = train_ids + val_ids + test_ids

    assert set(combined) == set(case_ids)
    assert len(combined) == len(case_ids)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)


def test_create_train_val_test_split_is_reproducible_for_same_seed():
    case_ids = [str(i) for i in range(20)]

    split_1 = create_train_val_test_split(case_ids, seed=15)
    split_2 = create_train_val_test_split(case_ids, seed=15)

    assert split_1 == split_2


def test_create_train_val_test_split_changes_with_different_seed():
    case_ids = [str(i) for i in range(20)]

    split_1 = create_train_val_test_split(case_ids, seed=15)
    split_2 = create_train_val_test_split(case_ids, seed=99)

    assert split_1 != split_2


def test_create_train_val_test_split_converts_case_ids_to_strings():
    case_ids = [1, 2, 3, 4, 5, 6]

    train_ids, val_ids, test_ids = create_train_val_test_split(case_ids, seed=15)

    assert all(isinstance(case_id, str) for case_id in train_ids)
    assert all(isinstance(case_id, str) for case_id in val_ids)
    assert all(isinstance(case_id, str) for case_id in test_ids)
