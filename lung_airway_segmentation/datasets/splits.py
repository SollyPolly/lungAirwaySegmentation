"""Deterministic helpers for train, validation, and test case splits.

This module keeps experiment split logic in one place so case allocation stays
reproducible across training runs. It provides the original three-way
train/validation/test split (used by AeroPath) and a four-way semi-supervised
split (used by ATM'22) that additionally partitions the train pool into a
labelled subset and an unlabelled subset.
"""

import random

from sklearn.model_selection import train_test_split

def create_train_val_test_split(
    case_ids,
    *,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    seed=15
):
    """Split case IDs into non-overlapping train, validation, and test lists."""
    if abs((train_split + val_split + test_split) - 1.0 ) > 1e-8:
        raise ValueError("Splits must sum to 1.0")
    
    if train_split < 0 or val_split < 0 or test_split < 0:
        raise ValueError("Split fractions must be non-negative")
    
    case_ids = [str(case_id) for case_id in case_ids]

    train_ids, temp_ids = train_test_split(
        case_ids,
        train_size=train_split,
        random_state=seed,
        shuffle=True
    )

    relative_val_split = val_split / (val_split + test_split)

    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=relative_val_split,
        random_state=seed,
        shuffle=True
    )

    return train_ids, val_ids, test_ids


def create_semisupervised_split(
    case_ids,
    *,
    test_count,
    val_count,
    labelled_count,
    seed=15,
):
    """Partition case IDs into disjoint test, val, labelled-train, unlabelled-train sets.

    Test and val are carved out first from a seed-shuffled ordering and held
    fixed; the remaining train pool is split into a labelled subset of size
    ``labelled_count`` and an unlabelled subset (everything left). Because the
    ordering is deterministic and test/val are sliced before the labelled
    boundary, sweeping ``labelled_count`` for a label-efficiency curve never
    disturbs the sacred test/val sets — it only slides the labelled/unlabelled
    boundary within the train pool.

    Returns a dict with keys ``labelled_train``, ``unlabelled_train``, ``val``,
    ``test`` (sorted lists of string IDs).
    """
    ids = sorted(str(case_id) for case_id in case_ids)
    total = len(ids)

    if test_count < 0 or val_count < 0:
        raise ValueError("test_count and val_count must be non-negative.")
    if labelled_count <= 0:
        raise ValueError("labelled_count must be positive.")
    if test_count + val_count + labelled_count > total:
        raise ValueError(
            f"test_count + val_count + labelled_count ({test_count + val_count + labelled_count}) "
            f"exceeds the number of available cases ({total})."
        )

    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)

    test_ids = shuffled[:test_count]
    val_ids = shuffled[test_count:test_count + val_count]
    train_pool = shuffled[test_count + val_count:]
    labelled_train_ids = train_pool[:labelled_count]
    unlabelled_train_ids = train_pool[labelled_count:]

    return {
        "labelled_train": sorted(labelled_train_ids),
        "unlabelled_train": sorted(unlabelled_train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
