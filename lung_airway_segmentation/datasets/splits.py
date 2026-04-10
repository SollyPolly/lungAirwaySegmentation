"""Deterministic helpers for train, validation, and test case splits.

This module keeps experiment split logic in one place so case allocation stays
reproducible across training runs. The current implementation creates one
random but seed-controlled train/validation/test split from a list of case
identifiers.
"""

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
