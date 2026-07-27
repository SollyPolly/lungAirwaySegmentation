"""Deterministic ATM'22 split helper for nnU-Net and SSL experiments."""

import random


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


def create_split_from_config(case_ids, split_config: dict) -> dict[str, list[str]]:
    """Build the canonical split from the compact nnU-Net split YAML."""
    counts = split_config["labelled_split"]
    return create_semisupervised_split(
        case_ids,
        test_count=int(counts["test_count"]),
        val_count=int(counts["val_count"]),
        labelled_count=int(counts["labelled_count"]),
        seed=int(split_config.get("seed", 15)),
    )


def cases_for_split(split: dict[str, list[str]], name: str) -> list[str]:
    """Return val, test, or the combined train pool from a canonical split."""
    if name == "train":
        return sorted(split["labelled_train"] + split["unlabelled_train"])
    if name not in {"val", "test"}:
        raise ValueError(f"Unsupported split: {name}")
    return list(split[name])
