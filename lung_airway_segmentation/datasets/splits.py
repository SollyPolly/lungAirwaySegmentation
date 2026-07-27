"""Deterministic and frozen ATM'22 split helpers.

Count-based splitting is useful while designing a new experiment, but it is
unsafe once files are added to the dataset root: the same seed shuffles a
different population and silently changes every role. Reportable experiments
therefore use explicit ``splits`` lists in their YAML configuration.
"""

import random


SPLIT_KEYS = ("labelled_train", "unlabelled_train", "val", "test")


def _normalise_case_id(case_id) -> str:
    value = str(case_id).strip()
    if value.upper().startswith("ATM_"):
        value = value[4:]
    if not value.isdigit():
        raise ValueError(f"Invalid ATM case identifier: {case_id!r}")
    return f"{int(value):03d}"


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


def _frozen_split_from_config(case_ids, split_config: dict) -> dict[str, list[str]]:
    raw_split = split_config["splits"]
    if not isinstance(raw_split, dict):
        raise ValueError("split config 'splits' must be a mapping.")

    split: dict[str, list[str]] = {}
    for key in SPLIT_KEYS:
        raw_values = raw_split.get(key)
        if not isinstance(raw_values, list):
            raise ValueError(f"split config splits.{key} must be a list.")
        values = [_normalise_case_id(value) for value in raw_values]
        if len(values) != len(set(values)):
            raise ValueError(f"split config splits.{key} contains duplicate case IDs.")
        split[key] = sorted(values)

    groups = {key: set(values) for key, values in split.items()}
    for index, left in enumerate(SPLIT_KEYS):
        for right in SPLIT_KEYS[index + 1:]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(
                    f"Frozen split groups {left} and {right} overlap: {sorted(overlap)}"
                )

    expected_counts = split_config.get("expected_counts", {})
    if not isinstance(expected_counts, dict):
        raise ValueError("split config expected_counts must be a mapping.")
    for key in SPLIT_KEYS:
        if key in expected_counts and len(split[key]) != int(expected_counts[key]):
            raise ValueError(
                f"Frozen split expected {expected_counts[key]} {key} cases, "
                f"got {len(split[key])}."
            )

    available_values = [_normalise_case_id(value) for value in case_ids]
    if len(available_values) != len(set(available_values)):
        raise ValueError("Dataset inventory contains duplicate ATM case IDs.")
    available = set(available_values)
    configured = set().union(*(groups[key] for key in SPLIT_KEYS))
    missing = configured - available
    if missing:
        raise ValueError(f"Frozen split cases are missing from the dataset: {sorted(missing)}")

    raw_allowed = split_config.get("allowed_additional_case_ids", [])
    if not isinstance(raw_allowed, list):
        raise ValueError("split config allowed_additional_case_ids must be a list.")
    allowed_additional = {_normalise_case_id(value) for value in raw_allowed}
    unknown_extra = (available - configured) - allowed_additional
    if unknown_extra:
        raise ValueError(
            "Dataset has cases not declared by the frozen split or its allowed additions: "
            f"{sorted(unknown_extra)}"
        )
    return split


def create_split_from_config(case_ids, split_config: dict) -> dict[str, list[str]]:
    """Resolve a frozen split, or a size-pinned count split for new experiments.

    A count-only configuration must declare ``expected_total_cases``. This
    makes adding a second archive fail loudly instead of silently reshuffling
    labelled, validation, and test membership.
    """
    if "splits" in split_config:
        return _frozen_split_from_config(case_ids, split_config)

    if "labelled_split" not in split_config:
        raise ValueError("Split config must define explicit 'splits' or 'labelled_split'.")
    if "expected_total_cases" not in split_config:
        raise ValueError(
            "Count-based split configs must set expected_total_cases; use explicit "
            "'splits' lists for reportable experiments."
        )
    case_ids = list(case_ids)
    expected_total = int(split_config["expected_total_cases"])
    if len(case_ids) != expected_total:
        raise ValueError(
            f"Count-based split expected {expected_total} cases, found {len(case_ids)}. "
            "Refusing to reshuffle a changed dataset inventory."
        )
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
