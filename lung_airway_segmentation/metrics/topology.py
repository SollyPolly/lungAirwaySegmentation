"""Topology-aware metrics for binary airway segmentation masks.

Tree-length detected (TD) and branch detected (BD) follow the public ATM'22
evaluation definitions: TD is reference-skeleton recall, while a parsed
reference branch is detected when at least 80% of its skeleton is covered.

Reference evaluator:
https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/tree/main/evaluation
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


# v2 fixes the hard-clDice extension at TPrec == TSens == 0: non-empty,
# fully-disjoint masks score 0 rather than 1. Target-component policies are
# otherwise unchanged from the original scorer.
TOPOLOGY_METRIC_VERSION = "hard-cldice-v2"


def _as_binary_mask(mask) -> np.ndarray:
    binary_mask = np.asarray(mask) > 0
    if binary_mask.ndim != 3:
        raise ValueError(f"Topology metrics require 3D masks, got shape {binary_mask.shape}.")
    return binary_mask


def _validate_matching_masks(predictions, targets) -> tuple[np.ndarray, np.ndarray]:
    prediction_mask = _as_binary_mask(predictions)
    target_mask = _as_binary_mask(targets)
    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "Prediction and target masks must have the same shape: "
            f"{prediction_mask.shape} != {target_mask.shape}"
        )
    return prediction_mask, target_mask


def _crop_to_foreground_union(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    padding: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    foreground = np.logical_or(prediction_mask, target_mask)
    coordinates = np.where(foreground)
    if coordinates[0].size == 0:
        return prediction_mask, target_mask

    slices = tuple(
        slice(max(0, int(axis.min()) - padding), min(size, int(axis.max()) + padding + 1))
        for axis, size in zip(coordinates, foreground.shape)
    )
    return prediction_mask[slices], target_mask[slices]


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.copy()

    components, component_count = ndimage.label(
        mask,
        structure=ndimage.generate_binary_structure(3, 1),
    )
    if component_count == 1:
        largest_component = mask.copy()
    else:
        component_sizes = np.bincount(components.ravel())
        component_sizes[0] = 0
        largest_component = components == int(component_sizes.argmax())
    return np.asarray(ndimage.binary_fill_holes(largest_component), dtype=bool)


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.asarray(skeletonize(mask), dtype=bool)


def _fraction(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else float(numerator / denominator)


def _harmonic_mean(first: float, second: float) -> float:
    """Return the harmonic mean, continuously extended to zero at (0, 0)."""
    denominator = first + second
    return 0.0 if denominator == 0 else float(2.0 * first * second / denominator)


def _foreground_slices(mask: np.ndarray, padding: int = 4) -> tuple[slice, slice, slice]:
    coordinates = np.where(mask)
    if coordinates[0].size == 0:
        return tuple(slice(0, size) for size in mask.shape)
    return tuple(
        slice(max(0, int(axis.min()) - padding), min(size, int(axis.max()) + padding + 1))
        for axis, size in zip(coordinates, mask.shape)
    )


def topology_precision_from_masks(predictions, targets) -> float:
    """Return the fraction of predicted centerline contained in the target."""
    prediction_mask, target_mask = _validate_matching_masks(predictions, targets)
    prediction_mask, target_mask = _crop_to_foreground_union(prediction_mask, target_mask)
    prediction_skeleton = _skeletonize(prediction_mask)
    return _fraction(
        int(np.logical_and(prediction_skeleton, target_mask).sum()),
        int(prediction_skeleton.sum()),
    )


def topology_sensitivity_from_masks(predictions, targets) -> float:
    """Return the fraction of target centerline contained in the prediction."""
    prediction_mask, target_mask = _validate_matching_masks(predictions, targets)
    prediction_mask, target_mask = _crop_to_foreground_union(prediction_mask, target_mask)
    target_skeleton = _skeletonize(target_mask)
    return _fraction(
        int(np.logical_and(target_skeleton, prediction_mask).sum()),
        int(target_skeleton.sum()),
    )


def cldice_score_from_masks(predictions, targets) -> float:
    """Compute binary centerline Dice (clDice)."""
    prediction_mask, target_mask = _validate_matching_masks(predictions, targets)
    prediction_mask, target_mask = _crop_to_foreground_union(prediction_mask, target_mask)
    prediction_skeleton = _skeletonize(prediction_mask)
    target_skeleton = _skeletonize(target_mask)

    topology_precision = _fraction(
        int(np.logical_and(prediction_skeleton, target_mask).sum()),
        int(prediction_skeleton.sum()),
    )
    topology_sensitivity = _fraction(
        int(np.logical_and(target_skeleton, prediction_mask).sum()),
        int(target_skeleton.sum()),
    )
    return _harmonic_mean(topology_precision, topology_sensitivity)


def hard_centerline_metrics_from_masks(
    predictions,
    targets,
    max_ratio: float | None = None,
) -> dict[str, float | int | bool | None]:
    """clDice, topology precision, and tree-length detected in one skeletonisation pass.

    A lean, selection-oriented combination of ``cldice_score_from_masks``,
    ``topology_precision_from_masks`` and ``tree_length_detected_from_masks`` that
    skeletonises the prediction once (instead of twice across the separate calls)
    and skips the expensive ATM'22 branch parsing of
    ``airway_topology_metrics_from_masks``. Intended for per-epoch validation
    selection (hard mask, no LCC).

    ``max_ratio`` defaults to ``None`` (NO gate) — this metric is computed on the
    RAW prediction, and a healthy topology model deliberately produces a large,
    fragmented raw mask at 0.5 (commonly 8-35x GT volume before LCC), so a
    raw-volume gate would reject exactly the models worth selecting. (analyse_distal's
    6x gate is applied AFTER LCC, where the disconnected false-positive bulk is
    already removed — it does NOT transfer to raw selection.) ``max_ratio`` is kept
    only as an optional *catastrophic* guard (set it well above mature raw volumes,
    e.g. 50x) to skip skeletonising a pathological blob; when it fires, clDice and
    topology precision are returned as ``None`` (``gated=True``) — meaning INVALID /
    excluded, NOT a genuine 0.0. TLD (target skeleton only) and the diagnostics are
    always reported.
    """
    prediction_mask, target_mask = _validate_matching_masks(predictions, targets)
    prediction_mask, target_mask = _crop_to_foreground_union(prediction_mask, target_mask)
    prediction_voxels = int(prediction_mask.sum())
    target_voxels = int(target_mask.sum())

    # Connectivity diagnostic (6-connectivity, matching the LCC post-processing):
    # number of predicted components. clDice does NOT measure connectivity, so this
    # is tracked separately as a diagnostic — never used for selection. Cheap
    # (labelling, not skeletonisation), so computed even when skeletonisation gates.
    _, component_count = ndimage.label(
        prediction_mask, structure=ndimage.generate_binary_structure(3, 1)
    )

    # TLD (reference-skeleton recall) needs only the target LCC skeleton.
    target_component = _largest_connected_component(target_mask)
    target_slices = _foreground_slices(target_component)
    target_lcc_skeleton = _skeletonize(target_component[target_slices])
    tree_length_detected = _fraction(
        int(np.logical_and(target_lcc_skeleton, prediction_mask[target_slices]).sum()),
        int(target_lcc_skeleton.sum()),
    )

    if (
        max_ratio is not None
        and target_voxels
        and prediction_voxels > max_ratio * target_voxels
    ):
        return {
            "cldice": None,  # invalid / excluded, NOT a genuine 0.0
            "topology_precision": None,
            "tree_length_detected": float(tree_length_detected),
            "component_count": int(component_count),
            "gated": True,
        }

    prediction_skeleton = _skeletonize(prediction_mask)
    target_skeleton = _skeletonize(target_mask)
    topology_precision = _fraction(
        int(np.logical_and(prediction_skeleton, target_mask).sum()),
        int(prediction_skeleton.sum()),
    )
    topology_sensitivity = _fraction(
        int(np.logical_and(target_skeleton, prediction_mask).sum()),
        int(target_skeleton.sum()),
    )
    cldice = _harmonic_mean(topology_precision, topology_sensitivity)
    return {
        "cldice": float(cldice),
        "topology_precision": float(topology_precision),
        "tree_length_detected": float(tree_length_detected),
        "component_count": int(component_count),
        "gated": False,
    }


def tree_length_detected_from_masks(predictions, targets) -> float:
    """Compute ATM'22-style tree-length detected as reference-skeleton recall."""
    prediction_mask, target_mask = _validate_matching_masks(predictions, targets)
    prediction_mask, target_mask = _crop_to_foreground_union(prediction_mask, target_mask)
    target_component = _largest_connected_component(target_mask)
    target_slices = _foreground_slices(target_component)
    prediction_mask = prediction_mask[target_slices]
    target_skeleton = _skeletonize(target_component[target_slices])
    return _fraction(
        int(np.logical_and(target_skeleton, prediction_mask).sum()),
        int(target_skeleton.sum()),
    )


def _split_reference_skeleton(
    skeleton: np.ndarray,
    minimum_branch_voxels: int,
) -> tuple[np.ndarray, int]:
    """Remove junctions and short fragments using the public ATM'22 parser rules."""
    if minimum_branch_voxels < 1:
        raise ValueError("minimum_branch_voxels must be at least 1.")
    if not skeleton.any():
        return np.zeros_like(skeleton, dtype=np.int32), 0

    neighbour_count = ndimage.convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    split_skeleton = np.logical_and(skeleton, neighbour_count <= 3)
    components, component_count = ndimage.label(
        split_skeleton,
        structure=ndimage.generate_binary_structure(3, 3),
    )
    if component_count == 0:
        return np.zeros_like(skeleton, dtype=np.int32), 0

    component_sizes = np.bincount(components.ravel(), minlength=component_count + 1)
    retained = component_sizes >= minimum_branch_voxels
    retained[0] = False
    split_skeleton = np.logical_and(split_skeleton, retained[components])
    return ndimage.label(
        split_skeleton,
        structure=ndimage.generate_binary_structure(3, 3),
    )


def _nearest_branch_tree_parsing(
    split_skeleton_labels: np.ndarray,
    target_component: np.ndarray,
) -> np.ndarray:
    if split_skeleton_labels.max() == 0:
        return np.zeros_like(split_skeleton_labels, dtype=np.int32)
    nearest_indices = ndimage.distance_transform_edt(
        split_skeleton_labels == 0,
        return_distances=False,
        return_indices=True,
    )
    return np.asarray(
        split_skeleton_labels[tuple(nearest_indices)] * target_component,
        dtype=np.int32,
    )


def _locate_trachea(tree_parsing: np.ndarray, branch_count: int) -> int:
    branch_volumes = np.bincount(tree_parsing.ravel(), minlength=branch_count + 1)
    branch_volumes[0] = 0
    return int(branch_volumes.argmax())


def _expand_slices(
    slices: tuple[slice, slice, slice],
    shape: tuple[int, int, int],
    padding: int = 4,
) -> tuple[slice, slice, slice]:
    return tuple(
        slice(max(0, axis_slice.start - padding), min(size, axis_slice.stop + padding))
        for axis_slice, size in zip(slices, shape)
    )


def _branch_adjacency(tree_parsing: np.ndarray, branch_count: int) -> np.ndarray:
    adjacency = np.zeros((branch_count, branch_count), dtype=bool)
    object_slices = ndimage.find_objects(tree_parsing, max_label=branch_count)
    dilation_structure = ndimage.generate_binary_structure(3, 1)

    for branch_index, object_slice in enumerate(object_slices):
        if object_slice is None:
            continue
        local_slice = _expand_slices(object_slice, tree_parsing.shape)
        local_tree = tree_parsing[local_slice]
        branch_id = branch_index + 1
        branch_mask = local_tree == branch_id
        boundary = np.logical_and(
            ndimage.binary_dilation(branch_mask, structure=dilation_structure),
            np.logical_not(branch_mask),
        )
        adjacent_ids = np.unique(local_tree[boundary])
        adjacent_ids = adjacent_ids[adjacent_ids > 0]
        adjacency[branch_index, adjacent_ids - 1] = True
    return adjacency


def _parent_children_maps(
    adjacency: np.ndarray,
    trachea: int,
) -> tuple[np.ndarray, np.ndarray]:
    branch_count = adjacency.shape[0]
    parent_map = np.zeros((branch_count, branch_count), dtype=bool)
    children_map = np.zeros((branch_count, branch_count), dtype=bool)
    generation = np.zeros(branch_count, dtype=np.int32)
    processing = [trachea - 1]
    parent_map[trachea - 1, trachea - 1] = True

    while processing:
        iteration = processing
        processing = []
        while iteration:
            current = iteration.pop()
            for child in np.flatnonzero(adjacency[current]):
                if not parent_map[child].any():
                    parent_map[child, current] = True
                    children_map[current, child] = True
                    generation[child] = generation[current] + 1
                    processing.append(int(child))
                elif generation[current] + 1 == generation[child]:
                    parent_map[child, current] = True
                    children_map[current, child] = True
    return parent_map, children_map


def _refine_tree_parsing(
    tree_parsing: np.ndarray,
    parent_map: np.ndarray,
    children_map: np.ndarray,
) -> tuple[np.ndarray, int, bool]:
    branch_count = parent_map.shape[0]
    delete_ids: list[int] = []

    for branch_index in np.flatnonzero(parent_map.sum(axis=1) > 1):
        parent_indices = np.flatnonzero(parent_map[branch_index])
        retained_parent_id = int(parent_indices[0] + 1)
        for parent_index in parent_indices[1:]:
            parent_id = int(parent_index + 1)
            tree_parsing[tree_parsing == parent_id] = retained_parent_id
            if parent_id not in delete_ids:
                delete_ids.append(parent_id)

    for parent_index in np.flatnonzero(children_map.sum(axis=1) == 1):
        parent_id = int(parent_index + 1)
        child_id = int(np.flatnonzero(children_map[parent_index])[0] + 1)
        if parent_id not in delete_ids and child_id not in delete_ids:
            tree_parsing[tree_parsing == child_id] = parent_id
            delete_ids.append(child_id)

    if not delete_ids:
        return tree_parsing, branch_count, False

    retained_ids = [branch_id for branch_id in range(1, branch_count + 1) if branch_id not in delete_ids]
    relabel = np.zeros(branch_count + 1, dtype=np.int32)
    for new_id, old_id in enumerate(retained_ids, start=1):
        relabel[old_id] = new_id
    return relabel[tree_parsing], len(retained_ids), True


def _atm22_reference_branch_labels(
    target_component: np.ndarray,
    target_skeleton: np.ndarray,
    minimum_branch_voxels: int,
) -> np.ndarray:
    split_labels, branch_count = _split_reference_skeleton(
        target_skeleton,
        minimum_branch_voxels,
    )
    if branch_count == 0:
        return np.zeros_like(target_component, dtype=np.int32)

    tree_parsing = _nearest_branch_tree_parsing(split_labels, target_component)
    while True:
        trachea = _locate_trachea(tree_parsing, branch_count)
        adjacency = _branch_adjacency(tree_parsing, branch_count)
        parent_map, children_map = _parent_children_maps(adjacency, trachea)
        tree_parsing, branch_count, changed = _refine_tree_parsing(
            tree_parsing,
            parent_map,
            children_map,
        )
        if not changed:
            break
    return np.asarray(tree_parsing * target_skeleton, dtype=np.int32)


def parse_reference_skeleton_branches(
    targets,
    *,
    minimum_branch_voxels: int = 5,
) -> np.ndarray:
    """Parse an airway reference mask into ATM'22-compatible skeleton branches."""
    target_mask = _as_binary_mask(targets)
    target_component = _largest_connected_component(target_mask)
    target_slices = _foreground_slices(target_component)
    target_component = target_component[target_slices]
    target_skeleton = _skeletonize(target_component)
    return _atm22_reference_branch_labels(
        target_component,
        target_skeleton,
        minimum_branch_voxels,
    )


def branch_detection_metrics_from_masks(
    predictions,
    targets,
    *,
    detection_threshold: float = 0.8,
    minimum_branch_voxels: int = 5,
) -> dict[str, float | int]:
    """Compute ATM'22-style reference branch detection statistics."""
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError("detection_threshold must be between 0 and 1.")

    prediction_mask, target_mask = _validate_matching_masks(predictions, targets)
    prediction_mask, target_mask = _crop_to_foreground_union(prediction_mask, target_mask)
    target_component = _largest_connected_component(target_mask)
    target_slices = _foreground_slices(target_component)
    prediction_mask = prediction_mask[target_slices]
    target_component = target_component[target_slices]
    target_skeleton = _skeletonize(target_component)
    branch_labels = _atm22_reference_branch_labels(
        target_component,
        target_skeleton,
        minimum_branch_voxels,
    )
    reference_branch_count = int(branch_labels.max())
    if reference_branch_count == 0:
        return {
            "reference_branch_count": 0,
            "detected_branch_count": 0,
            "branch_detected": 1.0,
        }

    reference_counts = np.bincount(
        branch_labels[branch_labels > 0],
        minlength=reference_branch_count + 1,
    )[1:]
    detected_counts = np.bincount(
        branch_labels[np.logical_and(branch_labels > 0, prediction_mask)],
        minlength=reference_branch_count + 1,
    )[1:]
    detected_branch_count = int(np.count_nonzero(detected_counts / reference_counts >= detection_threshold))

    return {
        "reference_branch_count": reference_branch_count,
        "detected_branch_count": detected_branch_count,
        "branch_detected": float(detected_branch_count / reference_branch_count),
    }


def airway_topology_metrics_from_masks(
    predictions,
    targets,
    *,
    branch_detection_threshold: float = 0.8,
    minimum_branch_voxels: int = 5,
) -> dict[str, float | int]:
    """Compute clDice and ATM'22-style TD/BD metrics in one pass."""
    if not 0.0 <= branch_detection_threshold <= 1.0:
        raise ValueError("branch_detection_threshold must be between 0 and 1.")
    prediction_mask, target_mask = _validate_matching_masks(predictions, targets)
    prediction_mask, target_mask = _crop_to_foreground_union(prediction_mask, target_mask)

    prediction_skeleton = _skeletonize(prediction_mask)
    target_skeleton = _skeletonize(target_mask)

    topology_precision = _fraction(
        int(np.logical_and(prediction_skeleton, target_mask).sum()),
        int(prediction_skeleton.sum()),
    )
    topology_sensitivity = _fraction(
        int(np.logical_and(target_skeleton, prediction_mask).sum()),
        int(target_skeleton.sum()),
    )
    cldice = _harmonic_mean(topology_precision, topology_sensitivity)
    target_component = _largest_connected_component(target_mask)
    target_slices = _foreground_slices(target_component)
    atm_prediction_mask = prediction_mask[target_slices]
    target_component = target_component[target_slices]
    atm_target_skeleton = _skeletonize(target_component)
    tree_length_detected = _fraction(
        int(np.logical_and(atm_target_skeleton, atm_prediction_mask).sum()),
        int(atm_target_skeleton.sum()),
    )

    branch_labels = _atm22_reference_branch_labels(
        target_component,
        atm_target_skeleton,
        minimum_branch_voxels,
    )
    reference_branch_count = int(branch_labels.max())
    if reference_branch_count == 0:
        detected_branch_count = 0
        branch_detected = 1.0
    else:
        reference_counts = np.bincount(
            branch_labels[branch_labels > 0],
            minlength=reference_branch_count + 1,
        )[1:]
        detected_counts = np.bincount(
            branch_labels[np.logical_and(branch_labels > 0, atm_prediction_mask)],
            minlength=reference_branch_count + 1,
        )[1:]
        detected_branch_count = int(
            np.count_nonzero(detected_counts / reference_counts >= branch_detection_threshold)
        )
        branch_detected = float(detected_branch_count / reference_branch_count)

    return {
        "tree_length_detected": tree_length_detected,
        "branch_detected": branch_detected,
        "cldice": cldice,
        "topology_precision": topology_precision,
        "topology_sensitivity": topology_sensitivity,
        "reference_skeleton_voxels": int(atm_target_skeleton.sum()),
        "prediction_skeleton_voxels": int(prediction_skeleton.sum()),
        "reference_branch_count": reference_branch_count,
        "detected_branch_count": detected_branch_count,
    }
