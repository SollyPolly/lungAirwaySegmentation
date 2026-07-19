"""Tests for topology-aware airway metrics."""

import numpy as np

from lung_airway_segmentation.metrics.topology import (
    airway_topology_metrics_from_masks,
    branch_detection_metrics_from_masks,
    cldice_score_from_masks,
    hard_centerline_metrics_from_masks,
    topology_precision_from_masks,
    topology_sensitivity_from_masks,
    tree_length_detected_from_masks,
)


def _three_branch_target() -> np.ndarray:
    target = np.zeros((15, 15, 15), dtype=np.uint8)
    target[2:13, 7, 7] = 1
    target[7, 7:14, 7] = 1
    target[7, 7, 7:14] = 1
    return target


def test_cldice_is_one_for_perfect_tubular_prediction():
    target = _three_branch_target()

    assert np.isclose(cldice_score_from_masks(target, target), 1.0)
    assert np.isclose(tree_length_detected_from_masks(target, target), 1.0)


def test_cldice_penalizes_disconnected_false_positive_skeleton():
    target = _three_branch_target()
    prediction = target.copy()
    prediction[2:13, 2, 2] = 1

    metrics = airway_topology_metrics_from_masks(prediction, target)

    assert np.isclose(metrics["topology_sensitivity"], 1.0)
    assert metrics["topology_precision"] < 1.0
    assert metrics["cldice"] < 1.0


def test_cldice_is_zero_for_nonempty_disjoint_trees_across_metric_apis():
    target = np.zeros((12, 12, 12), dtype=np.uint8)
    target[1:11, 3, 3] = 1
    prediction = np.zeros_like(target)
    prediction[1:11, 8, 8] = 1

    combined = hard_centerline_metrics_from_masks(prediction, target)
    airway = airway_topology_metrics_from_masks(prediction, target)

    assert topology_precision_from_masks(prediction, target) == 0.0
    assert topology_sensitivity_from_masks(prediction, target) == 0.0
    assert cldice_score_from_masks(prediction, target) == 0.0
    assert combined["cldice"] == 0.0
    assert combined["tree_length_detected"] == 0.0
    assert airway["cldice"] == 0.0
    assert airway["topology_precision"] == 0.0
    assert airway["topology_sensitivity"] == 0.0
    assert airway["tree_length_detected"] == 0.0


def test_cldice_empty_mask_conventions_are_explicit_and_consistent():
    empty = np.zeros((9, 9, 9), dtype=np.uint8)
    tree = np.zeros_like(empty)
    tree[1:8, 4, 4] = 1

    # Both empty represents perfect agreement. One-sided empty masks do not.
    assert cldice_score_from_masks(empty, empty) == 1.0
    assert hard_centerline_metrics_from_masks(empty, empty)["cldice"] == 1.0
    assert airway_topology_metrics_from_masks(empty, empty)["cldice"] == 1.0

    for prediction, target in ((empty, tree), (tree, empty)):
        assert cldice_score_from_masks(prediction, target) == 0.0
        assert hard_centerline_metrics_from_masks(prediction, target)["cldice"] == 0.0
        assert airway_topology_metrics_from_masks(prediction, target)["cldice"] == 0.0


def test_branch_detection_uses_eighty_percent_coverage_threshold():
    target = np.zeros((9, 9, 20), dtype=np.uint8)
    target[4, 4, 2:18] = 1
    prediction_above_threshold = np.zeros_like(target)
    prediction_above_threshold[4, 4, 2:15] = 1
    prediction_below_threshold = np.zeros_like(target)
    prediction_below_threshold[4, 4, 2:14] = 1

    above = branch_detection_metrics_from_masks(prediction_above_threshold, target)
    below = branch_detection_metrics_from_masks(prediction_below_threshold, target)

    assert above["reference_branch_count"] == 1
    assert above["branch_detected"] == 1.0
    assert below["branch_detected"] == 0.0


def test_tree_length_detected_measures_reference_skeleton_coverage():
    target = np.zeros((9, 9, 12), dtype=np.uint8)
    target[4, 4, 1:11] = 1
    prediction = np.zeros_like(target)
    prediction[4, 4, 1:6] = 1

    assert np.isclose(tree_length_detected_from_masks(prediction, target), 0.5)


def test_hard_centerline_metrics_match_the_individual_functions():
    target = _three_branch_target()
    prediction = target.copy()
    prediction[2:13, 2, 2] = 1  # a disconnected false branch

    combined = hard_centerline_metrics_from_masks(prediction, target)

    assert np.isclose(combined["cldice"], cldice_score_from_masks(prediction, target))
    assert np.isclose(
        combined["topology_precision"], topology_precision_from_masks(prediction, target)
    )
    assert np.isclose(
        combined["tree_length_detected"],
        tree_length_detected_from_masks(prediction, target),
    )
    assert combined["gated"] is False
    # Two components: the (overlapping) tree plus the detached false branch.
    assert combined["component_count"] == 2


def test_hard_centerline_metrics_default_has_no_volume_gate():
    """A large fragmented raw mask (the healthy topology-model case) is NOT gated."""
    target = _three_branch_target()
    prediction = np.ones_like(target)  # 10x+ GT volume

    result = hard_centerline_metrics_from_masks(prediction, target)  # max_ratio=None

    assert result["gated"] is False
    assert result["cldice"] is not None  # skeletonised and scored, not rejected


def test_hard_centerline_metrics_catastrophic_guard_marks_invalid_not_zero():
    target = _three_branch_target()
    prediction = np.ones_like(target)  # massive over-segmentation (whole volume)

    gated = hard_centerline_metrics_from_masks(prediction, target, max_ratio=6.0)

    # Skeletonisation skipped: clDice/precision are INVALID (None), NOT a genuine
    # 0.0; TLD (target-only skeleton) is still reported.
    assert gated["gated"] is True
    assert gated["cldice"] is None
    assert gated["topology_precision"] is None
    assert np.isclose(gated["tree_length_detected"], 1.0)
