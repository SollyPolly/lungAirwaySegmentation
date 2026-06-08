"""Tests for topology-aware airway metrics."""

import numpy as np

from lung_airway_segmentation.metrics.topology import (
    airway_topology_metrics_from_masks,
    branch_detection_metrics_from_masks,
    cldice_score_from_masks,
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
