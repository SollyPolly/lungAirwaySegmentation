"""Derived-target generation for topology-aware training.

This file should build supervision targets that are not present directly in the
raw dataset.

Examples:
- centerlines
- Euclidean distance transforms
- signed distance fields
- anatomy-aware airway group labels

Keep target generation here so training code can stay focused on optimization.
"""
