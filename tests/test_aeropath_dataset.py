"""Tests for the AeroPath dataset wrapper.

This module verifies that the dataset turns preprocessed cases into a stable
sample dictionary and preserves the configuration passed in at construction
time.
"""

from pathlib import Path

import numpy as np

from lung_airway_segmentation.datasets.aeropath import AeroPathDataset
from lung_airway_segmentation.schemas import PreprocessedCase


def make_preprocessed_case(*, case_id: str, include_lung_mask: bool) -> PreprocessedCase:
    return PreprocessedCase(
        case_id=case_id,
        ct=np.ones((4, 5, 6), dtype=np.float32),
        airway_mask=np.ones((4, 5, 6), dtype=np.uint8),
        lung_mask=np.ones((4, 5, 6), dtype=np.uint8) if include_lung_mask else None,
        spacing=(1.0, 1.0, 1.0),
        affine=np.eye(4, dtype=np.float64),
        crop_box=((0, 4), (0, 5), (0, 6)),
        metadata={
            "supervision": "labeled",
            "case_dir": Path("data/AeroPath/1"),
            "ct_path": Path("data/AeroPath/1/1_CT_HR.nii.gz"),
            "lung_mask_path": Path("data/AeroPath/1/1_CT_HR_label_lungs.nii.gz"),
            "airway_mask_path": Path("data/AeroPath/1/1_CT_HR_label_airways.nii.gz"),
            "original_shape": (4, 5, 6),
            "processed_shape": (4, 5, 6),
            "spacing": (1.0, 1.0, 1.0),
            "original_affine": np.eye(4, dtype=np.float64),
            "cropped_affine": np.eye(4, dtype=np.float64),
            "crop_margin": (5, 5, 5),
            "hu_window": (-1024.0, 600.0),
            "crop_source": "lung_mask",
        },
    )


def test_aeropath_dataset_returns_expected_sample_keys(monkeypatch):
    def fake_preprocess_case(case_id, **kwargs):
        return make_preprocessed_case(case_id=str(case_id), include_lung_mask=True)

    monkeypatch.setattr(
        "lung_airway_segmentation.datasets.aeropath.preprocess_case",
        fake_preprocess_case,
    )

    dataset = AeroPathDataset(case_ids=["1"], include_lung_mask=True)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["case_id"] == "1"
    assert set(sample.keys()) == {
        "case_id",
        "image",
        "airway_mask",
        "lung_mask",
        "spacing",
        "affine",
        "crop_box",
        "metadata",
    }
    assert isinstance(sample["image"], np.ndarray)
    assert isinstance(sample["airway_mask"], np.ndarray)
    assert isinstance(sample["lung_mask"], np.ndarray)
    assert sample["image"].shape == (32, 32, 32)
    assert sample["airway_mask"].shape == (32, 32, 32)
    assert sample["lung_mask"].shape == (16, 16, 16)


def test_aeropath_dataset_keeps_lung_mask_none_when_not_requested(monkeypatch):
    def fake_preprocess_case(case_id, **kwargs):
        return make_preprocessed_case(case_id=str(case_id), include_lung_mask=False)

    monkeypatch.setattr(
        "lung_airway_segmentation.datasets.aeropath.preprocess_case",
        fake_preprocess_case,
    )

    dataset = AeroPathDataset(case_ids=["1"], include_lung_mask=False)
    sample = dataset[0]

    assert sample["lung_mask"] is None
    assert sample["spacing"] == (1.0, 1.0, 1.0)
    assert sample["crop_box"] == ((0, 4), (0, 5), (0, 6))


def test_aeropath_dataset_applies_optional_transform(monkeypatch):
    def fake_preprocess_case(case_id, **kwargs):
        return make_preprocessed_case(case_id=str(case_id), include_lung_mask=False)

    def transform(sample):
        sample["transformed"] = True
        return sample

    monkeypatch.setattr(
        "lung_airway_segmentation.datasets.aeropath.preprocess_case",
        fake_preprocess_case,
    )

    dataset = AeroPathDataset(case_ids=["1"], transform=transform)
    sample = dataset[0]

    assert sample["transformed"] is True


def test_aeropath_dataset_uses_list_case_ids_when_case_ids_not_provided(monkeypatch):
    def fake_list_case_ids(data_root):
        return ["2", "5"]

    monkeypatch.setattr(
        "lung_airway_segmentation.datasets.aeropath.list_case_ids",
        fake_list_case_ids,
    )

    dataset = AeroPathDataset(case_ids=None)

    assert len(dataset) == 2
    assert dataset.case_ids == ["2", "5"]
