"""Tests for dataset path conventions and case discovery.

This file should verify:
- raw AeroPath file naming rules
- case ID discovery and sorting
- clear failures for missing CT or mask files

Keep this file focused on path and validation behavior only.
"""

from pathlib import Path

import pytest

from lung_airway_segmentation.io.case_layout import (
    list_case_ids,
    resolve_case_paths,
    validate_case_paths,
)

from lung_airway_segmentation.schemas import UnlabelledCasePaths


def test_list_case_ids_sorts_numeric_case_directories(tmp_path):
    (tmp_path / "10").mkdir()
    (tmp_path / "2").mkdir()
    (tmp_path / "1").mkdir()
    (tmp_path / "notes").mkdir()

    case_ids = list_case_ids(tmp_path)

    assert case_ids == ["1", "2", "10"]

def test_resolve_case_paths_builds_expected_paths(tmp_path):
    case_dir = tmp_path / "1"
    case_dir.mkdir()

    paths = resolve_case_paths(1, data_root=tmp_path)

    assert paths["case_id"] == "1"
    assert paths["case_dir"] == case_dir
    assert paths["ct"] == case_dir / "1_CT_HR.nii.gz"

def test_validate_case_paths_raises_when_ct_is_missing(tmp_path):
    case_dir = tmp_path / "1"
    case_dir.mkdir()

    paths: UnlabelledCasePaths= {
        "case_id": "1",
        "case_dir": case_dir,
        "ct": case_dir / "1_CT_HR.nii.gz",
        "lung": None,
        "airway": None
    }

    with pytest.raises(FileNotFoundError, match="CT file does not exist"):
        validate_case_paths(paths)

def test_resolve_case_paths_returns_labelled_case_when_airway_exists(tmp_path):
    case_dir = tmp_path / "1"
    case_dir.mkdir()

    (case_dir / "1_CT_HR.nii.gz").touch()
    (case_dir / "1_CT_HR_label_lungs.nii.gz").touch()
    (case_dir / "1_CT_HR_label_airways.nii.gz").touch()

    paths = resolve_case_paths(1, data_root=tmp_path)

    assert paths["lung"] == case_dir / "1_CT_HR_label_lungs.nii.gz"
    assert paths["airway"] == case_dir / "1_CT_HR_label_airways.nii.gz"