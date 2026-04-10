"""Path resolution and lightweight validation for on-disk case data.

This module defines how a case is represented on disk for the current dataset.
It lists available case IDs, builds the expected CT and mask paths for a case,
and checks that required files exist before downstream code tries to load them.
It is the disk-level entry point for dataset access and does not inspect image
contents.
"""

from pathlib import Path

from lung_airway_segmentation.settings import RAW_AEROPATH_ROOT
from lung_airway_segmentation.schemas import (
    CasePaths,
    LabelledCasePaths,
    UnlabelledCasePaths,
)


def list_case_ids(data_root = RAW_AEROPATH_ROOT):
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}.")
    
    case_ids = [
        path.name for path in data_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]

    if not case_ids:
        raise ValueError(f"No numeric case directories were found in {data_root}.\n"
                         "Currently only working for AeroPath")
    
    return sorted(case_ids, key=int)

def resolve_case_paths(case_id, data_root=RAW_AEROPATH_ROOT) -> CasePaths:
    case_str = str(case_id)
    case_dir = Path(data_root) / case_str

    # AeroPath-specific filename conventions live here. A new dataset should get
    # its own layout resolver rather than being forced through these names.
    ct_path = case_dir / f"{case_str}_CT_HR.nii.gz"
    lung_path = case_dir / f"{case_str}_CT_HR_label_lungs.nii.gz"
    airway_path = case_dir / f"{case_str}_CT_HR_label_airways.nii.gz"

    # Cases with an airway mask are treated as fully labelled inputs.
    if airway_path.exists():
        labelled_paths: LabelledCasePaths = {
            "case_id": case_str,
            "case_dir": case_dir,
            "ct": ct_path,
            "lung": lung_path,
            "airway": airway_path
        }
        validate_case_paths(labelled_paths)
        return labelled_paths
    
    unlabelled_paths: UnlabelledCasePaths = {
        "case_id": case_str,
        "case_dir": case_dir,
        "ct": ct_path,
        "lung": lung_path if lung_path.exists() else None,
        "airway": None
        }
    return unlabelled_paths

def validate_case_paths(paths: CasePaths) -> None:
    if not paths["case_dir"].is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {paths['case_dir']}")
    
    if not paths["ct"].is_file():
        raise FileNotFoundError(f"CT file does not exist: {paths['ct']}")

    lung_path = paths["lung"]
    if lung_path is not None and not lung_path.is_file():
        raise FileNotFoundError(f"Lung mask does not exist: {lung_path}")
    
    airway_path = paths["airway"]
    if airway_path is not None and not airway_path.is_file():
        raise FileNotFoundError(f"Airway mask does not exist: {airway_path}")
