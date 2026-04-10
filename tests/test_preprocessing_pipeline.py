"""Integration checks for the preprocessing pipeline.

This module exercises the full case-level preprocessing flow on a real AeroPath
case when the dataset is available locally. It verifies that path resolution,
image loading, spatial cropping, CT normalization, and metadata assembly remain
consistent when used together.
"""

import numpy as np
import pytest

from lung_airway_segmentation.preprocessing.pipeline import preprocess_case
from lung_airway_segmentation.settings import RAW_AEROPATH_ROOT


CASE_1_DIR = RAW_AEROPATH_ROOT / "1"


@pytest.mark.skipif(
    not CASE_1_DIR.exists(),
    reason="AeroPath case 1 is not available in the local data directory.",
)
def test_preprocess_case_smoke_check_on_real_case():
    case = preprocess_case(1, include_lung_mask=True)

    assert case.case_id == "1"
    assert case.ct.ndim == 3
    assert case.airway_mask is not None
    assert case.airway_mask.ndim == 3
    assert case.lung_mask is not None
    assert case.lung_mask.ndim == 3

    assert case.ct.shape == case.airway_mask.shape
    assert case.ct.shape == case.lung_mask.shape
    assert case.affine.shape == (4, 4)
    assert case.crop_box is not None
    assert len(case.spacing) == 3

    assert np.isfinite(case.ct).all()
    assert case.ct.dtype == np.float32
    assert case.ct.min() >= 0.0
    assert case.ct.max() <= 1.0

    assert case.metadata["supervision"] == "labeled"
    assert case.metadata["case_dir"] == CASE_1_DIR
    assert case.metadata["ct_path"] == CASE_1_DIR / "1_CT_HR.nii.gz"
    assert case.metadata["lung_mask_path"] == CASE_1_DIR / "1_CT_HR_label_lungs.nii.gz"
    assert case.metadata["airway_mask_path"] == CASE_1_DIR / "1_CT_HR_label_airways.nii.gz"
    assert case.metadata["original_shape"] == (512, 512, 767)
    assert case.metadata["processed_shape"] == case.ct.shape
    assert case.metadata["crop_margin"] == (5, 5, 5)
    assert case.metadata["hu_window"] == (-1024.0, 600.0)
    assert case.metadata["crop_source"] == "lung_mask"
