"""Case-level preprocessing orchestration.

This module ties together path resolution, NIfTI loading, spatial cropping, CT
normalization, and metadata assembly to produce a consistent preprocessed case
record. It owns the case-level workflow while delegating low-level geometry,
intensity, and image I/O details to the dedicated helper modules.
"""
import numpy as np

from lung_airway_segmentation.io.case_layout import resolve_case_paths
from lung_airway_segmentation.io.nifti import (
    load_canonical_image, ensure_3d, image_to_array, 
    spacing_from_image, affine_from_image, verify_alignment
)

from lung_airway_segmentation.preprocessing.geometry import (
    crop_box_from_mask, crop_volume, affine_after_crop, 
    validate_3d_shape, normalize_margin
)

from lung_airway_segmentation.preprocessing.intensity import normalize_ct

from lung_airway_segmentation.settings import (
    RAW_AEROPATH_ROOT, 
    DEFAULT_HU_WINDOW, 
    DEFAULT_CROP_MARGIN
)
from lung_airway_segmentation.schemas import PreprocessedCase, PreprocessedMetadata



def preprocess_case(
        case_id,
        data_root=RAW_AEROPATH_ROOT,
        include_lung_mask=False,
        hu_window=DEFAULT_HU_WINDOW,
        crop_margin=DEFAULT_CROP_MARGIN
) -> PreprocessedCase:
    """Load, crop, and normalize one labelled AeroPath case at native spacing."""

    paths = resolve_case_paths(case_id, data_root=data_root)

    # This preprocessing path is AeroPath-specific today: it assumes labelled
    # airway masks are present and uses the lung mask to define the crop ROI.
    if paths["lung"] is None or paths["airway"] is None:
        raise ValueError(f"Case {case_id} is missing required masks for labelled preprocessing.")
    
    ct_img = load_canonical_image(paths["ct"])
    lung_img = load_canonical_image(paths["lung"])
    airway_img = load_canonical_image(paths["airway"])
    
    # Validate image structures before array conversion.
    ensure_3d(ct_img, "CT")
    ensure_3d(lung_img, "lung mask")
    ensure_3d(airway_img, "airway mask")

    # Verify images are alligned.
    verify_alignment(ct_img, lung_img, reference_name="CT", other_name="lung mask")
    verify_alignment(ct_img, airway_img, reference_name="CT", other_name="airway mask")

    # Record metadata that is defined on the original CT image.
    spacing = spacing_from_image(ct_img)
    original_affine = affine_from_image(ct_img, "CT")

    # Load arrays only after image-level checks have passed.
    ct_raw = image_to_array(ct_img, dtype=np.float32)
    lung_raw = image_to_array(lung_img)
    airway_raw = image_to_array(airway_img)
    
    # Downstream code expects binary masks rather than label intensities.
    lung_mask = (lung_raw > 0).astype(np.uint8, copy=False)
    airway_mask = (airway_raw > 0).astype(np.uint8, copy=False)
    
    # AeroPath preprocessing currently uses the lung mask as the shared crop ROI.
    crop_box = crop_box_from_mask(lung_mask, crop_margin=crop_margin)
    
    ct_cropped = crop_volume(ct_raw, crop_box)
    lung_mask_cropped = crop_volume(lung_mask, crop_box)
    airway_mask_cropped = crop_volume(airway_mask, crop_box)
    
    # Cropping changes the voxel origin, so the affine must be translated too.
    affine_crop = affine_after_crop(original_affine, crop_box)
    
    # Intensity normalization happens after cropping to avoid extra work.
    ct_cropped_normalized = normalize_ct(ct_cropped, hu_window)

    metadata = build_preprocessed_metadata(
        paths=paths,
        spacing=spacing,
        original_affine=original_affine,
        cropped_affine=affine_crop,
        ct_raw=ct_raw,
        ct_cropped=ct_cropped,
        crop_margin=crop_margin,
        hu_window=hu_window,
    )

    return PreprocessedCase(
        case_id = str(case_id),
        ct=ct_cropped_normalized,
        airway_mask=airway_mask_cropped,
        lung_mask=lung_mask_cropped if include_lung_mask else None,
        spacing=spacing,
        affine=affine_crop,
        crop_box=crop_box,
        metadata=metadata
    )


def build_preprocessed_metadata(
    *,
    paths,
    spacing,
    original_affine,
    cropped_affine,
    ct_raw,
    ct_cropped,
    crop_margin,
    hu_window,
) -> PreprocessedMetadata:
    """Assemble the metadata record stored alongside a preprocessed case."""
    return {
        "supervision": "labeled",
        "case_dir": paths["case_dir"],
        "ct_path": paths["ct"],
        "lung_mask_path": paths["lung"],
        "airway_mask_path": paths["airway"],
        "original_shape": validate_3d_shape(ct_raw, "CT"),
        "processed_shape": validate_3d_shape(ct_cropped, "cropped CT"),
        "spacing": spacing,
        "original_affine": original_affine,
        "cropped_affine": cropped_affine,
        "crop_margin": normalize_margin(crop_margin),
        "hu_window": hu_window,
        "crop_source": "lung_mask",
    }
