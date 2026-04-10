"""Nibabel-based helpers for loading and inspecting NIfTI images.

This module provides the image-level operations used by the preprocessing
pipeline: loading files, canonicalizing orientation, converting images into
NumPy arrays, and extracting spatial metadata such as affine matrices, voxel
spacing, and header-derived properties. It assumes that dataset-specific path
resolution has already happened elsewhere.
"""

from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
from nibabel.orientations import aff2axcodes

from lung_airway_segmentation.schemas import Spacing3D, ImageMetadata


def load_nifti(path) -> Nifti1Image:
    image = nib.load(str(path))

    if not isinstance(image, Nifti1Image):
        raise TypeError(f"Expected Nifti1Image, got {type(image).__name__}")
    
    return image

def ensure_3d(image, name):
    if len(image.shape) != 3:
        raise ValueError(f"{name} must be 3D, but got shape {image.shape}.")


def load_canonical_image(path) -> Nifti1Image:
    image = load_nifti(path)
    canonical_image = nib.as_closest_canonical(image)

    if not isinstance(canonical_image, Nifti1Image):
        raise TypeError(f"Expected canonical image to be Nifti1Image, got {type(canonical_image).__name__} ")

    return canonical_image

def affine_from_image(image: SpatialImage, name) -> np.ndarray:
    affine = image.affine
    if affine is None:
        raise ValueError(f"{name} does not have an affine matrix")
    return np.asarray(affine, dtype=np.float64)

def spacing_from_image(image: SpatialImage) -> Spacing3D:
    zooms = image.header.get_zooms()[:3]
    if len(zooms) != 3:
        raise ValueError(f"Expected 3 spatial zooms, but got {zooms}")
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))

def image_to_array(image: SpatialImage, dtype=None) -> np.ndarray:
    return np.asarray(image.dataobj, dtype=dtype)

def load_image_array(path, dtype=None) -> np.ndarray:
    canonical_image = load_canonical_image(path)
    return image_to_array(canonical_image, dtype=dtype)


def get_image_metadata(image: Nifti1Image) -> ImageMetadata:
    """Return header-derived metadata for an already loaded NIfTI image."""
    header = image.header
    return ImageMetadata(
        shape=image.shape,
        dtype=image.get_data_dtype(),
        zooms=tuple(float(z) for z in header.get_zooms()[:3]),
        xyzt_units=header.get_xyzt_units(),
        qform_code=int(header["qform_code"]),
        sform_code=int(header["sform_code"]),
        orientation=aff2axcodes(image.affine),
    )

def load_image_metadata(path) -> ImageMetadata:
    """Load a canonical image and return its metadata in one step."""
    image = load_canonical_image(path)
    return get_image_metadata(image)

def verify_alignment(
    reference_image,
    other_image,
    *,
    reference_name,
    other_name,
    atol=1e-4,
    rtol=1e-5,
) -> None:
    if reference_image.shape != other_image.shape:
        raise ValueError(
            f"{other_name} is not aligned with {reference_name}: "
            f"shape {other_image.shape} != {reference_image.shape}."
        )

    reference_affine = affine_from_image(reference_image, reference_name)
    other_affine = affine_from_image(other_image, other_name)

    if not np.allclose(reference_affine, other_affine, atol=atol, rtol=rtol):
        raise ValueError(
            f"{other_name} is not aligned with {reference_name}: affine matrices differ."
        )
