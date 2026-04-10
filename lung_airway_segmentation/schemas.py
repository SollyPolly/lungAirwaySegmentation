"""Shared type aliases and structured containers used across the package.

This module holds the small, reusable schema definitions that multiple layers
of the project depend on: path records, shape aliases, preprocessing metadata,
and lightweight in-memory result objects. It is intentionally declarative and
does not contain pipeline logic.
"""

from pathlib import Path
from typing import TypedDict, Literal
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


Spacing3D = tuple[float, float, float]
Shape3D = tuple[int, int, int]
CropBox3D = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

FloatVolume = npt.NDArray[np.float32]
BinaryMask = npt.NDArray[np.uint8]
Affine4x4 = npt.NDArray[np.float64]

SupervisionKind = Literal["labeled", "unlabeled", "pseudo_labeled"]
CropSource = Literal["lung_mask", "full_volume", "external_roi"]


class LabelledCasePaths(TypedDict):
    case_id: str
    case_dir: Path
    ct: Path
    lung: Path
    airway: Path

class UnlabelledCasePaths(TypedDict):
    case_id: str
    case_dir: Path
    ct: Path
    lung: Path | None
    airway: None

CasePaths = LabelledCasePaths | UnlabelledCasePaths


class PreprocessedMetadata(TypedDict):
    supervision: SupervisionKind
    case_dir: Path
    ct_path: Path
    lung_mask_path: Path | None
    airway_mask_path: Path | None
    original_shape: Shape3D
    processed_shape: Shape3D
    spacing: Spacing3D
    original_affine: Affine4x4
    cropped_affine: Affine4x4
    crop_margin: Shape3D
    hu_window: tuple[float, float]
    crop_source: CropSource

@dataclass(frozen=True)
class PreprocessedCase:
    case_id: str
    ct: np.ndarray
    airway_mask: np.ndarray | None
    lung_mask: np.ndarray | None
    spacing: tuple[float, float, float]
    affine: np.ndarray
    crop_box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None
    metadata: PreprocessedMetadata

@dataclass(frozen=True)
class ImageMetadata:
    shape: tuple[int, ...]
    dtype: np.dtype
    zooms: tuple[float, ...]
    xyzt_units: tuple[str, str]
    qform_code: int
    sform_code: int
    orientation: tuple[str | None, ...]

class SavedCasePaths(TypedDict):
    case_dir: Path
    ct: Path
    lung_mask: Path | None
    airway_mask: Path | None
    metadata_json: Path

@dataclass(frozen=True)
class PseudoLabelAtrifacts:
    airway_mask: BinaryMask
    confidence_map: FloatVolume | None = None
    uncertainty_map: FloatVolume | None  = None
