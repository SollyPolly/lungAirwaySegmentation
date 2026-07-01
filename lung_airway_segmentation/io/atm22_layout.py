"""Path Resolution for the ATM'22 airway segmentation dataset.

ATM'22 uses a flat layout:
    <batch_root>/
        imagesTr/ ATM_001_0000.nii.gz
        labelsTr/ ATM_001.nii.gz or ATM_001_0000.nii.gz
"""

import re
from pathlib import Path

from lung_airway_segmentation.schemas import CasePaths, LabelledCasePaths, UnlabelledCasePaths

_ID_PATTERN = re.compile(r"^ATM_(\d{3})_0000\.nii\.gz$")

def list_case_ids(batch_root: Path) -> list[str]:
    images_dir = Path(batch_root) / "imagesTr"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"ATM'22 imagesTr not found: {images_dir}")
    
    ids = []
    for f in images_dir.iterdir():
        m = _ID_PATTERN.match(f.name)
        if m:
            ids.append(m.group(1))

    if not ids:
        raise ValueError(f"No ATM cases found in {images_dir}")
    
    return sorted(ids)

def resolve_case_paths(case_id: str, batch_root: Path) -> CasePaths:
    batch_root = Path(batch_root)
    padded = str(case_id).zfill(3)

    ct_path = batch_root / "imagesTr" / f"ATM_{padded}_0000.nii.gz"
    label_candidates = (
        batch_root / "labelsTr" / f"ATM_{padded}.nii.gz",
        batch_root / "labelsTr" / f"ATM_{padded}_0000.nii.gz",
    )
    airway_path = next((path for path in label_candidates if path.is_file()), None)

    if not ct_path.is_file():
        raise FileNotFoundError(f"ATM'22 CT not found: {ct_path}")
    
    if airway_path is not None:
        return LabelledCasePaths(
            case_id=padded,
            case_dir=batch_root,
            ct=ct_path,
            lung=None,
            airway=airway_path,
        )

    return UnlabelledCasePaths(
        case_id=padded,
        case_dir=batch_root,
        ct=ct_path,
        lung=None,
        airway=None,
    )


def resolve_lung_mask_path(
    case_id: str,
    *,
    batch_root: Path,
    lung_root: Path | None = None,
) -> Path:
    """Path to the precomputed binary lung mask for a case.

    Written by ``scripts/precompute_lung_masks.py`` (lungmask) and loaded at train time
    to crop CT + airway to the lung bounding box (tighter than the CT-foreground crop).
    Defaults to a ``lungTr/`` sibling of the images/labels dirs (must be writable;
    override with ``lung_root``).
    """
    padded = str(case_id).zfill(3)
    root = Path(lung_root) if lung_root is not None else Path(batch_root) / "lungTr"
    return root / f"ATM_{padded}_lung.nii.gz"


def resolve_distal_classes_path(
    case_id: str,
    *,
    batch_root: Path,
    radius: float,
    classes_root: Path | None = None,
) -> Path:
    """Path to the precomputed distal crop-class map for a case + EDT radius.

    Written by ``scripts/precompute_distal_classes.py`` and loaded at train time
    (see ``build_atm22_labelled_transforms``) instead of skeletonising every epoch.
    Keyed by ``radius`` because the distal class (2) depends on the EDT cutoff, so
    different radii must not collide. Defaults to a ``distalTr/`` sibling of the
    images/labels dirs (must be writable; override with ``classes_root``).
    """
    padded = str(case_id).zfill(3)
    root = Path(classes_root) if classes_root is not None else Path(batch_root) / "distalTr"
    return root / f"ATM_{padded}_distal_r{radius:g}.nii.gz"
