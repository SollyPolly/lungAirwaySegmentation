"""Precompute distal crop-class maps to disk (airway hard-mining).

Writes a uint8 NIfTI per labelled ATM'22 case, aligned to its airway mask, with
``0 = background`` / ``1 = proximal airway`` / ``2 = distal skeleton`` (EDT radius
<= ``--radius``). Training loads these instead of skeletonising every epoch, which
removes the per-epoch ``skeletonize`` + EDT (the transient-allocation source behind
the OOM) and speeds every epoch. Run ONCE per (dataset, radius) before launching a
distal-sampling run; re-running skips files that already exist (use ``--overwrite``).

Behaviour-preserving: the supervised pipeline does not resample, and the CT-foreground
crop never cuts the airway, so skeletonising the native-resolution mask here reproduces
the legacy on-the-fly ``crop_classes`` exactly (see ``compute_distal_crop_classes``).

Example:
    python -u -m scripts.precompute_distal_classes --batch-root data/ATM22 --radius 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.datasets.monai_atm22 import compute_distal_crop_classes
from lung_airway_segmentation.io.atm22_layout import (
    list_case_ids,
    resolve_case_paths,
    resolve_distal_classes_path,
)
from lung_airway_segmentation.training.config import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--batch-root",
        required=True,
        help="ATM'22 root containing imagesTr/ + labelsTr/ (project-relative paths are resolved).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="EDT radius cutoff in voxels (<= this on the skeleton = distal). Default 2.0.",
    )
    parser.add_argument(
        "--classes-root",
        default=None,
        help="Output directory (default: <batch-root>/distalTr). Must be writable.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Specific case ids (e.g. 016 027); default = every labelled case under batch-root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even when the output map already exists.",
    )
    args = parser.parse_args()

    if args.radius <= 0.0:
        parser.error("--radius must be positive.")

    batch_root = resolve_project_path(args.batch_root)
    classes_root = resolve_project_path(args.classes_root) if args.classes_root else None
    case_ids = args.case_ids if args.case_ids else list_case_ids(batch_root)

    written = skipped = no_label = 0
    for case_id in case_ids:
        paths = resolve_case_paths(case_id, batch_root=batch_root)
        if paths["airway"] is None:
            no_label += 1
            continue

        out_path = resolve_distal_classes_path(
            paths["case_id"], batch_root=batch_root, radius=args.radius, classes_root=classes_root
        )
        if out_path.is_file() and not args.overwrite:
            skipped += 1
            continue

        mask_img = nib.load(str(paths["airway"]))
        mask = np.asanyarray(mask_img.dataobj)
        classes = compute_distal_crop_classes(mask, distal_radius_voxels=args.radius)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Fresh header from the mask's affine (no scl_slope/inter rescaling on reload);
        # same grid as the mask so the index-based CropForegroundd/SpatialPadd stay aligned.
        out_img = nib.Nifti1Image(classes.astype(np.uint8), affine=mask_img.affine)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(out_path))
        written += 1
        n_distal = int((classes == 2).sum())
        n_airway = int((classes >= 1).sum())
        print(f"  {paths['case_id']}: {out_path.name}  distal={n_distal}  airway={n_airway}")

    resolved_root = classes_root if classes_root is not None else (batch_root / "distalTr")
    print(
        f"Done. written={written} skipped={skipped} no_label={no_label} "
        f"radius={args.radius:g} -> {resolved_root}"
    )


if __name__ == "__main__":
    main()
