"""Gate native nnU-Net airway masks to a lung-bbox ROI without overwriting them.

The ROI matches the training-time ``lung_with_trachea_extension`` crop: a bounding
box around a CT-derived lungmask, a modest margin on all non-superior faces, and a
larger superior extension for the cervical trachea.  Output masks remain on the full
input grid, with voxels outside the box set to zero, so the standard external-mask
evaluator can score them directly.

Example:
    python -u -m scripts.apply_nnunet_lung_roi \
      --pred-dir data/nnunet/predict_out/Dataset122_val_mt_warm_sym \
      --out-dir data/nnunet/predict_out/Dataset122_val_mt_warm_sym_lungroi_m8_s120 \
      --batch-root data/ATM22
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.inference.postprocess import lung_bbox_slices
from lung_airway_segmentation.io.atm22_layout import resolve_lung_mask_path


_PREDICTION_PATTERN = re.compile(r"^ATM_(\d{3})\.nii(?:\.gz)?$", re.IGNORECASE)


def _prediction_cases(pred_dir: Path) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    for path in pred_dir.iterdir():
        if not path.is_file():
            continue
        match = _PREDICTION_PATTERN.match(path.name)
        if match:
            cases.append((match.group(1), path))
    cases = sorted(cases)
    case_ids = [case_id for case_id, _ in cases]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(
            f"Multiple native prediction files found for cases: {', '.join(duplicates)}. "
            "Keep exactly one of .nii or .nii.gz for each case."
        )
    return cases


def _same_geometry(first: nib.Nifti1Image, second: nib.Nifti1Image) -> bool:
    return first.shape == second.shape and np.allclose(first.affine, second.affine, atol=1e-5)


def _zero_outside(volume: np.ndarray, bounds: tuple[slice, slice, slice]) -> None:
    """Zero the complement of an axis-aligned box in place without another full array."""
    for axis, bound in enumerate(bounds):
        if bound.start:
            index = [slice(None)] * 3
            index[axis] = slice(0, bound.start)
            volume[tuple(index)] = 0
        if bound.stop < volume.shape[axis]:
            index = [slice(None)] * 3
            index[axis] = slice(bound.stop, volume.shape[axis])
            volume[tuple(index)] = 0


def apply_case(
    prediction_path: Path,
    lung_path: Path,
    output_path: Path,
    *,
    margin_voxels: int,
    superior_margin_voxels: int,
    overwrite: bool,
) -> dict:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    prediction_image = nib.load(str(prediction_path))
    lung_image = nib.load(str(lung_path))
    if not _same_geometry(prediction_image, lung_image):
        raise ValueError(
            f"Prediction/lung geometry mismatch: {prediction_path.name} {prediction_image.shape} "
            f"vs {lung_path.name} {lung_image.shape}."
        )

    lung = np.asarray(lung_image.dataobj, dtype=np.uint8)
    bounds = lung_bbox_slices(
        lung,
        affine=prediction_image.affine,
        margin_voxels=margin_voxels,
        superior_margin_voxels=superior_margin_voxels,
    )
    del lung
    if bounds is None:
        raise ValueError(f"Lung mask is empty: {lung_path}")

    gated = np.asarray(prediction_image.dataobj, dtype=np.uint8)
    np.not_equal(gated, 0, out=gated)
    original_voxels = int(gated.sum(dtype=np.int64))
    _zero_outside(gated, bounds)
    retained_voxels = int(gated.sum(dtype=np.int64))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = prediction_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(gated, prediction_image.affine, header), str(output_path))

    roi_voxels = int(np.prod([bound.stop - bound.start for bound in bounds], dtype=np.int64))
    full_voxels = int(np.prod(prediction_image.shape, dtype=np.int64))
    return {
        "prediction": str(prediction_path),
        "lung_mask": str(lung_path),
        "output": str(output_path),
        "bbox": [[int(bound.start), int(bound.stop)] for bound in bounds],
        "roi_fraction": roi_voxels / full_voxels,
        "prediction_voxels_before": original_voxels,
        "prediction_voxels_after": retained_voxels,
        "prediction_retained_fraction": retained_voxels / max(original_voxels, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True, help="Native nnU-Net prediction directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for full-grid ROI-gated masks.")
    parser.add_argument("--batch-root", type=Path, required=True, help="ATM'22 root containing lungTr/.")
    parser.add_argument("--lung-root", type=Path, default=None, help="Optional override for precomputed lung masks.")
    parser.add_argument("--case-ids", nargs="*", default=None, help="Optional case subset; default is every ATM mask.")
    parser.add_argument("--margin-voxels", type=int, default=8, help="Non-superior bbox margin (default: 8).")
    parser.add_argument(
        "--superior-margin-voxels", type=int, default=120,
        help="Superior/trachea bbox extension (default: 120).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace files already present in --out-dir.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pred_dir = args.pred_dir.resolve()
    out_dir = args.out_dir.resolve()
    batch_root = args.batch_root.resolve()
    lung_root = args.lung_root.resolve() if args.lung_root else None
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
    if pred_dir == out_dir:
        raise ValueError("--out-dir must differ from --pred-dir; source masks are never overwritten.")
    if args.margin_voxels < 0 or args.superior_margin_voxels < 0:
        raise ValueError("ROI margins must be >= 0.")

    available = dict(_prediction_cases(pred_dir))
    requested = [str(case_id).zfill(3) for case_id in args.case_ids] if args.case_ids else sorted(available)
    if not requested:
        raise ValueError(f"No native ATM prediction masks found in {pred_dir}")
    duplicate_requests = sorted({case_id for case_id in requested if requested.count(case_id) > 1})
    if duplicate_requests:
        raise ValueError(f"Duplicate --case-ids: {', '.join(duplicate_requests)}")
    missing = [case_id for case_id in requested if case_id not in available]
    if missing:
        raise FileNotFoundError(f"Missing source predictions for cases: {', '.join(missing)}")

    rows = []
    for case_id in requested:
        lung_path = resolve_lung_mask_path(case_id, batch_root=batch_root, lung_root=lung_root)
        if not lung_path.is_file():
            raise FileNotFoundError(f"Missing lung mask for case {case_id}: {lung_path}")
        source_path = available[case_id]
        output_path = out_dir / source_path.name
        row = apply_case(
            source_path,
            lung_path,
            output_path,
            margin_voxels=args.margin_voxels,
            superior_margin_voxels=args.superior_margin_voxels,
            overwrite=args.overwrite,
        )
        row["case_id"] = case_id
        rows.append(row)
        print(
            f"{case_id}: retained {row['prediction_retained_fraction']:.2%} prediction foreground; "
            f"ROI is {row['roi_fraction']:.2%} of the original volume",
            flush=True,
        )

    manifest = {
        "operation": "lung_bbox_roi_with_superior_trachea_extension",
        "source_prediction_dir": str(pred_dir),
        "output_dir": str(out_dir),
        "batch_root": str(batch_root),
        "lung_root": str(lung_root) if lung_root else str(batch_root / "lungTr"),
        "parameters": {
            "margin_voxels": int(args.margin_voxels),
            "superior_margin_voxels": int(args.superior_margin_voxels),
        },
        "cases": rows,
        "mean_roi_fraction": float(np.mean([row["roi_fraction"] for row in rows])),
        "mean_prediction_retained_fraction": float(
            np.mean([row["prediction_retained_fraction"] for row in rows])
        ),
    }
    manifest_path = out_dir / "lung_roi_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(rows)} gated masks and manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
