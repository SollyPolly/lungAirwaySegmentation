"""Build an nnU-Net predict-input folder from our ATM cases (CTs named <id>_0000.nii.gz).

Our val/test cases are excluded from the training export (Dataset111 = train only), so there
is no ready folder to predict on. This links each case's CT into <out-dir> as
``ATM_<id>_0000.nii.gz`` (channel 0), which ``nnUNetv2_predict -i <out-dir>`` consumes; the
ensemble then writes ``ATM_<id>.nii.gz``, ready for ``scripts.evaluate_nnunet_predictions``.

A single bare command (PBS-friendly) — replaces the inline python blob in the predict flow.
Develop on val, seal test: default ``--report-split val``.

Usage:
    python -u -m scripts.make_nnunet_predict_input \
      --split-run-dir runs/atm-l110-supervised/2026-06-26__06-12-47__cldice-w1-cbdice-w2-p96-l110__baseline_unet \
      --report-split val --out-dir data/nnunet/predict_in/val
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from lung_airway_segmentation.io.atm22_layout import resolve_case_paths, resolve_lung_mask_path
from lung_airway_segmentation.io.nnunet_lungcrop import write_lung_roi_ct
from lung_airway_segmentation.training.config import load_yaml_config, resolve_project_path

_SPLIT_KEYS = {"val": "val_case_ids", "test": "test_case_ids", "train": "train_case_ids"}


def _place(src: str, dst: Path, mode: str) -> None:
    """Symlink (default) or copy src -> dst, replacing any existing dst; copy fallback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:  # Windows without symlink privilege / cross-device
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True, help="nnU-Net predict-input folder to create.")
    ap.add_argument("--data-config", type=Path, default=Path("configs/data/atm22.yaml"),
                    help="ATM'22 data YAML (batch_root). Default: configs/data/atm22.yaml.")
    ap.add_argument("--split-run-dir", type=Path, default=None,
                    help="A run dir whose run_metadata.json holds the split (val/test/train ids).")
    ap.add_argument("--report-split", choices=("val", "test", "train"), default="val",
                    help="Which split to build (default val — develop on val, seal test).")
    ap.add_argument("--cases", type=str, default=None, help="Comma-separated ids (overrides --split-run-dir).")
    ap.add_argument("--prefix", type=str, default="ATM_", help="Filename prefix (default 'ATM_').")
    ap.add_argument("--mode", choices=("symlink", "copy"), default="symlink",
                    help="Link (default, saves space) or copy the CTs.")
    ap.add_argument("--lung-roi", action="store_true",
                    help="Zero outside the affine-aware lung/trachea ROI instead of linking the full CT.")
    ap.add_argument("--lung-root", type=Path, default=None, help="Lung masks (default <batch_root>/lungTr).")
    ap.add_argument("--margin-voxels", type=int, default=8)
    ap.add_argument("--superior-margin-voxels", type=int, default=120)
    args = ap.parse_args()

    if args.report_split == "test" and not args.cases:
        print("WARNING: building input for the SEALED TEST split — final runs only.", flush=True)

    batch_root = resolve_project_path(load_yaml_config(args.data_config)["batch_root"])
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    elif args.split_run_dir:
        meta = json.loads((args.split_run_dir / "run_metadata.json").read_text())
        cases = [str(c) for c in meta.get("splits", {}).get(_SPLIT_KEYS[args.report_split], [])]
    else:
        raise SystemExit("Provide --cases or --split-run-dir (to read the split from run_metadata.json).")
    if not cases:
        raise SystemExit(f"No {args.report_split} case ids resolved.")

    roi_records = {}
    for cid in cases:
        paths = resolve_case_paths(cid, batch_root=batch_root)
        if paths["ct"] is None:
            raise FileNotFoundError(f"ATM case {cid} has no CT.")
        destination = args.out_dir / f"{args.prefix}{paths['case_id']}_0000.nii.gz"
        if args.lung_roi:
            lung_path = resolve_lung_mask_path(
                paths["case_id"], batch_root=batch_root, lung_root=args.lung_root
            )
            if not lung_path.is_file():
                raise FileNotFoundError(f"Precomputed lung mask not found: {lung_path}")
            roi_records[f"ATM_{paths['case_id']}"] = write_lung_roi_ct(
                Path(paths["ct"]),
                lung_path,
                destination,
                margin_voxels=args.margin_voxels,
                superior_margin_voxels=args.superior_margin_voxels,
            )
        else:
            _place(str(paths["ct"]), destination, args.mode)

    if args.lung_roi:
        (args.out_dir / "lung_crop_manifest.json").write_text(json.dumps({
            "method": "full_grid_zero_outside_lung_bbox",
            "margin_voxels": args.margin_voxels,
            "superior_margin_voxels": args.superior_margin_voxels,
            "cases": roi_records,
        }, indent=2), encoding="utf-8")

    mode = "lung-roi" if args.lung_roi else args.mode
    print(f"placed {len(cases)} CT(s) -> {args.out_dir}  (split={args.report_split}, mode={mode})")


if __name__ == "__main__":
    main()
