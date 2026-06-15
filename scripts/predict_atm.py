"""Save viewer-compatible NIfTI predictions for ATM'22 cases (raw + LCC).

Writes the same layout mask_visualisation.py expects, so saved runs appear in the
prediction-run dropdown:

    <run-dir>/predictions/<case>/
        prediction_metadata.json     (gate file the viewer looks for)
        airway_prob_full.nii.gz      probability [0,1]
        airway_pred_full.nii.gz      binary prediction at --threshold
        airway_pred_lcc_full.nii.gz  + largest connected component (--connectivity)

CT/GT are loaded in canonical (RAS+) orientation via load_canonical_image — the
SAME path the viewer uses — so the saved masks line up with the CT it displays
(it re-loads CT/GT from the original data and checks shapes match).

Usage:
    python -m scripts.predict_atm --run-dir runs/supervised-atm-l20-cldice/<run> \
        --cases 002,012 --threshold 0.90
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from lung_airway_segmentation.inference.postprocess import keep_component_containing_trachea
from lung_airway_segmentation.inference.sliding_window import predict_logits_for_volume
from lung_airway_segmentation.io.atm22_layout import resolve_case_paths
from lung_airway_segmentation.io.nifti import load_canonical_image
from lung_airway_segmentation.training.builders import build_model, resolve_checkpoint_path
from lung_airway_segmentation.training.config import resolve_device, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated IDs; default: first 2 test cases.")
    parser.add_argument("--threshold", type=float, default=0.90, help="Binarisation threshold (LCC operating point ~0.90).")
    parser.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=6, help="LCC connectivity (6 = face).")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap (0.5 gives cleaner masks than 0.25).")
    parser.add_argument("--checkpoint", choices=("best", "dice", "topology", "last"), default="best",
                        help="best = Dice-selected (alias of dice); topology = hard-clDice@0.5 selection; last = final epoch.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prediction-set", type=str, default="predictions", help="Subfolder name under the run dir.")
    return parser.parse_args()


def save_nifti(array: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(array), affine), str(path))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "resolved_config.json").read_text())
    metadata = json.loads((run_dir / "run_metadata.json").read_text())

    data_config = cfg["data"]
    if str(data_config["dataset_name"]).lower() != "atm22":
        raise SystemExit("predict_atm only supports ATM'22 runs (data.dataset_name == atm22).")
    lo, hi = (float(v) for v in data_config["preprocessing"]["hu_window"])
    atm_root = resolve_project_path(data_config["batch_root"])

    validation = cfg["training"]["validation"]
    roi_size = tuple(int(v) for v in validation["roi_size"])

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = [str(c) for c in metadata.get("splits", {}).get("test_case_ids", [])][:2]
    if not cases:
        raise SystemExit("No cases: pass --cases or ensure run_metadata has a test split.")

    device = resolve_device(args.device)
    ckpt_path = resolve_checkpoint_path(run_dir, args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(device, cfg["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    checkpoint_epoch = checkpoint.get("epoch")

    print(f"model: {metadata.get('experiment_name')}  | cases: {cases}")
    print(f"threshold {args.threshold}  | LCC-{args.connectivity}  | overlap {args.overlap}\n")

    for cid in cases:
        paths = resolve_case_paths(cid, batch_root=atm_root)
        if paths["airway"] is None:
            raise FileNotFoundError(f"ATM case {cid} has no airway label.")

        # Canonical (RAS+) orientation — matches mask_visualisation.py's CT/GT loading.
        ct_image = load_canonical_image(paths["ct"])
        affine = np.asarray(ct_image.affine, dtype=np.float64)
        ct = np.asarray(ct_image.dataobj, dtype=np.float32)
        gt = np.asarray(load_canonical_image(paths["airway"]).dataobj) > 0

        ct_norm = np.clip((ct - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        logits = predict_logits_for_volume(
            model, torch.from_numpy(ct_norm), device=device, roi_size=roi_size,
            sw_batch_size=4, overlap=args.overlap, use_amp=False,
        )
        prob = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
        if prob.shape != ct.shape:
            raise ValueError(f"prob shape {prob.shape} != CT shape {ct.shape}")
        pred = (prob >= args.threshold).astype(np.uint8)
        pred_lcc = keep_component_containing_trachea(
            pred, connectivity=args.connectivity, affine=affine
        ).astype(np.uint8)

        def dice(a):
            return float(2 * int((a.astype(bool) & gt).sum()) / ((int(a.sum()) + int(gt.sum())) or 1))

        case_dir = run_dir / args.prediction_set / str(cid)
        save_nifti(prob, affine, case_dir / "airway_prob_full.nii.gz")
        save_nifti(pred, affine, case_dir / "airway_pred_full.nii.gz")
        save_nifti(pred_lcc, affine, case_dir / "airway_pred_lcc_full.nii.gz")
        (case_dir / "prediction_metadata.json").write_text(json.dumps({
            "case_id": str(cid),
            "checkpoint_epoch": checkpoint_epoch,
            "threshold": args.threshold,
            "lcc_connectivity": args.connectivity,
            "overlap": args.overlap,
            "checkpoint_path": str(ckpt_path),
        }, indent=2))
        print(f"case {cid}: Dice raw {dice(pred):.3f} -> LCC {dice(pred_lcc):.3f}  "
              f"| pred voxels {int(pred.sum()):,} -> LCC {int(pred_lcc.sum()):,}  | {case_dir}")


if __name__ == "__main__":
    main()
