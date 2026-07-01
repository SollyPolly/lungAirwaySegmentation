"""Generate topology-filtered pseudo-labels for the unlabelled ATM'22 pool.

The first stage of topology-filtered self-training (SSL). A frozen supervised
labeller (default: the cl1+cb2 topology checkpoint) predicts airway masks for the
90 *unlabelled* ATM'22 cases, each mask is cleaned with trachea-seeded LCC, and a
per-case quality gate decides which masks are trustworthy enough to train on.

Unlike ``predict_atm.py`` (which loads canonical RAS+ for the viewer), this writes
masks in the CT's **native** orientation + affine + shape, so the labelled training
pipeline's ``LoadImaged`` aligns CT and pseudo-mask voxel-for-voxel (the trainer reads
the native ATM files; canonical reorientation would misalign).

Output layout (consumed by scripts/train_selftraining.py via the manifest):

    <run-dir>/<out>/manifest.json                  accept/reject + stats + provenance
    <run-dir>/<out>/<case>/airway_pseudo_full.nii.gz   uint8 mask, native space

Usage:
    python -m scripts.pseudo_label_atm --run-dir runs/atm-l20-supervised/<cl1+cb2> \
        --checkpoint topology --threshold 0.60
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from lung_airway_segmentation.inference.postprocess import keep_component_containing_trachea
from lung_airway_segmentation.inference.sliding_window import predict_logits_for_volume
from lung_airway_segmentation.io.atm22_layout import resolve_case_paths
from lung_airway_segmentation.training.builders import build_model, resolve_case_splits, resolve_checkpoint_path
from lung_airway_segmentation.training.config import resolve_device, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Labeller run directory (the frozen supervised model).")
    parser.add_argument("--checkpoint", choices=("best", "dice", "topology", "last"), default="topology",
                        help="Which labeller checkpoint to predict with (default: topology).")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Binarisation threshold for the pseudo-label (default 0.60: cleaner/higher-precision labels).")
    parser.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=6, help="Trachea-LCC connectivity.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap.")
    parser.add_argument("--sw-batch", type=int, default=4, help="Sliding-window batch size (raise on big GPUs to speed inference).")
    parser.add_argument("--amp", action="store_true",
                        help="fp16 autocast for inference (~2x faster on modern GPUs). Default OFF for bit-comparable probabilities.")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated IDs to pseudo-label; default: the resolved unlabelled-train split.")
    # Case-level quality gate (no GT available). Permissive defaults; tune from the manifest.
    parser.add_argument("--min-lcc-frac", type=float, default=0.30,
                        help="Reject if trachea-LCC keeps < this fraction of the raw prediction (fragmented tree).")
    parser.add_argument("--min-vox", type=int, default=20000, help="Reject if the LCC mask is smaller than this (empty/broken).")
    parser.add_argument("--max-vox", type=int, default=2000000, help="Reject if the LCC mask is larger than this (blow-up/leak).")
    parser.add_argument("--out", type=str, default="pseudo_labels", help="Subfolder under the run dir for masks + manifest.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def save_nifti(array: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(array), affine), str(path))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "resolved_config.json").read_text())

    data_config = cfg["data"]
    training_config = cfg["training"]
    if str(data_config["dataset_name"]).lower() != "atm22":
        raise SystemExit("pseudo_label_atm only supports ATM'22 runs (data.dataset_name == atm22).")
    lo, hi = (float(v) for v in data_config["preprocessing"]["hu_window"])
    atm_root = resolve_project_path(data_config["batch_root"])
    roi_size = tuple(int(v) for v in training_config["validation"]["roi_size"])

    # The unlabelled-train split is the only legitimate pseudo-label pool (val/test
    # are sealed). resolve_case_splits is deterministic from labelled_split + seed.
    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = resolve_case_splits(data_config, training_config)["unlabelled_train"]
    if not cases:
        raise SystemExit("No unlabelled cases resolved; pass --cases or check labelled_split.")

    device = resolve_device(args.device)
    ckpt_path = resolve_checkpoint_path(run_dir, args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(device, cfg["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    out_dir = run_dir / args.out
    print(f"labeller: {run_dir.name} [{args.checkpoint} ep{checkpoint.get('epoch')}]"
          f"  | threshold {args.threshold}  LCC-{args.connectivity}  overlap {args.overlap}  sw_batch {args.sw_batch}  amp {bool(args.amp)}")
    print(f"pseudo-labelling {len(cases)} cases -> {out_dir}\n")

    entries = []
    accepted = 0
    for cid in cases:
        paths = resolve_case_paths(cid, batch_root=atm_root)
        # Native orientation (NOT canonical) so the mask aligns with the trainer's LoadImaged.
        ct_image = nib.load(str(paths["ct"]))
        affine = np.asarray(ct_image.affine, dtype=np.float64)
        ct = np.asarray(ct_image.dataobj, dtype=np.float32)

        ct_norm = np.clip((ct - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        logits = predict_logits_for_volume(
            model, torch.from_numpy(ct_norm), device=device, roi_size=roi_size,
            sw_batch_size=args.sw_batch, overlap=args.overlap, use_amp=args.amp,
        )
        prob = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
        if prob.shape != ct.shape:
            raise ValueError(f"case {cid}: prob shape {prob.shape} != CT shape {ct.shape}")

        pred = (prob >= args.threshold).astype(np.uint8)
        pred_lcc = keep_component_containing_trachea(
            pred, connectivity=args.connectivity, affine=affine
        ).astype(np.uint8)

        raw_vox = int(pred.sum())
        lcc_vox = int(pred_lcc.sum())
        lcc_frac = float(lcc_vox / raw_vox) if raw_vox else 0.0

        reasons = []
        if lcc_vox < args.min_vox:
            reasons.append(f"lcc_vox<{args.min_vox}")
        if lcc_vox > args.max_vox:
            reasons.append(f"lcc_vox>{args.max_vox}")
        if lcc_frac < args.min_lcc_frac:
            reasons.append(f"lcc_frac<{args.min_lcc_frac}")
        accept = not reasons

        if accept:
            save_nifti(pred_lcc, affine, out_dir / str(cid) / "airway_pseudo_full.nii.gz")
            accepted += 1

        entries.append({
            "case_id": str(cid),
            "accepted": accept,
            "reject_reasons": reasons,
            "raw_vox": raw_vox,
            "lcc_vox": lcc_vox,
            "lcc_retained_fraction": lcc_frac,
            "mask_path": (str(out_dir / str(cid) / "airway_pseudo_full.nii.gz") if accept else None),
        })
        flag = "OK " if accept else "REJECT"
        print(f"  {flag} case {cid}: raw {raw_vox:,} -> LCC {lcc_vox:,} "
              f"(kept {lcc_frac:.2f})" + (f"  [{', '.join(reasons)}]" if reasons else ""))

    manifest = {
        "labeller_run": str(run_dir),
        "labeller_run_name": run_dir.name,
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "threshold": args.threshold,
        "connectivity": args.connectivity,
        "overlap": args.overlap,
        "gate": {"min_lcc_frac": args.min_lcc_frac, "min_vox": args.min_vox, "max_vox": args.max_vox},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_total": len(entries),
        "n_accepted": accepted,
        "accepted_case_ids": sorted(e["case_id"] for e in entries if e["accepted"]),
        "cases": entries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\naccepted {accepted}/{len(entries)} cases -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
