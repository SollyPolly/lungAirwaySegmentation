"""Build the SELF-TRAINING nnU-Net raw dataset: real GT for the 20 labelled, PSEUDO for the 90.

This is the SSL label-efficiency experiment. The @20 floor (Dataset120) and the @110 ceiling
(Dataset111) are both scored; this dataset is the middle rung — the same 110-case pool as @110,
but 90 of its labels come from the @20 model instead of the annotator. Training fold 0 on it with
the identical trainer/plans makes "fraction of the label gap recovered" a directly readable number.

⚠️ WHY THIS SCRIPT EXISTS (do not substitute convert_atm_to_nnunet):
   The 90 "unlabelled" cases are ATM cases whose labels we WITHHOLD — the GT is still on disk.
   convert_atm_to_nnunet takes every label from batch_root, so pointing it at the 110-case pool
   would export the ORACLE labels and silently reproduce the @110 ceiling. This script takes GT
   ONLY for the labelled 20 and requires a pseudo-mask for every one of the 90.

The sealed TEST and the held-out VAL are never exported (asserted, not just intended).

Stage 1 — CTs of the 90 unlabelled cases, to predict pseudo-labels from:
    python -u -m scripts.build_selftraining_nnunet_dataset \
      --training-config configs/training/supervised_atm.yaml \
      --emit-predict-input data/nnunet/predict_in/unlabelled90

Stage 2 — assemble the dataset once those predictions exist:
    python -u -m scripts.build_selftraining_nnunet_dataset \
      --training-config configs/training/supervised_atm.yaml \
      --assemble --pseudo-dir data/nnunet/predict_out/Dataset120_unlabelled90_dice \
      --nnunet-raw "$nnUNet_raw" --dataset-id 121 --dataset-name ATM22SSL
"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.datasets.splits import create_semisupervised_split
from lung_airway_segmentation.inference.postprocess import keep_component_containing_trachea
from lung_airway_segmentation.io.atm22_layout import list_case_ids, resolve_case_paths
from lung_airway_segmentation.io.nnunet_export import _place, nnunet_dataset_json
from lung_airway_segmentation.training.config import load_yaml_config, resolve_project_path


def _resolve_split(data_config, training_config):
    batch_root = resolve_project_path(data_config["batch_root"])
    counts = training_config["labelled_split"]
    split = create_semisupervised_split(
        list_case_ids(batch_root),
        test_count=int(counts["test_count"]),
        val_count=int(counts["val_count"]),
        labelled_count=int(counts["labelled_count"]),
        seed=int(training_config.get("seed", 15)),
    )
    return batch_root, split


def _clean_pseudo(mask: np.ndarray, affine) -> np.ndarray:
    """Trachea-LCC the pseudo-label: keep the connected tree, drop free-floating FP blobs.

    Matched to how every prediction in this project is scored (evaluate_nnunet_predictions
    reports +LCC), so the pseudo-labels carry the same postprocessing the metrics assume.
    Cleaning is applied ONLY to pseudo-labels; real GT is passed through untouched.
    """
    return np.asarray(keep_component_containing_trachea(mask, affine=affine) > 0, dtype=np.uint8)


def emit_predict_input(out_dir: Path, batch_root, unlabelled, mode: str) -> None:
    for cid in unlabelled:
        paths = resolve_case_paths(cid, batch_root=batch_root)
        if paths["ct"] is None:
            raise FileNotFoundError(f"ATM case {cid} has no CT.")
        _place(Path(paths["ct"]), out_dir / f"ATM_{paths['case_id']}_0000.nii.gz", mode)
    print(f"placed {len(unlabelled)} unlabelled CT(s) -> {out_dir}  (mode={mode})")
    print("Next: nnUNetv2_predict from the @20 Dice model into a pseudo-label dir, then --assemble.")


def assemble(args, batch_root, split) -> None:
    labelled, unlabelled = list(split["labelled_train"]), list(split["unlabelled_train"])
    sealed = set(split["test"]) | set(split["val"])
    pool = labelled + unlabelled
    leaked = sealed & set(pool)
    if leaked:
        raise SystemExit(f"REFUSING: val/test cases in the training pool: {sorted(leaked)}")

    out_dir = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_dir, labels_dir = out_dir / "imagesTr", out_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    provenance = {}
    for cid in pool:
        paths = resolve_case_paths(cid, batch_root=batch_root)
        padded = paths["case_id"]
        _place(Path(paths["ct"]), images_dir / f"ATM_{padded}_0000.nii.gz", args.mode)

        if cid in set(labelled):
            if paths["airway"] is None:
                raise SystemExit(f"Labelled case {cid} has no GT airway mask.")
            _place(Path(paths["airway"]), labels_dir / f"ATM_{padded}.nii.gz", args.mode)
            provenance[padded] = "gt"
            continue

        # Unlabelled: the label MUST come from the pseudo-dir, never from batch_root.
        pseudo_path = Path(args.pseudo_dir) / f"ATM_{padded}.nii.gz"
        if not pseudo_path.is_file():
            raise SystemExit(
                f"No pseudo-label for unlabelled case {cid} at {pseudo_path}. "
                f"Run nnUNetv2_predict over the 90 unlabelled CTs first."
            )
        img = nib.load(str(pseudo_path))
        mask = np.asanyarray(img.dataobj) > 0
        kept = mask.astype(np.uint8)
        if not args.no_lcc:
            kept = _clean_pseudo(mask, img.affine)
        before, after = int(mask.sum()), int(kept.sum())
        out = nib.Nifti1Image(kept, img.affine, img.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, str(labels_dir / f"ATM_{padded}.nii.gz"))
        provenance[padded] = "pseudo"
        print(f"  pseudo {padded}: {before:,} -> {after:,} vox "
              f"(lcc kept {after / max(before, 1):.3f})", flush=True)

    (out_dir / "dataset.json").write_text(json.dumps(nnunet_dataset_json(len(pool)), indent=2))
    # Provenance is the audit trail that this is NOT the oracle @110 dataset.
    (out_dir / "label_provenance.json").write_text(json.dumps({
        "seed_model": args.seed_model,
        "pseudo_dir": str(args.pseudo_dir),
        "pseudo_postprocessing": "none" if args.no_lcc else "trachea_lcc",
        "num_gt": sum(v == "gt" for v in provenance.values()),
        "num_pseudo": sum(v == "pseudo" for v in provenance.values()),
        "excluded_val": sorted(split["val"]),
        "excluded_test": sorted(split["test"]),
        "labels": provenance,
    }, indent=2))

    n_gt = sum(v == "gt" for v in provenance.values())
    n_ps = sum(v == "pseudo" for v in provenance.values())
    print(f"\nDataset{args.dataset_id:03d}_{args.dataset_name}: {len(pool)} cases "
          f"({n_gt} GT + {n_ps} pseudo) -> {out_dir}")
    print(f"Excluded VAL (held-out): {split['val']}")
    print(f"Excluded TEST (sealed):  {split['test']}")
    print("\nNext (plans must be MOVED, not re-planned — plan_and_preprocess would clobber them):\n"
          f"  nnUNetv2_move_plans_between_datasets -s 111 -t {args.dataset_id} "
          f"-sp nnUNetPlans -tp nnUNetPlans\n"
          f"  nnUNetv2_preprocess -d {args.dataset_id} -plans_name nnUNetPlans -c 3d_fullres -np 8\n"
          f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0 "
          f"-tr nnUNetTrainer_NoDeepSupervision_NoMirroring")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", type=Path, default=Path("configs/data/atm22.yaml"))
    ap.add_argument("--training-config", type=Path, required=True,
                    help="Provides labelled_split counts + seed (use the same one as the @20 floor).")
    ap.add_argument("--emit-predict-input", type=Path, default=None,
                    help="Stage 1: write the 90 unlabelled CTs to this folder and exit.")
    ap.add_argument("--assemble", action="store_true", help="Stage 2: build the raw dataset.")
    ap.add_argument("--pseudo-dir", type=Path, default=None,
                    help="nnU-Net predictions for the 90 unlabelled cases (required with --assemble).")
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--dataset-id", type=int, default=121)
    ap.add_argument("--dataset-name", default="ATM22SSL")
    ap.add_argument("--seed-model", default="Dataset120 nnUNetTrainer_NoDeepSupervision_NoMirroring fold 0",
                    help="Recorded in label_provenance.json — which model produced the pseudo-labels.")
    ap.add_argument("--no-lcc", action="store_true",
                    help="Do NOT trachea-LCC the pseudo-labels (default: clean them).")
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    args = ap.parse_args()

    data_config = load_yaml_config(args.data_config)
    training_config = load_yaml_config(args.training_config)
    batch_root, split = _resolve_split(data_config, training_config)

    if args.emit_predict_input:
        emit_predict_input(args.emit_predict_input, batch_root, split["unlabelled_train"], args.mode)
        return
    if not args.assemble:
        raise SystemExit("Pick a stage: --emit-predict-input <dir>  or  --assemble --pseudo-dir <dir>.")
    if not args.pseudo_dir:
        raise SystemExit("--assemble requires --pseudo-dir (the pseudo-labels for the 90).")
    if not args.nnunet_raw:
        raise SystemExit("Set --nnunet-raw or export $nnUNet_raw.")
    assemble(args, batch_root, split)


if __name__ == "__main__":
    main()
