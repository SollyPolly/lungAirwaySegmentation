"""Build the MEAN-TEACHER nnU-Net raw dataset: real GT for the 20 labelled, IGNORE for the 90.

This is the ONLINE semi-supervised (Mean-Teacher) arm of the SSL label-efficiency study, the
counterpart to the OFFLINE self-training rung (Dataset121). Same 110-case pool as @110/@121;
the difference is how the 90 "unlabelled" cases are handled:

  - self-training (Dataset121): the 90 carry a PSEUDO-label baked into labelsTr (offline).
  - mean-teacher  (Dataset122): the 90 carry an all-IGNORE label, so nnU-Net's supervised
    DC+CE loss skips their voxels (via LabelManager.ignore_label) and the Mean-Teacher trainer
    consumes them ONLINE through the EMA-teacher consistency loss instead.

nnU-Net's dataset.json requires every imagesTr case to have a matching labelsTr file, so we
cannot simply omit labels for the 90. The ignore label is nnU-Net's native mechanism for
un-/partially-annotated data: it is declared as the HIGHEST label value and masked out of the
supervised loss. We therefore write, for each of the 90, a label volume filled entirely with
the ignore index (2), matched to the CT geometry.

⚠️ WHY THIS SCRIPT EXISTS (do not substitute convert_atm_to_nnunet):
   The 90 "unlabelled" cases are ATM cases whose GT we WITHHOLD — the GT is still on disk.
   convert_atm_to_nnunet / a plain export would take every label from batch_root and silently
   leak the oracle labels. This script takes GT ONLY for the labelled 20 and writes an ignore
   label for every one of the 90. The sealed TEST and held-out VAL are never exported (asserted).

Usage (same split/seed as the @20 floor so the arms are paired):
    python -u -m scripts.build_meanteacher_nnunet_dataset \
      --training-config configs/training/supervised_atm.yaml \
      --nnunet-raw "$nnUNet_raw" --dataset-id 122 --dataset-name ATM22MT

Then (plans MOVED from 111, not re-planned — see PROJECT_STATE / hpc-nnunet-new-dataset-recipe):
    nnUNetv2_extract_fingerprint -d 122
    cp "$nnUNet_raw/Dataset122_ATM22MT/dataset.json" "$nnUNet_preprocessed/Dataset122_ATM22MT/dataset.json"
    nnUNetv2_move_plans_between_datasets -s 111 -t 122 -sp nnUNetPlans -tp nnUNetPlans
    nnUNetv2_preprocess -d 122 -plans_name nnUNetPlans -c 3d_fullres -np 6   # mem>=96gb
"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from lung_airway_segmentation.datasets.splits import create_semisupervised_split
from lung_airway_segmentation.io.atm22_layout import list_case_ids, resolve_case_paths
from lung_airway_segmentation.io.nnunet_export import _place, nnunet_dataset_json
from lung_airway_segmentation.training.config import load_yaml_config, resolve_project_path

# nnU-Net convention: the ignore label is the highest integer value. background=0, airway=1.
IGNORE_INDEX = 2
MT_LABELS = {"background": 0, "airway": 1, "ignore": IGNORE_INDEX}


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


def _write_ignore_label(ct_path: Path, dst: Path) -> tuple[int, ...]:
    """Write an all-ignore label matched to the CT geometry. Returns the CT shape.

    No CT voxels are loaded (only header/affine), so this stays memory-light; a constant
    volume also compresses to almost nothing on disk.
    """
    ct = nib.load(str(ct_path))
    ignore = np.full(ct.shape, IGNORE_INDEX, dtype=np.uint8)
    out = nib.Nifti1Image(ignore, ct.affine, ct.header)
    out.set_data_dtype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    return tuple(ct.shape)


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
        if paths["ct"] is None:
            raise SystemExit(f"ATM case {cid} has no CT.")
        _place(Path(paths["ct"]), images_dir / f"ATM_{padded}_0000.nii.gz", args.mode)

        if cid in set(labelled):
            if paths["airway"] is None:
                raise SystemExit(f"Labelled case {cid} has no GT airway mask.")
            _place(Path(paths["airway"]), labels_dir / f"ATM_{padded}.nii.gz", args.mode)
            provenance[padded] = "gt"
            continue

        # Unlabelled: never take the on-disk GT — write an all-ignore label instead.
        shape = _write_ignore_label(Path(paths["ct"]), labels_dir / f"ATM_{padded}.nii.gz")
        provenance[padded] = "ignore"
        print(f"  ignore {padded}: all-{IGNORE_INDEX} label ({'x'.join(map(str, shape))})", flush=True)

    (out_dir / "dataset.json").write_text(json.dumps(
        nnunet_dataset_json(len(pool), labels=MT_LABELS), indent=2))
    # Provenance is the audit trail that this is NOT the oracle @110 dataset.
    (out_dir / "label_provenance.json").write_text(json.dumps({
        "method": "mean_teacher_ignore_label",
        "ignore_index": IGNORE_INDEX,
        "num_gt": sum(v == "gt" for v in provenance.values()),
        "num_ignore": sum(v == "ignore" for v in provenance.values()),
        "excluded_val": sorted(split["val"]),
        "excluded_test": sorted(split["test"]),
        "labels": provenance,
    }, indent=2))

    n_gt = sum(v == "gt" for v in provenance.values())
    n_ig = sum(v == "ignore" for v in provenance.values())
    print(f"\nDataset{args.dataset_id:03d}_{args.dataset_name}: {len(pool)} cases "
          f"({n_gt} GT + {n_ig} ignore) -> {out_dir}")
    print(f"Excluded VAL (held-out): {split['val']}")
    print(f"Excluded TEST (sealed):  {split['test']}")
    print("\nNext (plans MOVED, not re-planned — plan_and_preprocess would clobber them):\n"
          f"  nnUNetv2_extract_fingerprint -d {args.dataset_id}\n"
          f"  cp {out_dir / 'dataset.json'} $nnUNet_preprocessed/Dataset{args.dataset_id:03d}_{args.dataset_name}/dataset.json\n"
          f"  nnUNetv2_move_plans_between_datasets -s 111 -t {args.dataset_id} -sp nnUNetPlans -tp nnUNetPlans\n"
          f"  nnUNetv2_preprocess -d {args.dataset_id} -plans_name nnUNetPlans -c 3d_fullres -np 6\n"
          f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0 "
          f"-tr nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring "
          f"-pretrained_weights <@20 Dataset120 fold_0 checkpoint_final.pth>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", type=Path, default=Path("configs/data/atm22.yaml"))
    ap.add_argument("--training-config", type=Path, required=True,
                    help="Provides labelled_split counts + seed (use the same one as the @20 floor).")
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--dataset-id", type=int, default=122)
    ap.add_argument("--dataset-name", default="ATM22MT")
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: cap the number of labelled+unlabelled cases assembled (local smoke test).")
    args = ap.parse_args()

    if not args.nnunet_raw:
        raise SystemExit("Set --nnunet-raw or export $nnUNet_raw.")

    data_config = load_yaml_config(args.data_config)
    training_config = load_yaml_config(args.training_config)
    batch_root, split = _resolve_split(data_config, training_config)

    if args.limit is not None:
        # Keep a couple of each provenance class for a cheap local integrity check.
        keep_lab = list(split["labelled_train"])[: max(1, args.limit // 2)]
        keep_unl = list(split["unlabelled_train"])[: max(1, args.limit - len(keep_lab))]
        split = dict(split, labelled_train=keep_lab, unlabelled_train=keep_unl)

    assemble(args, batch_root, split)


if __name__ == "__main__":
    main()
