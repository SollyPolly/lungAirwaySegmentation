"""Convert ATM'22 -> nnU-Net v2 raw dataset (Track A: stock nnU-Net baseline-strength control).

Separates the PIPELINE gap from the METHOD gap and pre-empts the "weak baseline" critique:
nnU-Net's rule-based pipeline (plan_and_preprocess -> train 5-fold -> predict) trains on the
SAME ATM data, then we score it on OUR held-out val/test with OUR topology metrics. nnU-Net
optimises Dice (topology-blind), so it is a CONTROL, not the clDice method.

TEST HYGIENE: the sealed TEST split (seed 15) is NEVER exported. VAL is also excluded by
default so nnU-Net can be evaluated on our held-out val (dev) + test (sealed) cleanly.

Case pool (default = every labelled case except our val+test = the strong control):
  --only-labelled : export ONLY our 20 labelled_train cases (data-matched to our runs)
  --include-val   : also export our val cases (NOT recommended — loses the held-out dev set)

Example (HPC, with nnUNet_raw exported):
  python -u -m scripts.convert_atm_to_nnunet \
    --data-config configs/data/atm22.yaml \
    --training-config configs/training/supervised_atm_topoloss_cldice_w1_cbdice_w2.yaml \
    --nnunet-raw "$nnUNet_raw" --dataset-id 111 --mode symlink
Then:
  nnUNetv2_plan_and_preprocess -d 111 --verify_dataset_integrity
  nnUNetv2_train 111 3d_fullres 0   # ... folds 0..4
"""

import argparse
import os
from pathlib import Path

from lung_airway_segmentation.datasets.splits import create_semisupervised_split
from lung_airway_segmentation.io.atm22_layout import list_case_ids
from lung_airway_segmentation.io.nnunet_export import export_atm_to_nnunet
from lung_airway_segmentation.training.config import load_yaml_config, resolve_project_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", type=Path, required=True,
                    help="ATM'22 data YAML (provides batch_root).")
    ap.add_argument("--training-config", type=Path, required=True,
                    help="Training YAML (provides labelled_split counts + seed for the sealed split).")
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"),
                    help="nnU-Net raw root (default: $nnUNet_raw).")
    ap.add_argument("--dataset-id", type=int, default=111, help="nnU-Net dataset id (-> DatasetXXX_...).")
    ap.add_argument("--dataset-name", default="ATM22", help="nnU-Net dataset name suffix.")
    ap.add_argument("--only-labelled", action="store_true",
                    help="Export ONLY our 20 labelled_train cases (data-matched control).")
    ap.add_argument("--include-val", action="store_true",
                    help="Also export our val cases (NOT recommended — loses the held-out dev set).")
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink",
                    help="How to place images/labels (symlink saves the 30 GB copy on HPC).")
    args = ap.parse_args()

    if not args.nnunet_raw:
        raise SystemExit("Set --nnunet-raw or export the nnUNet_raw environment variable.")

    data_config = load_yaml_config(args.data_config)
    training_config = load_yaml_config(args.training_config)
    batch_root = resolve_project_path(data_config["batch_root"])
    labelled_split = training_config["labelled_split"]
    split = create_semisupervised_split(
        list_case_ids(batch_root),
        test_count=int(labelled_split["test_count"]),
        val_count=int(labelled_split["val_count"]),
        labelled_count=int(labelled_split["labelled_count"]),
        seed=int(training_config.get("seed", 15)),
    )

    if args.only_labelled:
        pool = list(split["labelled_train"])
    else:
        pool = list(split["labelled_train"]) + list(split["unlabelled_train"])
    if args.include_val:
        pool += list(split["val"])
    # Belt-and-braces: the sealed test is NEVER exported.
    pool = sorted(set(pool) - set(split["test"]))

    out_dir = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    summary = export_atm_to_nnunet(pool, batch_root=batch_root, out_dir=out_dir, mode=args.mode)

    print(f"Exported {summary['num_training']} cases ({args.mode}) -> {summary['dataset_dir']}")
    print(f"Excluded TEST (sealed, never exported): {split['test']}")
    if not args.include_val:
        print(f"Excluded VAL (held-out for eval): {split['val']}")
    print(f"\nNext:\n"
          f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity\n"
          f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0   # ... folds 0..4")


if __name__ == "__main__":
    main()
