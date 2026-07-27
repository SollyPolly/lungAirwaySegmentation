# Active configuration

Only configuration used by the nnU-Net workflows remains here:

- `data/atm22.yaml`: local/HPC ATM'22 dataset root.
- `nnunet/atm22_split_l20.yaml`: explicit frozen Batch-1 membership for the
  existing 20-label + 90-unlabelled experiments. Batch-2 files may coexist in
  `data/ATM22`, but this manifest deliberately excludes them.
- `nnunet/atm22_split_l20_u240.yaml`: expanded, still-frozen membership for
  20 labelled + 240 unlabelled cases while preserving the same external
  validation, sealed test, and four-case nnU-Net fold-0 validation. It also
  owns the explicit stored-value intensity override for the ten Batch-2
  `uint16` CTs.

Model architecture, augmentation, optimisation, and plans are owned by nnU-Net
and its trainer classes under `nnunet_trainers/`; they do not use the retired
MONAI model/training YAMLs.
