# Active configuration

Only configuration used by the nnU-Net workflows remains here:

- `data/atm22.yaml`: local/HPC ATM'22 dataset root.
- `nnunet/atm22_split_l20.yaml`: the sealed seed-15 split shared by the
  20-label nnU-Net, Mean Teacher, and pseudo-label/self-training arms.

Model architecture, augmentation, optimisation, and plans are owned by nnU-Net
and its trainer classes under `nnunet_trainers/`; they do not use the retired
MONAI model/training YAMLs.
