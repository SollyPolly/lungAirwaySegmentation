# Lung Airway Segmentation

This repository contains the nnU-Net workflows used for ATM'22 airway
segmentation, including supervised training, online Mean Teacher training,
offline pseudo-label/self-training, lung-ROI preparation, topology evaluation,
and prediction visualisation.

## Active layout

- `nnunet_trainers/`: custom nnU-Net trainers, including the two-stream Mean
  Teacher implementation and matched controls.
- `scripts/`: ATM-to-nnU-Net conversion, MT/SSL dataset creation, prediction
  preparation, lung-ROI handling, evaluation, report merging, and viewer export.
- `lung_airway_segmentation/`: shared ATM paths/splits, nnU-Net export and
  lung-ROI helpers, topology metrics, post-processing, and viewer utilities.
- `configs/data/atm22.yaml`: ATM'22 dataset location.
- `configs/nnunet/atm22_split_l20.yaml`: canonical sealed split for the
  label-efficiency and SSL experiments.
- `docs/NNUNET_LUNGCROP_MT_EXPERIMENT.md`: current lung-ROI Mean Teacher protocol.

The retired simple MONAI U-Net experiments are stored only in the local,
Git-ignored `legacy/` mirror. They are not part of an HPC checkout.

## Typical workflow

Create a supervised nnU-Net dataset:

```powershell
python -m scripts.convert_atm_to_nnunet `
  --data-config configs/data/atm22.yaml `
  --training-config configs/nnunet/atm22_split_l20.yaml `
  --nnunet-raw $env:nnUNet_raw --dataset-id 111
```

Create the lung-ROI supervised and Mean Teacher pair:

```powershell
python -m scripts.build_lungcrop_meanteacher_nnunet `
  --data-config configs/data/atm22.yaml `
  --training-config configs/nnunet/atm22_split_l20.yaml
```

Create predict inputs and score native nnU-Net masks:

```powershell
python -m scripts.make_nnunet_predict_input `
  --report-split val --out-dir data/nnunet/predict_in/val

python -m scripts.evaluate_nnunet_predictions `
  --pred-dir data/nnunet/predict_out/Dataset111_val `
  --report-split val --branch
```

Launch the local prediction viewer with:

```powershell
marimo run mask_visualisation.py
```
