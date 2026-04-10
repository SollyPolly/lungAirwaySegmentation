# Project Structure

This document defines the target layout for the repo as it grows from
preprocessing and visualization into a full airway-segmentation research codebase.

## Design Rules

- Keep one responsibility per module.
- Put reusable logic in the package, not in notebooks or one-off scripts.
- Let CLI scripts stay thin: they should call package functions instead of
  holding core logic themselves.
- Avoid rebuilding file-path rules, cropping logic, metrics, or losses in more
  than one place.

## Target Tree

```text
lungAirwaySegmentation/
  configs/
    data/
    model/
    training/
  docs/
  learning/
  scripts/
  tests/
  lung_airway_segmentation/
    io/
    preprocessing/
    datasets/
    models/
    losses/
    metrics/
    training/
    inference/
    visualization/
```

## How Current Files Map Into The New Layout

- `preprocessing.py`
  - split low-level file handling into `lung_airway_segmentation/io/`
  - split crop and affine math into `lung_airway_segmentation/preprocessing/geometry.py`
  - split HU clipping and normalization into `lung_airway_segmentation/preprocessing/intensity.py`
  - keep orchestration in `lung_airway_segmentation/preprocessing/pipeline.py`
- `utils.py`
  - move NIfTI metadata helpers into `lung_airway_segmentation/io/nifti.py`
  - move slice display helpers into `lung_airway_segmentation/visualization/slices.py`
- `mask_visualisation.py`
  - move reusable plotting and mesh logic into
    `lung_airway_segmentation/visualization/`
  - keep notebook or app-specific UI wiring outside the reusable modules

## Suggested Build Order

1. Move preprocessing helpers into the new package without changing behavior.
2. Add tests for crop boxes, affine updates, and intensity normalization.
3. Add dataset wrappers and config loading.
4. Add the baseline model and training loop.
5. Add topology-aware targets, losses, and metrics.
6. Add semi-supervised training, inference fusion, and distal refinement.

## Folder Ownership

- `configs/`
  - experiment settings only
- `scripts/`
  - thin entrypoints only
- `tests/`
  - behavioral verification only
- `lung_airway_segmentation/io/`
  - all path, NIfTI, JSON, CSV, and artifact I/O
- `lung_airway_segmentation/preprocessing/`
  - raw CT and mask preparation
- `lung_airway_segmentation/datasets/`
  - dataset classes, splits, patch sampling
- `lung_airway_segmentation/models/`
  - network definitions only
- `lung_airway_segmentation/losses/`
  - training losses only
- `lung_airway_segmentation/metrics/`
  - evaluation metrics only
- `lung_airway_segmentation/training/`
  - loops, checkpointing, logging, teacher-student training
- `lung_airway_segmentation/inference/`
  - sliding-window inference, postprocessing, stage fusion
- `lung_airway_segmentation/visualization/`
  - reusable slice and mesh helpers

## Migration Note

The new package is a scaffold, not a forced rewrite. Keep the current
root-level scripts working while you gradually move stable logic into the new
modules.
