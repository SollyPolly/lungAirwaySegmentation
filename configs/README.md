# Config Layout

Use this folder to keep experiment settings separate from implementation.

Rules:

- `configs/data/`: dataset roots, splits, preprocessing defaults
- `configs/model/`: architecture and channel/depth choices
- `configs/training/`: optimizer, scheduler, batch size, epochs, losses
- keep each config focused on one concern
- prefer composing small configs over one huge file

The main reason for this folder is repeatability: you should be able to rerun
an experiment from a saved config without editing Python code.

`scripts/train_baseline.py` now reads one YAML from each of these groups:

- `configs/data/` for dataset roots
- `configs/model/` for the shared model definition
- `configs/training/` for the training regime and optimizer settings

The CLI is intentionally small and should only be used for explicit runtime
overrides such as device selection, experiment naming, or short smoke-test
changes.
