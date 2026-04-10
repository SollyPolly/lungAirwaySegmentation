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
