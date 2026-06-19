"""Topology-filtered self-training CLI.

Retrains the supervised patch model on the labelled cases (oversampled) plus the
accepted pseudo-labelled unlabelled cases produced by ``scripts/pseudo_label_atm.py``.
Config-first: the loss recipe, split, and the ``selftraining`` block live in the
training YAML (default: configs/training/selftraining_atm.yaml); the CLI exposes
only the pseudo-label directory, the labelled oversample, and an optional warm-start.

Usage:
    python -m scripts.pseudo_label_atm --run-dir runs/.../<cl1+cb2> --checkpoint topology --threshold 0.60
    python -m scripts.train_selftraining \
        --data-config configs/data/atm22.yaml \
        --pseudo-label-dir runs/.../<cl1+cb2>/pseudo_labels \
        --init-checkpoint runs/.../<cl1+cb2>/best_topology_model.pt
"""

from lung_airway_segmentation.training.config import (
    build_selftraining_argument_parser,
    build_selftraining_config_path_parser,
)
from lung_airway_segmentation.training.engine import run_selftraining_training


def main() -> None:
    config_path_parser = build_selftraining_config_path_parser()
    config_args, _ = config_path_parser.parse_known_args()
    args = build_selftraining_argument_parser(config_args).parse_args()
    run_selftraining_training(args)


if __name__ == "__main__":
    main()
