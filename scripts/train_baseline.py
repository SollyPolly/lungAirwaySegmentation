"""Train the supervised baseline airway segmentation model.

This script is config-first: dataset, model, and training defaults come from
YAML files under ``configs/`` and the CLI only exposes a small set of explicit
runtime overrides. The same shared model, loss, and training loop can therefore
be compared across patch-based training and full-volume ablations by swapping
training configs rather than editing Python code.
"""

from lung_airway_segmentation.training.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    build_argument_parser,
    build_config_path_parser,
)
from lung_airway_segmentation.training.engine import run_supervised_training


def main() -> None:
    config_path_parser = build_config_path_parser()
    config_args, _ = config_path_parser.parse_known_args()
    args = build_argument_parser(config_args).parse_args()
    run_supervised_training(args)


if __name__ == "__main__":
    main()
