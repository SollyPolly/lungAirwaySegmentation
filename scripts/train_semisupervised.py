"""Semi-supervised training CLI.

This entrypoint should own the teacher-student or related semi-supervised
experiment setup.

What to put here:
- load labeled and unlabeled splits
- build student and teacher models
- launch uncertainty-aware training

What not to put here:
- baseline-only training
- generic metric implementations
- low-level patch-sampling code
"""

from lung_airway_segmentation.training.config import (
    build_semisupervised_argument_parser,
    build_semisupervised_config_path_parser,
)
from lung_airway_segmentation.training.engine import run_semisupervised_training


def main() -> None:
    config_path_parser = build_semisupervised_config_path_parser()
    config_args, _ = config_path_parser.parse_known_args()
    args = build_semisupervised_argument_parser(config_args).parse_args()
    run_semisupervised_training(args)

if __name__ == "__main__":
    main()
