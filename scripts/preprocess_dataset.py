"""Dataset preprocessing CLI.

This script should stay thin and do only command-line orchestration.

What to put here:
- parse a config or CLI flags
- iterate cases
- call `lung_airway_segmentation.preprocessing.pipeline`
- save outputs and summary artifacts

What not to put here:
- crop-box math
- NIfTI loading internals
- HU normalization logic
"""


def main() -> None:
    print("TODO: wire this script to the preprocessing pipeline.")


if __name__ == "__main__":
    main()
