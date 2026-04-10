"""Offline target-generation CLI.

This script should build derived supervision targets that are expensive enough
to precompute once and reuse in training.

What to put here:
- centerline target generation
- distance transform or signed-distance target generation
- anatomy-aware label derivation

What not to put here:
- model definitions
- training loops
- visualization code
"""


def main() -> None:
    print("TODO: build offline centerline, distance, and anatomy targets.")


if __name__ == "__main__":
    main()
