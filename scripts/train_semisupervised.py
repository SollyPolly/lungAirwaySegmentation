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


def main() -> None:
    print("TODO: train the semi-supervised model.")


if __name__ == "__main__":
    main()
