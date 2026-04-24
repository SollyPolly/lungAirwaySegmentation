"""Rebuild the shared runs/run_index.csv table from saved run artifacts."""

import argparse
from pathlib import Path

from lung_airway_segmentation.reporting.run_index import RUN_INDEX_FILENAME, collect_run_index_rows, refresh_run_index
from lung_airway_segmentation.settings import RUNS_ROOT


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for run-index regeneration."""
    parser = argparse.ArgumentParser(
        description="Rebuild the shared run_index.csv summary from saved run directories.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT,
        help="Root directory containing saved training runs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=f"Optional explicit CSV output path. Defaults to <runs-root>/{RUN_INDEX_FILENAME}.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    output_path = refresh_run_index(
        runs_root=args.runs_root.resolve(),
        output_path=args.output_path.resolve() if args.output_path is not None else None,
    )
    row_count = len(collect_run_index_rows(args.runs_root.resolve()))

    print(f"Run index written to: {output_path}")
    print(f"Indexed runs: {row_count}")


if __name__ == "__main__":
    main()
