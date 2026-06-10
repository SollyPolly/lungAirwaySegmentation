"""Project-wide paths and default settings.

This file should become the single home for stable project defaults so they are
not redefined across scripts, notebooks, and modules.

"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_ROOT = PROJECT_ROOT / "data"
RAW_AEROPATH_ROOT = DATA_ROOT / "AeroPath"
RAW_ATM22_ROOT = DATA_ROOT / "ATM22"

PROCESSED_ROOT = DATA_ROOT / "processed"
CONFIG_ROOT = PROJECT_ROOT / "configs"
RUNS_ROOT = PROJECT_ROOT / "runs"

DEFAULT_HU_WINDOW: tuple[float, float] = (-1024.0, 600.0)
DEFAULT_CROP_MARGIN = 5
