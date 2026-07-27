"""Path record types shared by the active ATM'22/nnU-Net utilities."""

from pathlib import Path
from typing import TypedDict


class LabelledCasePaths(TypedDict):
    case_id: str
    case_dir: Path
    ct: Path
    lung: Path | None
    airway: Path

class UnlabelledCasePaths(TypedDict):
    case_id: str
    case_dir: Path
    ct: Path
    lung: Path | None
    airway: None

CasePaths = LabelledCasePaths | UnlabelledCasePaths
