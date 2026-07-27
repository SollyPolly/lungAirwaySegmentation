"""Small configuration helpers used by the active nnU-Net utilities."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML file and require a top-level mapping."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected config file {config_path} to contain a mapping.")
    return data
