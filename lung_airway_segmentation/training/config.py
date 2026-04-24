"""Config loading, CLI parsing, and validation for supervised baseline runs."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from lung_airway_segmentation.settings import CONFIG_ROOT, PROJECT_ROOT


DEFAULT_DATA_CONFIG = CONFIG_ROOT / "data" / "aeropath.yaml"
DEFAULT_MODEL_CONFIG = CONFIG_ROOT / "model" / "baseline_unet.yaml"
DEFAULT_TRAINING_CONFIG = CONFIG_ROOT / "training" / "baseline.yaml"


def build_config_path_parser() -> argparse.ArgumentParser:
    """Build the lightweight parser used to locate YAML config files."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to the model YAML config.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Path to the training YAML config.",
    )
    return parser


def load_yaml_config(path: Path) -> dict:
    """Load one YAML file and require a top-level mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected config file {path} to contain a mapping.")
    return data


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative or absolute path from config."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_argument_parser(config_args: argparse.Namespace) -> argparse.ArgumentParser:
    """Build the main CLI with only a small set of runtime overrides."""
    parser = argparse.ArgumentParser(
        description="Train the supervised baseline airway segmentation model from YAML configs.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=config_args.data_config,
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=config_args.model_config,
        help="Path to the model YAML config.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=config_args.training_config,
        help="Path to the training YAML config.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional override for the experiment name stored in run metadata.",
    )
    parser.add_argument(
        "--run-description",
        type=str,
        default=None,
        help="Optional free-text description stored in run metadata.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Optional override for the number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for the training batch size from config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for the DataLoader worker count.",
    )
    parser.add_argument(
        "--cache-rate",
        type=float,
        default=None,
        help="Optional override for the MONAI patch-cache rate.",
    )
    return parser


def build_resolved_training_config(
    base_training_config: dict,
    args: argparse.Namespace,
) -> dict:
    """Apply explicit CLI overrides to the loaded training config."""
    resolved = deepcopy(base_training_config)

    if args.experiment_name is not None:
        resolved["experiment_name"] = args.experiment_name
    if args.num_epochs is not None:
        resolved["epochs"] = args.num_epochs
    if args.batch_size is not None:
        resolved["batch_size"] = args.batch_size
    if args.num_workers is not None:
        resolved["num_workers"] = args.num_workers
    if args.cache_rate is not None:
        resolved["sampling"]["cache_rate"] = args.cache_rate

    return resolved


def validate_model_config(model_config: dict) -> None:
    """Validate one model config before building the network."""
    model_name = str(model_config.get("model_name", "")).lower()

    if model_name == "baseline_unet":
        if int(model_config["spatial_dims"]) != 3:
            raise ValueError("baseline_unet currently requires spatial_dims = 3.")
        if int(model_config["in_channels"]) <= 0:
            raise ValueError("baseline_unet.in_channels must be positive.")
        if int(model_config["out_channels"]) <= 0:
            raise ValueError("baseline_unet.out_channels must be positive.")
        if len(model_config["channels"]) < 2:
            raise ValueError("baseline_unet.channels must define at least two stages.")
        if len(model_config["strides"]) != len(model_config["channels"]) - 1:
            raise ValueError("baseline_unet.strides must be one shorter than channels.")
        if int(model_config["num_res_units"]) < 0:
            raise ValueError("baseline_unet.num_res_units must be non-negative.")
        if float(model_config["dropout"]) < 0.0:
            raise ValueError("baseline_unet.dropout must be non-negative.")
        return

    if model_name == "ct_fm_segresnet":
        if int(model_config["spatial_dims"]) != 3:
            raise ValueError("ct_fm_segresnet currently requires spatial_dims = 3.")
        if int(model_config["in_channels"]) != 1:
            raise ValueError("ct_fm_segresnet currently requires in_channels = 1.")
        if int(model_config["out_channels"]) <= 0:
            raise ValueError("ct_fm_segresnet.out_channels must be positive.")
        if int(model_config["init_filters"]) <= 0:
            raise ValueError("ct_fm_segresnet.init_filters must be positive.")
        if len(model_config["blocks_down"]) == 0:
            raise ValueError("ct_fm_segresnet.blocks_down must not be empty.")
        if int(model_config.get("dsdepth", 1)) != 1:
            raise ValueError("ct_fm_segresnet.dsdepth must be 1 for the current training loop.")

        pretrained = model_config.get("pretrained", {})
        if not isinstance(pretrained, dict):
            raise ValueError("ct_fm_segresnet.pretrained must be a mapping.")
        if not isinstance(pretrained.get("enabled", True), bool):
            raise ValueError("ct_fm_segresnet.pretrained.enabled must be a boolean.")

        if pretrained.get("enabled", True):
            source = str(pretrained.get("source", "local")).lower()
            variant = str(pretrained.get("variant", "encoder")).lower()

            if source not in {"local", "hub"}:
                raise ValueError("ct_fm_segresnet.pretrained.source must be 'local' or 'hub'.")
            if variant not in {"encoder", "segresnet"}:
                raise ValueError(
                    "ct_fm_segresnet.pretrained.variant must be 'encoder' or 'segresnet'."
                )
            if source == "local":
                if variant != "encoder":
                    raise ValueError(
                        "Local CT-FM loading currently supports only pretrained.variant = 'encoder'."
                    )
                path_value = pretrained.get("path")
                if path_value is None:
                    raise ValueError("Local CT-FM loading requires ct_fm_segresnet.pretrained.path.")

                resolved_path = resolve_project_path(path_value)
                if not resolved_path.is_dir():
                    raise ValueError(f"Local CT-FM checkpoint directory does not exist: {resolved_path}")

        return

    raise ValueError(f"Unsupported model_name: {model_name}")


def validate_training_config(training_config: dict) -> None:
    """Validate the resolved training config before running experiments."""
    training_regime = training_config["training_regime"]
    if training_regime not in {"patch", "full_volume"}:
        raise ValueError(f"Unsupported training_regime: {training_regime}")

    if int(training_config["epochs"]) <= 0:
        raise ValueError("epochs must be positive.")
    if int(training_config["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive.")
    if int(training_config["num_workers"]) < 0:
        raise ValueError("num_workers must be non-negative.")
    if training_regime == "full_volume" and int(training_config["batch_size"]) != 1:
        raise ValueError("full_volume training currently requires batch_size = 1.")

    splits = training_config["splits"]
    total_fraction = (
        float(splits["train_fraction"])
        + float(splits["val_fraction"])
        + float(splits["test_fraction"])
    )
    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError("Train/val/test fractions must sum to 1.0.")

    sampling = training_config["sampling"]
    if len(sampling["patch_size"]) != 3:
        raise ValueError("sampling.patch_size must have three dimensions.")
    if int(sampling["patches_per_case"]) <= 0:
        raise ValueError("sampling.patches_per_case must be positive.")
    if not 0.0 <= float(sampling["foreground_probability"]) <= 1.0:
        raise ValueError("sampling.foreground_probability must be in [0.0, 1.0].")
    if not 0.0 <= float(sampling["cache_rate"]) <= 1.0:
        raise ValueError("sampling.cache_rate must be in [0.0, 1.0].")

    validation = training_config["validation"]
    if int(validation["validate_every"]) <= 0:
        raise ValueError("validation.validate_every must be positive.")
    if len(validation["roi_size"]) != 3:
        raise ValueError("validation.roi_size must have three dimensions.")
    if int(validation["sw_batch_size"]) <= 0:
        raise ValueError("validation.sw_batch_size must be positive.")
    if not 0.0 <= float(validation["inference_overlap"]) < 1.0:
        raise ValueError("validation.inference_overlap must be in [0.0, 1.0).")

    optimizer_config = training_config["optimizer"]
    if optimizer_config["name"].lower() != "adamw":
        raise ValueError("Only AdamW is currently supported.")
    if float(optimizer_config["lr"]) <= 0.0:
        raise ValueError("optimizer.lr must be positive.")
    if float(optimizer_config["weight_decay"]) < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative.")

    scheduler_name = training_config["scheduler"]["name"].lower()
    if scheduler_name not in {"none", "cosine"}:
        raise ValueError("scheduler.name must be 'none' or 'cosine'.")
    if int(training_config["scheduler"].get("warmup_epochs", 0)) < 0:
        raise ValueError("scheduler.warmup_epochs must be non-negative.")

    amp_config = training_config.get("amp", {"enabled": False})
    if not isinstance(amp_config["enabled"], bool):
        raise ValueError("amp.enabled must be a boolean.")

    loss_config = training_config["loss"]
    if float(loss_config["bce_weight"]) < 0.0:
        raise ValueError("loss.bce_weight must be non-negative.")
    if float(loss_config["dice_weight"]) < 0.0:
        raise ValueError("loss.dice_weight must be non-negative.")


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested device string to a torch device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
