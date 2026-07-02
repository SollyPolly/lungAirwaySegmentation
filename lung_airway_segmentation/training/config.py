"""Config loading, CLI parsing, and validation for training runs."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from lung_airway_segmentation.settings import CONFIG_ROOT, PROJECT_ROOT


DEFAULT_DATA_CONFIG = CONFIG_ROOT / "data" / "aeropath.yaml"
DEFAULT_ATM22_CONFIG = CONFIG_ROOT / "data" / "atm22.yaml"
DEFAULT_MODEL_CONFIG = CONFIG_ROOT / "model" / "baseline_unet.yaml"
DEFAULT_TRAINING_CONFIG = CONFIG_ROOT / "training" / "baseline.yaml"
DEFAULT_SEMISUPERVISED_TRAINING_CONFIG = CONFIG_ROOT / "training" / "mean_teacher_atm.yaml"
DEFAULT_SELFTRAINING_TRAINING_CONFIG = CONFIG_ROOT / "training" / "selftraining_atm.yaml"


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


def build_semisupervised_config_path_parser() -> argparse.ArgumentParser:
    """Build the lightweight parser used to locate Mean Teacher configs."""
    parser = build_config_path_parser()
    parser.set_defaults(training_config=DEFAULT_SEMISUPERVISED_TRAINING_CONFIG)
    parser.add_argument(
        "--atm22-config",
        type=Path,
        default=DEFAULT_ATM22_CONFIG,
        help="Path to the unlabeled ATM'22 dataset YAML config.",
    )
    return parser


def build_selftraining_config_path_parser() -> argparse.ArgumentParser:
    """Build the lightweight parser used to locate self-training configs."""
    parser = build_config_path_parser()
    parser.set_defaults(training_config=DEFAULT_SELFTRAINING_TRAINING_CONFIG)
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
        "--study-name",
        type=str,
        default=None,
        help="Optional override for the stable study directory grouping.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional override for the semantic run variant included in the directory name.",
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
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Optional override for loss.positive_class_weight (BCE positive-class weight).",
    )
    parser.add_argument(
        "--cldice-weight",
        type=float,
        default=None,
        help="Optional override for loss.cldice_weight (clDice term weight).",
    )
    parser.add_argument(
        "--cbdice-weight",
        type=float,
        default=None,
        help="Optional override for loss.cbdice_weight (cbDice radius-aware term weight; warm-up uses loss.cbdice_warmup_epochs/rampup_epochs from config).",
    )
    parser.add_argument(
        "--val-threshold",
        type=float,
        default=None,
        help="Optional override for validation.threshold (checkpoint-selection threshold; pair with --pos-weight).",
    )
    parser.add_argument(
        "--topo-weight",
        type=float,
        default=None,
        help="Optional override for loss.topo_weight (EXPERIMENTAL persistent-homology term; 0 = off, needs torch-topological).",
    )
    parser.add_argument(
        "--calibre-weight-max",
        type=float,
        default=None,
        help="Optional override for loss.calibre_weight_max (calibre/depth-aware BCE weighting; max per-voxel weight on the thinnest distal airway, 1.0 = off).",
    )
    parser.add_argument(
        "--calibre-radius-voxels",
        type=float,
        default=None,
        help="Optional override for loss.calibre_radius_voxels (GT EDT radius at/above which the calibre BCE weight is 1.0; below it ramps to calibre_weight_max at radius<=1).",
    )
    parser.add_argument(
        "--distal-sampling",
        dest="distal_sampling",
        action="store_true",
        default=None,
        help="Enable distal-skeleton biased patch sampling (sets sampling.distal_sampling.enabled=true).",
    )
    parser.add_argument(
        "--no-distal-sampling",
        dest="distal_sampling",
        action="store_false",
        help="Disable distal-skeleton biased patch sampling (sets sampling.distal_sampling.enabled=false).",
    )
    parser.add_argument(
        "--distal-radius",
        type=float,
        default=None,
        help="Optional override for sampling.distal_sampling.distal_radius_voxels (EDT voxels; <= this = distal).",
    )
    parser.add_argument(
        "--lung-crop",
        dest="lung_crop",
        action="store_true",
        default=None,
        help="Enable lung-bbox cropping (sets sampling.lung_crop.enabled=true; needs precompute_lung_masks).",
    )
    parser.add_argument(
        "--no-lung-crop",
        dest="lung_crop",
        action="store_false",
        help="Disable lung-bbox cropping (sets sampling.lung_crop.enabled=false; crops CT foreground instead).",
    )
    parser.add_argument(
        "--lung-crop-margin",
        type=int,
        default=None,
        help="Optional override for sampling.lung_crop.margin_voxels (in-plane + inferior voxel margin around the lung bbox).",
    )
    parser.add_argument(
        "--lung-crop-superior-margin",
        type=int,
        default=None,
        help="Optional override for sampling.lung_crop.superior_margin_voxels (superior extension for the cervical trachea).",
    )
    return parser


def build_semisupervised_argument_parser(
    config_args: argparse.Namespace,
) -> argparse.ArgumentParser:
    """Build the Mean Teacher CLI while preserving baseline runtime overrides."""
    parser = build_argument_parser(config_args)
    parser.description = "Train the Mean Teacher airway segmentation model from YAML configs."
    parser.add_argument(
        "--atm22-config",
        type=Path,
        default=config_args.atm22_config,
        help="Path to the unlabeled ATM'22 dataset YAML config.",
    )
    parser.add_argument(
        "--batch-size-unlabelled",
        type=int,
        default=None,
        help="Optional override for the ATM'22 batch size.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint to warm-start both student and teacher "
            "(e.g. the supervised baseline best_model.pt). Overrides the "
            "init_checkpoint field in the training config."
        ),
    )
    return parser


def build_selftraining_argument_parser(
    config_args: argparse.Namespace,
) -> argparse.ArgumentParser:
    """Build the self-training CLI while preserving baseline runtime overrides."""
    parser = build_argument_parser(config_args)
    parser.description = "Train the topology-filtered self-training model from YAML configs."
    parser.add_argument(
        "--pseudo-label-dir",
        type=Path,
        default=None,
        help=(
            "Directory of generated pseudo-labels (contains manifest.json), written "
            "by scripts/pseudo_label_atm.py. Overrides selftraining.pseudo_label_dir."
        ),
    )
    parser.add_argument(
        "--labelled-oversample",
        type=int,
        default=None,
        help="How many times to duplicate each real labelled case vs the pseudo pool. Overrides selftraining.labelled_oversample.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start the student (e.g. the labeller's best_topology_model.pt). Overrides init_checkpoint.",
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
    if getattr(args, "study_name", None) is not None:
        resolved["study_name"] = args.study_name
    if getattr(args, "run_label", None) is not None:
        resolved["run_label"] = args.run_label
    if args.num_epochs is not None:
        resolved["epochs"] = args.num_epochs
    if args.batch_size is not None:
        resolved["batch_size"] = args.batch_size
    if args.num_workers is not None:
        resolved["num_workers"] = args.num_workers
    if args.cache_rate is not None:
        resolved["sampling"]["cache_rate"] = args.cache_rate
    if args.pos_weight is not None:
        resolved["loss"]["positive_class_weight"] = args.pos_weight
    if args.cldice_weight is not None:
        resolved["loss"]["cldice_weight"] = args.cldice_weight
    if args.cbdice_weight is not None:
        resolved["loss"]["cbdice_weight"] = args.cbdice_weight
    if args.topo_weight is not None:
        resolved["loss"]["topo_weight"] = args.topo_weight
    if args.calibre_weight_max is not None:
        resolved["loss"]["calibre_weight_max"] = args.calibre_weight_max
    if args.calibre_radius_voxels is not None:
        resolved["loss"]["calibre_radius_voxels"] = args.calibre_radius_voxels
    if getattr(args, "distal_sampling", None) is not None:
        resolved["sampling"].setdefault("distal_sampling", {})["enabled"] = bool(args.distal_sampling)
    if getattr(args, "distal_radius", None) is not None:
        resolved["sampling"].setdefault("distal_sampling", {})["distal_radius_voxels"] = float(args.distal_radius)
    if getattr(args, "lung_crop", None) is not None:
        resolved["sampling"].setdefault("lung_crop", {})["enabled"] = bool(args.lung_crop)
    if getattr(args, "lung_crop_margin", None) is not None:
        resolved["sampling"].setdefault("lung_crop", {})["margin_voxels"] = int(args.lung_crop_margin)
    if getattr(args, "lung_crop_superior_margin", None) is not None:
        resolved["sampling"].setdefault("lung_crop", {})["superior_margin_voxels"] = int(args.lung_crop_superior_margin)
    if args.val_threshold is not None:
        resolved["validation"]["threshold"] = args.val_threshold
    if getattr(args, "batch_size_unlabelled", None) is not None:
        resolved["batch_size_unlabelled"] = args.batch_size_unlabelled
    if getattr(args, "init_checkpoint", None) is not None:
        resolved["init_checkpoint"] = str(args.init_checkpoint)
    if getattr(args, "pseudo_label_dir", None) is not None:
        resolved.setdefault("selftraining", {})["pseudo_label_dir"] = str(args.pseudo_label_dir)
    if getattr(args, "labelled_oversample", None) is not None:
        resolved.setdefault("selftraining", {})["labelled_oversample"] = int(args.labelled_oversample)

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
    for field_name in ("study_name", "run_label"):
        if field_name in training_config and (
            not isinstance(training_config[field_name], str)
            or not training_config[field_name].strip()
        ):
            raise ValueError(f"{field_name} must be a non-empty string when provided.")

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

    # Splits are dataset-dependent: AeroPath uses fractional three-way splits,
    # ATM'22 uses a count-based four-way semi-supervised split. Validate whichever
    # the config declares; require at least one.
    splits = training_config.get("splits")
    labelled_split = training_config.get("labelled_split")
    if splits is None and labelled_split is None:
        raise ValueError(
            "Training config must define either 'splits' (fractions) or "
            "'labelled_split' (counts)."
        )
    if splits is not None:
        total_fraction = (
            float(splits["train_fraction"])
            + float(splits["val_fraction"])
            + float(splits["test_fraction"])
        )
        if abs(total_fraction - 1.0) > 1e-8:
            raise ValueError("Train/val/test fractions must sum to 1.0.")
    if labelled_split is not None:
        if int(labelled_split["test_count"]) < 0 or int(labelled_split["val_count"]) < 0:
            raise ValueError("labelled_split test_count and val_count must be non-negative.")
        if int(labelled_split["labelled_count"]) <= 0:
            raise ValueError("labelled_split.labelled_count must be positive.")

    sampling = training_config["sampling"]
    if len(sampling["patch_size"]) != 3:
        raise ValueError("sampling.patch_size must have three dimensions.")
    if int(sampling["patches_per_case"]) <= 0:
        raise ValueError("sampling.patches_per_case must be positive.")
    if not 0.0 <= float(sampling["foreground_probability"]) <= 1.0:
        raise ValueError("sampling.foreground_probability must be in [0.0, 1.0].")
    if not 0.0 <= float(sampling["cache_rate"]) <= 1.0:
        raise ValueError("sampling.cache_rate must be in [0.0, 1.0].")

    distal_sampling = sampling.get("distal_sampling")
    if distal_sampling is not None:
        if not isinstance(distal_sampling, dict):
            raise ValueError("sampling.distal_sampling must be a mapping.")
        if not isinstance(distal_sampling.get("enabled", False), bool):
            raise ValueError("sampling.distal_sampling.enabled must be a boolean.")
        if float(distal_sampling.get("distal_radius_voxels", 2.0)) <= 0.0:
            raise ValueError("sampling.distal_sampling.distal_radius_voxels must be positive.")
        ratios = distal_sampling.get("ratios", [0.3, 0.3, 0.4])
        if len(ratios) != 3 or any(float(r) < 0.0 for r in ratios) or sum(float(r) for r in ratios) <= 0.0:
            raise ValueError(
                "sampling.distal_sampling.ratios must be three non-negative numbers summing to > 0 "
                "([background, proximal-airway, distal-skeleton])."
            )

    lung_crop = sampling.get("lung_crop")
    if lung_crop is not None:
        if not isinstance(lung_crop, dict):
            raise ValueError("sampling.lung_crop must be a mapping.")
        if not isinstance(lung_crop.get("enabled", False), bool):
            raise ValueError("sampling.lung_crop.enabled must be a boolean.")
        strategy = lung_crop.get("strategy", "lung_with_trachea_extension")
        if strategy not in ("lung_with_trachea_extension", "lung_union_airway"):
            raise ValueError(
                "sampling.lung_crop.strategy must be 'lung_with_trachea_extension' "
                "(inference-valid) or 'lung_union_airway' (oracle diagnostic)."
            )
        if int(lung_crop.get("margin_voxels", 0)) < 0:
            raise ValueError("sampling.lung_crop.margin_voxels must be >= 0.")
        if int(lung_crop.get("superior_margin_voxels", 0)) < 0:
            raise ValueError("sampling.lung_crop.superior_margin_voxels must be >= 0.")

    validation = training_config["validation"]
    if int(validation["validate_every"]) <= 0:
        raise ValueError("validation.validate_every must be positive.")
    if len(validation["roi_size"]) != 3:
        raise ValueError("validation.roi_size must have three dimensions.")
    if int(validation["sw_batch_size"]) <= 0:
        raise ValueError("validation.sw_batch_size must be positive.")
    if not 0.0 <= float(validation["inference_overlap"]) < 1.0:
        raise ValueError("validation.inference_overlap must be in [0.0, 1.0).")
    if not 0.0 <= float(validation.get("threshold", 0.5)) <= 1.0:
        raise ValueError("validation.threshold must be in [0.0, 1.0].")
    if not 0.0 <= float(validation.get("topology_threshold", 0.5)) <= 1.0:
        raise ValueError("validation.topology_threshold must be in [0.0, 1.0].")
    topology_max_ratio = validation.get("topology_max_ratio")
    if topology_max_ratio is not None and float(topology_max_ratio) <= 1.0:
        raise ValueError(
            "validation.topology_max_ratio is an optional catastrophic guard and "
            "must be > 1.0 (set it well above mature raw prediction volumes, e.g. "
            "50); omit it for no gate (the default)."
        )

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

    if not isinstance(training_config.get("deterministic", True), bool):
        raise ValueError("deterministic must be a boolean.")

    loss_config = training_config["loss"]
    if float(loss_config["bce_weight"]) < 0.0:
        raise ValueError("loss.bce_weight must be non-negative.")
    if float(loss_config["dice_weight"]) < 0.0:
        raise ValueError("loss.dice_weight must be non-negative.")
    if float(loss_config.get("positive_class_weight", 1.0)) <= 0.0:
        raise ValueError("loss.positive_class_weight must be positive.")
    if float(loss_config.get("cldice_weight", 0.0)) < 0.0:
        raise ValueError("loss.cldice_weight must be non-negative.")
    if int(loss_config.get("cldice_iterations", 10)) < 1:
        raise ValueError("loss.cldice_iterations must be >= 1.")
    if int(loss_config.get("cldice_warmup_epochs", 0)) < 0:
        raise ValueError("loss.cldice_warmup_epochs must be non-negative.")
    if int(loss_config.get("cldice_rampup_epochs", 0)) < 0:
        raise ValueError("loss.cldice_rampup_epochs must be non-negative.")
    if float(loss_config.get("cbdice_weight", 0.0)) < 0.0:
        raise ValueError("loss.cbdice_weight must be non-negative.")
    if int(loss_config.get("cbdice_iterations", 10)) < 1:
        raise ValueError("loss.cbdice_iterations must be >= 1.")
    if int(loss_config.get("cbdice_warmup_epochs", 0)) < 0:
        raise ValueError("loss.cbdice_warmup_epochs must be non-negative.")
    if int(loss_config.get("cbdice_rampup_epochs", 0)) < 0:
        raise ValueError("loss.cbdice_rampup_epochs must be non-negative.")
    if float(loss_config.get("topo_weight", 0.0)) < 0.0:
        raise ValueError("loss.topo_weight must be non-negative.")
    if int(loss_config.get("topo_warmup_epochs", 0)) < 0:
        raise ValueError("loss.topo_warmup_epochs must be non-negative.")
    if int(loss_config.get("topo_rampup_epochs", 0)) < 0:
        raise ValueError("loss.topo_rampup_epochs must be non-negative.")
    if float(loss_config.get("calibre_weight_max", 1.0)) < 1.0:
        raise ValueError("loss.calibre_weight_max must be >= 1.0 (1.0 = off).")
    if float(loss_config.get("calibre_radius_voxels", 3.0)) <= 1.0:
        raise ValueError("loss.calibre_radius_voxels must be > 1.0.")


def validate_semisupervised_training_config(training_config: dict) -> None:
    """Validate Mean Teacher settings in addition to shared baseline settings."""
    validate_training_config(training_config)

    if training_config["training_regime"] != "patch":
        raise ValueError("Mean Teacher training currently requires training_regime = 'patch'.")
    if int(training_config["batch_size_unlabelled"]) <= 0:
        raise ValueError("batch_size_unlabelled must be positive.")

    unlabelled_sampling = training_config.get("unlabelled_sampling", {})
    if not 0.0 <= float(unlabelled_sampling.get("cache_rate", 0.0)) <= 1.0:
        raise ValueError("unlabelled_sampling.cache_rate must be in [0.0, 1.0].")

    teacher_config = training_config["teacher"]
    if not 0.0 <= float(teacher_config["ema_decay"]) < 1.0:
        raise ValueError("teacher.ema_decay must be in [0.0, 1.0).")
    if int(teacher_config["warm_start_epochs"]) < 0:
        raise ValueError("teacher.warm_start_epochs must be non-negative.")
    if int(teacher_config.get("consistency_rampup_epochs", 0)) < 0:
        raise ValueError("teacher.consistency_rampup_epochs must be non-negative.")
    foreground_threshold = float(teacher_config["foreground_confidence_threshold"])
    background_threshold = float(teacher_config["background_confidence_threshold"])
    if not 0.0 <= background_threshold < foreground_threshold <= 1.0:
        raise ValueError(
            "Teacher confidence thresholds must satisfy "
            "0 <= background < foreground <= 1."
        )

    if float(training_config["loss"]["consistency_weight"]) < 0.0:
        raise ValueError("loss.consistency_weight must be non-negative.")


def validate_selftraining_training_config(training_config: dict) -> None:
    """Validate topology-filtered self-training settings on top of the baseline ones."""
    validate_training_config(training_config)

    if training_config["training_regime"] != "patch":
        raise ValueError("Self-training currently requires training_regime = 'patch'.")

    selftraining = training_config.get("selftraining")
    if not isinstance(selftraining, dict) or not selftraining.get("pseudo_label_dir"):
        raise ValueError(
            "Self-training requires a 'selftraining.pseudo_label_dir' (set it in the "
            "training config or pass --pseudo-label-dir)."
        )
    if int(selftraining.get("labelled_oversample", 1)) < 1:
        raise ValueError("selftraining.labelled_oversample must be >= 1.")

    manifest_path = resolve_project_path(selftraining["pseudo_label_dir"]) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Pseudo-label manifest not found: {manifest_path}. "
            "Run scripts/pseudo_label_atm.py before self-training."
        )


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested device string to a torch device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)
