"""Train the supervised baseline airway segmentation model.

This script is config-first: dataset, model, and training defaults come from
YAML files under ``configs/`` and the CLI only exposes a small set of explicit
runtime overrides. The same shared model, loss, and training loop can therefore
be compared across patch-based training and full-volume ablations by swapping
training configs rather than editing Python code.
"""

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import yaml
from monai.data import DataLoader, list_data_collate

from lung_airway_segmentation.datasets.aeropath import AeroPathDataset
from lung_airway_segmentation.datasets.monai_aeropath import build_monai_aeropath_datasets
from lung_airway_segmentation.datasets.splits import create_train_val_test_split
from lung_airway_segmentation.io.case_layout import list_case_ids
from lung_airway_segmentation.losses.segmentation import CombinedSegmentationLoss
from lung_airway_segmentation.models.baseline_unet import build_baseline_unet
from lung_airway_segmentation.settings import CONFIG_ROOT, PROJECT_ROOT, RUNS_ROOT
from lung_airway_segmentation.training.loops import train_one_epoch, validate_one_epoch


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


def build_case_splits(data_root: Path, training_config: dict) -> tuple[list[str], list[str], list[str]]:
    """Create deterministic train, validation, and test case splits."""
    case_ids = list_case_ids(data_root)
    splits = training_config["splits"]
    return create_train_val_test_split(
        case_ids,
        train_split=float(splits["train_fraction"]),
        val_split=float(splits["val_fraction"]),
        test_split=float(splits["test_fraction"]),
        seed=int(training_config["seed"]),
    )


def build_datasets(
    train_ids,
    val_ids,
    data_root: Path,
    data_config: dict,
    training_config: dict,
):
    """Build train and validation datasets for the configured training regime."""
    training_regime = training_config["training_regime"]
    sampling = training_config["sampling"]
    preprocessing = data_config["preprocessing"]
    crop_margin_voxels = int(data_config["preprocessing"]["crop_margin_voxels"])
    hu_window = tuple(float(value) for value in preprocessing["hu_window"])

    if training_regime == "full_volume":
        train_dataset = AeroPathDataset(
            case_ids=train_ids,
            data_root=data_root,
            include_lung_mask=False,
            hu_window=hu_window,
            crop_margin=crop_margin_voxels,
        )
        val_dataset = AeroPathDataset(
            case_ids=val_ids,
            data_root=data_root,
            include_lung_mask=False,
            hu_window=hu_window,
            crop_margin=crop_margin_voxels,
        )
        return train_dataset, val_dataset

    train_dataset, val_dataset = build_monai_aeropath_datasets(
        train_ids=train_ids,
        val_ids=val_ids,
        data_root=data_root,
        patch_size=tuple(int(value) for value in sampling["patch_size"]),
        patches_per_case=int(sampling["patches_per_case"]),
        foreground_probability=float(sampling["foreground_probability"]),
        cache_rate=float(sampling["cache_rate"]),
        crop_margin_voxels=crop_margin_voxels,
        hu_window=hu_window,
    )
    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, batch_size: int, num_workers: int):
    """Build train and validation dataloaders for baseline training."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )
    return train_loader, val_loader


def build_scheduler(optimizer, training_config: dict):
    """Build the configured LR scheduler, if any."""
    scheduler_config = training_config["scheduler"]
    scheduler_name = scheduler_config["name"].lower()
    num_epochs = int(training_config["epochs"])

    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine":
        warmup_epochs = int(scheduler_config.get("warmup_epochs", 0))
        if warmup_epochs > 0 and warmup_epochs < num_epochs:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(num_epochs - warmup_epochs, 1),
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(num_epochs, 1),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_training_components(device, model_config: dict, training_config: dict):
    """Build the configured model, loss function, optimizer, and scheduler."""
    model = build_baseline_unet(
        spatial_dims=int(model_config["spatial_dims"]),
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        channels=tuple(int(value) for value in model_config["channels"]),
        strides=tuple(int(value) for value in model_config["strides"]),
        num_res_units=int(model_config["num_res_units"]),
        dropout=float(model_config["dropout"]),
        norm=str(model_config["norm"]).upper(),
        act=str(model_config["act"]).upper(),
    ).to(device)

    loss_config = training_config["loss"]
    loss_fn = CombinedSegmentationLoss(
        bce_weight=float(loss_config["bce_weight"]),
        dice_weight=float(loss_config["dice_weight"]),
    )

    optimizer_config = training_config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )

    scheduler = build_scheduler(optimizer, training_config)

    return model, loss_fn, optimizer, scheduler


def get_optimizer_learning_rates(optimizer) -> list[float]:
    """Return one learning-rate value per optimizer parameter group."""
    return [float(group["lr"]) for group in optimizer.param_groups]


def save_checkpoint(model, optimizer, epoch, metrics, output_path, scheduler=None):
    """Save model, optimizer, and summary metrics for one training checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()

    torch.save(payload, output_path)


def write_json(data, output_path):
    """Write one JSON artifact to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def build_run_dir(experiment_name):
    """Create a timestamped run directory for one baseline experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_ROOT / experiment_name / timestamp


def initialize_run_artifacts(run_dir, run_metadata, resolved_config):
    """Write static run artifacts before training starts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_metadata, run_dir / "run_metadata.json")
    write_json(resolved_config, run_dir / "resolved_config.json")

    notes_path = run_dir / "notes.txt"
    if not notes_path.exists():
        notes_path.write_text(
            "Add baseline run notes here.\n",
            encoding="utf-8",
        )


def main() -> None:
    config_path_parser = build_config_path_parser()
    config_args, _ = config_path_parser.parse_known_args()

    data_config = load_yaml_config(config_args.data_config)
    model_config = load_yaml_config(config_args.model_config)
    training_config = load_yaml_config(config_args.training_config)

    args = build_argument_parser(config_args).parse_args()
    resolved_training_config = build_resolved_training_config(training_config, args)
    validate_training_config(resolved_training_config)

    device = resolve_device(args.device)
    torch.manual_seed(int(resolved_training_config["seed"]))

    data_root = resolve_project_path(data_config["raw_data_root"])
    train_ids, val_ids, test_ids = build_case_splits(data_root, resolved_training_config)

    train_dataset, val_dataset = build_datasets(
        train_ids,
        val_ids,
        data_root,
        data_config,
        resolved_training_config,
    )
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=int(resolved_training_config["batch_size"]),
        num_workers=int(resolved_training_config["num_workers"]),
    )

    model, loss_fn, optimizer, scheduler = build_training_components(
        device,
        model_config,
        resolved_training_config,
    )
    use_amp = bool(resolved_training_config.get("amp", {}).get("enabled", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_dir = build_run_dir(resolved_training_config["experiment_name"])
    best_val_dice = -1.0
    best_epoch = 0
    history = []

    training_regime = resolved_training_config["training_regime"]
    sampling_config = resolved_training_config["sampling"]
    validation_config = resolved_training_config["validation"]

    run_description = args.run_description
    if run_description is None:
        if training_regime == "patch":
            run_description = "Supervised MONAI patch-based baseline."
        else:
            run_description = "Supervised full-volume ablation baseline."

    resolved_config_artifact = {
        "data": data_config,
        "model": model_config,
        "training": resolved_training_config,
    }

    run_metadata = {
        "experiment_name": resolved_training_config["experiment_name"],
        "description": run_description,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_files": {
            "data": str(args.data_config),
            "model": str(args.model_config),
            "training": str(args.training_config),
        },
        "data_root": str(data_root),
        "device": str(device),
        "data_pipeline": training_regime,
        "amp_enabled": use_amp,
        "model_name": model_config["model_name"],
        "optimizer_name": resolved_training_config["optimizer"]["name"],
        "scheduler_name": resolved_training_config["scheduler"]["name"],
        "effective_batch_size": (
            int(resolved_training_config["batch_size"])
            if training_regime == "full_volume"
            else int(resolved_training_config["batch_size"]) * int(sampling_config["patches_per_case"])
        ),
        "splits": {
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_case_ids": train_ids,
            "val_case_ids": val_ids,
            "test_case_ids": test_ids,
        },
    }
    initialize_run_artifacts(run_dir, run_metadata, resolved_config_artifact)

    for epoch in range(int(resolved_training_config["epochs"])):
        learning_rates_before_epoch = get_optimizer_learning_rates(optimizer)
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        epoch_summary = {
            "epoch": epoch + 1,
            "learning_rate": learning_rates_before_epoch[0],
            "learning_rates_before_epoch": learning_rates_before_epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
        }

        should_validate = (
            (epoch + 1) % int(validation_config["validate_every"]) == 0
            or epoch + 1 == int(resolved_training_config["epochs"])
        )
        if should_validate:
            val_metrics = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                roi_size=tuple(int(value) for value in validation_config["roi_size"]),
                sw_batch_size=int(validation_config["sw_batch_size"]),
                overlap=float(validation_config["inference_overlap"]),
                use_amp=use_amp,
            )
            epoch_summary["val_loss"] = val_metrics["loss"]
            epoch_summary["val_dice"] = val_metrics["dice"]

            print(
                f"Epoch {epoch + 1} / {resolved_training_config['epochs']}"
                f" - train_loss: {train_metrics['loss']:.4f}"
                f" - train_dice: {train_metrics['dice']:.4f}"
                f" - val_loss: {val_metrics['loss']:.4f}"
                f" - val_dice: {val_metrics['dice']:.4f}"
            )
        else:
            val_metrics = None
            print(
                f"Epoch {epoch + 1} / {resolved_training_config['epochs']}"
                f" - train_loss: {train_metrics['loss']:.4f}"
                f" - train_dice: {train_metrics['dice']:.4f}"
                " - validation skipped"
            )

        if scheduler is not None:
            scheduler.step()
            epoch_summary["scheduler_stepped"] = True
        else:
            epoch_summary["scheduler_stepped"] = False

        epoch_summary["learning_rates_after_epoch"] = get_optimizer_learning_rates(optimizer)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=epoch_summary,
            output_path=run_dir / "last_model.pt",
            scheduler=scheduler,
        )

        if val_metrics is not None and val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch + 1
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=epoch_summary,
                output_path=run_dir / "best_model.pt",
                scheduler=scheduler,
            )

        history.append(epoch_summary)
        write_json(
            {
                "best": {
                    "epoch": best_epoch,
                    "val_dice": best_val_dice,
                },
                "history": history,
            },
            run_dir / "history.json",
        )

    print("Training finished.")
    print(f"Run directory: {run_dir}")
    print(f"Held-out test cases: {len(test_ids)}")
    print(f"Best val dice: {best_val_dice:.4f}")
    print(f"Best epoch: {best_epoch}")


if __name__ == "__main__":
    main()
