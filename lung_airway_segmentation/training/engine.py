"""Training orchestration, checkpointing, and run artifacts for supervised baselines."""

import argparse
import json
from datetime import datetime

import torch

from lung_airway_segmentation.settings import RUNS_ROOT
from lung_airway_segmentation.training.builders import (
    build_case_splits,
    build_dataloaders,
    build_datasets,
    build_training_components,
    get_optimizer_learning_rates,
)
from lung_airway_segmentation.training.config import (
    build_resolved_training_config,
    load_yaml_config,
    resolve_device,
    resolve_project_path,
    validate_training_config,
)
from lung_airway_segmentation.training.loops import train_one_epoch, validate_one_epoch


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


def run_supervised_training(args: argparse.Namespace) -> None:
    """Run one supervised baseline experiment from parsed CLI args."""
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)
    training_config = load_yaml_config(args.training_config)

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
