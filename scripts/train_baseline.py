"""Supervised baseline training CLI.

This entrypoint should launch the first clean reference experiment only.

What to put here:
- config loading
- dataset and dataloader construction
- baseline model creation
- supervised training loop kickoff

What not to put here:
- semi-supervised teacher-student logic
- distal refinement fusion
- ad hoc notebook-only code
"""

import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from lung_airway_segmentation.datasets.aeropath import AeroPathDataset
from lung_airway_segmentation.datasets.splits import create_train_val_test_split
from lung_airway_segmentation.io.case_layout import list_case_ids
from lung_airway_segmentation.losses.segmentation import CombinedSegmentationLoss
from lung_airway_segmentation.models.baseline_unet import build_baseline_unet
from lung_airway_segmentation.settings import RAW_AEROPATH_ROOT, RUNS_ROOT
from lung_airway_segmentation.training.loops import train_one_epoch, validate_one_epoch


def build_case_splits(data_root, seed, train_split, val_split, test_split):
    """Create deterministic train, validation, and test case ID splits."""
    case_ids = list_case_ids(data_root)
    train_ids, val_ids, test_ids = create_train_val_test_split(
        case_ids,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed = seed
    )
    return train_ids, val_ids, test_ids


def build_datasets(train_ids, val_ids, data_root):
    """Build train and validation datasets from the selected case IDs."""
    train_dataset = AeroPathDataset(
        case_ids=train_ids,
        data_root=data_root,
        include_lung_mask=False
    )

    val_dataset = AeroPathDataset(
        case_ids=val_ids,
        data_root=data_root,
        include_lung_mask=False
    )
    return train_dataset, val_dataset

def build_dataloaders(train_dataset, val_dataset, batch_size):
    """Build train and validation dataloaders for baseline training."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader

def build_training_components(device, learning_rate):
    """Build the baseline model, loss function, and optimizer."""

    model = build_baseline_unet().to(device)
    loss_fn = CombinedSegmentationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer

def save_checkpoint(model, optimizer, epoch, metrics, output_path):
    """Save model, optimizer, and summary metrics for one training checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics
    }
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


def initialize_run_artifacts(run_dir, run_metadata):
    """Write run metadata and a notes file before training starts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_metadata, run_dir / "run_metadata.json")

    notes_path = run_dir / "notes.txt"
    if not notes_path.exists():
        notes_path.write_text(
            "Add baseline run notes here.\n",
            encoding="utf-8",
        )


def main() -> None:
    # Basic experiment settings
    data_root = RAW_AEROPATH_ROOT
    seed = 15
    batch_size = 1
    num_epochs = 25
    learning_rate = 1e-3
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    experiment_name = "baseline_unet"
    run_description = "First run on the imperial HPC. With the aim to understand a very very early baseline."

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reproducibility
    torch.manual_seed(seed)

    train_ids, val_ids, test_ids = build_case_splits(
        data_root=data_root,
        seed=seed,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
    )
    train_dataset, val_dataset = build_datasets(train_ids, val_ids, data_root)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, batch_size)
    model, loss_fn, optimizer = build_training_components(device, learning_rate)

    # Saving
    run_dir = build_run_dir(experiment_name)
    best_val_dice = -1.0
    history = []
    run_metadata = {
        "experiment_name": experiment_name,
        "description": run_description,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "device": str(device),
        "seed": seed,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "splits": {
            "train_fraction": train_split,
            "val_fraction": val_split,
            "test_fraction": test_split,
            "train_case_ids": train_ids,
            "val_case_ids": val_ids,
            "test_case_ids": test_ids,
        },
        "model": {
            "name": "baseline_unet",
            "library": "MONAI",
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [16, 32, 64, 128, 256],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
        },
        "loss": {
            "name": "CombinedSegmentationLoss",
            "bce_weight": loss_fn.bce_weight,
            "dice_weight": loss_fn.dice_weight,
        },
    }
    initialize_run_artifacts(run_dir, run_metadata)

    # Training loop
    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch {epoch + 1} / {num_epochs}"
            f" - train_loss: {train_metrics['loss']:.4f}"
            f" - train_dice: {train_metrics['dice']:.4f}"
            f" - val_loss: {val_metrics['loss']:.4f}"
            f" - val_dice: {val_metrics['dice']:.4f}"
        )

        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
        }
        history.append(epoch_summary)
        write_json(
            {
                "run_metadata": run_metadata,
                "history": history,
            },
            run_dir / "history.json",
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics={
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"]
            },
            output_path=run_dir / "last_model.pt"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics = {
                    "train_loss": train_metrics["loss"],
                    "train_dice": train_metrics["dice"],
                    "val_loss": val_metrics["loss"],
                    "val_dice": val_metrics["dice"]
                },
                output_path=run_dir / "best_model.pt"
            )


    print("Training finished.")
    print(f"Run directory: {run_dir}")
    print(f"Held-out test cases: {len(test_ids)}")



if __name__ == "__main__":
    main()
