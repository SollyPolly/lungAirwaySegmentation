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

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lung_airway_segmentation.datasets.aeropath import AeroPathDataset, AeroPathPatchDataset
from lung_airway_segmentation.datasets.splits import create_train_val_test_split
from lung_airway_segmentation.io.case_layout import list_case_ids
from lung_airway_segmentation.losses.segmentation import CombinedSegmentationLoss
from lung_airway_segmentation.models.baseline_unet import build_baseline_unet
from lung_airway_segmentation.settings import RAW_AEROPATH_ROOT, RUNS_ROOT
from lung_airway_segmentation.training.loops import train_one_epoch, validate_one_epoch
from lung_airway_segmentation.datasets.transforms import build_train_patch_transforms



def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for baseline training runs."""
    parser = argparse.ArgumentParser(
        description="Train the baseline supervised 3D U-Net airway segmentation model.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=RAW_AEROPATH_ROOT,
        help="Path to the AeroPath case root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=15,
        help="Random seed for dataset splits and torch initialization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training and validation dataloaders.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the AdamW optimizer.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of cases assigned to the training split.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of cases assigned to the validation split.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction of cases assigned to the test split.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="baseline_unet",
        help="Name used for the timestamped run directory.",
    )
    parser.add_argument(
        "--run-description",
        type=str,
        default="First run on the Imperial HPC. Early full-volume supervised baseline.",
        help="Free-text description stored in run metadata.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=1.0,
        help="Weight for BCE-with-logits in the combined segmentation loss.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight for Dice loss in the combined segmentation loss.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(96, 96, 96),
        metavar=("Z", "Y", "X"),
        help="3D patch size used for patch-based training.",
    )
    parser.add_argument(
        "--use-full-volumes",
        action="store_true",
        help="Train on full preprocessed volumes instead of sampled patches.",
    )
    parser.add_argument(
        "--patches-per-case",
        type=int,
        default=4,
        help="Number of training patch samples exposed per case in each epoch.",
    )
    parser.add_argument(
        "--foreground-probability",
        type=float,
        default=0.7,
        help="How often a training patch is forced to be airway-centered"
    )


    return parser


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested device string to a torch device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


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




def build_datasets(
        train_ids, 
        val_ids, 
        data_root,
        *,
        use_full_volumes,
        patch_size,
        patches_per_case,
        foreground_probability
    ):
    """Build train and validation datasets from the selected case IDs."""

    

    if use_full_volumes:
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
    
    else:

        train_transform = build_train_patch_transforms(include_lung_mask=True)
        
        train_dataset = AeroPathPatchDataset(
        case_ids=train_ids,
        data_root=data_root,
        include_lung_mask=True,
        patch_size=tuple(patch_size),
        patches_per_case=patches_per_case,
        foreground_probability=foreground_probability,
        transform=train_transform
        )

        val_dataset = AeroPathDataset(
            case_ids=val_ids,
            data_root=data_root,
            include_lung_mask=False
        )       
        return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, batch_size, num_workers):
    """Build train and validation dataloaders for baseline training."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader


def build_training_components(device, learning_rate, bce_weight, dice_weight):
    """Build the baseline model, loss function, and optimizer."""

    model = build_baseline_unet().to(device)
    loss_fn = CombinedSegmentationLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
    )
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
    args = build_argument_parser().parse_args()

    # Device
    device = resolve_device(args.device)
    # Reproducibility
    torch.manual_seed(args.seed)

    train_ids, val_ids, test_ids = build_case_splits(
        data_root=args.data_root,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    train_dataset, val_dataset = build_datasets(
        train_ids, 
        val_ids, 
        args.data_root,
        use_full_volumes=args.use_full_volumes,
        patch_size=args.patch_size,
        patches_per_case=args.patches_per_case,
        foreground_probability=args.foreground_probability
        )
    
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        args.num_workers,
    )


    model, loss_fn, optimizer = build_training_components(
        device,
        args.learning_rate,
        args.bce_weight,
        args.dice_weight,
    )

    # Saving
    run_dir = build_run_dir(args.experiment_name)
    best_val_dice = -1.0
    best_epoch = 0
    history = []
    run_metadata = {
        "experiment_name": args.experiment_name,
        "description": args.run_description,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "splits": {
            "train_fraction": args.train_split,
            "val_fraction": args.val_split,
            "test_fraction": args.test_split,
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
    for epoch in range(args.num_epochs):
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
            device=device,
            roi_size=tuple(args.patch_size),
            sw_batch_size=1,
            overlap=0.5
        )

        print(
            f"Epoch {epoch + 1} / {args.num_epochs}"
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
            best_epoch = epoch + 1
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

        history.append(epoch_summary)
        write_json(
            {
                "run_metadata": run_metadata,
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
