"""Dataset, dataloader, model, and optimizer builders for supervised training."""

import torch
from monai.data import DataLoader, list_data_collate

from lung_airway_segmentation.datasets.monai_aeropath import (
    build_monai_aeropath_datasets,
    build_monai_aeropath_full_volume_datasets,
)
from lung_airway_segmentation.datasets.splits import create_train_val_test_split
from lung_airway_segmentation.io.case_layout import list_case_ids
from lung_airway_segmentation.losses.segmentation import CombinedSegmentationLoss
from lung_airway_segmentation.models.baseline_unet import build_baseline_unet
from lung_airway_segmentation.models.ct_fm_segresnet import build_ct_fm_segresnet


def build_model(device, model_config: dict):
    """Build the configured model and move it to the target device."""
    model_name = str(model_config["model_name"]).lower()

    if model_name == "baseline_unet":
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
        )
    elif model_name == "ct_fm_segresnet":
        pretrained_config = model_config.get("pretrained", {})
        model = build_ct_fm_segresnet(
            spatial_dims=int(model_config["spatial_dims"]),
            in_channels=int(model_config["in_channels"]),
            out_channels=int(model_config["out_channels"]),
            init_filters=int(model_config["init_filters"]),
            blocks_down=tuple(int(value) for value in model_config["blocks_down"]),
            dsdepth=int(model_config.get("dsdepth", 1)),
            act=str(model_config.get("act", "relu")),
            norm=str(model_config.get("norm", "batch")),
            pretrained_enabled=bool(pretrained_config.get("enabled", True)),
            pretrained_source=str(pretrained_config.get("source", "local")),
            pretrained_variant=str(pretrained_config.get("variant", "encoder")),
            pretrained_path=pretrained_config.get("path"),
            pretrained_repo_id=pretrained_config.get("repo_id"),
            freeze_encoder=bool(pretrained_config.get("freeze_encoder", False)),
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model.to(device)


def build_case_splits(data_root, training_config: dict) -> tuple[list[str], list[str], list[str]]:
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
    data_root,
    data_config: dict,
    training_config: dict,
):
    """Build train and validation datasets for the configured training regime."""
    training_regime = training_config["training_regime"]
    sampling = training_config["sampling"]
    preprocessing = data_config["preprocessing"]
    crop_margin_voxels = int(preprocessing["crop_margin_voxels"])
    hu_window = tuple(float(value) for value in preprocessing["hu_window"])

    if training_regime == "full_volume":
        return build_monai_aeropath_full_volume_datasets(
            train_ids=train_ids,
            val_ids=val_ids,
            data_root=data_root,
            cache_rate=float(sampling["cache_rate"]),
            crop_margin_voxels=crop_margin_voxels,
            hu_window=hu_window,
        )

    return build_monai_aeropath_datasets(
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
    model = build_model(device, model_config)

    loss_config = training_config["loss"]
    loss_fn = CombinedSegmentationLoss(
        bce_weight=float(loss_config["bce_weight"]),
        dice_weight=float(loss_config["dice_weight"]),
    ).to(device)

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
