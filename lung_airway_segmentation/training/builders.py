"""Dataset, dataloader, model, and optimizer builders for training."""

import copy

import torch
import torch.nn as nn
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate

from lung_airway_segmentation.datasets.monai_atm22 import (
    build_atm22_labelled_records,
    build_atm22_labelled_val_transforms,
    build_monai_atm22_dataset,
    build_monai_atm22_labelled_datasets,
    build_monai_atm22_selftraining_dataset,
)
from lung_airway_segmentation.io.atm22_layout import resolve_case_paths
from lung_airway_segmentation.datasets.monai_aeropath import (
    build_monai_aeropath_datasets,
    build_monai_aeropath_full_volume_datasets,
)
from lung_airway_segmentation.datasets.splits import (
    create_semisupervised_split,
    create_train_val_test_split,
)
from lung_airway_segmentation.io.atm22_layout import list_case_ids as list_atm22_case_ids
from lung_airway_segmentation.io.case_layout import list_case_ids as list_aeropath_case_ids
from lung_airway_segmentation.losses.segmentation import CombinedSegmentationLoss
from lung_airway_segmentation.models.baseline_unet import build_baseline_unet
from lung_airway_segmentation.models.ct_fm_segresnet import build_ct_fm_segresnet
from lung_airway_segmentation.preprocessing.geometry import normalize_margin
from lung_airway_segmentation.reproducibility import make_seeded_generator, seed_worker
from lung_airway_segmentation.training.config import resolve_project_path


# --checkpoint choices for the analysis/prediction scripts → on-disk filenames.
# 'best' is the Dice-selected checkpoint, kept as a compatibility alias for
# best_dice_model.pt. 'topology' is the hard-clDice@0.5 selection (see
# training/loops.py + engine.py); both 'dice'/'topology' are produced by runs
# from 2026-06-15 onward.
CHECKPOINT_FILENAMES = {
    "best": "best_model.pt",
    "dice": "best_dice_model.pt",
    "topology": "best_topology_model.pt",
    "last": "last_model.pt",
}


def is_strict_improvement(value, best_value) -> bool:
    """Strict-improvement test for checkpoint selection.

    Returns ``True`` only when ``value`` is a real number strictly greater than
    ``best_value``. ``None`` (an absent/invalid metric — e.g. clDice during the
    topology warm-up or a fully volume-gated epoch) never counts as an improvement,
    so best_dice_model and best_topology_model are each updated only on their own
    valid signal and can land on different epochs.
    """
    return value is not None and value > best_value


def resolve_checkpoint_path(run_dir, choice: str):
    """Map a ``--checkpoint`` choice to its file under ``run_dir``.

    Runs that predate three-checkpoint saving only have ``best_model.pt`` /
    ``last_model.pt``; for those, ``dice``/``topology`` fall back to
    ``best_model.pt`` with a printed warning (topology falls back to the
    Dice-selected weights, so the warning is load-bearing — the result is NOT a
    topology-selected checkpoint).
    """
    from pathlib import Path

    run_dir = Path(run_dir)
    path = run_dir / CHECKPOINT_FILENAMES[choice]
    if not path.exists() and choice in ("dice", "topology"):
        legacy = run_dir / "best_model.pt"
        if legacy.exists():
            print(
                f"[checkpoint] {path.name} not found in {run_dir.name} (run predates "
                f"three-checkpoint saving); falling back to best_model.pt"
                + (" — Dice-selected, NOT topology-selected." if choice == "topology" else ".")
            )
            return legacy
    return path


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


def resolve_case_splits(data_config: dict, training_config: dict) -> dict:
    """Resolve case splits for either dataset as a unified mapping.

    Returns keys ``labelled_train``, ``unlabelled_train``, ``val``, ``test``.
    AeroPath uses fractional three-way splits and has an empty
    ``unlabelled_train``; ATM'22 uses a count-based four-way semi-supervised
    split so the same test/val sets are shared by the supervised baseline and
    the Mean Teacher run (paired comparison).
    """
    dataset_name = str(data_config["dataset_name"]).lower()
    seed = int(training_config["seed"])

    if dataset_name == "aeropath":
        data_root = resolve_project_path(data_config["raw_data_root"])
        case_ids = list_aeropath_case_ids(data_root)
        splits = training_config["splits"]
        train_ids, val_ids, test_ids = create_train_val_test_split(
            case_ids,
            train_split=float(splits["train_fraction"]),
            val_split=float(splits["val_fraction"]),
            test_split=float(splits["test_fraction"]),
            seed=seed,
        )
        return {
            "labelled_train": train_ids,
            "unlabelled_train": [],
            "val": val_ids,
            "test": test_ids,
        }

    if dataset_name == "atm22":
        batch_root = resolve_project_path(data_config["batch_root"])
        case_ids = list_atm22_case_ids(batch_root)
        labelled_split = training_config["labelled_split"]
        return create_semisupervised_split(
            case_ids,
            test_count=int(labelled_split["test_count"]),
            val_count=int(labelled_split["val_count"]),
            labelled_count=int(labelled_split["labelled_count"]),
            seed=seed,
        )

    raise ValueError(f"Unsupported dataset_name for splits: {dataset_name}")


def build_datasets(
    train_ids,
    val_ids,
    data_config: dict,
    training_config: dict,
):
    """Build labelled train and validation datasets for the configured dataset."""
    dataset_name = str(data_config["dataset_name"]).lower()
    training_regime = training_config["training_regime"]
    sampling = training_config["sampling"]
    preprocessing = data_config["preprocessing"]
    hu_window = tuple(float(value) for value in preprocessing["hu_window"])

    if dataset_name == "atm22":
        if training_regime != "patch":
            raise ValueError(
                "ATM'22 labelled training currently supports only training_regime = 'patch'."
            )
        batch_root = resolve_project_path(data_config["batch_root"])
        return build_monai_atm22_labelled_datasets(
            train_ids=train_ids,
            val_ids=val_ids,
            batch_root=batch_root,
            patch_size=tuple(int(value) for value in sampling["patch_size"]),
            patches_per_case=int(sampling["patches_per_case"]),
            foreground_probability=float(sampling["foreground_probability"]),
            cache_rate=float(sampling["cache_rate"]),
            hu_window=hu_window,
            distal_sampling=sampling.get("distal_sampling"),
        )

    if dataset_name != "aeropath":
        raise ValueError(f"Unsupported dataset_name for datasets: {dataset_name}")

    data_root = resolve_project_path(data_config["raw_data_root"])
    crop_margin_voxels = normalize_margin(preprocessing["crop_margin_voxels"])

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


def build_selftraining_records(
    labelled_ids,
    pseudo_entries,
    *,
    batch_root,
    labelled_oversample: int = 1,
):
    """Assemble combined CT+mask records for topology-filtered self-training.

    Real labelled cases (ground-truth masks) are duplicated ``labelled_oversample``
    times so the 20 trusted cases stay influential against the larger pseudo pool;
    each accepted pseudo case contributes one record whose ``airway_mask`` points at
    its generated pseudo-mask file (``pseudo_entries`` = manifest ``cases`` with
    ``accepted`` true, each carrying ``case_id`` and ``mask_path``). The CT for a
    pseudo case is the same native ATM'22 file the real pipeline reads.
    """
    if labelled_oversample < 1:
        raise ValueError("labelled_oversample must be >= 1.")

    labelled_records = build_atm22_labelled_records(labelled_ids, batch_root=batch_root)
    records = list(labelled_records) * int(labelled_oversample)

    for entry in pseudo_entries:
        case_id = str(entry["case_id"])
        mask_path = entry["mask_path"]
        if not mask_path:
            raise ValueError(f"Accepted pseudo case {case_id} has no mask_path in the manifest.")
        paths = resolve_case_paths(case_id, batch_root=batch_root)
        records.append({
            "case_id": case_id,
            "image": str(paths["ct"]),
            "airway_mask": str(mask_path),
        })

    return records


def build_selftraining_datasets(
    labelled_ids,
    val_ids,
    pseudo_entries,
    data_config: dict,
    training_config: dict,
    *,
    labelled_oversample: int = 1,
):
    """Build the combined self-training train dataset and the real-labelled val dataset."""
    if str(data_config["dataset_name"]).lower() != "atm22":
        raise ValueError("Self-training currently supports only the ATM'22 dataset.")
    if training_config["training_regime"] != "patch":
        raise ValueError("Self-training currently supports only training_regime = 'patch'.")

    batch_root = resolve_project_path(data_config["batch_root"])
    sampling = training_config["sampling"]
    hu_window = tuple(float(value) for value in data_config["preprocessing"]["hu_window"])
    cache_rate = float(sampling["cache_rate"])

    train_records = build_selftraining_records(
        labelled_ids,
        pseudo_entries,
        batch_root=batch_root,
        labelled_oversample=int(labelled_oversample),
    )
    train_dataset = build_monai_atm22_selftraining_dataset(
        train_records,
        patch_size=tuple(int(value) for value in sampling["patch_size"]),
        patches_per_case=int(sampling["patches_per_case"]),
        foreground_probability=float(sampling["foreground_probability"]),
        cache_rate=cache_rate,
        hu_window=hu_window,
    )

    # Validation stays on the real labelled val set (full-volume transforms).
    val_records = build_atm22_labelled_records(val_ids, batch_root=batch_root)
    dataset_class = CacheDataset if cache_rate > 0.0 else Dataset
    cache_kwargs = {"cache_rate": cache_rate} if cache_rate > 0.0 else {}
    val_dataset = dataset_class(
        data=val_records,
        transform=build_atm22_labelled_val_transforms(hu_window=hu_window),
        **cache_kwargs,
    )

    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    seed: int | None = None,
):
    """Build train and validation dataloaders for baseline training.

    Passing ``seed`` makes the train loader's shuffle order and per-worker RNG
    reproducible across runs (the val loader is unshuffled but still gets the
    worker seeding so any random transforms it runs are deterministic).
    """
    loader_generator = make_seeded_generator(seed) if seed is not None else None
    worker_init_fn = seed_worker if seed is not None else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        generator=loader_generator,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        worker_init_fn=worker_init_fn,
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
        positive_class_weight=float(loss_config.get("positive_class_weight", 1.0)),
        cldice_weight=float(loss_config.get("cldice_weight", 0.0)),
        cldice_iterations=int(loss_config.get("cldice_iterations", 10)),
        cbdice_weight=float(loss_config.get("cbdice_weight", 0.0)),
        cbdice_iterations=int(loss_config.get("cbdice_iterations", 10)),
        topo_weight=float(loss_config.get("topo_weight", 0.0)),
        calibre_weight_max=float(loss_config.get("calibre_weight_max", 1.0)),
        calibre_radius_voxels=float(loss_config.get("calibre_radius_voxels", 3.0)),
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


def build_teacher(student: nn.Module) -> nn.Module:
    """Build a frozen EMA teacher initialized from the student."""
    teacher = copy.deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()
    return teacher


def build_unlabelled_dataloader(
    atm22_config: dict,
    training_config: dict,
    case_ids,
) -> DataLoader:
    """Build the ATM'22 loader over an explicit set of unlabelled case IDs.

    ``case_ids`` must be the unlabelled-train subset only — never the val or
    test cases — so held-out cases do not leak into training as unlabelled data.
    """
    atm22_root = resolve_project_path(atm22_config["batch_root"])
    atm22_ids = list(case_ids)
    if not atm22_ids:
        raise ValueError("build_unlabelled_dataloader received an empty case_ids set.")
    sampling = training_config["sampling"]
    unlabelled_sampling = training_config.get("unlabelled_sampling", {})
    preprocessing = atm22_config.get("preprocessing", {})
    hu_window = tuple(float(value) for value in preprocessing.get("hu_window", (-1024, 600)))

    unlabelled_dataset = build_monai_atm22_dataset(
        atm22_ids,
        batch_root=atm22_root,
        patch_size=tuple(int(value) for value in sampling["patch_size"]),
        patches_per_case=int(sampling["patches_per_case"]),
        cache_rate=float(unlabelled_sampling.get("cache_rate", 0.0)),
        hu_window=hu_window,
    )

    return DataLoader(
        unlabelled_dataset,
        batch_size=int(training_config["batch_size_unlabelled"]),
        num_workers=int(training_config["num_workers"]),
        shuffle=True,
        collate_fn=list_data_collate,
        generator=make_seeded_generator(int(training_config["seed"])),
        worker_init_fn=seed_worker,
    )
