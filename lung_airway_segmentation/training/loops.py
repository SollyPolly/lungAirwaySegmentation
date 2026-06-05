"""Supervised training and validation loops.

This file should define the generic epoch and step logic for:
- forward passes
- loss computation
- optimizer steps
- validation summaries

Keep this loop generic so different models and configs can reuse it.
"""
import torch

from monai.inferers import sliding_window_inference

from lung_airway_segmentation.inference.postprocess import binarize_logits
from lung_airway_segmentation.metrics.segmentation import (
    binary_dice_score_from_logits,
    binary_dice_score_from_masks,
)


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    scaler,
    use_amp=False,
    threshold=0.5,
):
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for batch in dataloader:
        # get inputs and targets from batch
        images = batch["image"]
        targets = batch["airway_mask"]

        if images.ndim == 4:
            images = images.unsqueeze(1)
        elif images.ndim != 5:
            raise ValueError(f"Expected image batch to be 4D or 5D, got shape {images.shape}")

        if targets.ndim == 4:
            targets = targets.unsqueeze(1)
        elif targets.ndim != 5:
            raise ValueError(f"Expected target batch to be 4D or 5D, got shape {targets.shape}")

        # move to device and ensure float dtype, add channel dimension
        images = images.to(device).float()
        targets = targets.to(device).float()

        # clear old gradients
        optimizer.zero_grad(set_to_none=True)

        # forward pass
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            logits = model(images)
            loss = loss_fn(logits, targets)
        dice = binary_dice_score_from_logits(logits.detach(), targets, threshold=threshold)

        # backward pass
        scaler.scale(loss).backward()

        # optimizer step
        scaler.step(optimizer)
        scaler.update()

        # accumulate loss
        running_loss += loss.item()
        running_dice += float(dice.item())
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Dataloader is empty.")

    epoch_loss = running_loss / num_batches
    epoch_dice = running_dice / num_batches
    return {"loss": epoch_loss, "dice": epoch_dice}


def validate_one_epoch(
    model,
    dataloader,
    loss_fn,
    device,
    roi_size,
    sw_batch_size=1,
    overlap=0.75,
    use_amp=False,
    threshold=0.5,
):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0
    per_case_dice = {}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"]
            targets = batch["airway_mask"]

            if images.ndim == 4:
                images = images.unsqueeze(1)
            elif images.ndim != 5:
                raise ValueError(f"Expected image batch to be 4D or 5D, got shape {images.shape}")

            if targets.ndim == 4:
                targets = targets.unsqueeze(1)
            elif targets.ndim != 5:
                raise ValueError(f"Expected target batch to be 4D or 5D, got shape {targets.shape}")

            images = images.to(device).float()
            targets = targets.to(device).float()

            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian"
                )
                loss = loss_fn(logits, targets)

            # Binarize logits to get the binary prediction mask.
            # We do NOT apply the voxel-level lung parenchyma mask here because
            # the trachea and mainstem bronchi sit in the mediastinum, outside
            # the lung parenchyma mask, so masking would zero out central airway
            # predictions and collapse Dice. The bounding-box crop uses the lung
            # mask only as an in-plane ROI and preserves the full superior-
            # inferior axis so the trachea remains available.
            predictions = binarize_logits(logits.float(), threshold=threshold)

            # Compute and record per-case Dice so we can see which cases are failing.
            case_ids = batch.get("case_id", [None] * images.shape[0])
            if isinstance(case_ids, str):
                case_ids = [case_ids]

            batch_dice = 0.0
            for i, case_id in enumerate(case_ids):
                case_dice = float(binary_dice_score_from_masks(
                    predictions[i].cpu().numpy(),
                    targets[i].cpu().numpy(),
                ))
                batch_dice += case_dice
                if case_id is not None:
                    per_case_dice[str(case_id)] = case_dice

            running_loss += loss.item()
            running_dice += batch_dice / images.shape[0]
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Dataloader is empty.")

    return {
        "loss": running_loss / num_batches,
        "dice": running_dice / num_batches,
        "per_case_dice": per_case_dice,
    }
