"""Supervised training and validation loops.

This file should define the generic epoch and step logic for:
- forward passes
- loss computation
- optimizer steps
- validation summaries

Keep this loop generic so different models and configs can reuse it.
"""
import torch

from lung_airway_segmentation.metrics.segmentation import binary_dice_score_from_logits


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    threshold=0.5,
):
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    for batch in dataloader:
        # get inputs and targets from batch
        images = batch["image"].unsqueeze(1)
        targets = batch["airway_mask"].unsqueeze(1)

        # move to device and ensure float dtype, add channel dimension
        images = images.to(device).float()
        targets = targets.to(device).float()

        # clear old gradients
        optimizer.zero_grad()

        # forward pass
        logits = model(images)

        # compute loss
        loss = loss_fn(logits, targets)
        dice = binary_dice_score_from_logits(logits.detach(), targets, threshold=threshold)

        # backward pass
        loss.backward()

        # optimizer step
        optimizer.step()

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
    threshold=0.5,
):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            
            images = batch["image"].unsqueeze(1)
            targets = batch["airway_mask"].unsqueeze(1)

            # move to device and ensure float dtype
            images = images.to(device).float()
            targets = targets.to(device).float()

            # forward pass
            logits = model(images)

            # compute loss
            loss = loss_fn(logits, targets)
            dice = binary_dice_score_from_logits(logits, targets, threshold=threshold)

            # accumulate loss
            running_loss += loss.item()
            running_dice += float(dice.item())
            num_batches += 1

    if num_batches == 0:
        raise ValueError("Dataloader is empty.")

    epoch_loss = running_loss / num_batches
    epoch_dice = running_dice / num_batches
    return {"loss": epoch_loss, "dice": epoch_dice}
