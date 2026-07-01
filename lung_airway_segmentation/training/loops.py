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
from lung_airway_segmentation.metrics.topology import hard_centerline_metrics_from_masks

# Hard-mask centreline metrics for checkpoint selection are computed at 0.5 by
# default — the operating point at which clDice/TLD are reported (see
# analyse_distal). Decoupled from the validation Dice threshold (0.99 for the
# saturated baseline), which selects best_dice_model.pt. Override per run via
# validation.topology_threshold. Selection runs on the RAW prediction WITHOUT LCC
# and WITHOUT a volume gate (a healthy topology model is legitimately large and
# fragmented at 0.5, 8-35x GT before LCC) — see hard_centerline_metrics_from_masks.
TOPOLOGY_SELECTION_THRESHOLD = 0.5


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
        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float()

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
    compute_topology=True,
    topology_threshold=TOPOLOGY_SELECTION_THRESHOLD,
    topology_max_ratio=None,
    compute_soft_cbdice=False,
):
    """Validate over the full validation set.

    ``dice`` is the hard Dice at ``threshold`` (the configured validation
    threshold, 0.99 for the saturated baseline) and drives ``best_dice_model``.
    When ``compute_topology`` is set, additionally report — at
    ``topology_threshold``, on the RAW prediction WITHOUT LCC — the hard-mask
    centreline metrics that drive ``best_topology_model``: ``cldice`` (the
    selector), topology precision and tree-length detected, plus raw diagnostics
    (predicted/ground-truth foreground-volume ratio and predicted component count,
    which expose how messy / fragmented a high-clDice checkpoint is) and the soft
    loss components (BCE, Dice, soft clDice, and soft cbDice when requested).

    The topology metrics deliberately exclude LCC — selection must not deepen the
    post-processing reliance (the whole point is to surface models that need LCC
    *less*). ``topology_max_ratio`` is ``None`` (no volume gate) by default; it is
    only an optional catastrophic guard (a healthy raw mask at 0.5 is large), and
    gated cases are EXCLUDED from the clDice mean, not scored 0.
    """
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0
    num_cases = 0
    per_case_dice = {}

    running_cldice = 0.0
    running_topology_precision = 0.0
    num_cldice_cases = 0  # cases with a valid (non-gated) clDice
    running_tree_length_detected = 0.0
    running_foreground_volume_ratio = 0.0
    running_component_count = 0.0
    num_gated = 0
    per_case_cldice = {}

    running_bce = 0.0
    running_dice_loss = 0.0
    running_soft_cldice = 0.0
    running_soft_cbdice = 0.0

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

            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()

            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian"
                )

            logits = logits.float()

            # Loss + unweighted diagnostic components in ONE pass (float, for stable
            # soft-skeleton / scipy-EDT). force_* makes validation log the soft terms
            # even during warm-up when their weight is 0, while the weighted total
            # still uses the current weights — so the expensive cbDice EDT runs once,
            # not once for the loss and again for logging.
            total_loss, components = loss_fn.compute_components(
                logits,
                targets,
                force_cldice=compute_topology,
                force_cbdice=compute_topology and compute_soft_cbdice,
            )

            # Binarize logits to get the binary prediction mask.
            # We do NOT apply the voxel-level lung parenchyma mask here because
            # the trachea and mainstem bronchi sit in the mediastinum, outside
            # the lung parenchyma mask, so masking would zero out central airway
            # predictions and collapse Dice. The bounding-box crop uses the lung
            # mask only as an in-plane ROI and preserves the full superior-
            # inferior axis so the trachea remains available.
            predictions = binarize_logits(logits, threshold=threshold)

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

            running_loss += float(total_loss.item())
            running_dice += batch_dice / images.shape[0]
            num_batches += 1

            if compute_topology:
                running_bce += float(components["bce"].item())
                running_dice_loss += float(components["dice"].item())
                running_soft_cldice += float(components["soft_cldice"].item())
                if compute_soft_cbdice and "soft_cbdice" in components:
                    running_soft_cbdice += float(components["soft_cbdice"].item())

                # Hard-mask centreline metrics at the topology threshold, RAW (no
                # LCC) — these drive best_topology_model selection. One combined
                # helper (single prediction skeletonisation). No volume gate by
                # default; a gated (catastrophic-guard) case is excluded from the
                # clDice/precision means, not scored 0.
                topology_predictions = binarize_logits(
                    logits, threshold=topology_threshold
                )
                for i, case_id in enumerate(case_ids):
                    prediction_mask = (
                        topology_predictions[i].squeeze(0).cpu().numpy() > 0
                    )
                    target_mask = targets[i].squeeze(0).cpu().numpy() > 0
                    case_metrics = hard_centerline_metrics_from_masks(
                        prediction_mask,
                        target_mask,
                        max_ratio=topology_max_ratio,
                    )
                    running_tree_length_detected += float(case_metrics["tree_length_detected"])
                    running_component_count += float(case_metrics["component_count"])
                    target_volume = float(target_mask.sum())
                    running_foreground_volume_ratio += (
                        float(prediction_mask.sum()) / target_volume
                        if target_volume > 0
                        else 0.0
                    )
                    num_cases += 1
                    if case_metrics["gated"]:
                        num_gated += 1
                        if case_id is not None:
                            per_case_cldice[str(case_id)] = None
                    else:
                        case_cldice = float(case_metrics["cldice"])
                        running_cldice += case_cldice
                        running_topology_precision += float(case_metrics["topology_precision"])
                        num_cldice_cases += 1
                        if case_id is not None:
                            per_case_cldice[str(case_id)] = case_cldice

    if num_batches == 0:
        raise ValueError("Dataloader is empty.")

    metrics = {
        "loss": running_loss / num_batches,
        "dice": running_dice / num_batches,
        "per_case_dice": per_case_dice,
    }

    if compute_topology and num_cases > 0:
        # soft_cldice_loss / soft_cbdice_loss are LOSSES (lower is better), unlike
        # cldice (a score, higher is better) — named with the _loss suffix so the
        # direction is unambiguous in history.json. Always-valid raw diagnostics
        # (TLD, fg-volume ratio, component count, gated count) average over all
        # cases; clDice / topology precision average over the non-gated subset.
        metrics.update(
            {
                "tree_length_detected": running_tree_length_detected / num_cases,
                "foreground_volume_ratio": running_foreground_volume_ratio / num_cases,
                "predicted_component_count": running_component_count / num_cases,
                "gated_case_count": num_gated,
                "per_case_cldice": per_case_cldice,
                "bce_loss": running_bce / num_batches,
                "dice_loss": running_dice_loss / num_batches,
                "soft_cldice_loss": running_soft_cldice / num_batches,
            }
        )
        # Omit clDice entirely if every case was gated (no valid selection signal)
        # — the engine guards on its presence so best_topology_model just isn't
        # updated that epoch.
        if num_cldice_cases > 0:
            metrics["cldice"] = running_cldice / num_cldice_cases
            metrics["topology_precision"] = running_topology_precision / num_cldice_cases
        if compute_soft_cbdice:
            metrics["soft_cbdice_loss"] = running_soft_cbdice / num_batches

    return metrics
