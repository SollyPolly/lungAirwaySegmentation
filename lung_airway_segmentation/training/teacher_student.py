"""Mean Teacher training utilities."""

import torch
import torch.nn as nn

from lung_airway_segmentation.metrics.segmentation import binary_dice_score_from_logits


def prepare_segmentation_batch(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a labelled MONAI batch to the device with channel-first tensors."""
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

    return images.to(device).float(), targets.to(device).float()


def prepare_unlabelled_views(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return strong student and weak teacher views from one ATM'22 batch."""
    student_images = batch["image"]
    teacher_images = batch.get("teacher_image", student_images)

    if student_images.ndim == 4:
        student_images = student_images.unsqueeze(1)
    elif student_images.ndim != 5:
        raise ValueError(
            f"Expected unlabeled student batch to be 4D or 5D, got shape {student_images.shape}"
        )

    if teacher_images.ndim == 4:
        teacher_images = teacher_images.unsqueeze(1)
    elif teacher_images.ndim != 5:
        raise ValueError(
            f"Expected unlabeled teacher batch to be 4D or 5D, got shape {teacher_images.shape}"
        )

    return student_images.to(device).float(), teacher_images.to(device).float()


def update_ema(student: nn.Module, teacher: nn.Module, alpha: float) -> None:
    """Update teacher parameters by EMA and copy non-trainable buffers."""
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("EMA alpha must be in [0.0, 1.0).")

    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())
    if student_parameters.keys() != teacher_parameters.keys():
        raise ValueError("Student and teacher parameters do not match.")

    student_buffers = dict(student.named_buffers())
    teacher_buffers = dict(teacher.named_buffers())
    if student_buffers.keys() != teacher_buffers.keys():
        raise ValueError("Student and teacher buffers do not match.")

    with torch.no_grad():
        for name, teacher_parameter in teacher_parameters.items():
            teacher_parameter.mul_(alpha).add_(
                student_parameters[name],
                alpha=1.0 - alpha,
            )
        for name, teacher_buffer in teacher_buffers.items():
            teacher_buffer.copy_(student_buffers[name])


def generate_teacher_probabilities(
    teacher: nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Generate detached soft teacher targets for consistency training."""
    teacher.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=torch.float16,
        ):
            logits = teacher(images)
        return torch.sigmoid(logits.float())


def next_unlabelled_batch(unlabelled_loader, unlabelled_iterator):
    """Read the next unlabeled batch, restarting the loader when necessary."""
    try:
        return next(unlabelled_iterator), unlabelled_iterator
    except StopIteration:
        unlabelled_iterator = iter(unlabelled_loader)
        return next(unlabelled_iterator), unlabelled_iterator


def train_semisupervised_epoch(
    student: nn.Module,
    teacher: nn.Module,
    labelled_loader,
    unlabelled_loader,
    loss_fn,
    consistency_loss_fn,
    optimizer,
    scaler,
    device,
    ema_alpha,
    consistency_weight,
    use_consistency,
    use_amp=False,
    threshold=0.5,
):
    """Train one student epoch while updating the EMA teacher."""
    student.train()
    teacher.eval()

    running_total_loss = 0.0
    running_supervised_loss = 0.0
    running_consistency_loss = 0.0
    running_dice = 0.0
    running_confident_fraction = 0.0
    running_confident_foreground_fraction = 0.0
    running_confident_background_fraction = 0.0
    num_steps = 0
    unlabelled_iterator = iter(unlabelled_loader) if use_consistency else None

    for labelled_batch in labelled_loader:
        images_labelled, targets_labelled = prepare_segmentation_batch(labelled_batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
            dtype=torch.float16,
        ):
            student_logits_labelled = student(images_labelled)
            supervised_loss = loss_fn(student_logits_labelled, targets_labelled)

        consistency_loss = supervised_loss.new_zeros(())
        confident_fraction = 0.0
        confident_foreground_fraction = 0.0
        confident_background_fraction = 0.0
        if use_consistency:
            unlabelled_batch, unlabelled_iterator = next_unlabelled_batch(
                unlabelled_loader,
                unlabelled_iterator,
            )
            student_images, teacher_images = prepare_unlabelled_views(unlabelled_batch, device)
            teacher_probabilities = generate_teacher_probabilities(
                teacher,
                teacher_images,
                device=device,
                use_amp=use_amp,
            )

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=torch.float16,
            ):
                student_logits_unlabelled = student(student_images)
                consistency_loss = consistency_loss_fn(
                    student_logits_unlabelled,
                    teacher_probabilities,
                )

            foreground_mask, background_mask = consistency_loss_fn.confidence_masks(
                teacher_probabilities
            )
            confident_fraction = float(
                (foreground_mask | background_mask).float().mean().item()
            )
            confident_foreground_fraction = float(foreground_mask.float().mean().item())
            confident_background_fraction = float(background_mask.float().mean().item())

        total_loss = supervised_loss + float(consistency_weight) * consistency_loss
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        update_ema(student, teacher, float(ema_alpha))

        dice = binary_dice_score_from_logits(
            student_logits_labelled.detach(),
            targets_labelled,
            threshold=threshold,
        )
        running_total_loss += float(total_loss.item())
        running_supervised_loss += float(supervised_loss.item())
        running_consistency_loss += float(consistency_loss.item())
        running_dice += float(dice.item())
        running_confident_fraction += confident_fraction
        running_confident_foreground_fraction += confident_foreground_fraction
        running_confident_background_fraction += confident_background_fraction
        num_steps += 1

    if num_steps == 0:
        raise ValueError("Labelled dataloader is empty.")

    return {
        "loss": running_total_loss / num_steps,
        "supervised_loss": running_supervised_loss / num_steps,
        "consistency_loss": running_consistency_loss / num_steps,
        "dice": running_dice / num_steps,
        "confident_fraction": running_confident_fraction / num_steps,
        "confident_foreground_fraction": (
            running_confident_foreground_fraction / num_steps
        ),
        "confident_background_fraction": (
            running_confident_background_fraction / num_steps
        ),
        "consistency_weight": float(consistency_weight),
    }
