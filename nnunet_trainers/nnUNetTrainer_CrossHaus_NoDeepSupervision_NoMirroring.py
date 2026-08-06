"""Isolated Dataset126 CrossHaus and matched dual-head control trainers.

This is a causal, nnU-Net-adapted implementation of Zhu et al., MICCAI 2024,
"Semi-supervised Tubular Structure Segmentation with Cross Geometry and
Hausdorff Distance Consistency".  It deliberately does not modify the shared
K1, K3, five-fold, or SoftCLDice trainer modules.

Both public trainers use the established Dataset126 K1 envelope: one real-GT
and one provenance-unlabelled patch per step, the Dataset123 fold-matched warm
start, 500 epochs at 1e-3, the 5/20 Gaussian ramp, strong-view augmentation,
and EMA *weight averaging* for final deployment.  Unlike Mean Teacher, no EMA
prediction is evaluated and no teacher output is a target.

The common network adds a full-resolution non-negative distance-regression
head to the stock two-class nnU-Net decoder.  Labeled patches supervise both
the normal nnU-Net segmentation loss and an exact foreground EDT target.  The
CrossHaus arm additionally applies, on the unlabelled patch only::

    L_cross = MSE(p, tanh(gamma * r)) + MSE(r, D(p))
    L_hdc   = mean((p - tanh(gamma * r))**2 * (r + D(p)))

where ``p`` is foreground softmax probability, ``r`` is the learned unsigned
distance map, and ``D`` is a differentiable distance approximation.  The
control has the identical architecture, labeled objectives, full-batch
forward, augmentation, optimizer schedule, and EMA averaging, but contributes
exactly zero unlabelled gradient.

Zhu et al. provide no source code and leave several numerical details
unspecified.  Their printed 33^3 Euclidean-kernel log-sum-exp transform is also
impractical for this experiment's 128x160x112 patches (about 82 billion MACs
per map) and prints a temperature sign that would produce a soft maximum.  We
therefore use an explicitly versioned, bounded, six-neighbour differentiable
morphological transform.  It sums successive soft erosions, is finite at empty
and patch-edge masks, uses voxel units like the paper, and is checked against
exact SciPy EDT targets in the test suite.  The precise algorithm and all
hyperparameters are persisted in ``<checkpoint>.crosshaus``.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast, nn

if __package__ == "nnunet_trainers":  # repository-local tests
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _TwoStreamBase,
    )
else:  # installed beside the unchanged parent trainers in nnU-Net
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _TwoStreamBase,
    )


CROSSHAUS_PROTOCOL_VERSION = 1
SOFT_DISTANCE_ALGORITHM = "six_neighbour_iterative_soft_erosion_v1"
DISTANCE_CAP_VOXELS = 16
SOFT_BINARIZATION_SLOPE = 10.0
DISTANCE_HEAVISIDE_GAMMA = 5.0


def _normalised_soft_binarize(probability: torch.Tensor, slope: float) -> torch.Tensor:
    """Smoothly sharpen [0, 1] probabilities while preserving exact endpoints."""
    if probability.ndim != 5 or probability.shape[1] != 1:
        raise ValueError(
            "CrossHaus soft binarization expects (B,1,D,H,W), got "
            f"{tuple(probability.shape)}."
        )
    if slope <= 0:
        raise ValueError(f"Soft-binarization slope must be positive, got {slope}.")
    low = probability.new_tensor(-0.5 * float(slope)).sigmoid()
    high = probability.new_tensor(0.5 * float(slope)).sigmoid()
    sharpened = (torch.sigmoid(float(slope) * (probability - 0.5)) - low) / (high - low)
    return sharpened.clamp(0.0, 1.0)


def _soft_erode_six_neighbour(mask: torch.Tensor) -> torch.Tensor:
    """Differentiable binary erosion with an explicit zero-valued patch exterior."""
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"Soft erosion expects (B,1,D,H,W), got {tuple(mask.shape)}.")

    # max_pool's implicit padding is -inf.  Negating that to implement a min
    # pool would incorrectly make the exterior +inf, so pad explicit zeros.
    along_d = -F.max_pool3d(
        -F.pad(mask, (0, 0, 0, 0, 1, 1), value=0.0),
        kernel_size=(3, 1, 1),
        stride=1,
    )
    along_h = -F.max_pool3d(
        -F.pad(mask, (0, 0, 1, 1, 0, 0), value=0.0),
        kernel_size=(1, 3, 1),
        stride=1,
    )
    along_w = -F.max_pool3d(
        -F.pad(mask, (1, 1, 0, 0, 0, 0), value=0.0),
        kernel_size=(1, 1, 3),
        stride=1,
    )
    return torch.minimum(torch.minimum(along_d, along_h), along_w)


def _soft_morphological_distance_transform(
    foreground_probability: torch.Tensor,
    iterations: int = DISTANCE_CAP_VOXELS,
    *,
    binarization_slope: float = SOFT_BINARIZATION_SLOPE,
) -> torch.Tensor:
    """Return a bounded differentiable unsigned interior-distance map.

    For a binary mask, summing its successive six-neighbour erosions yields a
    city-block morphological distance to the exterior, clipped at
    ``iterations``.  For probabilities, the normalized sigmoid sharpening and
    min-pooling retain a gradient path to the segmentation head.  This is an
    intentional runtime-safe approximation, not the paper's unreleased 33^3
    convolutional implementation.
    """
    if int(iterations) != iterations or iterations < 1:
        raise ValueError(f"Distance iterations must be a positive integer, got {iterations}.")
    current = _normalised_soft_binarize(
        foreground_probability.float(), float(binarization_slope)
    )
    distance = torch.zeros_like(current)
    for _ in range(int(iterations)):
        distance = distance + current
        current = _soft_erode_six_neighbour(current)
    return distance.clamp_(0.0, float(iterations))


def _distance_to_segmentation(distance: torch.Tensor, gamma: float) -> torch.Tensor:
    """Zhu et al.'s smooth non-negative-DT-to-segmentation conversion H(r)."""
    if gamma <= 0:
        raise ValueError(f"Distance Heaviside gamma must be positive, got {gamma}.")
    return torch.tanh(float(gamma) * distance.float().clamp_min(0.0))


def _crosshaus_terms(
    foreground_probability: torch.Tensor,
    predicted_distance: torch.Tensor,
    *,
    distance_iterations: int = DISTANCE_CAP_VOXELS,
    binarization_slope: float = SOFT_BINARIZATION_SLOPE,
    distance_heaviside_gamma: float = DISTANCE_HEAVISIDE_GAMMA,
) -> dict[str, torch.Tensor]:
    """Compute mean-reduced Cross Geometry and Hausdorff consistency terms."""
    if foreground_probability.shape != predicted_distance.shape:
        raise ValueError(
            "CrossHaus expects matching foreground/distance maps, got "
            f"{tuple(foreground_probability.shape)} and {tuple(predicted_distance.shape)}."
        )
    if foreground_probability.ndim != 5 or foreground_probability.shape[1] != 1:
        raise ValueError("CrossHaus expects maps shaped (B,1,D,H,W).")

    probability = foreground_probability.float()
    distance = predicted_distance.float().clamp(0.0, float(distance_iterations))
    pseudo_segmentation = _distance_to_segmentation(
        distance, float(distance_heaviside_gamma)
    )
    pseudo_distance = _soft_morphological_distance_transform(
        probability,
        int(distance_iterations),
        binarization_slope=float(binarization_slope),
    )
    disagreement_sq = (probability - pseudo_segmentation).square()
    segmentation_reference = disagreement_sq.mean()
    distance_reference = (distance - pseudo_distance).square().mean()
    hausdorff_consistency = (
        disagreement_sq * (distance + pseudo_distance)
    ).mean()
    return {
        "segmentation_reference": segmentation_reference,
        "distance_reference": distance_reference,
        "hausdorff_consistency": hausdorff_consistency,
        "cross_geometry": segmentation_reference + distance_reference,
        "pseudo_segmentation": pseudo_segmentation,
        "pseudo_distance": pseudo_distance,
    }


def _exact_foreground_edt_target(
    target: torch.Tensor,
    *,
    foreground_label: int = 1,
    distance_cap: int = DISTANCE_CAP_VOXELS,
) -> torch.Tensor:
    """Compute exact voxel-space foreground EDTs for labeled augmented patches.

    Padding by one zero voxel defines a background exterior even for a mask
    touching every patch face.  Ignore and background labels are both outside
    the foreground.  This function must only receive the provenance-labeled
    sample; it never inspects the all-ignore unlabelled target.
    """
    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError(f"EDT target expects (B,1,D,H,W), got {tuple(target.shape)}.")
    if distance_cap < 1:
        raise ValueError(f"distance_cap must be positive, got {distance_cap}.")
    from scipy.ndimage import distance_transform_edt

    masks = target.detach().cpu().numpy()[:, 0] == int(foreground_label)
    distances: list[np.ndarray] = []
    for mask in masks:
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        edt = distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]
        distances.append(np.minimum(edt, float(distance_cap)).astype(np.float32, copy=False))
    stacked = np.stack(distances, axis=0)[:, None]
    return torch.from_numpy(stacked)


class _CrossHausDistanceHead(nn.Module):
    """One-channel ReLU regression head attached to the final decoder feature map.

    The attribute is intentionally named ``seg_layers``.  nnU-Net's official
    warm-start loader skips keys containing ``.seg_layers.``, so the new head
    initializes independently while every compatible Dataset123 backbone key
    remains mandatory and loadable.
    """

    def __init__(self, input_channels: int, distance_cap: int):
        super().__init__()
        self.seg_layers = nn.Conv3d(int(input_channels), 1, kernel_size=1, bias=True)
        nn.init.normal_(self.seg_layers.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.seg_layers.bias, 0.1)
        self.distance_cap = int(distance_cap)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(self.seg_layers(features)).clamp_max(float(self.distance_cap))


def _crosshaus_network_forward(network: nn.Module, data: torch.Tensor):
    """Instance-bound forward: dual output in train mode, stock logits in eval."""
    # The instance has this function bound as ``forward``; resolving forward on
    # its unchanged class calls the original nnU-Net implementation.
    original_forward = type(network).forward
    if not network.training:
        return original_forward(network, data)

    captured: list[torch.Tensor] = []

    def capture_final_decoder_feature(_module, inputs):
        if len(inputs) != 1:
            raise RuntimeError("Unexpected nnU-Net segmentation-head inputs.")
        captured.append(inputs[0])

    segmentation_layer = network.decoder.seg_layers[-1]
    handle = segmentation_layer.register_forward_pre_hook(capture_final_decoder_feature)
    try:
        segmentation_logits = original_forward(network, data)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            "CrossHaus could not capture exactly one final decoder feature map; "
            f"captured {len(captured)}."
        )
    if not torch.is_tensor(segmentation_logits):
        raise RuntimeError("CrossHaus requires NoDeepSupervision tensor segmentation output.")
    distance = network.crosshaus_auxiliary(captured[0])
    return segmentation_logits, distance


def _attach_crosshaus_head(network: nn.Module, distance_cap: int = DISTANCE_CAP_VOXELS) -> nn.Module:
    """Attach the warm-start-safe auxiliary head and train/eval-aware forward."""
    decoder = getattr(network, "decoder", None)
    segmentation_layers = getattr(decoder, "seg_layers", None)
    if segmentation_layers is None or len(segmentation_layers) < 1:
        raise RuntimeError("CrossHaus requires an nnU-Net decoder.seg_layers sequence.")
    final_layer = segmentation_layers[-1]
    if not isinstance(final_layer, nn.Conv3d):
        raise RuntimeError(
            "CrossHaus fold-0 was designed for a 3D nnU-Net Conv3d segmentation head, got "
            f"{type(final_layer).__name__}."
        )
    if hasattr(network, "crosshaus_auxiliary"):
        raise RuntimeError("Network already has a crosshaus_auxiliary module.")
    network.crosshaus_auxiliary = _CrossHausDistanceHead(
        final_layer.in_channels, int(distance_cap)
    )
    network.forward = MethodType(_crosshaus_network_forward, network)
    return network


def _trainer_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_crosshaus_protocol(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Fail closed on a missing, stale, or cross-arm protocol sidecar."""
    if not isinstance(actual, dict):
        raise RuntimeError("Invalid CrossHaus protocol sidecar payload.")
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}: stored={stored!r}, current={current!r}"
            for key, (stored, current) in sorted(mismatches.items())
        )
        raise RuntimeError(f"CrossHaus protocol/code mismatch on resume: {detail}")


class _CrossHausTwoStreamBase(_TwoStreamBase):
    """Shared dual-head implementation; public subclasses select U consistency."""

    enable_crosshaus_consistency = False
    configured_consistency_max = 0.0
    crosshaus_objective = "abstract"
    protocol_exposure = "K1"
    expected_labelled_per_step = 1
    expected_unlabelled_per_step = 1
    expected_local_batch_size = 2

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.consistency_mode = "crosshaus_task_consistency_no_teacher_target"
        self.consistency_max = float(type(self).configured_consistency_max)
        self.distance_supervision_weight = 1.0
        self.distance_cap_voxels = DISTANCE_CAP_VOXELS
        self.soft_distance_iterations = DISTANCE_CAP_VOXELS
        self.soft_binarization_slope = SOFT_BINARIZATION_SLOPE
        self.distance_heaviside_gamma = DISTANCE_HEAVISIDE_GAMMA
        self._ensure_crosshaus_state()

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if enable_deep_supervision:
            raise RuntimeError("CrossHaus requires NoDeepSupervision.")
        architecture_name = str(configuration_manager.network_arch_class_name)
        if not architecture_name.endswith("PlainConvUNet"):
            raise RuntimeError(
                "CrossHaus fold-0 is pinned to Dataset126 PlainConvUNet, got "
                f"{architecture_name!r}."
            )
        network = _TwoStreamBase.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return _attach_crosshaus_head(network, DISTANCE_CAP_VOXELS)

    def _ensure_crosshaus_state(self) -> None:
        defaults: dict[str, float | int] = {
            "_crosshaus_steps": 0,
            "_crosshaus_seg_supervised": 0.0,
            "_crosshaus_distance_supervised": 0.0,
            "_crosshaus_seg_reference": 0.0,
            "_crosshaus_distance_reference": 0.0,
            "_crosshaus_hdc": 0.0,
            "_crosshaus_probability_mass": 0.0,
            "_crosshaus_distance_mass": 0.0,
            "_crosshaus_edt_seconds": 0.0,
            "_crosshaus_consistency_seconds": 0.0,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def _crosshaus_protocol(self) -> dict[str, Any]:
        return {
            "protocol_version": CROSSHAUS_PROTOCOL_VERSION,
            "objective": type(self).crosshaus_objective,
            "crosshaus_enabled": bool(type(self).enable_crosshaus_consistency),
            "consistency_scope": "provenance_unlabelled_only",
            "protocol_exposure": type(self).protocol_exposure,
            "labelled_per_step": type(self).expected_labelled_per_step,
            "unlabelled_per_step": type(self).expected_unlabelled_per_step,
            "local_batch_size": type(self).expected_local_batch_size,
            "distance_supervision_weight": float(self.distance_supervision_weight),
            "consistency_max": float(self.consistency_max),
            "consistency_warmup_epochs": int(self.consistency_warmup_epochs),
            "consistency_rampup": float(self.consistency_rampup),
            "distance_units": "network_voxels",
            "distance_cap_voxels": int(self.distance_cap_voxels),
            "soft_distance_algorithm": SOFT_DISTANCE_ALGORITHM,
            "soft_distance_iterations": int(self.soft_distance_iterations),
            "soft_binarization_slope": float(self.soft_binarization_slope),
            "distance_heaviside_gamma": float(self.distance_heaviside_gamma),
            "teacher_prediction_target": False,
            "ema_role": "weight_averaging_and_deployment_only",
            "trainer_source_sha256": _trainer_source_sha256(),
        }

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank == 0 and not self.disable_checkpointing:
            torch.save(self._crosshaus_protocol(), filename + ".crosshaus")

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        # Resuming from an in-memory checkpoint dictionary would bypass both
        # sidecar contracts and silently rebuild the EMA copy from the student.
        # Require a filename, fail closed before mutating trainer state, and
        # normalize PathLike values because the inherited MT loader recognizes
        # strings only.
        if not isinstance(filename_or_checkpoint, (str, os.PathLike)):
            raise RuntimeError(
                "CrossHaus resumes require a checkpoint filename plus its .mt "
                "and .crosshaus sidecars."
            )
        filename = os.fspath(filename_or_checkpoint)
        mt_sidecar = filename + ".mt"
        sidecar = filename + ".crosshaus"
        if not os.path.isfile(mt_sidecar):
            raise RuntimeError(
                f"Cannot resume CrossHaus: missing Mean-Teacher sidecar {mt_sidecar}."
            )
        if not os.path.isfile(sidecar):
            raise RuntimeError(
                f"Cannot resume CrossHaus: missing protocol sidecar {sidecar}."
            )
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
        _validate_crosshaus_protocol(payload, self._crosshaus_protocol())
        super().load_checkpoint(filename)
        self.print_to_log_file("[CrossHaus] restored and verified both resume sidecars.")

    def on_train_start(self) -> None:
        super().on_train_start()
        if getattr(self, "is_ddp", False):
            raise RuntimeError("CrossHaus K1 is a single-GPU protocol and refuses DDP.")
        if not hasattr(self.network, "crosshaus_auxiliary"):
            raise RuntimeError("CrossHaus auxiliary distance head is missing from the network.")
        self.print_to_log_file(
            f"[CrossHaus] objective={type(self).crosshaus_objective} protocol=K1 "
            "dual_head=segmentation+unsigned_distance consistency_scope=U-only "
            f"distance_supervision_weight={self.distance_supervision_weight} "
            f"consistency_max={self.consistency_max} soft_dt={SOFT_DISTANCE_ALGORITHM} "
            f"iterations={self.soft_distance_iterations} gamma={self.distance_heaviside_gamma}; "
            "clDice_training=off teacher_forwards=0 EMA=weight-averaging-only."
        )

    def train_step(self, batch: dict) -> dict:
        self._ensure_crosshaus_state()
        target_cpu = batch["target"]
        if isinstance(target_cpu, list):
            raise RuntimeError("CrossHaus requires NoDeepSupervision (one target tensor).")
        labelled_idx, unlabelled_idx = self._batch_stream_indices(list(batch["keys"]))
        labelled_cpu_idx = labelled_idx.detach().cpu()

        edt_started = time.perf_counter()
        distance_target = _exact_foreground_edt_target(
            target_cpu.index_select(0, labelled_cpu_idx),
            distance_cap=int(self.distance_cap_voxels),
        ).to(self.device, non_blocking=True)
        self._crosshaus_edt_seconds += time.perf_counter() - edt_started

        data = batch["data"].to(self.device, non_blocking=True)
        target = target_cpu.to(self.device, non_blocking=True)
        if self.teacher is None:
            self._build_teacher()

        weight = self._consistency_weight()
        use_consistency = bool(type(self).enable_crosshaus_consistency and weight > 0.0)
        use_strong_view = self.current_epoch >= self.consistency_warmup_epochs
        student_in = (
            self._perturb(data, self.student_noise_std, self.student_scale, self.student_shift)
            if use_strong_view
            else data
        )

        self.optimizer.zero_grad(set_to_none=True)
        amp_context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with amp_context:
            network_output = self.network(student_in)
            if not isinstance(network_output, tuple) or len(network_output) != 2:
                raise RuntimeError(
                    "CrossHaus network must return (segmentation_logits, distance) in train mode."
                )
            segmentation_logits, predicted_distance = network_output
            supervised_segmentation = self.loss(
                segmentation_logits.index_select(0, labelled_idx),
                target.index_select(0, labelled_idx),
            )

        with autocast(self.device.type, enabled=False):
            labelled_distance = predicted_distance.index_select(0, labelled_idx).float()
            supervised_distance = F.mse_loss(labelled_distance, distance_target.float())
            supervised_loss = (
                supervised_segmentation.float()
                + float(self.distance_supervision_weight) * supervised_distance
            )

            unlabelled_logits = segmentation_logits.index_select(0, unlabelled_idx).float()
            unlabelled_distance = predicted_distance.index_select(0, unlabelled_idx).float()
            if use_consistency:
                # This deliberately measures host enqueue time only. CUDA
                # synchronization at every training step would perturb the
                # protocol and reduce throughput.
                consistency_started = time.perf_counter()
                foreground_probability = torch.softmax(unlabelled_logits, dim=1)[:, 1:2]
                terms = _crosshaus_terms(
                    foreground_probability,
                    unlabelled_distance,
                    distance_iterations=int(self.soft_distance_iterations),
                    binarization_slope=float(self.soft_binarization_slope),
                    distance_heaviside_gamma=float(self.distance_heaviside_gamma),
                )
                consistency = terms["cross_geometry"] + terms["hausdorff_consistency"]
                self._crosshaus_consistency_seconds += time.perf_counter() - consistency_started
            else:
                # Keep the full-batch dual-head forward identical while making
                # the U contribution explicitly differentiable with zero grad.
                consistency = (
                    unlabelled_logits.sum() + unlabelled_distance.sum()
                ) * 0.0
                zero = consistency.detach()
                terms = {
                    "segmentation_reference": zero,
                    "distance_reference": zero,
                    "hausdorff_consistency": zero,
                }
                foreground_probability = torch.softmax(unlabelled_logits, dim=1)[:, 1:2]
            loss = supervised_loss + float(weight) * consistency

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        self._mt_step += 1
        self._update_ema()
        self._log_cons += float(consistency.detach())
        self._log_sup += float(supervised_loss.detach())
        self._log_w = float(weight) if use_consistency else 0.0
        self._log_n += 1
        self._stream_steps += 1
        self._stream_labelled_samples += int(labelled_idx.numel())
        self._stream_unlabelled_samples += int(unlabelled_idx.numel())

        self._crosshaus_steps += 1
        self._crosshaus_seg_supervised += float(supervised_segmentation.detach())
        self._crosshaus_distance_supervised += float(supervised_distance.detach())
        self._crosshaus_seg_reference += float(terms["segmentation_reference"].detach())
        self._crosshaus_distance_reference += float(terms["distance_reference"].detach())
        self._crosshaus_hdc += float(terms["hausdorff_consistency"].detach())
        self._crosshaus_probability_mass += float(foreground_probability.detach().mean())
        self._crosshaus_distance_mass += float(unlabelled_distance.detach().mean())
        return {"loss": loss.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs) -> None:
        self._ensure_crosshaus_state()
        steps = int(self._crosshaus_steps)
        if steps > 0:
            denominator = float(steps)
            self.print_to_log_file(
                f"[CrossHausLoss] seg_sup={self._crosshaus_seg_supervised / denominator:.5f} "
                f"dt_sup={self._crosshaus_distance_supervised / denominator:.5f} "
                f"seg_ref={self._crosshaus_seg_reference / denominator:.6f} "
                f"dt_ref={self._crosshaus_distance_reference / denominator:.6f} "
                f"hdc={self._crosshaus_hdc / denominator:.6f} "
                f"u_probability_mean={self._crosshaus_probability_mass / denominator:.7f} "
                f"u_distance_mean={self._crosshaus_distance_mass / denominator:.5f} "
                f"edt_seconds={self._crosshaus_edt_seconds:.2f} "
                f"consistency_host_enqueue_seconds={self._crosshaus_consistency_seconds:.2f}"
            )
        super().on_train_epoch_end(train_outputs)
        for name, value in {
            "_crosshaus_steps": 0,
            "_crosshaus_seg_supervised": 0.0,
            "_crosshaus_distance_supervised": 0.0,
            "_crosshaus_seg_reference": 0.0,
            "_crosshaus_distance_reference": 0.0,
            "_crosshaus_hdc": 0.0,
            "_crosshaus_probability_mass": 0.0,
            "_crosshaus_distance_mass": 0.0,
            "_crosshaus_edt_seconds": 0.0,
            "_crosshaus_consistency_seconds": 0.0,
        }.items():
            setattr(self, name, value)


class nnUNetTrainer_CrossHaus_WarmStart_TwoStream_NoDeepSupervision_NoMirroring(
    _CrossHausTwoStreamBase
):
    """Dual-head labeled training plus U-only CrossHaus consistency."""

    enable_crosshaus_consistency = True
    configured_consistency_max = 0.1
    crosshaus_objective = "cross_geometry_plus_hausdorff_u_consistency"


class nnUNetTrainer_CrossHaus_WarmStart_TwoStream_DualHeadControl_NoDeepSupervision_NoMirroring(
    _CrossHausTwoStreamBase
):
    """Matched dual-head labeled control with exactly zero U-stream gradient."""

    enable_crosshaus_consistency = False
    configured_consistency_max = 0.0
    crosshaus_objective = "dual_head_supervised_control"
