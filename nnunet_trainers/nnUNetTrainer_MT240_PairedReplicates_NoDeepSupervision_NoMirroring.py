"""Strictly paired Dataset126 continuation trainers.

The historical MT240 control and soft-clDice runs used nnU-Net's transfer
loader, which intentionally excludes every segmentation head. They therefore
started from independently randomised output heads even though both named the
same Dataset123 checkpoint. This module supplies the auditable replacement:

* the complete Dataset123 state dict, including ``.seg_layers.``, is loaded
  strictly into every arm;
* a required replicate seed controls Python, NumPy and Torch randomness;
* single-thread augmentation plus epoch-derived seeds makes data order and
  perturbations reproducible across arms and after an epoch-boundary resume;
* checkpoints bind student and EMA sidecar with a shared transaction ID.

Control, probability-target soft-clDice, and nominally matched voxel MSE then
share one complete initial state and one stochastic trajectory per replicate.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

if __package__ == "nnunet_trainers":
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring import (
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_ControlDiagnostics_NoDeepSupervision_NoMirroring as _ControlBase,
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring as _SoftBase,
    )
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_VoxelMSE_NoDeepSupervision_NoMirroring import (
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_PlainMSEMatchedWeight_NoDeepSupervision_NoMirroring as _MSEW01Base,
    )
else:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_ControlDiagnostics_NoDeepSupervision_NoMirroring as _ControlBase,
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring as _SoftBase,
    )
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_VoxelMSE_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_PlainMSEMatchedWeight_NoDeepSupervision_NoMirroring as _MSEW01Base,
    )


PAIRED_PROTOCOL_VERSION = "mt240_full_state_epoch_seeded_v1"
EXPECTED_DATASET123_FOLD0_SHA256 = (
    "2f7344a2cdab8d2fa4e43c600a8234f7c73585903df8068d92a25bb6c2e42c5e"
)
REPLICATE_SEEDS = {
    1: 2026082101,
    2: 2026082102,
    3: 2026082103,
    4: 2026082104,
    5: 2026082105,
}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _network_module(network: torch.nn.Module) -> torch.nn.Module:
    module = network.module if hasattr(network, "module") else network
    return module._orig_mod if hasattr(module, "_orig_mod") else module


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes and bytes in a stable key order."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_complete_network_state(
    network: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, object]:
    """Strictly load all tensors and prove that segmentation heads were included."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Paired initial checkpoint does not exist: {path}")
    checkpoint_sha256 = _sha256_file(path)
    if checkpoint_sha256.lower() != expected_checkpoint_sha256.lower():
        raise RuntimeError(
            "Refusing paired launch: Dataset123 checkpoint SHA-256 differs; "
            f"expected {expected_checkpoint_sha256}, found {checkpoint_sha256}"
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("network_weights")
    if not isinstance(saved, Mapping):
        raise RuntimeError(f"Checkpoint has no network_weights mapping: {path}")

    module = _network_module(network)
    current = module.state_dict()
    saved_keys = set(saved)
    current_keys = set(current)
    if saved_keys != current_keys:
        missing = sorted(current_keys - saved_keys)
        extra = sorted(saved_keys - current_keys)
        raise RuntimeError(
            "Full-state checkpoint is not architecture-identical; "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    mismatched = [
        name
        for name in sorted(current)
        if tuple(current[name].shape) != tuple(saved[name].shape)
    ]
    if mismatched:
        raise RuntimeError(f"Full-state tensor shapes differ for: {mismatched[:8]}")

    head_keys = sorted(name for name in saved if ".seg_layers." in name)
    if not head_keys:
        raise RuntimeError("Checkpoint contains no .seg_layers. tensors; full-head pairing is unproven")

    module.load_state_dict(saved, strict=True)
    loaded = module.state_dict()
    for name in saved:
        if not torch.equal(loaded[name].detach().cpu(), saved[name].detach().cpu()):
            raise RuntimeError(f"Strict full-state verification failed for tensor {name}")

    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": checkpoint_sha256,
        "network_state_sha256": state_dict_sha256(loaded),
        "tensor_count": len(saved),
        "segmentation_head_tensor_count": len(head_keys),
        "segmentation_head_keys": head_keys,
    }


class _PairedReplicateMixin:
    """Full-state loading, deterministic epoch seeding and bound checkpoints."""

    paired_arm = "abstract"
    paired_protocol_version = PAIRED_PROTOCOL_VERSION
    expected_init_checkpoint_sha256 = EXPECTED_DATASET123_FOLD0_SHA256

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        replicate_text = os.environ.get("MT_PAIRED_REPLICATE")
        seed_text = os.environ.get("MT_PAIRED_SEED")
        checkpoint_text = os.environ.get("MT_PAIRED_INIT_CHECKPOINT")
        if replicate_text is None or seed_text is None or checkpoint_text is None:
            raise RuntimeError(
                "Paired trainers require MT_PAIRED_REPLICATE, MT_PAIRED_SEED and "
                "MT_PAIRED_INIT_CHECKPOINT"
            )
        self.paired_replicate = int(replicate_text)
        self.paired_seed = int(seed_text)
        if self.paired_replicate not in REPLICATE_SEEDS:
            raise RuntimeError(f"Unsupported paired replicate: {self.paired_replicate}")
        expected_seed = REPLICATE_SEEDS[self.paired_replicate]
        if self.paired_seed != expected_seed:
            raise RuntimeError(
                f"Replicate {self.paired_replicate} requires seed {expected_seed}, "
                f"got {self.paired_seed}"
            )
        if str(fold) != "0":
            raise RuntimeError(f"Paired MT240 replication is fixed to Dataset126 fold 0, got {fold}")
        self.paired_init_checkpoint = str(Path(checkpoint_text).expanduser().resolve())
        self._paired_initial_state: dict[str, object] | None = None
        self._seed_phase("construct", epoch=0)
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _phase_seed(self, phase: str, epoch: int) -> int:
        offsets = {
            "construct": 11,
            "loader": 101,
            "train": 1009,
            "validation": 2003,
        }
        if phase not in offsets:
            raise ValueError(f"Unknown paired RNG phase: {phase}")
        return int(self.paired_seed + offsets[phase] + int(epoch) * 100_003)

    def _seed_phase(self, phase: str, epoch: int) -> int:
        seed = self._phase_seed(phase, epoch)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if os.environ.get("MT_PAIRED_DETERMINISTIC", "1") != "1":
            raise RuntimeError("MT_PAIRED_DETERMINISTIC must remain 1 for the paired protocol")
        torch.use_deterministic_algorithms(True, warn_only=False)
        return seed

    def initialize(self) -> None:
        if self.was_initialized:
            return
        self._seed_phase("construct", epoch=0)
        super().initialize()
        self._paired_initial_state = load_complete_network_state(
            self.network,
            self.paired_init_checkpoint,
            expected_checkpoint_sha256=self.expected_init_checkpoint_sha256,
        )

    def on_train_start(self) -> None:
        if get_allowed_n_proc_DA() != 0:
            raise RuntimeError(
                "Paired protocol requires nnUNet_n_proc_DA=0; nondeterministic worker "
                "scheduling would break cross-arm batch pairing"
            )
        if not self.was_initialized:
            self.initialize()
        self._seed_phase("loader", epoch=self.current_epoch)
        super().on_train_start()
        assert self._paired_initial_state is not None
        self.print_to_log_file(
            "[PairedProtocol] "
            f"version={self.paired_protocol_version} arm={self.paired_arm} "
            f"replicate={self.paired_replicate} seed={self.paired_seed} "
            f"init_checkpoint_sha256={self._paired_initial_state['checkpoint_sha256']} "
            f"initial_network_sha256={self._paired_initial_state['network_state_sha256']} "
            f"seg_head_tensors={self._paired_initial_state['segmentation_head_tensor_count']} "
            "augmentation_workers=0 deterministic_algorithms=true"
        )

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._seed_phase("train", epoch=self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._seed_phase("validation", epoch=self.current_epoch)

    def on_epoch_end(self) -> None:
        # Epochs 0-4 have zero consistency. This checkpoint must be identical
        # within a replicate pair and is the direct pre-treatment audit point.
        if self.current_epoch == self.consistency_warmup_epochs - 1:
            self.save_checkpoint(
                os.path.join(self.output_folder, "checkpoint_pretreatment_epoch005.pth")
            )
        super().on_epoch_end()

    def _paired_checkpoint_fields(self, transaction_id: str) -> dict[str, object]:
        assert self._paired_initial_state is not None
        return {
            "paired_protocol_version": self.paired_protocol_version,
            "paired_arm": self.paired_arm,
            "paired_replicate": self.paired_replicate,
            "paired_seed": self.paired_seed,
            "paired_initial_checkpoint_sha256": self._paired_initial_state[
                "checkpoint_sha256"
            ],
            "paired_initial_network_sha256": self._paired_initial_state[
                "network_state_sha256"
            ],
            "paired_transaction_id": transaction_id,
        }

    def save_checkpoint(self, filename: str) -> None:
        """Atomically replace both files and bind them with one transaction ID."""

        if self.local_rank != 0 or self.disable_checkpointing:
            return
        if self.teacher is None:
            raise RuntimeError("Cannot save paired checkpoint before the EMA teacher exists")
        module = _network_module(self.network)
        transaction_id = uuid.uuid4().hex
        paired = self._paired_checkpoint_fields(transaction_id)
        main = {
            "network_weights": module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": (
                self.grad_scaler.state_dict() if self.grad_scaler is not None else None
            ),
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            **paired,
        }
        teacher = {
            "teacher_weights": self.teacher.state_dict(),
            "_mt_step": self._mt_step,
            "_best_teacher_ema": self._best_teacher_ema,
            "current_epoch": self.current_epoch + 1,
            **paired,
        }

        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_main = destination.with_name(f".{destination.name}.{transaction_id}.tmp")
        temp_teacher = Path(str(temp_main) + ".mt")
        try:
            torch.save(main, temp_main)
            torch.save(teacher, temp_teacher)
            # Either interruption order produces a detectable transaction
            # mismatch; a mixed student/teacher pair is never silently loaded.
            os.replace(temp_teacher, Path(str(destination) + ".mt"))
            os.replace(temp_main, destination)
        finally:
            temp_main.unlink(missing_ok=True)
            temp_teacher.unlink(missing_ok=True)

        metadata = {
            **paired,
            "checkpoint": str(destination),
            "current_epoch": self.current_epoch + 1,
            "student_network_sha256": state_dict_sha256(module.state_dict()),
            "teacher_network_sha256": state_dict_sha256(self.teacher.state_dict()),
        }
        metadata_path = Path(str(destination) + ".paired.json")
        metadata_temp = metadata_path.with_name(f".{metadata_path.name}.{transaction_id}.tmp")
        metadata_temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(metadata_temp, metadata_path)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        if isinstance(filename_or_checkpoint, str):
            main = torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False)
            sidecar_path = filename_or_checkpoint + ".mt"
            if not os.path.isfile(sidecar_path):
                raise RuntimeError(f"Paired checkpoint has no EMA sidecar: {sidecar_path}")
            teacher = torch.load(sidecar_path, map_location="cpu", weights_only=False)
            main_id = main.get("paired_transaction_id")
            teacher_id = teacher.get("paired_transaction_id")
            if not main_id or main_id != teacher_id:
                raise RuntimeError(
                    "Refusing paired resume: student and EMA checkpoint transaction IDs differ"
                )
            for field, expected in (
                ("paired_protocol_version", self.paired_protocol_version),
                ("paired_arm", self.paired_arm),
                ("paired_replicate", self.paired_replicate),
                ("paired_seed", self.paired_seed),
            ):
                if main.get(field) != expected or teacher.get(field) != expected:
                    raise RuntimeError(
                        f"Refusing paired resume: {field} does not match this launch"
                    )
        super().load_checkpoint(filename_or_checkpoint)


class nnUNetTrainer_MT240Paired_Control_NoDeepSupervision_NoMirroring(
    _PairedReplicateMixin, _ControlBase
):
    """Zero-consistency member of each strictly paired replicate."""

    paired_arm = "control"


class nnUNetTrainer_MT240Paired_SoftCLDiceW010_NoDeepSupervision_NoMirroring(
    _PairedReplicateMixin, _SoftBase
):
    """Probability-target soft-clDice treatment at ``w_max=0.10``."""

    paired_arm = "softcldice_w010"


class nnUNetTrainer_MT240Paired_PlainMSEW010_NoDeepSupervision_NoMirroring(
    _PairedReplicateMixin, _MSEW01Base
):
    """Whole-patch probability MSE at the same nominal ``w_max=0.10``."""

    paired_arm = "plainmse_w010"
