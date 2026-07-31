"""Exposure-scaled K3 Mean Teacher for Dataset126.

This module deliberately leaves the established 1+1 two-stream trainer
untouched.  Each optimisation step draws one real-GT patch and three patches
from three distinct provenance-unlabelled cases.  The four samples are CPU
transport only: the GT patch and each unlabelled patch are forwarded and
back-propagated sequentially so GPU activation memory remains close to the
1+1 run.

The objective is::

    L = L_supervised + weight * mean(L_consistency_1, ..., L_consistency_3)

There is still exactly one optimiser step and one EMA update per GT patch.
Consequently K3 changes unlabelled sampling/exposure without tripling the
nominal consistency coefficient or the supervised optimisation schedule.

Deploy this file together with the three unchanged parent trainer modules in
nnU-Net's ``variants/network_architecture`` trainer directory.
"""

from __future__ import annotations

import os
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast

if __package__ == "nnunet_trainers":  # repository-local tests
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (
        _normalise_case_key,
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _TwoStreamBase,
    )
else:  # installed beside the parent trainer in nnU-Net
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (  # noqa: E501
        _normalise_case_key,
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _TwoStreamBase,
    )


class ProvenanceK3DataLoader(nnUNetDataLoader):
    """Patch loader with exactly one GT and three distinct unlabelled cases."""

    transport_batch_size = 4
    unlabelled_crops_per_step = 3
    labelled_foreground_probability = 0.5

    def __init__(self, *args, labelled_identifiers, unlabelled_identifiers, **kwargs):
        super().__init__(*args, **kwargs)
        if self.batch_size != self.transport_batch_size:
            raise RuntimeError(
                "The K3 transport loader requires batch_size=4 (one GT + three unlabelled), "
                f"got {self.batch_size}."
            )
        self.labelled_identifiers = np.asarray(sorted(labelled_identifiers), dtype=object)
        self.unlabelled_identifiers = np.asarray(sorted(unlabelled_identifiers), dtype=object)
        self._labelled_identifier_set = set(self.labelled_identifiers.tolist())
        if self.labelled_identifiers.size == 0:
            raise ValueError("The K3 labelled stream is empty.")
        if self.unlabelled_identifiers.size < self.unlabelled_crops_per_step:
            raise ValueError(
                "K3 requires at least three unlabelled cases so each macro-step can draw "
                "three distinct cases."
            )

        # This is intentionally 0.5 rather than a value derived from the
        # transport batch of four. It preserves the labelled-patch distribution
        # of the K1 run (one forced-foreground GT crop every other GT draw in
        # expectation) and avoids introducing a supervised-sampling confound.
        self.labelled_foreground_probability = type(self).labelled_foreground_probability
        self._selected_keys: list[str] = []
        self.get_do_oversample = self._k3_do_oversample

    def get_indices(self):
        labelled = str(np.random.choice(self.labelled_identifiers))
        unlabelled = [
            str(value)
            for value in np.random.choice(
                self.unlabelled_identifiers,
                size=self.unlabelled_crops_per_step,
                replace=False,
            )
        ]
        if len(set(unlabelled)) != self.unlabelled_crops_per_step:
            raise RuntimeError("K3 sampler produced duplicate unlabelled cases within one step.")
        self._selected_keys = [labelled, *unlabelled]
        np.random.shuffle(self._selected_keys)
        return list(self._selected_keys)

    def _k3_do_oversample(self, sample_idx: int) -> bool:
        key = self._selected_keys[sample_idx]
        if key not in self._labelled_identifier_set:
            return False
        return bool(np.random.random() < self.labelled_foreground_probability)


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_K3_NoDeepSupervision_NoMirroring(
    _TwoStreamBase
):
    """Dataset126 exposure arm: one GT plus three mean-reduced U crops per step."""

    labelled_crops_per_step = 1
    unlabelled_crops_per_step = 3
    transport_batch_size = 4
    expected_planned_batch_size = 2
    consistency_reduction = "mean"
    k3_protocol_version = 1

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # nnU-Net reflects over this exact signature; do not replace it with
        # *args/**kwargs.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._k3_epoch_selected_by_case: Counter[str] = Counter()
        self._k3_epoch_active_by_case: Counter[str] = Counter()
        self._k3_total_selected_by_case: Counter[str] = Counter()
        self._k3_total_active_by_case: Counter[str] = Counter()
        self._k3_total_labelled_samples = 0
        self._k3_total_unlabelled_selected = 0
        self._k3_total_unlabelled_active = 0

    def get_dataloaders(self):
        if self.batch_size != self.expected_planned_batch_size:
            raise RuntimeError(
                "MT240_K3 is calibrated for the existing local nnU-Net plan batch_size=2; "
                f"got {self.batch_size}."
            )
        if getattr(self, "is_ddp", False):
            raise RuntimeError("MT240_K3 is a single-GPU protocol and refuses DDP.")
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation, dummy_2d, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation,
            deep_supervision_scales,
            mirror_axes,
            dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        tr_keys, val_keys = self.do_split()
        tr_keys = [_normalise_case_key(key) for key in tr_keys]
        val_keys = [_normalise_case_key(key) for key in val_keys]
        provenance, contract = self._load_experiment_contract()
        self._assert_contract_matches_split(contract, self.fold, tr_keys, val_keys)

        unknown = (set(tr_keys) | set(val_keys)) - set(provenance)
        if unknown:
            raise RuntimeError(f"Missing provenance for cases: {sorted(unknown)}")
        if any(provenance[key] != "gt" for key in val_keys):
            raise RuntimeError("K3 validation must contain real-GT cases only.")
        labelled = [key for key in tr_keys if provenance[key] == "gt"]
        unlabelled = [key for key in tr_keys if provenance[key] == self.secondary_provenance]
        if not labelled or len(unlabelled) < self.unlabelled_crops_per_step:
            raise RuntimeError(
                f"K3 needs at least one GT and three unlabelled train cases, got "
                f"{len(labelled)} GT and {len(unlabelled)} unlabelled."
            )
        if not self.label_manager.has_ignore_label:
            raise RuntimeError("K3 Mean Teacher requires the Dataset126 ignore label.")
        self._case_provenance = provenance

        dataset_tr = self.dataset_class(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        dl_tr = ProvenanceK3DataLoader(
            dataset_tr,
            self.transport_batch_size,
            initial_patch_size,
            patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            labelled_identifiers=labelled,
            unlabelled_identifiers=unlabelled,
        )
        dl_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            patch_size,
            patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed = get_allowed_n_proc_DA()
        if allowed == 0:
            train_gen = SingleThreadedAugmenter(dl_tr, None)
            val_gen = SingleThreadedAugmenter(dl_val, None)
        else:
            train_gen = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed,
                num_cached=max(6, allowed // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            val_gen = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed // 2),
                num_cached=max(3, allowed // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(train_gen)
        _ = next(val_gen)
        self.print_to_log_file(
            f"[K3] fold={self.fold}: {len(labelled)} GT train + {len(unlabelled)} "
            f"unlabelled train; {len(val_keys)} GT-only validation; CPU transport=1+3, "
            "GPU microbatch=1; U reduction=mean."
        )
        return train_gen, val_gen

    def _batch_stream_positions(self, keys: list[str]) -> tuple[int, list[int], list[str]]:
        normalised = [_normalise_case_key(key) for key in keys]
        try:
            labelled = [
                i for i, key in enumerate(normalised) if self._case_provenance[key] == "gt"
            ]
            unlabelled = [
                i
                for i, key in enumerate(normalised)
                if self._case_provenance[key] == self.secondary_provenance
            ]
        except KeyError as exc:
            raise RuntimeError(f"K3 batch contains a case with no provenance: {exc.args[0]}") from exc
        unlabelled_keys = [normalised[i] for i in unlabelled]
        if (
            len(labelled) != self.labelled_crops_per_step
            or len(unlabelled) != self.unlabelled_crops_per_step
            or len(set(unlabelled_keys)) != self.unlabelled_crops_per_step
        ):
            raise RuntimeError(
                "Expected one GT plus three distinct unlabelled cases, "
                f"got keys={normalised}, GT indices={labelled}, U indices={unlabelled}."
            )
        return labelled[0], unlabelled, unlabelled_keys

    @staticmethod
    def _backward(term: torch.Tensor, grad_scaler) -> None:
        if grad_scaler is None:
            term.backward()
        else:
            grad_scaler.scale(term).backward()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        if isinstance(target, list):
            raise RuntimeError("MT240_K3 requires NoDeepSupervision (a single target tensor).")
        if len(data) != self.transport_batch_size or len(target) != self.transport_batch_size:
            raise RuntimeError(
                f"MT240_K3 expected a CPU transport batch of four, got data={len(data)}, "
                f"target={len(target)}."
            )
        labelled_position, unlabelled_positions, unlabelled_keys = self._batch_stream_positions(
            list(batch["keys"])
        )

        if self.teacher is None:
            self._build_teacher()
        weight = self._consistency_weight()
        use_consistency = weight > 0.0
        # Match the existing K1/control protocol: the strong student view starts
        # after the warm-up even when consistency_max is zero.
        use_strong_view = self.current_epoch >= self.consistency_warmup_epochs

        self.optimizer.zero_grad(set_to_none=True)
        labelled_data = data[labelled_position : labelled_position + 1].to(
            self.device, non_blocking=True
        )
        labelled_target = target[labelled_position : labelled_position + 1].to(
            self.device, non_blocking=True
        )
        labelled_input = (
            self._perturb(
                labelled_data,
                self.student_noise_std,
                self.student_scale,
                self.student_shift,
            )
            if use_strong_view
            else labelled_data
        )
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            labelled_output = self.network(labelled_input)
            supervised_loss = self.loss(labelled_output, labelled_target)
        self._backward(supervised_loss, self.grad_scaler)
        del labelled_input, labelled_output, labelled_data, labelled_target

        consistency_values: list[torch.Tensor] = []
        if use_consistency:
            for position in unlabelled_positions:
                clean_unlabelled = data[position : position + 1].to(
                    self.device, non_blocking=True
                )
                student_input = self._perturb(
                    clean_unlabelled,
                    self.student_noise_std,
                    self.student_scale,
                    self.student_shift,
                )
                teacher_input = (
                    clean_unlabelled
                    if self.teacher_noise_std <= 0
                    else self._perturb(clean_unlabelled, self.teacher_noise_std, 0.0, 0.0)
                )
                with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                    student_output = self.network(student_input)
                    with torch.no_grad():
                        self.teacher.eval()
                        teacher_output = self.teacher(teacher_input)
                    consistency_term = self._consistency(student_output, teacher_output)
                    micro_loss = weight * consistency_term / self.unlabelled_crops_per_step
                self._backward(micro_loss, self.grad_scaler)
                consistency_values.append(consistency_term.detach())
                del (
                    clean_unlabelled,
                    student_input,
                    teacher_input,
                    student_output,
                    teacher_output,
                    consistency_term,
                    micro_loss,
                )

        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        if self.grad_scaler is None:
            self.optimizer.step()
        else:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

        self._mt_step += 1
        self._update_ema()

        if consistency_values:
            consistency = torch.stack(consistency_values).mean()
        else:
            consistency = supervised_loss.detach().new_zeros(())
        reported_loss = supervised_loss.detach() + weight * consistency
        self._log_cons += float(consistency)
        self._log_sup += float(supervised_loss.detach())
        self._log_w = weight
        self._log_n += 1

        self._stream_steps += 1
        self._stream_labelled_samples += 1
        self._stream_unlabelled_samples += self.unlabelled_crops_per_step
        self._k3_total_labelled_samples += 1
        self._k3_total_unlabelled_selected += self.unlabelled_crops_per_step
        self._k3_epoch_selected_by_case.update(unlabelled_keys)
        self._k3_total_selected_by_case.update(unlabelled_keys)
        if use_consistency:
            self._k3_total_unlabelled_active += self.unlabelled_crops_per_step
            self._k3_epoch_active_by_case.update(unlabelled_keys)
            self._k3_total_active_by_case.update(unlabelled_keys)

        return {"loss": reported_loss.cpu().numpy()}

    def on_train_epoch_end(self, train_outputs) -> None:
        selected_values = list(self._k3_epoch_selected_by_case.values())
        active_values = list(self._k3_epoch_active_by_case.values())
        super().on_train_epoch_end(train_outputs)
        self.print_to_log_file(
            f"[K3 exposure] selected_U={sum(selected_values)} "
            f"selected_unique_cases={len(selected_values)} "
            f"selected_case_range={min(selected_values, default=0)}..{max(selected_values, default=0)}; "
            f"consistency_evaluated_U={sum(active_values)} "
            f"active_unique_cases={len(active_values)}; reduction={self.consistency_reduction}; "
            "optimizer_steps=GT_samples."
        )
        self._k3_epoch_selected_by_case.clear()
        self._k3_epoch_active_by_case.clear()

    def _k3_protocol(self) -> dict:
        return {
            "version": self.k3_protocol_version,
            "labelled_crops_per_step": self.labelled_crops_per_step,
            "unlabelled_crops_per_step": self.unlabelled_crops_per_step,
            "transport_batch_size": self.transport_batch_size,
            "expected_planned_batch_size": self.expected_planned_batch_size,
            "consistency_reduction": self.consistency_reduction,
            # The HPC installation is a live symlink to this repository file.
            # Pin its bytes in every sidecar so a walltime resume cannot silently
            # continue under edited K3 code with otherwise unchanged constants.
            "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def _validate_k3_protocol(self, payload: dict) -> None:
        observed = payload.get("protocol")
        expected = self._k3_protocol()
        if observed != expected:
            raise RuntimeError(
                f"K3 checkpoint protocol mismatch: expected {expected}, found {observed}."
            )

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        filename = os.fspath(filename)
        if (
            self.local_rank == 0
            and not self.disable_checkpointing
            and os.path.isfile(filename)
        ):
            payload = {
                "protocol": self._k3_protocol(),
                "total_labelled_samples": self._k3_total_labelled_samples,
                "total_unlabelled_selected": self._k3_total_unlabelled_selected,
                "total_unlabelled_active": self._k3_total_unlabelled_active,
                "total_selected_by_case": dict(self._k3_total_selected_by_case),
                "total_active_by_case": dict(self._k3_total_active_by_case),
            }
            temporary = filename + ".k3.tmp"
            torch.save(payload, temporary)
            os.replace(temporary, filename + ".k3")

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        if not isinstance(filename_or_checkpoint, (str, os.PathLike)):
            raise RuntimeError("MT240_K3 resumes require a checkpoint filename and its .k3 sidecar.")
        filename = os.fspath(filename_or_checkpoint)
        mt_sidecar = filename + ".mt"
        sidecar = filename + ".k3"
        if not os.path.isfile(mt_sidecar):
            raise RuntimeError(f"Cannot resume MT240_K3: missing Mean-Teacher sidecar {mt_sidecar}.")
        if not os.path.isfile(sidecar):
            raise RuntimeError(f"Cannot resume MT240_K3: missing protocol sidecar {sidecar}.")
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
        self._validate_k3_protocol(payload)
        super().load_checkpoint(filename)
        self._k3_total_labelled_samples = int(payload.get("total_labelled_samples", 0))
        self._k3_total_unlabelled_selected = int(payload.get("total_unlabelled_selected", 0))
        self._k3_total_unlabelled_active = int(payload.get("total_unlabelled_active", 0))
        self._k3_total_selected_by_case = Counter(payload.get("total_selected_by_case", {}))
        self._k3_total_active_by_case = Counter(payload.get("total_active_by_case", {}))
        self.print_to_log_file(
            f"[K3] restored protocol/exposure sidecar: "
            f"GT={self._k3_total_labelled_samples}, "
            f"U selected={self._k3_total_unlabelled_selected}, "
            f"U active={self._k3_total_unlabelled_active}."
        )

    def on_train_start(self) -> None:
        super().on_train_start()
        self.print_to_log_file(
            "[K3 contract] one GT + three distinct U crops; mean-reduced U consistency; "
            "sequential GPU microbatches; one optimiser/EMA update per macro-step; "
            "labelled forced-foreground probability=0.5."
        )
