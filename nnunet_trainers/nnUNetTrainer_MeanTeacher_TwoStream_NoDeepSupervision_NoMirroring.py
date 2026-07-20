"""Provenance-aware two-stream Mean-Teacher trainers for Dataset124.

The raw dataset still gives every unlabelled case an all-ignore segmentation so
that it can pass through stock nnU-Net preprocessing without ever exposing its
withheld ground truth.  Training decisions do *not* infer provenance from that
target.  Instead, the dataset's ``semi_supervised`` contract supplies explicit
case lists and this loader draws exactly one labelled and one unlabelled patch
per batch.

Deploy this file together with:

* ``nnUNetTrainer_NoDeepSupervision_NoMirroring.py``
* ``nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py``

in nnU-Net's ``variants/network_architecture`` trainer directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast

if __package__ == "nnunet_trainers":  # repository-local tests; avoid installed historical copy
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (
        nnUNetTrainer_MeanTeacher_WarmStart_NoDeepSupervision_NoMirroring as _WarmStartBase,
    )
else:  # installed beside this module in nnU-Net
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_MeanTeacher_WarmStart_NoDeepSupervision_NoMirroring as _WarmStartBase,
    )


def _normalise_case_key(value: str) -> str:
    """Normalise provenance keys to nnU-Net's ``ATM_XXX`` identifiers."""
    name = Path(str(value)).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_0000"):
        name = name[:-5]
    if name.upper().startswith("ATM_"):
        suffix = name[4:]
    else:
        suffix = name
    if not suffix.isdigit():
        raise ValueError(f"Invalid ATM case identifier in provenance: {value!r}")
    return f"ATM_{int(suffix):03d}"


def _normalise_provenance(mapping: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = _normalise_case_key(raw_key)
        value = str(raw_value).lower()
        if value not in {"gt", "ignore"}:
            raise ValueError(f"Provenance for {key} must be 'gt' or 'ignore', got {raw_value!r}.")
        if key in result and result[key] != value:
            raise ValueError(f"Conflicting provenance entries for {key}.")
        result[key] = value
    return result


class ProvenanceTwoStreamDataLoader(nnUNetDataLoader):
    """nnU-Net patch loader with a guaranteed 1:1 GT/unlabelled case mix.

    Dataset124 uses the inherited batch size of two.  The labelled patch uses
    the same expected foreground/random mixture as stock nnU-Net batch-2
    sampling (50% forced foreground, 50% random); the all-ignore patch is
    always sampled randomly.
    """

    def __init__(self, *args, labelled_identifiers, unlabelled_identifiers, **kwargs):
        super().__init__(*args, **kwargs)
        if self.batch_size != 2:
            raise RuntimeError(
                "The controlled two-stream MT protocol requires local batch_size=2 "
                f"(one GT + one unlabelled), got {self.batch_size}. Do not launch it with DDP "
                "or a plan that reduces the per-GPU batch size."
            )
        self.labelled_identifiers = np.asarray(sorted(labelled_identifiers), dtype=object)
        self.unlabelled_identifiers = np.asarray(sorted(unlabelled_identifiers), dtype=object)
        if self.labelled_identifiers.size == 0 or self.unlabelled_identifiers.size == 0:
            raise ValueError("Both labelled and unlabelled streams must contain at least one case.")

        # Stock nnU-Net with batch=2 and oversample=0.33 deterministically makes
        # one of its two patches foreground-centred. Across labelled patches that
        # is a 0.5 forced-foreground probability, which we preserve here.
        stock_forced = self.batch_size - round(self.batch_size * (1 - self.oversample_foreground_percent))
        self.labelled_foreground_probability = stock_forced / self.batch_size
        self._selected_keys: list[str] = []
        # nnUNetDataLoader installs get_do_oversample as an instance attribute.
        self.get_do_oversample = self._two_stream_do_oversample

    def get_indices(self):
        unlabelled = str(np.random.choice(self.unlabelled_identifiers))
        labelled = str(np.random.choice(self.labelled_identifiers))
        # Randomise order so no downstream operation can accidentally learn a
        # fixed stream position. Foreground selection uses case provenance.
        self._selected_keys = [unlabelled, labelled]
        np.random.shuffle(self._selected_keys)
        return list(self._selected_keys)

    def _two_stream_do_oversample(self, sample_idx: int) -> bool:
        key = self._selected_keys[sample_idx]
        if key not in self.labelled_identifiers:
            return False
        return bool(np.random.random() < self.labelled_foreground_probability)


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring(_WarmStartBase):
    """Warm-start MT with explicit one-GT/one-unlabelled batches.

    Supervised Dice+CE is evaluated only on the provenance-labelled patch.
    Geometry consistency is evaluated only on the provenance-unlabelled patch,
    so the MT term has a clean interpretation.  The all-ignore target remains a
    second, independent guard against accidental supervised GT use.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._case_provenance: dict[str, str] = {}
        self._stream_steps = 0
        self._stream_labelled_samples = 0
        self._stream_unlabelled_samples = 0

    def _load_experiment_contract(self) -> tuple[dict[str, str], dict]:
        contract = self.dataset_json.get("semi_supervised")
        if not isinstance(contract, dict):
            # Backward-compatible audit path for an explicitly copied sidecar.
            sidecar = os.path.join(self.preprocessed_dataset_folder_base, "label_provenance.json")
            if not os.path.isfile(sidecar):
                raise RuntimeError(
                    "Dataset is missing dataset.json['semi_supervised']; refusing to infer "
                    "labelled/unlabelled provenance from target contents."
                )
            with open(sidecar, encoding="utf-8") as handle:
                legacy = json.load(handle)
            provenance = _normalise_provenance(legacy.get("labels", {}))
            return provenance, {}

        provenance = _normalise_provenance(contract.get("case_provenance", {}))
        if not provenance:
            raise RuntimeError("The semi_supervised contract has no case_provenance entries.")
        return provenance, contract

    @staticmethod
    def _assert_contract_matches_split(contract: dict, fold, tr_keys: list[str], val_keys: list[str]) -> None:
        folds = contract.get("folds", {})
        expected = folds.get(str(fold)) if isinstance(folds, dict) else None
        if expected is None:
            return
        expected_train = {_normalise_case_key(k) for k in expected.get("train", [])}
        expected_val = {_normalise_case_key(k) for k in expected.get("val", [])}
        if expected_train != set(tr_keys) or expected_val != set(val_keys):
            raise RuntimeError(
                "splits_final.json does not match the dataset's semi-supervised contract. "
                "Refusing to train with a split that changes GT exposure or validates on ignore cases."
            )

    def get_dataloaders(self):
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
        tr_keys = [_normalise_case_key(k) for k in tr_keys]
        val_keys = [_normalise_case_key(k) for k in val_keys]
        provenance, contract = self._load_experiment_contract()
        self._assert_contract_matches_split(contract, self.fold, tr_keys, val_keys)

        unknown = (set(tr_keys) | set(val_keys)) - set(provenance)
        if unknown:
            raise RuntimeError(f"Missing provenance for cases: {sorted(unknown)}")
        if any(provenance[k] != "gt" for k in val_keys):
            raise RuntimeError("Validation must contain real-GT cases only; ignore-labelled validation is invalid.")
        labelled = [k for k in tr_keys if provenance[k] == "gt"]
        unlabelled = [k for k in tr_keys if provenance[k] == "ignore"]
        if not labelled or not unlabelled:
            raise RuntimeError(
                f"Two-stream training needs both streams, got {len(labelled)} GT and {len(unlabelled)} unlabelled."
            )
        if not self.label_manager.has_ignore_label:
            raise RuntimeError("Two-stream MT requires an nnU-Net ignore label in dataset.json.")
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
        dl_tr = ProvenanceTwoStreamDataLoader(
            dataset_tr,
            self.batch_size,
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
            f"[TwoStream] fold={self.fold}: {len(labelled)} GT train + {len(unlabelled)} "
            f"unlabelled train; {len(val_keys)} GT-only validation; batch=1+1."
        )
        return train_gen, val_gen

    def _batch_stream_indices(self, keys: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        normalised = [_normalise_case_key(k) for k in keys]
        try:
            labelled = [i for i, key in enumerate(normalised) if self._case_provenance[key] == "gt"]
            unlabelled = [i for i, key in enumerate(normalised) if self._case_provenance[key] == "ignore"]
        except KeyError as exc:
            raise RuntimeError(f"Batch contains a case with no provenance: {exc.args[0]}") from exc
        if len(labelled) != 1 or len(unlabelled) != 1:
            raise RuntimeError(
                f"Expected one GT and one unlabelled sample, got keys={normalised}, "
                f"GT indices={labelled}, unlabelled indices={unlabelled}."
            )
        return (
            torch.as_tensor(labelled, device=self.device, dtype=torch.long),
            torch.as_tensor(unlabelled, device=self.device, dtype=torch.long),
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            raise RuntimeError("Two-stream MT requires NoDeepSupervision (a single target tensor).")
        target = target.to(self.device, non_blocking=True)
        labelled_idx, unlabelled_idx = self._batch_stream_indices(list(batch["keys"]))

        if self.teacher is None:
            self._build_teacher()

        weight = self._consistency_weight()
        use_consistency = weight > 0.0
        # Decouple strong-view augmentation from consistency_max so the zero-
        # consistency subclass is a genuinely matched fine-tuning/EMA control.
        use_strong_view = self.current_epoch >= self.consistency_warmup_epochs
        student_in = (
            self._perturb(data, self.student_noise_std, self.student_scale, self.student_shift)
            if use_strong_view
            else data
        )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(student_in)
            supervised_loss = self.loss(
                output.index_select(0, labelled_idx),
                target.index_select(0, labelled_idx),
            )
            if use_consistency:
                teacher_in = data.index_select(0, unlabelled_idx)
                if self.teacher_noise_std > 0:
                    teacher_in = self._perturb(teacher_in, self.teacher_noise_std, 0.0, 0.0)
                with torch.no_grad():
                    self.teacher.eval()
                    teacher_output = self.teacher(teacher_in)
                consistency = self._consistency(output.index_select(0, unlabelled_idx), teacher_output)
                loss = supervised_loss + weight * consistency
            else:
                consistency = supervised_loss.new_zeros(())
                loss = supervised_loss

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
        self._log_w = weight
        self._log_n += 1
        self._stream_steps += 1
        self._stream_labelled_samples += int(labelled_idx.numel())
        self._stream_unlabelled_samples += int(unlabelled_idx.numel())
        return {"loss": loss.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs) -> None:
        super().on_train_epoch_end(train_outputs)
        self.print_to_log_file(
            f"[TwoStream] steps={self._stream_steps} GT samples={self._stream_labelled_samples} "
            f"unlabelled samples={self._stream_unlabelled_samples} consistency_scope=unlabelled-only"
        )
        self._stream_steps = 0
        self._stream_labelled_samples = 0
        self._stream_unlabelled_samples = 0


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_AsymCLDice_NoDeepSupervision_NoMirroring(
    nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring
):
    """Two-stream lung-crop arm retaining the historical asymmetric beta=2 loss."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.cldice_cons_beta = 2.0


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_Control_NoDeepSupervision_NoMirroring(
    nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring
):
    """Matched continued-training/augmentation/EMA control with no consistency gradient."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.consistency_max = 0.0


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_LRMirroring(
    nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring
):
    """Optional symmetric MT arm with anatomical left-right mirroring only."""

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch_size, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        if len(self.configuration_manager.patch_size) != 3:
            raise RuntimeError("The ATM lung-crop experiment expects the 3d_fullres configuration.")
        transpose = list(self.plans_manager.transpose_forward)
        if transpose != [1, 0, 2]:
            raise RuntimeError(
                f"LRMirroring was validated for transpose_forward=[1, 0, 2], got {transpose}. "
                "Re-derive the anatomical LR network axis before training."
            )
        mirror_axes = (2,)
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation, dummy_2d, initial_patch_size, mirror_axes


class nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_AsymCLDice_LRMirroring(
    nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_LRMirroring
):
    """Optional beta=2 MT arm with anatomical left-right mirroring only."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.cldice_cons_beta = 2.0
