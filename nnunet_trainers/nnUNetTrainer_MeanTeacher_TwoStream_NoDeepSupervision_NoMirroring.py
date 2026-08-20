"""Provenance-aware two-stream trainers for Dataset124 and Dataset125.

The raw dataset still gives every unlabelled case an all-ignore segmentation so
that it can pass through stock nnU-Net preprocessing without ever exposing its
withheld ground truth.  Training decisions do *not* infer provenance from that
target.  Instead, the dataset's ``semi_supervised`` contract supplies explicit
case lists and this loader draws exactly one labelled and one unlabelled patch
per batch.

Dataset125 reuses the same loader and training envelope for conventional
offline self-training: one real-GT patch plus one fixed-pseudo patch, real-GT
only validation, 500-epoch low-LR warm start, strong intensity perturbation,
and final EMA deployment. It does not run an online teacher forward pass.

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


def _normalise_provenance(mapping: dict, secondary_value: str = "ignore") -> dict[str, str]:
    allowed = {"gt", str(secondary_value).lower()}
    result: dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = _normalise_case_key(raw_key)
        value = str(raw_value).lower()
        if value not in allowed:
            choices = " or ".join(repr(item) for item in sorted(allowed))
            raise ValueError(f"Provenance for {key} must be {choices}, got {raw_value!r}.")
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

    experiment_contract_key = "semi_supervised"
    secondary_provenance = "ignore"
    secondary_stream_name = "unlabelled"
    secondary_loss_scope = "consistency_scope=unlabelled-only"
    requires_ignore_label = True

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
        contract = self.dataset_json.get(self.experiment_contract_key)
        if not isinstance(contract, dict):
            # Backward-compatible audit path for an explicitly copied sidecar.
            if self.experiment_contract_key != "semi_supervised":
                raise RuntimeError(
                    f"Dataset is missing dataset.json[{self.experiment_contract_key!r}]; "
                    "refusing to infer fixed-pseudo provenance from target contents."
                )
            sidecar = os.path.join(self.preprocessed_dataset_folder_base, "label_provenance.json")
            if not os.path.isfile(sidecar):
                raise RuntimeError(
                    "Dataset is missing dataset.json['semi_supervised']; refusing to infer "
                    "labelled/unlabelled provenance from target contents."
                )
            with open(sidecar, encoding="utf-8") as handle:
                legacy = json.load(handle)
            provenance = _normalise_provenance(
                legacy.get("labels", {}),
                secondary_value=self.secondary_provenance,
            )
            return provenance, {}

        provenance = _normalise_provenance(
            contract.get("case_provenance", {}),
            secondary_value=self.secondary_provenance,
        )
        if not provenance:
            raise RuntimeError(
                f"The {self.experiment_contract_key} contract has no case_provenance entries."
            )
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
            raise RuntimeError(
                "Validation must contain real-GT cases only; secondary-stream validation is invalid."
            )
        labelled = [k for k in tr_keys if provenance[k] == "gt"]
        secondary = [k for k in tr_keys if provenance[k] == self.secondary_provenance]
        if not labelled or not secondary:
            raise RuntimeError(
                f"Two-stream training needs both streams, got {len(labelled)} GT and "
                f"{len(secondary)} {self.secondary_stream_name}."
            )
        if self.requires_ignore_label and not self.label_manager.has_ignore_label:
            raise RuntimeError("Two-stream MT requires an nnU-Net ignore label in dataset.json.")
        if not self.requires_ignore_label and self.label_manager.has_ignore_label:
            raise RuntimeError("Offline pseudo-label training must use ordinary binary 0/1 targets.")
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
            unlabelled_identifiers=secondary,
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
            f"[TwoStream] fold={self.fold}: {len(labelled)} GT train + {len(secondary)} "
            f"{self.secondary_stream_name} train; {len(val_keys)} GT-only validation; batch=1+1."
        )
        return train_gen, val_gen

    def _batch_stream_indices(self, keys: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        normalised = [_normalise_case_key(k) for k in keys]
        try:
            labelled = [i for i, key in enumerate(normalised) if self._case_provenance[key] == "gt"]
            unlabelled = [
                i
                for i, key in enumerate(normalised)
                if self._case_provenance[key] == self.secondary_provenance
            ]
        except KeyError as exc:
            raise RuntimeError(f"Batch contains a case with no provenance: {exc.args[0]}") from exc
        if len(labelled) != 1 or len(unlabelled) != 1:
            raise RuntimeError(
                f"Expected one GT and one {self.secondary_stream_name} sample, got keys={normalised}, "
                f"GT indices={labelled}, {self.secondary_stream_name} indices={unlabelled}."
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
            f"{self.secondary_stream_name} samples={self._stream_unlabelled_samples} "
            f"{self.secondary_loss_scope}"
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


class nnUNetTrainer_OfflinePseudo_WarmStart_TwoStream_NoDeepSupervision_NoMirroring(
    nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring
):
    """Conventional fixed-pseudo self-training matched to Dataset124's envelope.

    Dataset125 supplies ordinary binary targets for both streams. Each step
    contains one real-GT patch and one fixed-pseudo patch; Dice+CE is evaluated
    separately on each and averaged with equal stream weight. The network is
    warm-started from the same Dataset123 seed, receives the same strong
    intensity perturbation from epoch 5, runs for 500 epochs at LR 1e-3, and
    deploys the same EMA weight average in ``checkpoint_final``.

    The inherited ``teacher`` is used only as a weight-averaging accumulator.
    It is never evaluated to create an online target, and there is no
    consistency gradient.
    """

    experiment_contract_key = "offline_self_training"
    secondary_provenance = "pseudo"
    secondary_stream_name = "pseudo"
    secondary_loss_scope = "supervised_scope=gt-plus-fixed-pseudo"
    requires_ignore_label = False
    # Subclasses that reuse this envelope with a different secondary target
    # relabel the log lines through these, so a training log never claims a
    # target source the arm does not have.
    arm_log_tag = "OfflinePseudo"
    secondary_target_description = "fixed Dataset123 argmax targets"

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
        self.pseudo_loss_weight = 1.0
        self._offline_log_gt = 0.0
        self._offline_log_pseudo = 0.0
        self._offline_log_n = 0

    def on_train_start(self) -> None:
        super().on_train_start()
        self.print_to_log_file(
            f"[{self.arm_log_tag}] {self.secondary_target_description}; no online teacher "
            f"forward; Dice+CE stream weights GT=1.0 "
            f"{self.secondary_stream_name}={self.pseudo_loss_weight}; "
            f"strong-view start={self.consistency_warmup_epochs}; "
            "EMA is used only for final weight averaging."
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            raise RuntimeError("Two-stream offline pseudo training requires NoDeepSupervision.")
        target = target.to(self.device, non_blocking=True)
        labelled_idx, pseudo_idx = self._batch_stream_indices(list(batch["keys"]))

        if self.teacher is None:
            self._build_teacher()

        use_strong_view = self.current_epoch >= self.consistency_warmup_epochs
        student_in = (
            self._perturb(data, self.student_noise_std, self.student_scale, self.student_shift)
            if use_strong_view
            else data
        )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(student_in)
            gt_loss = self.loss(
                output.index_select(0, labelled_idx),
                target.index_select(0, labelled_idx),
            )
            pseudo_loss = self.loss(
                output.index_select(0, pseudo_idx),
                target.index_select(0, pseudo_idx),
            )
            normaliser = 1.0 + self.pseudo_loss_weight
            loss = (gt_loss + self.pseudo_loss_weight * pseudo_loss) / normaliser

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
        self._offline_log_gt += float(gt_loss.detach())
        self._offline_log_pseudo += float(pseudo_loss.detach())
        self._offline_log_n += 1
        self._stream_steps += 1
        self._stream_labelled_samples += int(labelled_idx.numel())
        self._stream_unlabelled_samples += int(pseudo_idx.numel())
        return {"loss": loss.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs) -> None:
        super().on_train_epoch_end(train_outputs)
        if self._offline_log_n > 0:
            self.print_to_log_file(
                f"[{self.arm_log_tag}] gt_loss={self._offline_log_gt / self._offline_log_n:.4f} "
                f"{self.secondary_stream_name}_loss="
                f"{self._offline_log_pseudo / self._offline_log_n:.4f} "
                f"{self.secondary_stream_name}_weight={self.pseudo_loss_weight:.2f}"
            )
        self._offline_log_gt = 0.0
        self._offline_log_pseudo = 0.0
        self._offline_log_n = 0


class nnUNetTrainer_OracleGT_WarmStart_TwoStream_NoDeepSupervision_NoMirroring(
    nnUNetTrainer_OfflinePseudo_WarmStart_TwoStream_NoDeepSupervision_NoMirroring
):
    """Supervised label ceiling on the SAME continuation envelope as MT240.

    Dataset127 is Dataset126 with the 240 all-ignore targets replaced by their
    real ATM'22 annotation, so this arm is the offline-pseudo arm with the
    pseudo-labels swapped for oracle GT. Everything else is the shared
    continuation envelope: the same fold-matched Dataset123 seed, the same
    one-GT/one-secondary batch, the same strong intensity view from epoch 5,
    500 epochs at LR 1e-3, and the same EMA average deployed in
    ``checkpoint_final``.

    That makes the four arms differ in exactly one thing -- what the 240 scans
    contribute to the gradient:

    * control      -- nothing (all-ignore target, no consistency)
    * MT           -- online clDice consistency against the EMA teacher
    * offline SSL  -- Dice+CE against fixed seed pseudo-labels
    * this arm     -- Dice+CE against their real labels

    So ``(MT - control) / (oracle - control)`` is the fraction of the available
    label gap that consistency recovers, with every other factor held fixed.

    Note what is deliberately NOT changed: the 240 keep the secondary stream's
    purely random patch sampling, while the 16 GT keep the 50% forced-foreground
    draw. Real labels on the 240 would justify oversampling their foreground
    too, but that would change patch statistics against the three arms this one
    exists to be compared with. The un-handicapped "best supervised model at 260
    labels" is the separate from-scratch Dataset127 run, which uses stock
    uniform sampling over all 256 cases.
    """

    experiment_contract_key = "supervised_ceiling"
    secondary_provenance = "oracle_gt"
    secondary_stream_name = "oracle"
    secondary_loss_scope = "supervised_scope=gt-plus-oracle-gt"
    requires_ignore_label = False
    arm_log_tag = "OracleGT"
    secondary_target_description = "real ATM'22 labels on all 260 cases"


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
