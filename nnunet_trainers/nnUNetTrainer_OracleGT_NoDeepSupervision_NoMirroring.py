"""Dataset127 oracle-label continuation isolated from the shared MT240 parent.

Keeping this class in its own module prevents a future supervised-reference edit
from changing the source blob used by already-running Mean-Teacher experiments.
The arm remains a descriptive schedule-matched oracle reference; its two stream
losses are averaged by the inherited offline trainer and therefore it is not a
literal one-variable label-gap fraction without further redesign.
"""

from __future__ import annotations

if __package__ == "nnunet_trainers":
    from nnunet_trainers.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _MTBase,
        nnUNetTrainer_OfflinePseudo_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _OfflineBase,
    )
else:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _MTBase,
        nnUNetTrainer_OfflinePseudo_WarmStart_TwoStream_NoDeepSupervision_NoMirroring as _OfflineBase,
    )


class nnUNetTrainer_OracleGT_WarmStart_TwoStream_NoDeepSupervision_NoMirroring(
    _OfflineBase
):
    """One real-L20 patch plus one real-U240-label patch per optimizer step."""

    experiment_contract_key = "supervised_ceiling"
    secondary_provenance = "oracle_gt"
    secondary_stream_name = "oracle"
    secondary_loss_scope = "supervised_scope=gt-plus-oracle-gt"
    requires_ignore_label = False
    arm_log_tag = "OracleGT"
    secondary_target_description = "real ATM'22 labels on all 260 cases"

    def on_train_start(self) -> None:
        # Skip _OfflineBase's target-specific log line while preserving the
        # shared MT setup and teacher construction.
        _MTBase.on_train_start(self)
        self.print_to_log_file(
            "[OracleGT] real ATM'22 labels on all 260 cases; no online teacher "
            f"forward; Dice+CE stream weights GT=1.0 oracle={self.pseudo_loss_weight}; "
            f"strong-view start={self.consistency_warmup_epochs}; "
            "EMA is used only for final weight averaging."
        )

    def on_train_epoch_end(self, train_outputs) -> None:
        # As above, retain the shared accounting but label the secondary loss
        # honestly rather than calling the oracle targets pseudo-labels.
        _MTBase.on_train_epoch_end(self, train_outputs)
        if self._offline_log_n > 0:
            self.print_to_log_file(
                f"[OracleGT] gt_loss={self._offline_log_gt / self._offline_log_n:.4f} "
                f"oracle_loss={self._offline_log_pseudo / self._offline_log_n:.4f} "
                f"oracle_weight={self.pseudo_loss_weight:.2f}"
            )
        self._offline_log_gt = 0.0
        self._offline_log_pseudo = 0.0
        self._offline_log_n = 0
