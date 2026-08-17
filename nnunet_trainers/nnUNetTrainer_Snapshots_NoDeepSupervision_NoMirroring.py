"""Supervised seed trainer that RETAINS mid-trajectory checkpoints.

Purpose: supply the seed-maturity rungs for the Mean-Teacher ladder.  The
scientific quantity is

    Delta_k = MT(seed_k) - Control(seed_k)

i.e. the incremental benefit of the unlabelled objective at a given teacher
maturity, so we need several checkpoints drawn from ONE optimisation
trajectory.

WHY SNAPSHOTS AND NOT SHORT RUNS.  nnU-Net's polynomial schedule is
parameterised by ``num_epochs``, so a completed 25-epoch run has fully
annealed (lr -> 0) while epoch 25 of the 1000-epoch schedule still sits at
lr ~ 9.75e-3.  Only the latter is a mid-trajectory model; a short run changes
maturity AND optimisation dynamics together.  Hence: one 1000-epoch run, many
retained checkpoints.

WHY A SUBCLASS IS NEEDED.  The stock trainer writes ``checkpoint_latest.pth``
every ``save_every`` (50) epochs, OVERWRITES it each time, and then DELETES it
in ``on_train_end``.  Nothing mid-trajectory survives a completed run, which is
why the existing Dataset123 fold-0 run left only ``best`` and ``final``.

SNAPSHOT EPOCHS.  Chosen from the existing Dataset123 fold-0 log (smoothed over
21 epochs), which shows three regimes rather than a simple plateau:

  RAPID LEARNING (0-25)     84% of the total pseudo-Dice gain is banked by
                            epoch 25, and the steepest single stretch is
                            epoch 10 -> 15 (36% -> 72%).  Sampled at 3, 5, 10,
                            15, 25 so that transition is resolved.
  CONSOLIDATION (25-100)    train_loss crosses BELOW val_loss at ~epoch 75-100
                            (gap +0.0076 at ep50, -0.0012 at ep100): the onset
                            of overfitting on 16 training cases.  Sampled at
                            50 and 100 so the pair straddles the crossover.
  MEMORISATION (100-1000)   val_loss flat while train_loss keeps falling; the
                            gap deepens from -0.001 to -0.037 by epoch 900,
                            most of it after epoch 500.  Sampled at 250, 500,
                            750 -- this is where a model becomes MORE confident
                            without becoming better, which is exactly the
                            mechanism the teacher-halo hypothesis concerns.

Note that pseudo-Dice is a voxel-count metric and ~59.7% of airway voxels sit
at branch depth <= 2, so its plateau is a PROXIMAL plateau; whether distal
topology keeps improving is exactly what the retained checkpoints are for.  Do
not treat the pseudo-Dice plateau as evidence that the rungs are equivalent.

NAMING.  ``checkpoint_snapshot_ep0025.pth`` holds the network state AFTER the
epoch logged as ``Epoch 25`` completed, which is the same instant at which the
stock trainer would have written ``checkpoint_latest.pth``.  Files are written
through the base ``save_checkpoint`` so they are fully nnU-Net-loadable and can
be passed straight to ``nnUNetv2_train -pretrained_weights``.

Deploy on HPC into the ctfm nnU-Net site-packages alongside the other trainers:
    cp nnUNetTrainer_Snapshots_NoDeepSupervision_NoMirroring.py \
       "$NNUNET/training/nnUNetTrainer/variants/network_architecture/"
"""

from __future__ import annotations

import os

import torch

# Import the real train-safe base directly.  This repository also installs
# inference-only compatibility shims carrying the same class name, and
# recursive class discovery can silently select one of those for training.
if __package__ == "nnunet_trainers":  # repository-local import
    from nnunet_trainers.nnUNetTrainer_NoDeepSupervision_NoMirroring import (  # type: ignore[no-redef]
        nnUNetTrainer_NoDeepSupervision_NoMirroring as _Base,
    )
else:  # installed beside this module in the nnU-Net trainer variants directory
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_NoDeepSupervision_NoMirroring import (  # noqa: E501
        nnUNetTrainer_NoDeepSupervision_NoMirroring as _Base,
    )


class nnUNetTrainer_Snapshots_NoDeepSupervision_NoMirroring(_Base):
    """Stock supervised recipe, plus retained mid-trajectory checkpoints.

    Training is byte-for-byte the parent's: no change to loss, schedule,
    sampler, augmentation or epoch count.  The only added behaviour is extra
    ``save_checkpoint`` calls at fixed epochs, so a snapshot run is a valid
    replacement for the ordinary seed run.
    """

    # After the epoch of this index completes.  See the module docstring for
    # the three-regime derivation; ~2.35 GB in total, which is free next to the
    # ~7.5 h each paired MT/Control rung costs.
    snapshot_epochs: tuple[int, ...] = (3, 5, 10, 15, 25, 50, 100, 250, 500, 750)

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # nnUNetTrainer reflects over this exact signature; do not replace it
        # with *args/**kwargs.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._validate_snapshot_epochs()

    def _validate_snapshot_epochs(self) -> None:
        # Read through the instance, not type(self): a per-run override set on
        # the object (or by a PBS preflight) must be validated, not bypassed.
        epochs = tuple(int(e) for e in self.snapshot_epochs)
        if len(set(epochs)) != len(epochs):
            raise ValueError(f"snapshot_epochs contains duplicates: {epochs}")
        # The final epoch already lands in checkpoint_final.pth, and nnU-Net's
        # own periodic save skips it for the same reason, so requesting it is a
        # configuration mistake rather than a duplicate write.
        last_trainable = self.num_epochs - 1
        for epoch in epochs:
            if not 0 <= epoch < last_trainable:
                raise ValueError(
                    f"snapshot epoch {epoch} is outside [0, {last_trainable}) "
                    f"for num_epochs={self.num_epochs}. Epoch {last_trainable} is "
                    "already persisted as checkpoint_final.pth."
                )
        self.snapshot_epochs = epochs

    def snapshot_filename(self, epoch: int) -> str:
        return os.path.join(self.output_folder, f"checkpoint_snapshot_ep{int(epoch):04d}.pth")

    def on_train_start(self) -> None:
        super().on_train_start()
        self.print_to_log_file(
            f"[Snapshots] retaining checkpoints after epochs {list(self.snapshot_epochs)} "
            f"(num_epochs={self.num_epochs}, initial_lr={self.initial_lr}); "
            "files are checkpoint_snapshot_epNNNN.pth and survive on_train_end."
        )

    def on_epoch_end(self) -> None:
        # Capture BEFORE the parent runs: nnUNetTrainer.on_epoch_end increments
        # self.current_epoch as its last action, so afterwards it names the NEXT
        # epoch.  `epoch` is therefore the index of the epoch that just finished.
        epoch = int(self.current_epoch)
        super().on_epoch_end()
        if epoch not in self.snapshot_epochs:
            return
        if self.disable_checkpointing or self.local_rank != 0:
            return
        filename = self.snapshot_filename(epoch)
        self.save_checkpoint(filename)
        self.print_to_log_file(f"[Snapshots] wrote {filename}")


class nnUNetTrainer_SnapshotsDense_NoDeepSupervision_NoMirroring(
    nnUNetTrainer_Snapshots_NoDeepSupervision_NoMirroring
):
    """Denser early sampling, for the case where epoch 5 already looks mature.

    Not the default: nine extra checkpoints is ~2.1 GB and the first rungs are
    only worth resolving further if the epoch-5 probe says confidence saturates
    before it.
    """

    snapshot_epochs = (1, 2, 3, 5, 8, 10, 15, 25, 50, 100, 250, 500)
