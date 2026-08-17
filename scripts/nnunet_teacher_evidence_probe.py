"""Report Mean-Teacher teacher-evidence statistics for a BARE checkpoint.

WHY THIS EXISTS.  The quantities that carry the MT null -- the sub-threshold
"halo" ratio ``p>0.1 / p>0.5``, the sub-threshold probability/skeleton mass
shares, and ``soft_only_patches`` -- are currently emitted only from inside the
diagnostic trainer's ``train_step``, once per epoch, during a 7.5 h run.  The
seed-maturity ladder needs them for a set of seed snapshots BEFORE committing
GPU time to paired MT/Control runs, so they have to be obtainable from a
checkpoint alone.

WHAT IT MEASURES.  For each checkpoint it reproduces the numbers the MT run
would print at ITS OWN epoch 0: teacher := the checkpoint, student := the same
checkpoint under the trainer's configured intensity perturbation.  That is
exactly the warm-start state, where the EMA teacher is a copy of the seeded
student.  So ``[MTTeacherEvidence]`` here is directly comparable to the first
such line of a warm-started run, and to the 1.0373 halo ratio recorded for the
mature seed.

EXACTNESS.  The statistics are not reimplemented.  The script instantiates the
real diagnostic trainer, builds the real Dataset126 two-stream dataloader, and
calls the trainer's own ``_record_soft_diagnostics``.  If that code changes,
this probe changes with it.

PAIRING.  Unlabelled patches are drawn ONCE and replayed for every checkpoint,
and the student perturbation is re-seeded per step index, so differences
between checkpoints are not confounded with sampler noise.  Cache footprint is
about 2.3 GB at the default 250 steps (patch 128x160x112, float32, one U patch
per step); lower ``--steps`` if memory is tight.

Usage (HPC, inside the ctfm env, with the nnUNet_* variables exported)::

    python scripts/nnunet_teacher_evidence_probe.py \
        --dataset-id 126 --fold 0 \
        --checkpoint "$SEED_DIR/checkpoint_snapshot_ep0005.pth" \
                     "$SEED_DIR/checkpoint_snapshot_ep0025.pth" \
                     "$SEED_DIR/checkpoint_final.pth" \
        --out runs/mt_diagnostics/seed_snapshot_evidence.json

Nothing is written into any nnU-Net results folder: the trainer's output
folder is redirected to ``--scratch-folder``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_trainer_bits():
    """Prefer the installed nnU-Net copies; fall back to the repository ones.

    On the HPC the trainers are symlinked into nnU-Net's variants directory and
    the installed copy is authoritative.  Locally (tests, dry runs) only the
    repository copy exists.
    """
    try:
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring import (  # noqa: E501
            _soft_probability_cldice_terms,
            nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring as Trainer,
        )
    except ImportError:
        from nnunet_trainers.nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring import (  # noqa: E501
            _soft_probability_cldice_terms,
            nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring as Trainer,
        )
    return Trainer, _soft_probability_cldice_terms


def strip_compile_prefixes(state_dict: dict) -> dict:
    """Drop ``module.`` / ``_orig_mod.`` wrappers left by DDP or torch.compile."""
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    changed = True
        cleaned[name] = value
    return cleaned


def load_network_weights(network: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    """Load an nnU-Net checkpoint's network weights, returning its metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "network_weights" in checkpoint:
        state = checkpoint["network_weights"]
        meta = {
            "stored_current_epoch": checkpoint.get("current_epoch"),
            "stored_best_ema": checkpoint.get("_best_ema"),
        }
    else:  # a bare state_dict
        state = checkpoint
        meta = {"stored_current_epoch": None, "stored_best_ema": None}
    target = network._orig_mod if hasattr(network, "_orig_mod") else network
    target.load_state_dict(strip_compile_prefixes(state))
    return meta


def summarise_diagnostics(sums: dict, steps: int) -> dict:
    """Split the trainer's accumulators exactly as its own reporter does.

    Per-step summaries are divided by the step count; ``counterfactual_*`` is
    recorded once per epoch and must NOT be divided.
    """
    if steps <= 0:
        raise ValueError("summarise_diagnostics needs at least one recorded step")
    out = {
        name: float(value.detach().cpu()) / steps
        for name, value in sums.items()
        if not name.startswith("counterfactual_")
    }
    out.update(
        {
            name: float(value.detach().cpu())
            for name, value in sums.items()
            if name.startswith("counterfactual_")
        }
    )
    p_half = out.get("teacher_p_gt_0p5_fraction", 0.0)
    # The headline halo ratio.  Guard the degenerate empty-tree case rather
    # than emitting an infinity that would poison a downstream mean.
    out["halo_ratio_p0p1_over_p0p5"] = (
        out.get("teacher_p_gt_0p1_fraction", 0.0) / p_half if p_half > 0 else float("nan")
    )
    return out


def format_report_lines(mean: dict, steps: int) -> list[str]:
    """Reproduce the trainer's tagged log lines so old and new runs grep alike."""
    lines = [
        f"[MTSoftCLDice] samples={steps} "
        f"soft_loss={mean['soft_loss']:.5f} "
        f"soft_tprec={mean['soft_tprec']:.5f} soft_tsens={mean['soft_tsens']:.5f} "
        f"prob_mse={mean['probability_mse']:.7f} "
        f"teacher_prob_mass={mean['teacher_probability_mass']:.2f} "
        f"teacher_hard_voxels={mean['teacher_hard_voxels']:.2f} "
        f"teacher_soft_skel_mass={mean['teacher_soft_skeleton_mass']:.2f} "
        f"student_soft_skel_mass={mean['teacher_student_skeleton_mass']:.2f} "
        f"soft_self_loss={mean['soft_self_loss']:.5f}",
        f"[MTTeacherEvidence] p>0.1={mean['teacher_p_gt_0p1_fraction']:.7f} "
        f"p>0.3={mean['teacher_p_gt_0p3_fraction']:.7f} "
        f"p>0.5={mean['teacher_p_gt_0p5_fraction']:.7f} "
        f"p>0.8={mean['teacher_p_gt_0p8_fraction']:.7f} "
        f"subthr_prob_mass={mean['subthreshold_teacher_probability_mass_share']:.5f} "
        f"subthr_p>=0.05_prob_mass={mean['p0p05_to_0p5_teacher_probability_mass_share']:.5f} "
        f"subthr_skel_mass={mean['subthreshold_teacher_skeleton_mass_share']:.5f} "
        f"subthr_p>=0.05_skel_mass={mean['p0p05_to_0p5_teacher_skeleton_mass_share']:.5f} "
        f"skeleton_weighted_p={mean['teacher_skeleton_weighted_probability']:.5f} "
        f"hard_active_patches={mean['hard_positive_patch_fraction']:.3f} "
        f"soft_active_patches={mean['soft_skeleton_active_patch_fraction']:.3f} "
        f"soft_only_patches={mean['soft_evidence_without_hard_patch_fraction']:.3f}",
        f"[MTHaloRatio] p>0.1/p>0.5={mean['halo_ratio_p0p1_over_p0p5']:.4f}",
    ]
    if "counterfactual_hard_loss" in mean:
        lines.append(
            "[MTHardCounterfactual] samples=1 used_for_gradient=false "
            f"same_patch_soft_loss={mean['counterfactual_soft_loss']:.5f} "
            f"hard_loss={mean['counterfactual_hard_loss']:.5f} "
            f"hard_tprec={mean['counterfactual_hard_tprec']:.5f} "
            f"hard_tsens={mean['counterfactual_hard_tsens']:.5f}"
        )
    return lines


def build_trainer(dataset_id: int, fold: int, configuration: str, plans_identifier: str,
                  scratch_folder: str, device: torch.device):
    """Instantiate the diagnostic trainer without on_train_start's side effects."""
    from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    Trainer, _ = _import_trainer_bits()
    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    plans = load_json(join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json"))
    dataset_json = load_json(join(nnUNet_preprocessed, dataset_name, "dataset.json"))

    trainer = Trainer(plans, configuration, fold, dataset_json, device)
    # Redirect every write away from nnUNet_results: this probe must never
    # touch a real run's folder, and print_to_log_file creates files.
    trainer.output_folder_base = scratch_folder
    trainer.output_folder = join(scratch_folder, f"fold_{fold}")
    trainer.log_file = None
    maybe_mkdir_p(trainer.output_folder)

    trainer.initialize()
    trainer.set_deep_supervision_enabled(trainer.enable_deep_supervision)
    # Mirrors on_train_start's unpack step; a no-op once the folder is unpacked.
    trainer.dataset_class.unpack_dataset(
        trainer.preprocessed_dataset_folder, overwrite_existing=False, num_processes=2, verify=True
    )
    trainer.dataloader_train, trainer.dataloader_val = trainer.get_dataloaders()
    trainer.current_epoch = 0
    return trainer


def collect_unlabelled_patches(trainer, steps: int) -> list[torch.Tensor]:
    """Draw `steps` unlabelled patches once, on CPU, for replay across checkpoints."""
    patches = []
    while len(patches) < steps:
        batch = next(trainer.dataloader_train)
        data = batch["data"].to(trainer.device, non_blocking=True)
        _, unlabelled_idx = trainer._batch_stream_indices(list(batch["keys"]))
        if unlabelled_idx.numel() == 0:
            continue
        patches.append(data.index_select(0, unlabelled_idx).detach().cpu())
    return patches


def probe_checkpoint(trainer, terms_fn, patches: list[torch.Tensor], checkpoint_path: str,
                     perturb_seed: int) -> dict:
    """Run one checkpoint over the cached patches and summarise its diagnostics."""
    meta = load_network_weights(trainer.network, checkpoint_path, trainer.device)
    trainer.network.eval()
    trainer._diagnostic_sums = {}
    trainer._diagnostic_steps = 0
    trainer._diagnostic_hard_epoch = None

    with torch.no_grad():
        for step, patch in enumerate(patches):
            clean = patch.to(trainer.device, non_blocking=True)
            # Re-seed per step so every checkpoint sees the SAME perturbation.
            torch.manual_seed(perturb_seed + step)
            student_in = trainer._perturb(
                clean, trainer.student_noise_std, trainer.student_scale, trainer.student_shift
            )
            teacher_fg = torch.softmax(trainer.network(clean).float(), dim=1)[:, 1:2]
            student_fg = torch.softmax(trainer.network(student_in).float(), dim=1)[:, 1:2]
            soft_loss, tprec, tsens, student_skeleton, teacher_skeleton = terms_fn(
                student_fg, teacher_fg, trainer.cldice_iters, beta=trainer.cldice_cons_beta
            )
            trainer._record_soft_diagnostics(
                student_fg, teacher_fg, student_skeleton, teacher_skeleton, soft_loss, tprec, tsens
            )

    record = summarise_diagnostics(trainer._diagnostic_sums, trainer._diagnostic_steps)
    record["checkpoint"] = checkpoint_path
    record["checkpoint_name"] = os.path.basename(checkpoint_path)
    record["steps"] = trainer._diagnostic_steps
    record.update(meta)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", type=int, default=126,
                        help="Dataset supplying the UNLABELLED stream (default: 126).")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--plans-identifier", default="nnUNetPlans")
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="One or more checkpoints; the dataloader is built once and shared.")
    parser.add_argument("--steps", type=int, default=250,
                        help="Unlabelled patches per checkpoint (default 250 = one nnU-Net epoch).")
    parser.add_argument("--perturb-seed", type=int, default=1234)
    parser.add_argument("--scratch-folder", default=None,
                        help="Where the trainer may write. Defaults to a temp dir.")
    parser.add_argument("--out", default=None, help="Write the per-checkpoint records as JSON here.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    missing = [path for path in args.checkpoint if not os.path.isfile(path)]
    if missing:
        parser.error("missing checkpoint(s): " + ", ".join(missing))

    import tempfile

    device = torch.device(args.device)
    scratch = args.scratch_folder or tempfile.mkdtemp(prefix="mt_evidence_probe_")
    _, terms_fn = _import_trainer_bits()

    trainer = build_trainer(
        args.dataset_id, args.fold, args.configuration, args.plans_identifier, scratch, device
    )
    records = []
    try:
        print(f"[probe] caching {args.steps} unlabelled patches from Dataset{args.dataset_id} "
              f"fold {args.fold} for paired replay", flush=True)
        patches = collect_unlabelled_patches(trainer, args.steps)
        for path in args.checkpoint:
            print(f"\n[probe] === {os.path.basename(path)} ===", flush=True)
            record = probe_checkpoint(trainer, terms_fn, patches, path, args.perturb_seed)
            for line in format_report_lines(record, record["steps"]):
                print(line, flush=True)
            records.append(record)
    finally:
        for loader in (getattr(trainer, "dataloader_train", None), getattr(trainer, "dataloader_val", None)):
            finish = getattr(loader, "_finish", None)
            if callable(finish):
                finish()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"\n[probe] wrote {args.out}", flush=True)

    print("\n[probe] halo ratio by checkpoint (higher = more sub-threshold evidence):", flush=True)
    for record in records:
        print(f"  {record['checkpoint_name']:34s} "
              f"p>0.1/p>0.5={record['halo_ratio_p0p1_over_p0p5']:.4f}  "
              f"subthr_p>=0.05_skel_mass={record['p0p05_to_0p5_teacher_skeleton_mass_share']:.5f}  "
              f"soft_only_patches={record['soft_evidence_without_hard_patch_fraction']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
