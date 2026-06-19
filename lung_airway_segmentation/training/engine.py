"""Training orchestration, checkpointing, and run artifacts."""

import argparse
import json
import re
import time
from datetime import datetime

import torch

from lung_airway_segmentation.settings import RUNS_ROOT
from lung_airway_segmentation.training.builders import (
    build_dataloaders,
    build_datasets,
    build_selftraining_datasets,
    build_teacher,
    build_training_components,
    build_unlabelled_dataloader,
    get_optimizer_learning_rates,
    is_strict_improvement,
    resolve_case_splits,
)
from lung_airway_segmentation.training.config import (
    build_resolved_training_config,
    load_yaml_config,
    resolve_device,
    resolve_project_path,
    validate_model_config,
    validate_selftraining_training_config,
    validate_semisupervised_training_config,
    validate_training_config,
)
from lung_airway_segmentation.training.teacher_student import train_semisupervised_epoch
from lung_airway_segmentation.training.loops import train_one_epoch, validate_one_epoch
from lung_airway_segmentation.losses.semi_supervised import ConsistencyLoss
from lung_airway_segmentation.reporting.run_index import refresh_run_index
from lung_airway_segmentation.reproducibility import (
    collect_environment_metadata,
    seed_everything,
)


def save_checkpoint(model, optimizer, epoch, metrics, output_path, scheduler=None):
    """Save model, optimizer, and summary metrics for one training checkpoint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()

    torch.save(payload, output_path)


def save_semisupervised_checkpoint(
    student,
    teacher,
    optimizer,
    epoch,
    metrics,
    output_path,
    scheduler=None,
):
    """Save a Mean Teacher checkpoint compatible with the inference scripts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        # Inference should use the EMA teacher by default.
        "model_state": teacher.state_dict(),
        "student_model_state": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "checkpoint_model": "ema_teacher",
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, output_path)


def write_json(data, output_path):
    """Write one JSON artifact to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_accepted_pseudo_entries(pseudo_label_dir) -> tuple[list[dict], dict]:
    """Read a pseudo-label manifest and return the accepted (case_id, mask_path) entries.

    ``pseudo_label_dir`` is the directory written by ``scripts/pseudo_label_atm.py``
    (contains ``manifest.json``). Returns ``(entries, manifest)`` where ``entries`` is
    the list of accepted-case dicts and ``manifest`` is the full record for provenance.
    """
    manifest_path = resolve_project_path(pseudo_label_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Pseudo-label manifest not found: {manifest_path}. "
            "Run scripts/pseudo_label_atm.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [case for case in manifest.get("cases", []) if case.get("accepted")]
    if not entries:
        raise ValueError(f"No accepted pseudo-labelled cases in {manifest_path}.")
    return entries, manifest


def slugify_run_component(value: str) -> str:
    """Convert one run-directory label to a safe readable slug."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "run"


def build_run_dir(
    experiment_name,
    model_name,
    created_at=None,
    *,
    study_name=None,
    run_label=None,
):
    """Create a readable run directory, using semantic grouping when configured."""
    created_at = created_at or datetime.now()
    timestamp = created_at.strftime("%Y-%m-%d__%H-%M-%S")
    group_slug = slugify_run_component(str(study_name or experiment_name))
    model_slug = slugify_run_component(str(model_name))
    run_label_slug = slugify_run_component(str(run_label)) if run_label else None
    run_name_parts = [timestamp]
    if run_label_slug:
        run_name_parts.append(run_label_slug)
    run_name_parts.append(model_slug)
    return RUNS_ROOT / group_slug / "__".join(run_name_parts)


def initialize_run_artifacts(run_dir, run_metadata, resolved_config):
    """Write static run artifacts before training starts."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_metadata, run_dir / "run_metadata.json")
    write_json(resolved_config, run_dir / "resolved_config.json")

    notes_path = run_dir / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            "# Run notes\n\n"
            "**TL;DR:** _one-line result + verdict_\n\n"
            "## Config\n\n"
            "## Result\n\n"
            "## Insight (what this run tells us)\n\n"
            "## Follow-ups\n",
            encoding="utf-8",
        )

    refresh_run_index()


def run_supervised_training(args: argparse.Namespace) -> None:
    """Run one supervised baseline experiment from parsed CLI args."""
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)
    training_config = load_yaml_config(args.training_config)

    resolved_training_config = build_resolved_training_config(training_config, args)
    validate_model_config(model_config)
    if resolved_training_config.get("selftraining"):
        validate_selftraining_training_config(resolved_training_config)
    else:
        validate_training_config(resolved_training_config)

    device = resolve_device(args.device)
    deterministic = bool(resolved_training_config.get("deterministic", True))
    seed_everything(int(resolved_training_config["seed"]), deterministic=deterministic)

    data_root = resolve_project_path(
        data_config.get("raw_data_root") or data_config["batch_root"]
    )
    splits = resolve_case_splits(data_config, resolved_training_config)
    train_ids, val_ids, test_ids = splits["labelled_train"], splits["val"], splits["test"]

    # Self-training branch: mix the labelled cases with accepted pseudo-labelled
    # cases (topology-filtered). Absent the `selftraining` block this is a no-op and
    # the run is an ordinary supervised one — so existing configs are unchanged.
    selftraining_config = resolved_training_config.get("selftraining")
    selftraining_metadata = None
    if selftraining_config:
        pseudo_entries, pseudo_manifest = load_accepted_pseudo_entries(
            selftraining_config["pseudo_label_dir"]
        )
        labelled_oversample = int(selftraining_config.get("labelled_oversample", 1))
        train_dataset, val_dataset = build_selftraining_datasets(
            train_ids,
            val_ids,
            pseudo_entries,
            data_config,
            resolved_training_config,
            labelled_oversample=labelled_oversample,
        )
        selftraining_metadata = {
            "pseudo_label_dir": str(resolve_project_path(selftraining_config["pseudo_label_dir"])),
            "labelled_oversample": labelled_oversample,
            "labeller_run": pseudo_manifest.get("labeller_run"),
            "labeller_checkpoint": pseudo_manifest.get("checkpoint"),
            "pseudo_threshold": pseudo_manifest.get("threshold"),
            "n_pseudo_accepted": len(pseudo_entries),
            "n_pseudo_total": pseudo_manifest.get("n_total"),
            "pseudo_case_ids": [str(entry["case_id"]) for entry in pseudo_entries],
        }
    else:
        train_dataset, val_dataset = build_datasets(
            train_ids,
            val_ids,
            data_config,
            resolved_training_config,
        )
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=int(resolved_training_config["batch_size"]),
        num_workers=int(resolved_training_config["num_workers"]),
        seed=int(resolved_training_config["seed"]),
    )

    model, loss_fn, optimizer, scheduler = build_training_components(
        device,
        model_config,
        resolved_training_config,
    )

    # Optional warm-start (self-training): load a converged supervised checkpoint
    # into the model so the student begins from the labeller rather than scratch.
    # The optimizer keeps in-place references to the model parameters, so loading
    # after build_training_components is safe and optimizer state stays fresh.
    init_checkpoint = resolved_training_config.get("init_checkpoint")
    init_checkpoint_info = None
    if init_checkpoint:
        init_checkpoint_path = resolve_project_path(init_checkpoint)
        if not init_checkpoint_path.is_file():
            raise FileNotFoundError(f"init_checkpoint does not exist: {init_checkpoint_path}")
        checkpoint_payload = torch.load(init_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_payload["model_state"])
        init_checkpoint_info = {
            "path": str(init_checkpoint_path),
            "epoch": checkpoint_payload.get("epoch"),
        }
        print(
            f"Warm-started model from {init_checkpoint_path} "
            f"(checkpoint epoch {checkpoint_payload.get('epoch')})"
        )

    use_amp = bool(resolved_training_config.get("amp", {}).get("enabled", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_started_at = datetime.now()
    run_dir = build_run_dir(
        resolved_training_config["experiment_name"],
        model_config["model_name"],
        created_at=run_started_at,
        study_name=resolved_training_config.get("study_name"),
        run_label=resolved_training_config.get("run_label"),
    )
    best_val_dice = -1.0
    best_epoch = 0
    best_val_cldice = -1.0
    best_topology_epoch = 0
    history = []

    training_regime = resolved_training_config["training_regime"]
    sampling_config = resolved_training_config["sampling"]
    validation_config = resolved_training_config["validation"]

    # clDice warm-up: the soft skeleton of an untrained model's near-uniform
    # output is noise, so train Dice+BCE only for the first epochs, then ramp the
    # clDice term in. loss_fn.cldice_weight is mutated per epoch in the loop below.
    loss_config = resolved_training_config["loss"]
    max_cldice_weight = float(loss_config.get("cldice_weight", 0.0))
    cldice_warmup_epochs = int(loss_config.get("cldice_warmup_epochs", 0))
    cldice_rampup_epochs = int(loss_config.get("cldice_rampup_epochs", 0))
    # cbDice — same warm-up treatment as clDice (the soft skeleton of an untrained
    # model is noise). 0.0 = off (default).
    max_cbdice_weight = float(loss_config.get("cbdice_weight", 0.0))
    cbdice_warmup_epochs = int(loss_config.get("cbdice_warmup_epochs", 0))
    cbdice_rampup_epochs = int(loss_config.get("cbdice_rampup_epochs", 0))
    # EXPERIMENTAL persistent-homology term — same warm-up treatment as clDice
    # (topology of an untrained model is noise). 0.0 = off (default).
    max_topo_weight = float(loss_config.get("topo_weight", 0.0))
    topo_warmup_epochs = int(loss_config.get("topo_warmup_epochs", 0))
    topo_rampup_epochs = int(loss_config.get("topo_rampup_epochs", 0))

    # Topology validation metrics (best_topology_model selection) skeletonise the
    # RAW prediction, which is most expensive on the messy masks of early/warm-up
    # epochs that can never win topology selection anyway. So compute them only
    # once the topology-loss warm-up has elapsed (0 for a pure-Dice baseline → from
    # the first validation). The selection threshold and the optional catastrophic
    # volume guard (default None = no gate) are read once here.
    topology_metrics_warmup_epochs = max(cldice_warmup_epochs, cbdice_warmup_epochs)
    topology_threshold = float(validation_config.get("topology_threshold", 0.5))
    _topology_max_ratio = validation_config.get("topology_max_ratio")
    topology_max_ratio = (
        float(_topology_max_ratio) if _topology_max_ratio is not None else None
    )

    run_description = args.run_description
    if run_description is None:
        run_description = resolved_training_config.get("description")
    if run_description is None:
        if training_regime == "patch":
            run_description = "Supervised MONAI patch-based baseline."
        else:
            run_description = "Supervised full-volume ablation baseline."

    resolved_config_artifact = {
        "data": data_config,
        "model": model_config,
        "training": resolved_training_config,
    }

    run_metadata = {
        "study_name": resolved_training_config.get("study_name"),
        "run_label": resolved_training_config.get("run_label"),
        "experiment_name": resolved_training_config["experiment_name"],
        "description": run_description,
        "created_at": run_started_at.isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "config_files": {
            "data": str(args.data_config),
            "model": str(args.model_config),
            "training": str(args.training_config),
        },
        "data_root": str(data_root),
        "device": str(device),
        "deterministic": deterministic,
        "environment": collect_environment_metadata(),
        "data_pipeline": training_regime,
        "amp_enabled": use_amp,
        "model_name": model_config["model_name"],
        "optimizer_name": resolved_training_config["optimizer"]["name"],
        "scheduler_name": resolved_training_config["scheduler"]["name"],
        "effective_batch_size": (
            int(resolved_training_config["batch_size"])
            if training_regime == "full_volume"
            else int(resolved_training_config["batch_size"]) * int(sampling_config["patches_per_case"])
        ),
        "splits": {
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_case_ids": train_ids,
            "val_case_ids": val_ids,
            "test_case_ids": test_ids,
        },
        "init_checkpoint": init_checkpoint_info,
        "selftraining": selftraining_metadata,
    }
    initialize_run_artifacts(run_dir, run_metadata, resolved_config_artifact)

    for epoch in range(int(resolved_training_config["epochs"])):
        epoch_start_time = time.perf_counter()
        if max_cldice_weight > 0.0:
            epochs_after_warmup = max((epoch + 1) - cldice_warmup_epochs, 0)
            if epochs_after_warmup <= 0:
                loss_fn.cldice_weight = 0.0
            elif cldice_rampup_epochs > 0:
                loss_fn.cldice_weight = max_cldice_weight * min(
                    epochs_after_warmup / cldice_rampup_epochs, 1.0
                )
            else:
                loss_fn.cldice_weight = max_cldice_weight

        if max_cbdice_weight > 0.0:
            epochs_after_cbdice_warmup = max((epoch + 1) - cbdice_warmup_epochs, 0)
            if epochs_after_cbdice_warmup <= 0:
                loss_fn.cbdice_weight = 0.0
            elif cbdice_rampup_epochs > 0:
                loss_fn.cbdice_weight = max_cbdice_weight * min(
                    epochs_after_cbdice_warmup / cbdice_rampup_epochs, 1.0
                )
            else:
                loss_fn.cbdice_weight = max_cbdice_weight

        if max_topo_weight > 0.0:
            epochs_after_topo_warmup = max((epoch + 1) - topo_warmup_epochs, 0)
            if epochs_after_topo_warmup <= 0:
                loss_fn.topo_weight = 0.0
            elif topo_rampup_epochs > 0:
                loss_fn.topo_weight = max_topo_weight * min(
                    epochs_after_topo_warmup / topo_rampup_epochs, 1.0
                )
            else:
                loss_fn.topo_weight = max_topo_weight

        learning_rates_before_epoch = get_optimizer_learning_rates(optimizer)
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        epoch_summary = {
            "epoch": epoch + 1,
            "learning_rate": learning_rates_before_epoch[0],
            "learning_rates_before_epoch": learning_rates_before_epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "cldice_weight": float(getattr(loss_fn, "cldice_weight", 0.0)),
            "cbdice_weight": float(getattr(loss_fn, "cbdice_weight", 0.0)),
        }

        should_validate = (
            (epoch + 1) % int(validation_config["validate_every"]) == 0
            or epoch + 1 == int(resolved_training_config["epochs"])
        )
        if should_validate:
            validation_start_time = time.perf_counter()
            val_metrics = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                roi_size=tuple(int(value) for value in validation_config["roi_size"]),
                sw_batch_size=int(validation_config["sw_batch_size"]),
                overlap=float(validation_config["inference_overlap"]),
                use_amp=use_amp,
                threshold=float(validation_config.get("threshold", 0.5)),
                compute_topology=(epoch + 1) > topology_metrics_warmup_epochs,
                topology_threshold=topology_threshold,
                topology_max_ratio=topology_max_ratio,
                # Soft cbDice on full val volumes is EDT-heavy (~87% of a cbDice run's
                # walltime). Default: compute it iff cbDice is active; set
                # validation.compute_soft_cbdice: false to skip it (hard-clDice topology
                # selection is unaffected) and cut a cbDice run ~11h -> ~2h.
                compute_soft_cbdice=bool(
                    validation_config.get("compute_soft_cbdice", max_cbdice_weight > 0.0)
                ),
            )
            # Wall-clock of the whole validation pass (incl. the new raw
            # skeletonisation) — written to history.json each validation so the
            # cost is observable live on the shared FS (PBS spools train.out only
            # at job end). Watch it on the first post-warm-up validation.
            epoch_summary["val_seconds"] = time.perf_counter() - validation_start_time
            epoch_summary["val_loss"] = val_metrics["loss"]
            epoch_summary["val_dice"] = val_metrics["dice"]
            epoch_summary["val_per_case_dice"] = val_metrics.get("per_case_dice", {})
            # Topology-aware selection diagnostics (hard mask @ topology_threshold,
            # no LCC). *_loss keys are losses (lower better); cldice is a score.
            for key in (
                "cldice",
                "topology_precision",
                "tree_length_detected",
                "foreground_volume_ratio",
                "predicted_component_count",
                "gated_case_count",
                "bce_loss",
                "dice_loss",
                "soft_cldice_loss",
                "soft_cbdice_loss",
            ):
                if key in val_metrics:
                    epoch_summary[f"val_{key}"] = val_metrics[key]
            epoch_summary["val_per_case_cldice"] = val_metrics.get("per_case_cldice", {})

            per_case_str = "  ".join(
                f"{cid}:{d:.3f}"
                for cid, d in sorted(val_metrics.get("per_case_dice", {}).items())
            )
            print(
                f"Epoch {epoch + 1} / {resolved_training_config['epochs']}"
                f" - train_loss: {train_metrics['loss']:.4f}"
                f" - train_dice: {train_metrics['dice']:.4f}"
                f" - val_loss: {val_metrics['loss']:.4f}"
                f" - val_dice: {val_metrics['dice']:.4f}"
                + (
                    f" - val_cldice@{topology_threshold:g}: {val_metrics['cldice']:.4f}"
                    f" - val_fg_ratio@{topology_threshold:g}: {val_metrics['foreground_volume_ratio']:.3f}"
                    if "cldice" in val_metrics
                    else ""
                )
                + f" - val_seconds: {epoch_summary['val_seconds']:.1f}"
                + (f"\n  per-case: {per_case_str}" if per_case_str else "")
            )
        else:
            val_metrics = None
            print(
                f"Epoch {epoch + 1} / {resolved_training_config['epochs']}"
                f" - train_loss: {train_metrics['loss']:.4f}"
                f" - train_dice: {train_metrics['dice']:.4f}"
                " - validation skipped"
            )

        if scheduler is not None:
            scheduler.step()
            epoch_summary["scheduler_stepped"] = True
        else:
            epoch_summary["scheduler_stepped"] = False

        epoch_summary["learning_rates_after_epoch"] = get_optimizer_learning_rates(optimizer)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=epoch_summary,
            output_path=run_dir / "last_model.pt",
            scheduler=scheduler,
        )

        # best_dice_model.pt — highest val Dice at the configured threshold
        # (proximal-volume biased; the historical selection). best_model.pt is
        # kept as a compatibility alias for existing analysis scripts. Selected
        # INDEPENDENTLY of best_topology_model below — the two can land on
        # different epochs (the whole point: Dice rewards proximal volume, raw
        # clDice rewards the centreline tree).
        if val_metrics is not None and is_strict_improvement(
            val_metrics["dice"], best_val_dice
        ):
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch + 1
            for checkpoint_name in ("best_dice_model.pt", "best_model.pt"):
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    metrics=epoch_summary,
                    output_path=run_dir / checkpoint_name,
                    scheduler=scheduler,
                )

        # best_topology_model.pt — highest RAW hard-mask clDice at topology_threshold
        # (no LCC, no volume gate), the centreline selector. 'cldice' is absent until
        # after the topology-loss warm-up (and if every case was catastrophically
        # gated), so this epoch is simply skipped for topology selection then.
        if val_metrics is not None and is_strict_improvement(
            val_metrics.get("cldice"), best_val_cldice
        ):
            best_val_cldice = val_metrics["cldice"]
            best_topology_epoch = epoch + 1
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics=epoch_summary,
                output_path=run_dir / "best_topology_model.pt",
                scheduler=scheduler,
            )

        epoch_summary["epoch_seconds"] = time.perf_counter() - epoch_start_time
        history.append(epoch_summary)
        write_json(
            {
                "best": {
                    "epoch": best_epoch,
                    "val_dice": best_val_dice,
                },
                "best_topology": {
                    "epoch": best_topology_epoch,
                    "val_cldice": best_val_cldice,
                    "threshold": topology_threshold,
                    "lcc": False,
                },
                "history": history,
            },
            run_dir / "history.json",
        )

    refresh_run_index()
    print("Training finished.")
    print(f"Run directory: {run_dir}")
    print(f"Held-out test cases: {len(test_ids)}")
    print(f"Best val dice: {best_val_dice:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val clDice@0.5 (topology): {best_val_cldice:.4f}")
    print(f"Best topology epoch: {best_topology_epoch}")


def run_selftraining_training(args: argparse.Namespace) -> None:
    """Run topology-filtered self-training.

    Self-training is the supervised patch loop over a combined dataset of the
    labelled cases (oversampled) plus accepted pseudo-labelled unlabelled cases,
    optionally warm-started from the labeller. All of that is driven by the
    resolved config's ``selftraining`` block + ``init_checkpoint`` (populated from
    the training YAML and the CLI in build_resolved_training_config), so the
    supervised runner already does the work — including the three-checkpoint
    topology-aware selection and the clDice/cbDice warm-up ramps.
    """
    run_supervised_training(args)


def run_semisupervised_training(args: argparse.Namespace) -> None:
    """Run a Mean Teacher experiment using AeroPath labels and ATM'22 CTs."""
    data_config = load_yaml_config(args.data_config)
    atm22_config = load_yaml_config(args.atm22_config)
    model_config = load_yaml_config(args.model_config)
    training_config = load_yaml_config(args.training_config)

    resolved_training_config = build_resolved_training_config(training_config, args)
    validate_model_config(model_config)
    validate_semisupervised_training_config(resolved_training_config)

    device = resolve_device(args.device)
    deterministic = bool(resolved_training_config.get("deterministic", True))
    seed_everything(int(resolved_training_config["seed"]), deterministic=deterministic)

    data_root = resolve_project_path(
        data_config.get("raw_data_root") or data_config["batch_root"]
    )
    splits = resolve_case_splits(data_config, resolved_training_config)
    train_ids, val_ids, test_ids = splits["labelled_train"], splits["val"], splits["test"]
    unlabelled_ids = splits["unlabelled_train"]
    if not unlabelled_ids:
        raise ValueError(
            "Mean Teacher training requires unlabelled cases. Use an ATM'22 data-config "
            "with a 'labelled_split' that leaves cases unlabelled "
            "(labelled_count below the train-pool size)."
        )

    train_dataset, val_dataset = build_datasets(
        train_ids,
        val_ids,
        data_config,
        resolved_training_config,
    )
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=int(resolved_training_config["batch_size"]),
        num_workers=int(resolved_training_config["num_workers"]),
        seed=int(resolved_training_config["seed"]),
    )
    unlabelled_loader = build_unlabelled_dataloader(
        atm22_config,
        resolved_training_config,
        unlabelled_ids,
    )

    student, loss_fn, optimizer, scheduler = build_training_components(
        device,
        model_config,
        resolved_training_config,
    )

    # Warm-start: load supervised weights into the student before the teacher is
    # copied from it, so both networks begin from the converged baseline and the
    # teacher produces meaningful pseudo-labels from the first consistency step.
    # The optimizer keeps in-place references to the student parameters, so
    # loading after build_training_components is safe; optimizer state stays fresh.
    init_checkpoint = resolved_training_config.get("init_checkpoint")
    init_checkpoint_info = None
    if init_checkpoint:
        init_checkpoint_path = resolve_project_path(init_checkpoint)
        if not init_checkpoint_path.is_file():
            raise FileNotFoundError(f"init_checkpoint does not exist: {init_checkpoint_path}")
        checkpoint_payload = torch.load(init_checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint_payload["model_state"])
        init_checkpoint_info = {
            "path": str(init_checkpoint_path),
            "epoch": checkpoint_payload.get("epoch"),
        }
        print(
            f"Warm-started student from {init_checkpoint_path} "
            f"(checkpoint epoch {checkpoint_payload.get('epoch')})"
        )

    teacher = build_teacher(student)
    teacher_config = resolved_training_config["teacher"]
    consistency_loss_fn = ConsistencyLoss(
        foreground_threshold=float(teacher_config["foreground_confidence_threshold"]),
        background_threshold=float(teacher_config["background_confidence_threshold"]),
    ).to(device)

    use_amp = (
        bool(resolved_training_config.get("amp", {}).get("enabled", False))
        and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_started_at = datetime.now()
    run_dir = build_run_dir(
        resolved_training_config["experiment_name"],
        f"mean_teacher_{model_config['model_name']}",
        created_at=run_started_at,
        study_name=resolved_training_config.get("study_name"),
        run_label=resolved_training_config.get("run_label"),
    )
    best_val_dice = -1.0
    best_epoch = 0
    history = []

    sampling_config = resolved_training_config["sampling"]
    validation_config = resolved_training_config["validation"]
    warm_start_epochs = int(teacher_config["warm_start_epochs"])
    rampup_epochs = int(teacher_config.get("consistency_rampup_epochs", 0))
    maximum_consistency_weight = float(
        resolved_training_config["loss"]["consistency_weight"]
    )
    atm22_root = resolve_project_path(atm22_config["batch_root"])
    unlabelled_records = getattr(unlabelled_loader.dataset, "data", [])
    unlabelled_case_ids = [
        str(record["case_id"])
        for record in unlabelled_records
        if isinstance(record, dict) and "case_id" in record
    ]

    run_description = args.run_description or resolved_training_config.get("description")
    if run_description is None:
        run_description = "Mean Teacher training with AeroPath labels and ATM'22 unlabeled CTs."

    resolved_config_artifact = {
        "data": data_config,
        "unlabelled_data": atm22_config,
        "model": model_config,
        "training": resolved_training_config,
    }
    run_metadata = {
        "study_name": resolved_training_config.get("study_name"),
        "run_label": resolved_training_config.get("run_label"),
        "experiment_name": resolved_training_config["experiment_name"],
        "description": run_description,
        "created_at": run_started_at.isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "config_files": {
            "data": str(args.data_config),
            "unlabelled_data": str(args.atm22_config),
            "model": str(args.model_config),
            "training": str(args.training_config),
        },
        "data_root": str(data_root),
        "unlabelled_data_root": str(atm22_root),
        "device": str(device),
        "deterministic": deterministic,
        "environment": collect_environment_metadata(),
        "data_pipeline": "mean_teacher_patch",
        "amp_enabled": use_amp,
        "model_name": model_config["model_name"],
        "checkpoint_model": "ema_teacher",
        "init_checkpoint": init_checkpoint_info,
        "optimizer_name": resolved_training_config["optimizer"]["name"],
        "scheduler_name": resolved_training_config["scheduler"]["name"],
        "effective_labelled_batch_size": (
            int(resolved_training_config["batch_size"])
            * int(sampling_config["patches_per_case"])
        ),
        "effective_unlabelled_batch_size": (
            int(resolved_training_config["batch_size_unlabelled"])
            * int(sampling_config["patches_per_case"])
        ),
        "splits": {
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_case_ids": train_ids,
            "val_case_ids": val_ids,
            "test_case_ids": test_ids,
        },
        "unlabelled_cases": {
            "count": len(unlabelled_case_ids),
            "case_ids": unlabelled_case_ids,
        },
    }
    initialize_run_artifacts(run_dir, run_metadata, resolved_config_artifact)

    for epoch in range(int(resolved_training_config["epochs"])):
        epoch_number = epoch + 1
        epochs_after_warm_start = max(epoch_number - warm_start_epochs, 0)
        use_consistency = epochs_after_warm_start > 0 and maximum_consistency_weight > 0.0
        if not use_consistency:
            consistency_weight = 0.0
        elif rampup_epochs > 0:
            consistency_weight = maximum_consistency_weight * min(
                epochs_after_warm_start / rampup_epochs,
                1.0,
            )
        else:
            consistency_weight = maximum_consistency_weight

        learning_rates_before_epoch = get_optimizer_learning_rates(optimizer)
        train_metrics = train_semisupervised_epoch(
            student=student,
            teacher=teacher,
            labelled_loader=train_loader,
            unlabelled_loader=unlabelled_loader,
            loss_fn=loss_fn,
            consistency_loss_fn=consistency_loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            ema_alpha=float(teacher_config["ema_decay"]),
            consistency_weight=consistency_weight,
            use_consistency=use_consistency,
            use_amp=use_amp,
            threshold=float(validation_config.get("threshold", 0.5)),
        )

        epoch_summary = {
            "epoch": epoch_number,
            "learning_rate": learning_rates_before_epoch[0],
            "learning_rates_before_epoch": learning_rates_before_epoch,
            "train_loss": train_metrics["loss"],
            "train_supervised_loss": train_metrics["supervised_loss"],
            "train_consistency_loss": train_metrics["consistency_loss"],
            "train_dice": train_metrics["dice"],
            "teacher_confident_fraction": train_metrics["confident_fraction"],
            "teacher_confident_foreground_fraction": train_metrics[
                "confident_foreground_fraction"
            ],
            "teacher_confident_background_fraction": train_metrics[
                "confident_background_fraction"
            ],
            "consistency_enabled": use_consistency,
            "consistency_weight": consistency_weight,
        }

        should_validate = (
            epoch_number % int(validation_config["validate_every"]) == 0
            or epoch_number == int(resolved_training_config["epochs"])
        )
        if should_validate:
            val_metrics = validate_one_epoch(
                model=teacher,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                roi_size=tuple(int(value) for value in validation_config["roi_size"]),
                sw_batch_size=int(validation_config["sw_batch_size"]),
                overlap=float(validation_config["inference_overlap"]),
                use_amp=use_amp,
                threshold=float(validation_config.get("threshold", 0.5)),
                compute_topology=False,
            )
            epoch_summary["val_loss"] = val_metrics["loss"]
            epoch_summary["val_dice"] = val_metrics["dice"]
            epoch_summary["val_per_case_dice"] = val_metrics.get("per_case_dice", {})

            per_case_str = "  ".join(
                f"{case_id}:{dice:.3f}"
                for case_id, dice in sorted(val_metrics.get("per_case_dice", {}).items())
            )
            print(
                f"Epoch {epoch_number} / {resolved_training_config['epochs']}"
                f" - total_loss: {train_metrics['loss']:.4f}"
                f" - supervised_loss: {train_metrics['supervised_loss']:.4f}"
                f" - consistency_loss: {train_metrics['consistency_loss']:.4f}"
                f" - train_dice: {train_metrics['dice']:.4f}"
                f" - val_loss: {val_metrics['loss']:.4f}"
                f" - val_dice: {val_metrics['dice']:.4f}"
                + (f"\n  per-case: {per_case_str}" if per_case_str else "")
            )
        else:
            val_metrics = None
            print(
                f"Epoch {epoch_number} / {resolved_training_config['epochs']}"
                f" - total_loss: {train_metrics['loss']:.4f}"
                f" - supervised_loss: {train_metrics['supervised_loss']:.4f}"
                f" - consistency_loss: {train_metrics['consistency_loss']:.4f}"
                f" - train_dice: {train_metrics['dice']:.4f}"
                " - validation skipped"
            )

        if scheduler is not None:
            scheduler.step()
            epoch_summary["scheduler_stepped"] = True
        else:
            epoch_summary["scheduler_stepped"] = False
        epoch_summary["learning_rates_after_epoch"] = get_optimizer_learning_rates(optimizer)

        save_semisupervised_checkpoint(
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            epoch=epoch_number,
            metrics=epoch_summary,
            output_path=run_dir / "last_model.pt",
            scheduler=scheduler,
        )
        if val_metrics is not None and val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch_number
            save_semisupervised_checkpoint(
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                epoch=epoch_number,
                metrics=epoch_summary,
                output_path=run_dir / "best_model.pt",
                scheduler=scheduler,
            )

        history.append(epoch_summary)
        write_json(
            {
                "best": {
                    "epoch": best_epoch,
                    "val_dice": best_val_dice,
                },
                "history": history,
            },
            run_dir / "history.json",
        )

    refresh_run_index()
    print("Mean Teacher training finished.")
    print(f"Run directory: {run_dir}")
    print(f"Held-out test cases: {len(test_ids)}")
    print(f"Unlabelled ATM'22 cases: {len(unlabelled_case_ids)}")
    print(f"Best teacher val dice: {best_val_dice:.4f}")
    print(f"Best epoch: {best_epoch}")
