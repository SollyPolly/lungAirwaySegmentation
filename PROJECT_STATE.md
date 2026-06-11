# Lung Airway Segmentation — Project State

**Dissertation project, Imperial College London. Deadline: September 7, 2026.**

---

## Goal

Train a 3D U-Net for lung airway segmentation and show that semi-supervised
learning (Mean Teacher) improves over a supervised baseline, with clDice topology
analysis. Dissertation ablation: supervised / supervised+clDice / mean-teacher /
mean-teacher+clDice.

---

## DESIGN PIVOT (2026-06-11): ATM'22 is now the primary dataset

The original design (AeroPath labelled + ATM'22 unlabelled, evaluate on AeroPath)
had two fatal problems: (1) AeroPath's 27 cases give a 2-case test set —
statistically meaningless; (2) AeroPath↔ATM'22 is a domain gap, so it tested
cross-domain SSL, confounding the "does SSL help" claim.

**New design — single-domain SSL on ATM'22, AeroPath as an OOD test:**

| Dataset | Cases | Role |
|---------|-------|------|
| ATM'22 | 150 (→300 with batch 2) | **Primary.** Split into labelled-train / unlabelled-train / val / test. Both supervised and Mean Teacher train here. |
| AeroPath | 27 | **Held-out OOD test only** — "can a model trained on standard scans generalise to severe pathology?" Never in training. |

- The SSL claim is now clean: supervised-ATM(20 labels) vs Mean-Teacher-ATM(20 labels + 90 unlabelled), **paired** on the same ATM test set. Difference = the unlabelled data.
- Enables a **label-efficiency curve** (sweep `labelled_count` 10/20/40; test/val stay fixed).
- The supervised-AeroPath 0.665 model is retained as a *reference* in the AeroPath-OOD column (a model that actually trained on pathology), **not** the SSL baseline.
- Paths: ATM'22 `data/ATM22/` (`imagesTr/`, `labelsTr/`), layout `io/atm22_layout.py`; AeroPath `data/AeroPath/`, layout `io/case_layout.py`. Both on HPC (CX3/CX3-Phase2).

Work is on branch **`atm-primary-ssl`** (additive refactor; AeroPath pathway preserved). Merge to `main` after a clean run.

---

## Dataset-agnostic training architecture (branch `atm-primary-ssl`)

The labelled pathway is now driven by `data_config["dataset_name"]`, so supervised
training runs on **either** dataset and Mean Teacher runs single-domain on ATM'22:

- `datasets/splits.py::create_semisupervised_split` — count-based 4-way ATM split (test/val/labelled/unlabelled), seed-fixed; test/val invariant under `labelled_count` sweeps.
- `datasets/monai_atm22.py::build_monai_atm22_labelled_datasets` — labelled ATM (CT + airway mask), CT-intensity foreground crop (no lung masks).
- `training/builders.py::resolve_case_splits(data_config, training_config)` — returns unified `{labelled_train, unlabelled_train, val, test}`. AeroPath → fractional 3-way, empty `unlabelled_train`. ATM → 4-way counts.
- `build_datasets` dispatches on `dataset_name`; `build_unlabelled_dataloader(atm22_config, cfg, case_ids)` is restricted to the unlabelled-train set (**leakage guard** — verified disjoint from val/test).
- Config validation accepts either `splits` (fractions, AeroPath) or `labelled_split` (counts, ATM).

**How to run each job** (all reuse `configs/data/atm22.yaml` as the ATM data-config):

| Job | Command (key args) |
|-----|--------------------|
| Supervised-AeroPath (existing 0.665) | `train_baseline --data-config aeropath.yaml --training-config baseline.yaml` |
| Supervised-ATM (SSL baseline) | `train_baseline --data-config atm22.yaml --training-config supervised_atm.yaml` |
| Mean-Teacher-ATM | `train_semisupervised --data-config atm22.yaml --atm22-config atm22.yaml --training-config mean_teacher_atm.yaml` |

Split locked in: **test 20 / val 20 / train-pool 110 (labelled 20 + unlabelled 90)** for the 150-case dataset; revisit to ~test 40 / val 30 / labelled 30 / unlabelled ~200 once ATM batch 2 lands (code is count-parameterised, so only config numbers change).

---

## Phase Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Fix supervised baseline | **Complete** — val Dice = 0.665 at threshold=0.99 |
| 1 | ATM'22 integration | **Complete** — dataset, layout, dataloader, smoke tests all working |
| 2 | Mean-Teacher implementation | **Complete** — all components written and smoke-tested; ready to run on HPC |
| 3 | clDice topology loss | Not started |

---

## Supervised Baseline Result

- Best val Dice: **0.665** (epoch 46 of run `baseline-unet-patch-th99-fix`)
- Key finding: `pos_weight=10` in BCE shifts optimal threshold from 0.5 → 0.99. The model was always good; earlier 0.083 Dice was entirely due to wrong threshold at inference.
- Per-case Dice: case20=0.870, case25=0.707, case02=0.623, case03=0.461

---

## Architecture

```
configs/
  data/
    aeropath.yaml          — AeroPath paths and HU window
    atm22.yaml             — ATM'22 batch_root and HU window
  model/
    baseline_unet.yaml     — MONAI UNet config
  training/
    baseline.yaml          — supervised training config
    teacher_student.yaml   — mean-teacher config (threshold=0.99, warm_start=10, rampup=10)

lung_airway_segmentation/
  io/
    case_layout.py         — AeroPath path resolver
    atm22_layout.py        — ATM'22 path resolver (list_case_ids, resolve_case_paths)
  datasets/
    monai_aeropath.py      — labelled MONAI dataset (patch + full-volume)
    monai_atm22.py         — unlabelled MONAI dataset (CT-only, no labels)
    patches.py             — patch sampling utilities
    splits.py              — train/val/test split creation
  schemas.py               — TypedDicts: LabelledCasePaths, UnlabelledCasePaths, etc.
  losses/
    segmentation.py        — CombinedSegmentationLoss (Dice + BCE, pos_weight=10)
    semi_supervised.py     — ConsistencyLoss (confidence-masked MSE, fg/bg balanced)
    topology.py            — clDice stub (not yet implemented)
  models/
    baseline_unet.py       — MONAI UNet wrapper
    ct_fm_segresnet.py     — CT-FM pretrained SegResNet wrapper
  training/
    config.py              — YAML loading, CLI parsing, validation
    builders.py            — factories: model, datasets, dataloaders, teacher, optimizer
    loops.py               — train_one_epoch, validate_one_epoch (supervised)
    teacher_student.py     — update_ema, generate_teacher_probabilities, train_semisupervised_epoch
    engine.py              — run_supervised_training, run_semisupervised_training
  inference/
    sliding_window.py      — sliding window inference
    postprocess.py         — binarize_logits, LCC post-processing
  metrics/
    segmentation.py        — Dice, precision, recall
    topology.py            — topology metrics (tree length, branch count)

scripts/
  train_baseline.py        — supervised training entrypoint
  train_semisupervised.py  — mean-teacher entrypoint
  predict_case.py          — inference on one case
  evaluate_predictions.py  — Dice/precision/recall evaluation
  analyse_distal.py        — distal-radius probability stratification + threshold sweep (Dice/TD)
```

---

## Key Design Decisions

- **Config-first**: all hyperparameters in YAML. Each run saves `resolved_config.json`.
- **Threshold = 0.99**: required because `pos_weight=10` shifts BCE decision boundary to ~0.91–0.99.
- **Teacher validated, not student**: `validate_one_epoch` runs on the EMA teacher during semi-supervised training (more stable predictions).
- **Checkpoint format**: `model_state` = teacher weights (for inference compatibility), `student_model_state` = student weights, `checkpoint_model = "ema_teacher"`.
- **Unlabelled iterator cycles**: ATM'22 (150 cases) >> AeroPath train (~20 cases). The epoch length is determined by the labelled loader; `next_unlabelled_batch` restarts the ATM22 iterator when exhausted.
- **ConsistencyLoss**: confidence-masked MSE. Only voxels where teacher probability ≥ 0.99 (foreground) or ≤ 0.20 (background) contribute. If no confident foreground voxels exist in a batch, returns zero loss via `student_probs.sum() * 0.0` (keeps computation graph intact).
- **British English spelling** throughout: `labelled`, `unlabelled` (two l's).

---

## Mean-Teacher Run 1 Result (`mean-teacher-unet-scratch-50ep`)

- **Run**: 50 epochs, `pos_weight=3`, `threshold=0.75` at validation
- **Reported best val Dice**: 0.141 (epoch 50) — **this was wrong**
- **Threshold sweep result** (`analyse_distal.py`, overlap=0.25): best Dice = **0.345 at threshold=0.99**
- **Root cause**: threshold=0.75 at validation was wrong; same as the supervised baseline disaster (was 0.083 at 0.5 then 0.665 at 0.99)
- **Precision problem**: even at threshold=0.99, precision is only 16–29% (too many false positives). Mean output probability across the whole volume is ~0.47 — the model with `pos_weight=3` doesn't penalise false positives enough
- **Per-case Dice at 0.99**: case03=0.260, case25=0.377, case02=0.329, case20=0.414

## Decision: Run 2 is a Warm-Start Fine-Tune

Run 1 was undertrained and mis-configured (cold teacher from scratch, `pos_weight=3`, wrong threshold), so it cannot be compared to the 0.665 supervised baseline. Run 2 changes the experimental design to **warm-start both student and teacher from the supervised `best_model.pt`**, then fine-tune with ATM'22 consistency. This makes the teacher's pseudo-labels good from epoch 1 and directly tests the hypothesis "does ATM'22 consistency improve a converged model".

**New capability** (`engine.py` + `config.py`): `init_checkpoint` config field / `--init-checkpoint` CLI arg. When set, the supervised `model_state` is loaded into the student *before* `build_teacher` copies it, so both networks start converged. Optimizer state is intentionally **not** restored (fresh fine-tune). Recorded in `run_metadata.json` as `init_checkpoint`.

**`teacher_student.yaml` settings for the warm-start run:**
- `experiment_name`: `mean-teacher-warmstart-pw10-th99`
- `init_checkpoint`: `null` (set via `--init-checkpoint <supervised best_model.pt>` at submit time)
- `epochs`: **40** (fine-tune, not 100)
- `optimizer.lr`: **1e-4** (~1/3 of baseline 3e-4 — adapt without forgetting)
- `scheduler.warmup_epochs`: **0** (no LR warm-up when fine-tuning)
- `teacher.warm_start_epochs`: **0** (teacher already converged)
- `teacher.consistency_rampup_epochs`: **5** (gentle phase-in)
- `teacher.foreground_confidence_threshold`: **0.80** (pseudo-label distal branches, not just proximal — see below)
- `validation.threshold`: **0.99**, `loss.positive_class_weight`: **10.0** (match baseline)

**Distal-branch reasoning** (the `fg_threshold=0.80` choice): with `pos_weight=10` the teacher is confident (≥0.99) on proximal airways but only moderately confident (0.5–0.9) on fine distal branches. If the pseudo-label threshold equals the 0.99 validation threshold, distal branches get **zero** consistency signal — defeating the point of the 150 unlabelled cases. Decoupling the pseudo-label threshold (0.80) from the validation threshold (0.99) lets ATM'22 supervise distal airways. The (0.80, 0.99) gap is the "uncertain zone" — its size over training is worth tracking as an ablation point.

---

## Dataset Mismatch Notes

1. **HU window — RESOLVED.** `atm22.yaml` now uses `[-1024, 2048]`, matching AeroPath. This matters because the warm-started teacher *is* the AeroPath model: feeding it ATM'22 patches normalised with the old `[-1024, 600]` window made walls appear ~2× brighter than its training distribution (a −400 HU wall → 0.20 under AeroPath's window vs 0.38 under the old ATM'22 window), degrading pseudo-labels. Run 1 used the old window; this is now fixed (uncommitted change to `atm22.yaml`).

2. **Spatial resolution** (unresolved): AeroPath are HR CTs (`CT_HR`). If ATM'22 has coarser voxel spacing, a 96³ patch covers different anatomical scales. No resampling applied. Accept for now; candidate for a later spacing-aware ablation.

---

## Unlabelled Case Usage Per Epoch

Per epoch the labelled loader drives epoch length (21 cases × 4 patches = 84 steps); one ATM'22 batch is pulled per labelled step → ~84 of 600 ATM'22 batches per epoch (~14%). All 150 ATM'22 cases are eventually seen as the iterator cycles, but each ATM'22 case gets less per-case exposure than each AeroPath case. The unlabelled advantage is real but diluted per epoch — it acts as regularisation, not bulk gradient signal.

---

## What Needs to Be Done Next (in order)

1. **(Optional) Download ATM'22 batch 2** and consolidate into `data/ATM22/imagesTr` + `labelsTr` (layout auto-detects all `ATM_\d{3}` cases). Then bump the `labelled_split` counts in the ATM configs. Don't block on it — the pipeline works on the 150 now.

2. **Run supervised-ATM baseline** on HPC: `train_baseline --data-config configs/data/atm22.yaml --training-config configs/training/supervised_atm.yaml`. This is the in-domain SSL baseline (labelled_count=20) and produces the checkpoint to warm-start the MT from. Also run the upper bound (labelled_count = full train pool).

3. **Run Mean-Teacher-ATM**: `train_semisupervised --data-config configs/data/atm22.yaml --atm22-config configs/data/atm22.yaml --training-config configs/training/mean_teacher_atm.yaml`. Default config is canonical from-scratch; to warm-start from step 2 set `--init-checkpoint` + `warm_start_epochs: 0`, `lr: 1e-4`, `epochs: 40`. Watch early `teacher_confident_foreground_fraction` / `train_consistency_loss` (run 1 had consistency <1% of loss; raise `consistency_weight` if still negligible).

4. **Sweep + compare**: `analyse_distal.py` on each result; paired comparison of MT vs supervised-ATM on the ATM test set, plus the AeroPath-OOD column.

5. **AeroPath-OOD eval flow**: still to build — run an ATM-trained checkpoint over all 27 AeroPath cases via `predict_case.py` + `evaluate_predictions.py` (HU windows already match, both `[-1024, 2048]`).

6. **Phase 3**: implement clDice in `losses/topology.py` (port soft-skeleton from https://github.com/jocpae/clDice).

**Note:** the warm-start PBS (`train_meanteacher.pbs`) and the old AeroPath-labelled `teacher_student.yaml` predate the pivot. They still work for the cross-domain option but are no longer the primary path — the ATM configs above supersede them.

---

## Verified Notes (previously thought to be issues)

- **Teacher/student augmentation IS differentiated.** Empirically checked: the ATM'22 batch exposes a distinct `teacher_image` (weak, spatial-only) vs `image` (strong, + intensity perturbation); mean abs diff ≈ 0.047. The old "they see identical augmentations / teacher_image never exists" note was wrong — the consistency mechanism works. Perturbation is intensity-only (spatial augs are shared); strengthening it (different spatial views) is a possible future refinement, not a bug.
