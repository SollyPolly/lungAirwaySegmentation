# Lung Airway Segmentation — Project State

**Dissertation project, Imperial College London. Deadline: September 7, 2026.**
_Last updated: 2026-06-12._

---

## Goal & narrative (current)

The project began as *unsupervised*, moved to *semi-supervised* (Mean Teacher), and
has now settled on its strongest, evidence-backed shape: **topology-aware distal
airway segmentation**. The arc that holds the dissertation together:

1. **Diagnose** — airway segmentation fails on *distal* branches, and volumetric
   metrics (Dice) hide it because thin branches are ~nothing by volume.
2. **Fix it in the loss** — clDice (centerline/topology loss) recovers the distal
   tree. **This is the core positive contribution.**
3. **Unlock it with post-processing** — largest-connected-component (LCC) cleanup
   turns the topology gain into a *free* one (no Dice cost).
4. **Propagate it semi-supervised** — topology-filtered self-training on the
   unlabelled ATM'22 cases (the SSL route that *replaces* Mean Teacher).
5. **Mean Teacher is a documented negative result** (confirmation bias) that
   motivates the pivot — examiners value a well-diagnosed negative.

Everything is on **`main`** (the earlier `atm-primary-ssl` branch was merged).

---

## Datasets

| Dataset | Cases | Role |
|---------|-------|------|
| **ATM'22** | 150 (`data/ATM22/imagesTr` + `labelsTr`) | **Primary.** Split 20 labelled / 90 unlabelled / 20 val / 20 test (seed 15, count-based). Both supervised and SSL train here. |
| **AeroPath** | 27 (`data/AeroPath/`) | **Held-out OOD test only** — "can a model trained on standard scans generalise to severe pathology?" Never in training. |

- HU window matched at **`[-1024, 2048]`** for both (so a model trained on one isn't out-of-distribution on the other).
- Both on Imperial HPC (CX3/CX3-Phase2). ATM'22 **batch 2** (→~300 cases) is downloadable; the split is count-parameterised so only config numbers change when it lands.
- Layouts: `io/atm22_layout.py`, `io/case_layout.py`.

---

## Headline results so far

**The operating recipe:** `clDice loss → threshold ~0.85–0.90 → LCC-6 post-processing`.

| Model (ATM'22, in-domain) | Dice @ best op-point | TD (tree-length detected) | Verdict |
|---------------------------|---------------------:|--------------------------:|---------|
| Supervised-ATM (20 labels) | ~0.71 (+LCC @0.85) | **0.24** | in-domain baseline |
| **Supervised-ATM + clDice** | ~0.70 (+LCC @0.90) | **0.32** | **+~40% TD at equal Dice** |
| Mean-Teacher-ATM (both variants) | ≤ 0.50 | — | **degraded — negative result** |

*Numbers are means over 3 test cases (2/12/14) at sliding-window overlap 0.25–0.5; the full 20-case table is still to be produced.*

**Why clDice wins (measured, not asserted):** clDice produces a *connected* tree.
Under LCC at threshold 0.50, clDice's TD drops only ~5% (0.636→0.604) while the
baseline's drops ~32% (0.637→0.435) — the baseline finds distal voxels but leaves
them as disconnected islands that LCC deletes; clDice wires them into the tree.

**clDice trade is essentially free with LCC:** raw @0.99 it was −0.035 Dice for
+54% TD; with LCC at ~0.85–0.90 the Dice gap closes (~0.70 both) and clDice keeps
~+40% TD. The clDice model's Dice peaks at 0.99 *without* LCC (no hidden lower
optimum) — LCC is what unlocks the lower-threshold/higher-TD operating point.

**Methodological note:** TD is recall-like (ignores false positives), so its max at
low thresholds is degenerate — always compare TD at a *fixed* operating threshold,
paired with a precision-side metric (BD / topology precision, both in
`metrics/topology.py`).

### Diagnosis (updated 2026-06-12, n=20 val): distal tree is below threshold — but recovering it isn't free

Confirmed on the full 20 val cases (clDice model, `analyse_distal` dev run). The
distal voxels (r=1) are **~50% of all airway voxels**; the model predicts most of
them but at confidences the operating threshold discards:

| Distal (r=1) recall | @op 0.95 | @0.50 |
|---|---:|---:|
| recall | **39.9%** | **79.5%** |

The catch is **precision**. Threshold sweep, +LCC, mean over 20 val cases:

| threshold | Dice+LCC | TD+LCC | voxel-Prec+LCC |
|---|---:|---:|---:|
| 0.50 | 0.512 | **0.592** | 0.364 |
| 0.90 | **0.730** | 0.317 | 0.757 |
| 0.95 (op @ floor 0.80) | 0.721 | 0.278 | 0.805 |
| 0.99 | 0.658 | 0.210 | 0.860 |

The high TD at 0.50 comes with **voxel precision 0.36** — after LCC, ~⅔ of the
predicted tree's voxels are false. A precision-constrained op-point (≥0.80 voxel
precision) therefore lands at **~0.90–0.95**, TD ~0.28–0.32 — **not** the ~0.60 the
raw TD sweep implied. **This qualifies the earlier "operate low → ~double TD for
free" framing: TD's recall-like blindness to false positives hid heavy leakage.**

**Fork resolved (n=18 val, clDice model):** topology precision is **0.64 @0.50** vs
**0.86 @0.90**; voxel precision is 0.36 vs 0.76. So the low-threshold blow-up in voxel
precision is **mostly radial over-segmentation** (fat tubes — 64% of the centreline is
inside GT while only 36% of voxels are), i.e. **voxel precision was the wrong gate**.
But the 22-pt topology-precision drop (0.86→0.64) is a **real leakage/false-branch
component** — not purely benign.

**Key consequence — clDice favours a LOW operating point.** clDice (the core topology
metric, harmonic mean of topo-precision and TD) is **0.62 @0.50 vs 0.46 @0.90** —
higher low, because at 0.50 precision (0.64) and TD (0.60) are balanced, whereas at
0.90 TD (0.32) is the limiter. So the op-point should be selected by **clDice /
topology precision (~0.5–0.6)**, not the voxel-precision floor (which forced 0.95,
clDice 0.46). The residual leakage means **calibration (pos_weight)** is still a real
lever — push topo-precision up at low thresholds for an even better clDice.

Caveats: n=18, val, clDice-only; intermediate thresholds (0.6–0.7) unmeasured so the
clDice peak may sit slightly above 0.5; and the **baseline (supervised-ATM) still needs
the same run** — that comparison, not the clDice number alone, is the result.
Trade-off to state plainly: at ~0.5 the *mask* is poor (Dice+LCC ~0.50, voxel-prec
~0.36, fat walls) while the *tree* is good (clDice ~0.62) — a topology-first result,
not a segmentation-quality one. (A centreline-preserving thinning post-process could
later reclaim voxel precision without losing the tree — future scope.)

---

## Run inventory (key runs + verdicts)

Per-run detail lives in each run's `runs/<exp>/<ts>/notes.md` (local; `runs/` is gitignored).

| Run | What | Result / verdict |
|-----|------|------------------|
| `baseline-unet-patch-th99-fix` | Supervised AeroPath | **0.665** @0.99. Now the AeroPath-OOD *reference*, not the SSL baseline. |
| `ctfm-segresnet-corrected-50ep` | CT-FM pretrained SegResNet, AeroPath | **0.785** @0.99. **Reference ceiling only** (foundation model on a huge corpus) — NOT the method backbone. The method stays on the from-scratch UNet. |
| `supervised-atm-l20` | Supervised ATM, 20 labels | **0.609** val @0.99 (ep45). **The in-domain SSL baseline** / warm-start source. |
| `mean-teacher-unet-scratch-50ep` | MT run 1 (pw3/th75, scratch) | 0.141→0.345 swept. Mis-configured (wrong threshold + pos_weight). |
| `mean-teacher-warmstart-pw10-th99` | MT cross-domain (AeroPath lab + ATM unlab) | Warm-started 0.654 → **degraded to 0.499**. Domain gap. |
| `mean-teacher-atm-l20` | MT single-domain, from scratch | 0.459 @ep60, undertrained (still climbing). |
| `mean-teacher-atm-warmstart-l20` | MT single-domain, warm-started | 0.597 (ep5, best) → **0.499 (ep40)**. Consistency *erodes* it. |
| `supervised-atm-l20-cldice` (attempt 1) | clDice, **no warm-up** | 0.229, broke everything (proximal too). **Config bug** (clDice at full weight from epoch 1). |
| `supervised-atm-l20-cldice` (attempt 2) | clDice, **warm-up 15 + ramp 10**, 80ep | **0.539** val @0.99; **+54% TD** vs baseline; with LCC → the headline win above. |

**Mean Teacher = negative result.** Both single-domain MT runs converge to a
~0.50 val basin *below* the 0.609 supervised optimum (warm-start slides down into
it, from-scratch climbs up to it). Mechanism: confidence-masked MSE on an
over-predicting teacher reinforces its own false positives (confirmation bias);
the intensity-only perturbation is too weak to regularise. **Do not keep tuning MT.**

---

## Architecture

```
configs/
  data/        aeropath.yaml, atm22.yaml (dataset_name + batch_root + HU window)
  model/       baseline_unet.yaml, (ct_fm_segresnet)
  training/    baseline.yaml                 — supervised AeroPath
               supervised_atm.yaml           — supervised ATM (the SSL baseline)
               supervised_atm_cldice.yaml    — supervised ATM + clDice (warm-up 15, ramp 10, weight 1.0, iters 10, 80 epochs)
               supervised_atm_cldice_pw3.yaml — clDice + pos_weight=3, val threshold 0.5 (calibration ablation; ACTIVE next run)
               mean_teacher_atm.yaml          — MT single-domain (from scratch) [negative result]
               mean_teacher_atm_warmstart.yaml— MT single-domain (warm-started) [negative result]
               teacher_student.yaml           — pre-pivot cross-domain MT [superseded]

lung_airway_segmentation/
  io/          case_layout.py, atm22_layout.py, nifti.py (load_canonical_image)
  datasets/    monai_aeropath.py, monai_atm22.py (CT-only + build_monai_atm22_labelled_datasets),
               splits.py (create_train_val_test_split + create_semisupervised_split), patches.py
  losses/      segmentation.py  — CombinedSegmentationLoss = BCE(pos_weight) + Dice + (optional) clDice
               topology.py      — soft_skeleton / soft_cldice_loss / SoftClDiceLoss  [IMPLEMENTED]
               semi_supervised.py — ConsistencyLoss (MT; negative result)
  models/      baseline_unet.py, ct_fm_segresnet.py
  training/    config.py (YAML/CLI/validation), builders.py (resolve_case_splits, dataset-agnostic),
               loops.py, teacher_student.py, engine.py (run_supervised_training w/ clDice warm-up ramp,
               run_semisupervised_training w/ --init-checkpoint warm-start)
  inference/   sliding_window.py, postprocess.py (binarize_logits, keep_largest_connected_component)
  metrics/     segmentation.py (Dice/precision/recall), topology.py (clDice, TD, BD, branch parsing)

scripts/
  train_baseline.py / train_semisupervised.py   — training entrypoints
  predict_case.py                                — AeroPath inference (saves airway_pred*_full.nii.gz)
  predict_atm.py                                 — ATM inference, viewer-compatible (raw + LCC, canonical space)
  analyse_distal.py                              — distal-radius prob stratification + Dice/TD threshold sweep + LCC
  evaluate_predictions.py                        — Dice/precision/recall evaluation
mask_visualisation.py                            — marimo 3D/slice viewer (datasets + saved predictions)
train.pbs                                        — ONE reusable HPC job script (edit active block, → train.out)
```

---

## Tooling notes

- **`analyse_distal.py`** is the workhorse, with **test-set hygiene in the defaults**.
  It (1) **selects the operating threshold** — default `--select-by cldice`: max clDice
  over `--cldice-candidates` (default 0.4,0.5,0.6; warns + says to widen if the peak is
  at the candidate-range edge); `--select-by voxel-precision` is the fast no-skeletonise
  alternative (TD s.t. voxel-precision ≥ `--precision-floor`) — and (2) **reports Dice /
  TD / BD / clDice / TPrec (+LCC)** at that threshold, plus (3) distal radius
  stratification + the cheap threshold sweep (grid now includes 0.3/0.4 for de-saturated
  models). clDice/topology-precision are computed only at the candidates (or the chosen
  threshold); BD only on the test report. **Defaults to select+report on val** (develop
  here): `python -m scripts.analyse_distal --run-dir <run>` → `<run>/distal_analysis.json`.
  Use `--out distal_analysis__<tag>.json` so runs don't clobber. The sealed **test**
  table is opt-in: `--report-split test` (selects on val, reports on test, computes BD) —
  run once per frozen model, stack the `table_mean` rows. JSON keys: `operating_point`
  (+`cldice_candidates`), `cldice_candidate_scan`, `table_per_case`/`table_mean`,
  `selection_sweep`, `threshold_sweep`, `bins`, `dev_mode`. ~3–4 min/case in clDice mode
  → give ~2 h walltime.
- **`predict_atm.py`** writes the layout `mask_visualisation.py` expects
  (`predictions/<case>/prediction_metadata.json` + `airway_pred_full.nii.gz` +
  `airway_pred_lcc_full.nii.gz` + `airway_prob_full.nii.gz`), in **canonical
  (RAS+)** orientation via `load_canonical_image` so it overlays on the viewer's
  CT/GT. `--threshold 0.90 --connectivity 6`. (An earlier MONAI-space, non-metadata
  version did NOT appear in the viewer — that was the cause of the "can't select
  the clDice run" issue, now fixed.)
- **`train.pbs`** is a single reusable job script (user preference): edit the
  active command block, `qsub`, output overwrites `train.out`. CX3 sizing:
  `select=1:ncpus=8:mem=64gb:ngpus=1`, walltime 12h, gpu_type left empty.

---

## Key design decisions & lessons

- **Config-first**: all hyperparameters in YAML; each run saves `resolved_config.json`.
- **Threshold 0.99 for the supervised/baseline** because `pos_weight=10` saturates probabilities (the old "0.083 Dice" disasters were all wrong-threshold).
- **clDice needs a warm-up**: pure Dice+BCE for ~15 epochs, then ramp clDice in — the soft skeleton of an untrained model is noise. Implemented as an epoch ramp in `run_supervised_training` (`cldice_warmup_epochs`, `cldice_rampup_epochs`).
- **LCC (6-connectivity) is now standard post-processing** for airways (one connected tree). Most low-threshold false positives are disconnected blobs LCC removes.
- **TD/topology vs Dice**: report TD at a fixed operating threshold + a precision-side metric; never TD's max.
- **Dataset-agnostic training** via `resolve_case_splits(data_config, training_config)` → `{labelled_train, unlabelled_train, val, test}` (AeroPath = fractional 3-way w/ empty unlabelled; ATM = count-based 4-way). Leakage guard verified (unlabelled disjoint from val/test).
- **British English** throughout (`labelled`, `unlabelled`).
- **`pos_weight=10` + threshold 0.99 are coupled — and the coupling *is* the "funk".**
  `DiceLoss(sigmoid=True)` already handles class imbalance; stacking BCE
  `pos_weight=10` on top over-penalises false negatives and **saturates
  probabilities** toward 1 — which is *why* the threshold has to live at 0.99 and
  everything below is mush. clDice now supplies the distal-recall push that the high
  `pos_weight` was standing in for, so the principled config to test is **BCE pw 1–3
  + Dice + clDice**: it should de-saturate the output, move the natural threshold
  back toward ~0.5 (better-calibrated, less LCC-dependent, fewer border blobs).
  One-variable ablation, **but lower the val threshold at the same time** (next
  bullet) or the comparison is rigged against the better-calibrated model.
- **Checkpoint selection is biased toward proximal volume.** `best_model` is chosen
  by **val Dice @ 0.99** (`engine.py`), which rewards the thick-airway operating
  point — the opposite of the topology goal. Validate at ~0.5 (or select on
  Dice@0.5+LCC / a TD proxy) so the saved epoch is the best *tree*, not the best
  trachea. Also evaluate `last_model` — attempt 2 was still climbing at ep75/80.
- **Patch borders: gaussian blending is already on** (`sliding_window.py`,
  `mode="gaussian"`), so the seams are not naive constant-blend. Remaining causes:
  (a) **low inference overlap** — headline predictions must use 0.5–0.75, not the
  0.25 the analysis ran at; (b) **edge receptive-field** on 96³ patches with
  foreground-centred sampling (`foreground_probability=0.7`) — the model rarely sees
  a branch crossing a patch face with full context. The saturated-probability blobs
  that "only LCC-6 removes" are the visible symptom; de-saturating (lower
  `pos_weight`) + higher overlap + `foreground_probability≈0.5` all reduce them.
  Optional: 128³ patches for more context if GPU memory allows.
- **Test-set hygiene — develop on val, seal test.** With only 20 labelled cases the
  val set carries every decision (operating point, pos_weight, clDice weight, patch
  borders, SSL); the 20 test cases are touched **once**, at the end, on a frozen,
  pre-decided model set (baseline, +clDice, final SSL), all reported together.
  Repeated test peeking = implicit model selection = optimistic final numbers.
  `analyse_distal` defaults to reporting on val; `--report-split test` is the
  deliberate, rare final act (and the only mode that computes BD).

---

## What needs to be done next (in priority order)

**Tier 1 — lock the core result + de-funk the operating point (do first):**
1. **Full 20-case test table** (Dice + TD + BD), supervised-ATM vs +clDice, at a
   *fixed, defensible* operating point, overlap 0.5–0.75 + LCC. The dissertation's
   core table — but it is the **sealed final eval**: develop/compare configs on val
   first, freeze the models, then run `analyse_distal.py --report-split test` once for
   `supervised-atm-l20` and `supervised-atm-l20-cldice` and stack the two `table_mean`
   rows. (The default invocation reports on val, for development.)
2. **Report the TD–precision operating curve and choose the op-point for topology,
   not Dice.** Pick the threshold by a precision-side constraint (e.g. max TD s.t.
   BD/topology-precision ≥ bound), which lands well below 0.9 (~0.5–0.7 + LCC,
   TD ≈ 0.60). Free ~2× headline TD (see Diagnosis). Frame explicitly: "we operate
   where the tree is recovered; Dice is reported, not optimised."
3. **`pos_weight` ablation** on the clDice model (pw ∈ {1, 3, 10}). **pw3 wired** as
   the active train.pbs run (`supervised_atm_cldice_pw3.yaml`, val threshold 0.5);
   pw1 next. Goal: a calibrated model that holds topology precision at low thresholds
   (the pw10 model dropped to 0.64 @0.5) → higher clDice at the ~0.5–0.6 operating
   point. NB checkpoint selection is still by val Dice@0.5 (volume-biased); making it
   topology-aware (clDice/TD) is a separate engine change — offered, not yet done.

**Tier 2 — push the topology contribution:**
4. **clDice ablation:** `cldice_weight` ∈ {1, 1.5, 2}, +20 epochs (attempt 2 was
   still inching up at ep75/80). Pick best by a *topology* val metric, not Dice@0.99.
   Combine with (3) as a small grid, not a full factorial, to save GPU runs.
5. **Patch borders:** inference overlap 0.5→0.75 for headline numbers; try
   `foreground_probability≈0.5`; (optional) 128³ patches for more per-patch context.

**Tier 3 — the SSL chapter (replaces Mean Teacher):**
6. **Topology-filtered self-training.** clDice model pseudo-labels the 90 unlabelled
   ATM cases; keep only topology-clean labels (**LCC + high confidence**, optionally
   require TTA-flip agreement), add them (labelled cases up-weighted), retrain,
   evaluate on val/test. **Why it should work where MT didn't:** hard, topology-
   filtered labels instead of soft confidence-masked consistency on an over-
   predicting teacher — so it cannot reinforce the teacher's own false positives
   (the MT confirmation-bias mechanism). One round first; only iterate if round 1
   helps. Named fallback if it stalls: **Cross-Pseudo Supervision (CPS)** — two nets
   pseudo-label each other, diversity blunts confirmation bias (~2× compute).
7. **Label-efficiency curve** (`labelled_count` 10/20/40) — cheap, supports the SSL story.

**Tier 4 — generalisation + write-up:**
8. **AeroPath-OOD** column: ATM-trained clDice model over all 27 AeroPath cases
   (HU windows already match) — the "generalise to pathology?" result.
9. **(Optional) ATM'22 batch 2** to grow the pools.
10. **Write-up in parallel** — start the diagnosis/methods chapters now; the
    operating-point and calibration story is already evidenced and writes itself.

**Ruled out / parked:**
- **Full unsupervised** — too high-risk for the timeline; no working precedent for airways.
- **More Mean Teacher tuning** — two runs show it degrades (confirmation bias). Keep as the negative baseline.
- **Two-stage distal refiner** (another paper's idea) — derivative *and* redundant with clDice (both target distal). Future-work mention only.
- **Diffusion as segmenter** — still supervised, weaker for thin airways. (Diffusion/MAE *pretraining* on unlabelled CTs is the only unsupervised angle worth a stretch, after the core lands.)
- **Switching to CT-FM backbone** — it's the reference ceiling, not the method.
