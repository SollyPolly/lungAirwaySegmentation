# Mean Teacher Warm-Start / Checkpoint Seeding Review

## Purpose

Review the current Mean Teacher (MT) semi-supervised learning setup for the airway segmentation dissertation and determine whether the supervised checkpoint initialization is **too mature** before MT begins.

The main concern is:

> If the supervised seed has already converged (or nearly converged), the student may change very little after MT initialization. Since the teacher is initialized from the student and then updated as an EMA of it, teacher–student disagreement may remain extremely small. In that case, the consistency loss could contribute little useful gradient and the unlabelled data may have minimal effect.

Please use the full project memory, trainer history, existing diagnostics, checkpoint conventions, and available runs when assessing this.

---

## Current Understanding

The current training scheme is approximately:

1. Train a supervised nnU-Net model.
2. Initialize the Mean Teacher student from a supervised checkpoint.
3. Initialize the teacher from the student.
4. Begin Mean Teacher training.
5. Ramp the consistency loss to full weight over approximately 25 MT epochs.

The supervised seed has apparently been taken from something like:

- ~500 epochs, or
- ~1000 epochs.

The exact checkpoint(s), optimizer handling, learning-rate handling, and current production trainer should be verified from the repository/project memory rather than assumed.

---

## Core Question

Determine whether the current supervised seed is **too strongly trained** for Mean Teacher to provide a useful semi-supervised learning signal.

The issue is not simply that the student and teacher are identical at MT epoch 0. That is expected.

The important question is whether, once MT begins, there is enough useful divergence between teacher and student predictions on unlabelled data for the consistency objective to matter.

Conceptually:

\[
\theta_T^{(0)} = \theta_S^{(0)}
\]

and subsequently

\[
\theta_T^{(t)}
=
\alpha \theta_T^{(t-1)}
+
(1-\alpha)\theta_S^{(t)}.
\]

If the seeded student barely moves,

\[
\theta_S^{(t)} \approx \theta_S^{(t-1)},
\]

then

\[
\theta_T^{(t)} \approx \theta_S^{(t)}
\]

and potentially

\[
\mathcal{L}_{cons} \approx 0.
\]

This would mean that the nominal MT phase is functionally close to continued supervised fine-tuning rather than genuinely benefiting from unlabelled data.

---

# Review Tasks

## 1. Verify Exactly How Checkpoint Initialization Works

Inspect the current MT trainer and determine precisely what is restored from the supervised run.

Answer explicitly:

- Are **network weights only** loaded?
- Is the optimizer state restored?
- Is the learning-rate scheduler state restored?
- Is the epoch number restored?
- Is any nnU-Net polynomial LR schedule resumed from the supervised checkpoint?
- Is the MT phase given a fresh optimizer?
- Is the MT phase given a fresh LR schedule?
- Is the teacher initialized directly from the loaded student?
- Is the teacher kept in eval mode / gradients disabled as intended?
- Is EMA state initialized cleanly?

This distinction is critical.

### Desired distinction

#### Case A — weight initialization

```text
supervised checkpoint weights
        ↓
student weights
        ↓
fresh MT optimizer + fresh MT LR schedule
        ↓
teacher copied from student
```

This is a warm start.

#### Case B — full training resume

```text
supervised checkpoint
(weights + optimizer + scheduler + training state)
        ↓
continued training
        ↓
MT added late
```

This may strongly suppress student movement if the supervised LR has already decayed.

Determine which case the current implementation actually follows.

---

## 2. Check Whether the Supervised Seed Is Already Near Convergence

Use existing training logs/checkpoints where possible.

Compare candidate seed epochs, especially if available:

- 250
- 500
- 750
- 1000 / final

Do not require all of these if they do not already exist.

For each available checkpoint, inspect:

- validation Dice
- clDice / topology-aware metrics if available
- training loss
- validation loss
- airway-specific metrics
- peripheral airway performance if available
- whether performance has plateaued
- learning rate at that point
- evidence of overfitting or saturation

The relevant question is not merely which seed has the highest supervised Dice.

The question is:

> At what point is the model competent enough to provide meaningful teacher predictions, while still leaving room for unlabelled-data-driven MT learning?

---

## 3. Inspect Existing Mean Teacher Diagnostics

Search the existing MT diagnostics trainers/runs.

Relevant quantities include, if available:

### Teacher–student prediction disagreement

For unlabelled samples:

\[
D_p = \mathbb{E}\left[|p_T-p_S|\right]
\]

or another existing prediction-space discrepancy.

Inspect how it evolves across early MT training:

- epoch 0
- ~5
- ~10
- ~25
- later epochs

### Consistency loss magnitude

Inspect:

\[
\mathcal{L}_{cons}
\]

throughout training.

In particular:

- Is it effectively zero from the beginning?
- Does it spike and then decay?
- Does it remain meaningful after the 25-epoch ramp?
- Does it differ between central and peripheral airway regions if diagnostics exist?

### Relative loss contribution

Where possible evaluate:

\[
R =
\frac{\lambda \mathcal{L}_{cons}}
{\mathcal{L}_{sup}}
\]

or an equivalent gradient-scale comparison.

A consistency loss can look numerically nonzero while still being insignificant relative to the supervised term.

### Parameter disagreement

If cheap to obtain:

\[
D_\theta =
\frac{\|\theta_S-\theta_T\|_2}
{\|\theta_S\|_2}.
\]

This is secondary to prediction-space disagreement but may help diagnose whether the EMA teacher is effectively identical to the student.

---

## 4. Verify That MT Has a Genuine Source of Prediction Disagreement

Check exactly what inputs the teacher and student receive.

Determine:

- Do teacher and student receive different augmentations?
- Which augmentations are applied to each?
- Is the teacher given a weak/base view?
- Is the student given a stronger perturbation?
- Are spatial transforms synchronized where required?
- Are intensity transforms different?
- Is prediction alignment handled correctly after geometric transforms?
- Is there stochasticity from dropout or other model components?
- Is teacher inference deterministic?
- Is the consistency loss computed on the correct transformed coordinate system?

A teacher and student can have nearly identical weights and still provide a useful consistency objective if their inputs are perturbed appropriately.

Conversely, identical weights + nearly identical views can trivially produce near-zero consistency loss.

---

## 5. Review the 25-Epoch Consistency Warm-Up

Current understanding:

```text
MT epoch 0
consistency weight ≈ 0
        ↓
~25 epochs
        ↓
full consistency weight
```

Assess whether this is still justified when starting from a strong supervised checkpoint.

The original rationale for a long consistency ramp is strongest when the model begins from random or weak initialization and early teacher predictions are unreliable.

With a competent supervised seed, the teacher predictions may already be meaningful at MT epoch 0.

Therefore review whether:

- 25 epochs is appropriate,
- a shorter ramp would be better,
- no ramp is defensible,
- or ramp duration should depend on seed maturity.

Do not change this purely from theory; connect the recommendation to the existing diagnostics and observed prediction quality.

---

# Main Hypotheses to Test

## H1 — Current seed is too mature

The 500/1000 epoch supervised checkpoint has mostly converged.

After MT initialization:

- student moves very little,
- teacher tracks it closely,
- prediction disagreement is tiny,
- consistency loss contributes little,
- unlabelled data have little influence.

### Prediction

A less mature checkpoint may produce better final SSL gains even if its initial supervised score is lower.

---

## H2 — Seed maturity is fine, but LR handling is suppressing MT

The checkpoint itself may be appropriate, but the MT phase may inherit or effectively use an LR that is too low.

### Prediction

Resetting the optimizer/LR schedule while keeping the same checkpoint materially increases:

- student–teacher divergence,
- consistency signal,
- and possibly final validation performance.

---

## H3 — Seed maturity and LR are fine; perturbations are too weak

Student and teacher may receive views that are too similar.

### Prediction

Consistency loss is near-zero even with a healthy LR and nontrivial student updates.

---

## H4 — Current setup is actually healthy

It is possible that:

- the student moves sufficiently,
- EMA produces useful temporal ensembling,
- augmented views create prediction disagreement,
- consistency contributes meaningfully,
- and the current seed is not a problem.

Do not assume the hypothesis of "over-seeding" is correct.

Try to falsify it.

---

# Requested Run Planning

After reviewing the implementation and existing evidence, propose the **smallest useful set of additional runs** that can answer the question.

Prioritize experiments that isolate one factor at a time.

Avoid a large grid search unless existing evidence genuinely justifies it.

A likely structure to consider is below, but modify it based on the repository and prior runs.

---

## Candidate Ablation Dimension A — Seed Maturity

Potential checkpoints:

```text
250 epoch seed
500 epoch seed
1000/final epoch seed
```

Prefer using checkpoints that already exist.

If only 500 and 1000 exist, start there.

If an earlier checkpoint is already saved, use it rather than retraining unnecessarily.

---

## Candidate Ablation Dimension B — Consistency Ramp

Potential values:

```text
0 epochs
5–10 epochs
25 epochs (current)
```

Do not necessarily run all combinations.

Use diagnostics to decide whether this dimension is worth testing.

---

## Candidate Ablation Dimension C — MT Optimizer/LR Initialization

Verify first whether this is already controlled correctly.

If not, compare:

```text
checkpoint weights + fresh optimizer/LR
vs
current behaviour
```

This may be a higher-priority experiment than changing seed length.

---

# Suggested Minimum Experimental Logic

Please design the final run plan based on evidence, but aim for something like the following philosophy:

### Baseline

Current production configuration.

For example:

```text
seed = current checkpoint
consistency warm-up = 25
current EMA
current LR
current augmentations
```

### Earlier-seed comparison

Change only the supervised checkpoint.

```text
earlier seed
same MT settings
```

### LR/reset comparison

Only if current checkpoint initialization is carrying problematic optimizer/scheduler state.

### Warm-up comparison

Only if diagnostics suggest consistency is being unnecessarily delayed.

---

# Metrics to Compare Between Runs

Do not rely only on final Dice.

For each run, compare:

## Segmentation quality

- Dice
- clDice
- sensitivity / recall if available
- precision if available

## Airway-tree quality

Prefer project-specific topology metrics already used.

Potentially:

- tree length detected
- branch detection rate
- centerline completeness
- leakage / false branches
- peripheral airway recovery
- generation depth

Use the metrics that already exist in the project rather than introducing unnecessary new infrastructure.

## SSL behaviour

Track:

- supervised loss
- raw consistency loss
- weighted consistency loss
- teacher/student prediction discrepancy
- teacher vs student validation performance
- EMA divergence if available
- learning rate

The goal is to distinguish:

```text
better final segmentation
```

from:

```text
actual evidence that SSL is contributing.
```

---

# Important Diagnostic Question

For each candidate run, answer:

> What changed because of the unlabelled data?

A Mean Teacher run beating a supervised checkpoint is not automatically evidence that SSL helped if the student simply continued supervised optimization for longer.

Where feasible, compare against a **supervised-only continuation control** initialized from the same checkpoint for the same number of optimization steps.

This may be one of the most informative controls.

Example:

```text
500 epoch supervised seed
        ├── supervised-only continuation
        └── Mean Teacher continuation
```

with matched:

- number of epochs/iterations,
- labelled batches,
- LR schedule,
- augmentation policy where appropriate.

Then compare the final models.

This isolates the effect of the unlabelled consistency signal from simply training longer.

---

# Compute-Efficient Decision Order

Please prioritize analysis/runs approximately as:

1. **Inspect code**
   - checkpoint loading
   - optimizer
   - LR
   - EMA
   - augmentations
   - consistency ramp

2. **Inspect existing diagnostics**
   - consistency magnitude
   - teacher/student disagreement
   - LR
   - validation trajectory

3. **Inspect existing supervised checkpoints**
   - convergence / plateau

4. **Use existing completed runs wherever possible**

5. **Only then schedule new ablations**

Do not recommend expensive runs if the question can already be answered from logs/checkpoints.

---

# Deliverable Requested From Project Manager

Please return:

## A. Current implementation verdict

State clearly:

- whether checkpoint initialization is weights-only or a resume,
- whether LR is reset,
- how teacher/student are initialized,
- how consistency warm-up works,
- how perturbations differ between teacher/student.

## B. Risk assessment

Classify the current concern as one of:

- low concern,
- plausible concern,
- strong evidence of over-seeding,
- implementation issue more important than seed maturity.

Explain why.

## C. Evidence from existing runs

Summarize any available evidence showing whether MT is actually producing:

- meaningful consistency loss,
- teacher/student disagreement,
- SSL improvement.

## D. Recommended run matrix

Give a **small, prioritized table** containing:

- run name,
- seed checkpoint,
- MT warm-up,
- LR/reset behaviour,
- other changed variable,
- purpose,
- priority.

Keep unnecessary combinations out.

## E. Supervised continuation control

State whether a matched supervised-only continuation run is needed.

If yes, specify exactly how it should be matched.

## F. Stop criteria / decision rule

Define what evidence would convince us that:

### Current seed is fine

versus

### We should move to an earlier supervised checkpoint.

For example, use final airway metrics together with consistency diagnostics rather than Dice alone.

---

# Overall Goal

We are not trying to find the best hyperparameter combination blindly.

We are trying to answer a specific methodological question:

> **Does the current supervised checkpoint initialization leave enough room for Mean Teacher to extract useful information from the unlabelled airway CT data, or are we beginning SSL so late that teacher and student remain effectively indistinguishable?**

Use the existing project history and artifacts to answer this as efficiently as possible, then recommend only the experiments needed to resolve remaining uncertainty.
