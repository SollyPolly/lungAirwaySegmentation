# Project Manager Brief: 20-Label Seed vs Zero-Consistency Control

## Purpose

Before finalising the Mean Teacher interpretation, evaluate the original **20-label supervised seed** against the **matched zero-consistency control**.

This is primarily a sanity check.

The control is not necessarily identical to the seed even though its consistency weight is zero. Understanding the difference will make the SSL interpretation cleaner.

## 1. Experimental hierarchy

```text
20-label supervised seed
          |
          +----------------------+
          |                      |
          v                      v
zero-consistency control      Mean Teacher
          |                      |
 supervised continuation       supervised loss
 no unlabelled gradient         +
                              unlabelled consistency
```

Important comparisons:

| Comparison | Question |
|---|---|
| Seed -> control | What does additional supervised continuation / the MT training phase itself change? |
| Control -> Mean Teacher | What does the unlabelled consistency objective add? |
| Seed -> Mean Teacher | What is the total change produced by SSL continuation? |
| Mean Teacher -> 110-label reference | How competitive is SSL with substantially more labelled supervision? |

## 2. Why seed and control are not necessarily identical

The zero-consistency control has:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{sup}}
+
0 \cdot \mathcal{L}_{\mathrm{cons}}
\]

so unlabelled samples contribute no gradient.

However, the control can still differ from the original seed because it undergoes further optimisation.

Possible reasons:
- additional supervised epochs;
- a fresh optimiser state after warm start;
- a new learning-rate schedule;
- training-stage stochasticity;
- continued augmentation exposure;
- EMA averaging if predictions are taken from the final teacher.

Therefore `w_cons = 0` means **no unlabelled consistency gradient**, not **no change from the seed**.

## 3. Why this comparison matters

The primary SSL comparison remains:

\[
\text{Mean Teacher} - \text{zero-consistency control}
\]

because this isolates the effect of consistency training.

The seed tells us how much the control itself moved during continued supervised optimisation.

Example A:

```text
Seed TD     = 0.919
Control TD  = 0.920
MT TD       = 0.940
```

Interpretation:

> Additional supervised continuation changes little, while the consistency objective produces the major improvement.

Example B:

```text
Seed TD     = 0.900
Control TD  = 0.920
MT TD       = 0.940
```

Interpretation:

> A substantial part of the total seed-to-MT gain comes from supervised continuation, with a further gain from consistency training.

The control-to-MT comparison remains valid in either case.

## 4. Required evaluation

Evaluate the original 20-label seed on the **same held-out 20-case validation cohort** used for:
- zero-consistency control;
- soft/probability-target MT;
- hard/thresholded MT;
- hard 5-fold MT;
- 110-label supervised reference.

Use the same post-processing and metric pipeline.

At minimum calculate:
- overall tree detection (TD);
- overall voxel recall;
- branch detection (BD), if available;
- standard segmentation metrics already used elsewhere.

Do not include training cases merely to increase sample size.

## 5. Generation-depth sanity analysis

Run the seed through the same branch-depth evaluation pipeline.

Primary question:

> Does the seed approximately overlap with the zero-consistency control across branch depth?

Recommended outputs:
- seed TD by branch depth;
- control TD by branch depth;
- paired seed-to-control difference;
- wins/losses across cases.

There is no need for a major new Results section unless the difference is substantial.

If seed and control are nearly identical, report this briefly as evidence that continued supervised training alone does not explain the MT improvement.

If they differ materially, investigate before finalising the interpretation.

## 6. Important EMA check

Determine whether current control predictions are from:
- final student, or
- final EMA teacher.

If the reported control is the EMA teacher, separately evaluate:

1. original 20-label seed;
2. final zero-consistency control student;
3. final zero-consistency control EMA teacher.

Conceptually:

```text
20-label seed
      |
      v
continued control student
      |
      v
continued control EMA teacher
```

This identifies whether seed-to-control differences arise from continued supervised optimisation, EMA averaging, or both.

Do not turn this into a major experiment unless the effect is meaningful.

## 7. Recommended overall comparison table

| Model | Labelled training data | Unlabelled gradient | Overall TD | Overall recall | BD |
|---|---:|---|---:|---:|---:|
| 20-label supervised seed | 20 | No | [fill] | [fill] | [fill] |
| Zero-consistency control | 20 | No | 0.9202 | [fill] | [fill] |
| MT probability / soft-clDice | 20 | Yes | 0.9398 | [fill] | [fill] |
| MT thresholded | 20 | Yes | 0.9376 | [fill] | [fill] |
| MT thresholded, 5-fold | clarify fold usage | Yes | 0.9391 | [fill] | [fill] |
| 110-label supervised reference | 110 | No | 0.9333 | [fill] | [fill] |

Clarify exact label counts and fold usage from the training configuration rather than relying on model names.

## 8. Interpretation if seed and control are effectively equivalent

If the difference is very small:

> The zero-consistency control remained close to the original 20-label supervised seed despite additional training. The substantially larger improvement observed after introducing teacher-student consistency therefore cannot be explained by supervised continuation alone.

This makes the control-to-MT comparison especially clean.

## 9. Interpretation if control improves substantially over seed

If the control improves meaningfully, do not treat this as a failed experiment.

Separate:

\[
\text{seed} \rightarrow \text{control}
\]

as continued supervised optimisation / training-stage effects,

and:

\[
\text{control} \rightarrow \text{Mean Teacher}
\]

as the additional effect of the unlabelled consistency objective.

The dissertation should then distinguish:
- total improvement from seed to MT;
- supervised-continuation contribution;
- consistency-specific contribution.

## 10. Relationship to the 110-label supervised reference

The full experimental hierarchy becomes:

```text
20-label seed
      |
      | supervised continuation
      v
20-label zero-consistency control
      |
      | add unlabelled consistency
      v
20-label + unlabelled Mean Teacher
      |
      | compare label efficiency
      v
110-label supervised reference
```

These answer different questions:

### Seed vs control
Did additional training alone help?

### Control vs Mean Teacher
Did unlabelled consistency help?

### Mean Teacher vs 110-label
Can SSL recover performance normally associated with substantially more labelled supervision?

Do not collapse these into a single baseline-versus-proposed-method comparison.

## 11. Scope recommendation

Keep this as a **sanity/reference analysis**, not a new major experimental branch.

Recommended workflow:
1. evaluate the 20-label seed;
2. compare overall metrics with the zero-consistency control;
3. compare seed vs control by branch depth;
4. evaluate control student vs EMA teacher if necessary;
5. document the result;
6. stop unless an unexpected discrepancy appears.

If seed and control are close, no further investigation is needed.

Attention can then return to experiments capable of changing the methodological conclusion, particularly the supersampled soft-skeleton / x2 Mean Teacher experiment.

## 12. Strongest desired conclusion

If supported by the measurements:

> **The matched zero-consistency control showed little change relative to the original 20-label supervised seed, indicating that additional supervised continuation alone did not account for the observed improvement. Introducing the consistency objective produced a substantially larger increase in tree detection, particularly in the peripheral airway generations.**

If the seed-control difference is not negligible, revise this wording to quantify the supervised-continuation contribution rather than forcing the desired conclusion.
