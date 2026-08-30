# Discussion: Mean Teacher vs matched control vs 110-label supervised reference

## Purpose

This note summarises the interpretation of the generation-depth analysis when the Mean Teacher models are compared with both:

1. the **matched zero-consistency control**, and
2. the **more-supervised 110-label reference model**.

These comparisons answer different questions and should not be conflated.

---

## 1. What the matched control actually tests

The control is the Mean Teacher training setup with the consistency weight set to zero:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{sup}}
+
0 \cdot \mathcal{L}_{\mathrm{cons}}.
\]

Therefore, the unlabelled images contribute no gradient.

Conceptually:

```text
CONTROL

labelled batch   -> supervised loss       -> gradient
unlabelled batch -> consistency loss x 0  -> no gradient
```

By contrast:

```text
MEAN TEACHER

labelled batch   -> supervised loss     -> gradient
unlabelled batch -> consistency loss    -> gradient
```

This makes the comparison

\[
\text{Mean Teacher} - \text{control}
\]

the cleanest test of whether the **unlabelled consistency objective contributes useful learning beyond supervised continuation under the same training setup**.

The control should therefore remain the primary comparator for establishing the effect of semi-supervised learning.

---

## 2. What the 110-label model tests

The 110-label model answers a different question.

Rather than isolating the consistency mechanism, it asks:

> How does semi-supervised training with fewer labelled examples compare with a model trained using substantially more labelled data?

This is therefore better interpreted as a **more-supervised reference** or **label-efficiency reference**, rather than a matched control.

The name `ceiling110` may be misleading in the dissertation because the Mean Teacher models can exceed it on some metrics. A more neutral description would be:

> **110-label supervised reference**

---

## 3. Overall tree detection

The generation-analysis pipeline reproduces the same overall TD values as the calibre-analysis pipeline:

| Model | Overall TD |
|---|---:|
| Matched zero-consistency control | 0.9202 |
| 110-label supervised reference | 0.9333 |
| MT hard consistency, fold 0 | 0.9376 |
| MT hard consistency, 5-fold | 0.9391 |
| MT soft-clDice | **0.9398** |

This gives two distinct findings.

### Semi-supervised effect

For soft-clDice Mean Teacher:

\[
0.9398 - 0.9202 = +0.0196
\]

relative to the matched control.

This supports the conclusion that the consistency-based use of unlabelled data improves airway-tree detection beyond supervised continuation alone.

### Label-efficiency comparison

Soft-clDice Mean Teacher also exceeds the 110-label supervised reference:

\[
0.9398 - 0.9333 = +0.0065.
\]

All three Mean Teacher variants exceed the 110-label reference in overall TD.

This does **not** mean that semi-supervised learning is universally superior to using 110 labels, because other metrics and generation-specific behaviour are more nuanced.

It does show that the improvement in tree detection does not require the additional labelled data used by the 110-label model.

---

## 4. Generation-depth comparison

The most useful comparison is the paired change in TD relative to the matched control.

For soft-clDice Mean Teacher:

| Generation depth | Soft MT delta TD vs control |
|---|---:|
| 0 | -0.0016 |
| 1-2 | -0.0002 |
| 3 | -0.0004 |
| 4 | +0.0053 |
| 5 | +0.0139 |
| 6 | +0.0182 |
| 7 | +0.0251 |
| 8+ | +0.0283 |

The central pattern is:

\[
\Delta TD \approx 0
\]

for the proximal tree, followed by progressively larger gains with bifurcation depth.

The SSL benefit is therefore not distributed uniformly throughout the airway tree.

It is concentrated in the more peripheral generations.

---

## 5. Comparison with the 110-label supervised reference by depth

Because both the Mean Teacher models and the 110-label reference are reported relative to the same matched control, their generation-specific gains can be compared directly.

Approximate values for soft-clDice Mean Teacher are:

| Depth | Soft MT delta TD | 110-label delta TD | Soft MT - 110-label |
|---|---:|---:|---:|
| 0 | -0.0016 | -0.0206 | +0.0190 |
| 1-2 | -0.0002 | -0.0006 | approximately 0 |
| 3 | -0.0004 | -0.0004 | approximately 0 |
| 4 | +0.0053 | +0.0042 | +0.0011 |
| 5 | +0.0139 | +0.0116 | +0.0023 |
| 6 | +0.0182 | +0.0075 | **+0.0108** |
| 7 | +0.0251 | +0.0144 | **+0.0107** |
| 8+ | +0.0283 | +0.0389 | **-0.0105** |

### Interpretation

At generations 0-3, neither additional supervision nor Mean Teacher training materially changes TD because performance is already close to saturated.

At generations 4-7, soft-clDice Mean Teacher provides a larger improvement than the 110-label supervised reference.

The difference is especially clear around generations 6-7, where the Mean Teacher gain is roughly one percentage point greater than the 110-label gain.

At the pooled generation 8+ group, however, the 110-label model gains more TD than soft-clDice Mean Teacher.

This suggests that the additional labelled supervision may still provide an advantage in the most extreme periphery for this particular soft-clDice configuration.

---

## 6. Important qualification: the deepest-generation result is not a general failure of Mean Teacher

The comparison at generation 8+ is:

| Model | delta TD vs control, G8+ |
|---|---:|
| MT soft-clDice | +0.0283 |
| MT hard consistency, fold 0 | +0.0405 |
| MT hard consistency, 5-fold | **+0.0445** |
| 110-label supervised reference | +0.0389 |

Therefore, it would be inaccurate to conclude:

> The 110-label model is better than Mean Teacher in the deepest airways.

That is only true relative to the soft-clDice configuration.

Both hard-consistency Mean Teacher variants are comparable with or exceed the 110-label supervised reference in the G8+ group.

The stronger conclusion is therefore:

> Mean Teacher training as a whole remains highly competitive with additional labelled supervision in the peripheral tree, but the exact generation-specific behaviour depends on the consistency formulation.

---

## 7. Relation to the calibre analysis

The generation-depth analysis strengthens the earlier calibre-based result.

The calibre analysis showed that Mean Teacher particularly improves the smallest airway classes.

The generation-depth analysis now shows that the gain also increases with **topological depth**.

These are related but not identical properties.

The generation analysis found a moderate negative association between generation depth and calibre:

\[
\rho \approx -0.48.
\]

Later airways tend to be thinner, as expected.

However:

- approximately **94.3% of the 1-2 calibre centreline voxels occur at generation >= 5**, while
- approximately **81.0% of generation >= 5 centreline voxels are not in the 1-2 calibre group**.

Therefore:

```text
thin airway -> usually peripheral

but

peripheral airway -> not necessarily ultra-thin
```

The generation analysis is therefore not simply reproducing the calibre result under a different label.

---

## 8. The peripheral effect persists within fixed calibre

A particularly useful supporting result is that the Mean Teacher advantage tends to increase with depth even when calibre is held fixed.

For soft-clDice Mean Teacher:

### 1-2 calibre group

The increase in delta TD with generation gives approximately:

\[
\rho = 0.783,
\qquad
p = 0.0125.
\]

### 3-4 calibre group

The trend is even stronger:

\[
\rho = 0.976,
\qquad
p \approx 1.5 \times 10^{-6}.
\]

This is descriptively strong evidence that the Mean Teacher advantage is associated with **peripheral location itself**, rather than being explained solely by airway thickness.

This should not be presented as a causal multivariable result, but it supports the interpretation that calibre and generation contribute complementary information.

---

## 9. TD and voxel recall continue to tell different stories

The generation analysis again shows that improved tree recovery does not necessarily correspond to improved volumetric recall.

For soft-clDice Mean Teacher, voxel recall relative to control is lower at several later generations despite higher TD and branch detection.

For example:

- generation 5: voxel recall decreases while TD increases;
- generation 8+: TD increases substantially while voxel recall is lower on average.

This reinforces the interpretation that the Mean Teacher objective is not simply predicting more foreground.

Instead, it appears to redistribute performance toward more complete airway-tree recovery.

A safe wording is:

> Improved topological completeness was not accompanied by increased global or generation-wise voxel recall, indicating that the additional tree recovery was not simply the result of uniformly increased foreground prediction.

Avoid stating that the predictions are "thinner" unless this is measured directly.

---

## 10. Recommended dissertation interpretation

The cleanest experimental hierarchy is:

### A. Matched zero-consistency control

Question:

> Does the unlabelled consistency objective itself help?

Result:

\[
TD = 0.9202 \rightarrow 0.9398
\]

for soft-clDice Mean Teacher.

This isolates the effect of semi-supervised consistency training.

### B. Mean Teacher variants

Question:

> Where in the tree does SSL help?

Result:

The gain is negligible proximally and increases with generation depth.

This supports a **peripheral airway recovery** claim.

### C. 110-label supervised reference

Question:

> How does SSL compare with substantially more labelled supervision?

Result:

The Mean Teacher models match or exceed the 110-label model in overall TD, and remain highly competitive across peripheral generations.

This supports a **label-efficiency** interpretation.

---

## 11. Suggested discussion paragraph

> Compared with the matched zero-consistency control, all Mean Teacher configurations substantially improved airway-tree detection, demonstrating that the gain could not be explained by supervised continuation alone. Notably, overall TD also exceeded that of a reference model trained with 110 labelled scans. Generation-specific analysis showed that these gains were concentrated beyond the proximal tree: Mean Teacher and increased supervision produced little change at generations 0-3, while both improved later generations. Soft-clDice Mean Teacher exceeded the 110-label reference through much of generations 4-7, although the more-supervised model produced a larger gain in the pooled generation 8+ group. This behaviour was not universal across Mean Teacher configurations, as the hard-consistency variants remained comparable with or exceeded the 110-label reference at the deepest generations. These findings suggest that consistency-based learning from unlabelled images can recover a substantial proportion of the benefit normally associated with additional annotations, particularly for peripheral airway-tree completeness.

---

## 12. Strongest conclusion currently supported

The strongest defensible claim is not:

> Semi-supervised learning is better than using more labelled data.

Instead:

> **The unlabelled consistency objective produces a measurable improvement over an otherwise matched supervised control, with the largest gains occurring in the peripheral airway tree. Despite using fewer labelled data, the Mean Teacher models achieve overall tree detection comparable with or greater than the 110-label supervised reference, indicating that consistency-based learning can recover substantial peripheral-tree information from unlabelled scans.**

This wording keeps the mechanistic and label-efficiency conclusions separate and avoids overstating the 110-label comparison.

---

## 13. Terminology recommendation

For the dissertation, use:

- **matched zero-consistency control**
- **Mean Teacher / semi-supervised models**
- **110-label supervised reference**

Avoid calling the 110-label model a "ceiling", because the current results show that it is not an empirical upper bound on tree-detection performance.
