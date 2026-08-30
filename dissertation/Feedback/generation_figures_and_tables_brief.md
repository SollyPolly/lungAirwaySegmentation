# Project Manager Brief: Figures and Tables for Generation-Depth Analysis

## Purpose

This note defines the recommended figures and tables for presenting the airway generation-depth analysis in the dissertation.

The aim is to communicate three things clearly:

1. What branch/generation depth means.
2. Where the semi-supervised models improve airway-tree recovery.
3. Why generation depth is related to, but not equivalent to, airway calibre.

Avoid adding many redundant plots. The current figure set is already close to sufficient.

## 1. Main figure: reference skeleton coloured by branch depth

Use **ATM_016** as the main example in the dissertation.

ATM_016 is preferable because:
- it contains 288 branches;
- branch depth extends from 0 to 13;
- the progression from proximal to peripheral tree is visually clear;
- the coronal, sagittal and axial views show the branch-depth assignment well.

The ATM_027 figure is useful as a validation/sanity example but probably does not need to appear in the main text.

Recommended use:
- **Main dissertation:** ATM_016
- **Appendix:** ATM_027 and optionally one additional case

Suggested caption:

> Reference airway skeleton for ATM_016 coloured by branch depth. Depth was obtained by breadth-first traversal from the root branch, incrementing at bifurcations while absorbing single-child chains. Branch depth therefore provides a graph-based measure of position within the airway tree rather than a manually annotated anatomical generation.

Prefer the terms `branch depth`, `bifurcation depth`, or `generation depth`. Avoid implying manually defined anatomical generations unless that is truly supported.

## 2. Main result figure: airway recovery by branch depth

Keep the existing two-panel generation-depth recovery figure.

### Panel a

Absolute tree detection / tree-length detection versus branch depth.

Include:
- matched zero-consistency control;
- probability-target / soft-clDice Mean Teacher;
- thresholded-target Mean Teacher;
- thresholded 5-fold Mean Teacher;
- 110-label supervised reference.

Purpose:
- show proximal performance is already close to saturation;
- show performance decreases deeper in the tree;
- show SSL models retain more of the peripheral tree than the matched control;
- show how SSL compares with substantially more supervision.

### Panel b

Paired change in tree detection relative to the matched zero-consistency control:

\[
\Delta TD_g = TD_g(\mathrm{model}) - TD_g(\mathrm{control})
\]

This is arguably the most important panel.

It directly shows:
- gain is approximately zero at shallow depth;
- gain becomes positive around branch depth 4;
- improvement grows towards the peripheral tree.

### Important plotting change

If the final point combines depths 8-13, label it **8+**, not 8.

## 3. Dataset-context figure: branch-depth census and calibre relationship

Keep the current census/calibre figure.

### Panel a: branch-depth distribution

Show:
- share of total reference tree length at each branch depth;
- branch count per case;
- which depths are present in all cases and where the cohort becomes incomplete.

This explains why uncertainty increases at very late depths and why pooling as `8+` is reasonable.

### Panel b: calibre versus branch depth

Keep the heatmap showing operational thickness classes across branch depth.

Include the overall Spearman association:

\[
\rho \approx -0.48
\]

This shows later-generation airways tend to be thinner but calibre and depth are not equivalent.

Useful interpretation:
- about **94.3%** of `1-2` calibre centreline voxels occur at depth >= 5;
- about **81.0%** of centreline voxels at depth >= 5 are not in the `1-2` calibre class.

Therefore:

```text
thin airway -> usually peripheral

but

peripheral airway -> not necessarily ultra-thin
```

## 4. Optional supporting figure: fixed-calibre depth effect

This is the only additional figure currently worth considering.

Recommended two-panel figure:

### Panel a
Soft-clDice Mean Teacher delta TD versus branch depth for the `1-2` calibre group.

Current trend:

\[
\rho \approx 0.78
\]

### Panel b
Soft-clDice Mean Teacher delta TD versus branch depth for the `3-4` calibre group.

Current trend:

\[
\rho \approx 0.98
\]

This supports the interpretation that the peripheral effect is not explained solely by smaller airway calibre.

Recommended placement:
- main text if space allows;
- otherwise appendix.

Do not create equivalent plots for every calibre group unless there is a clear reason.

## 5. Figures that are probably unnecessary

Avoid separate main-text figures for:
- voxel recall versus generation;
- branch detection versus generation;
- each MT configuration separately;
- multiple example skeleton cases;
- direct MT-versus-110 plots;
- every calibre-by-depth combination.

Tree detection should remain the principal visual metric. Branch detection and voxel recall can support it numerically.

## 6. Main table: overall model comparison

Suggested structure:

| Model | Labelled training data | Uses unlabelled consistency | Overall TD | Overall voxel recall | Overall BD |
|---|---:|---|---:|---:|---:|
| 20-label supervised seed | [fill] | No | [fill] | [fill] | [fill] |
| Zero-consistency control | [same labelled set] | No | 0.9202 | [fill] | [fill] |
| MT probability / soft-clDice | [same labelled set] | Yes | **0.9398** | [fill] | [fill] |
| MT thresholded target | [same labelled set] | Yes | 0.9376 | [fill] | [fill] |
| MT thresholded, 5-fold | [same labelled set] | Yes | 0.9391 | [fill] | [fill] |
| 110-label supervised reference | 110 | No | 0.9333 | [fill] | [fill] |

Use the actual number of labelled cases for the seed/control/MT runs.

Do not call the 110-label model `ceiling110` in the dissertation. Prefer **110-label supervised reference**.

## 7. Main table: generation-specific paired changes

Suggested structure:

| Branch depth | Soft MT ΔTD | Hard MT ΔTD | Hard 5-fold ΔTD | 110-label ΔTD |
|---|---:|---:|---:|---:|
| 0 | -0.0016 | [fill] | [fill] | -0.0206 |
| 1-2 | -0.0002 | [fill] | [fill] | -0.0006 |
| 3 | -0.0004 | [fill] | [fill] | -0.0004 |
| 4 | +0.0053 | +0.0072 | +0.0050 | +0.0042 |
| 5 | +0.0139 | +0.0180 | +0.0180 | +0.0116 |
| 6 | +0.0182 | +0.0143 | +0.0193 | +0.0075 |
| 7 | +0.0251 | +0.0200 | +0.0220 | +0.0144 |
| 8+ | +0.0283 | +0.0405 | +0.0445 | +0.0389 |

Optionally report `mean ΔTD (wins / contributing cases)`, e.g. `+0.0283 (18/20)`.

Full median/IQR/Wilcoxon statistics can go in the appendix.

## 8. Small descriptive table: relationship between branch depth and calibre

| Quantity | Result |
|---|---:|
| Spearman correlation: branch depth vs calibre | -0.482 |
| `1-2` calibre centreline voxels occurring at depth >= 5 | 94.3% |
| Depth >= 5 centreline voxels not in `1-2` calibre | 81.0% |
| Fixed `1-2` calibre: depth vs soft-MT ΔTD | rho = 0.783 |
| Fixed `3-4` calibre: depth vs soft-MT ΔTD | rho = 0.976 |

This compactly supports the statement that calibre and branch depth are related but not interchangeable.

## 9. Recommended Results ordering

1. **Figure 1:** ATM_016 reference skeleton coloured by branch depth.
2. **Figure 2:** TD by branch depth and paired ΔTD versus control.
3. **Table 1:** overall model comparison.
4. **Figure 3:** branch-depth census and calibre-depth heatmap.
5. **Table 2:** generation-specific paired ΔTD.
6. **Optional Figure 4 / appendix:** fixed-calibre ΔTD versus branch depth.

## 10. Current recommendation

Recommended main-text figures:
1. ATM_016 branch-depth reference skeleton.
2. Generation-depth recovery figure.
3. Generation census + calibre heatmap.

Recommended optional supporting figure:
4. Fixed-calibre ΔTD versus branch depth.

ATM_027 should probably move to the appendix.

Do not continue adding exploratory figures unless they answer a genuinely new scientific question.
