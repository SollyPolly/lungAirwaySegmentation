# Dissertation figure assets

LaTeX includes vector PDF figures from `pdf/`. Matching `png/` files are previews
for review and are not the primary publication assets. JSON sidecars under
`provenance/` record checkpoint, crop and operator details for data-derived figures.

```text
Figures/
|-- branding/                         title-page and institutional artwork
|-- src/                              editable, title-free figure sources
|   `-- methods/mean_teacher/         native TikZ MT flow and nnU-Net backbone
|-- pdf/
|   |-- background/                   conceptual figures used in Background
|   |-- methods/cldice/               clDice implementation figures
|   |-- methods/mean_teacher/         compiled vector architecture figures
|   |-- analysis/teacher_targets/     checkpoint-derived mechanism analyses
|   `-- appendix/                     complete or pedagogical master plates
|-- png/                              preview mirror of pdf/
`-- provenance/                       machine-readable figure metadata
```

Generation commands, run from the repository root. The scripts that build these
assets live in `dissertation/scripts/`, alongside the LaTeX they serve; anything
in the top-level `scripts/` belongs to the training and evaluation pipeline.

```powershell
.\.venv\Scripts\python.exe dissertation\scripts\generate_cldice_explainer.py
.\.venv\Scripts\python.exe dissertation\scripts\generate_cldice_real_airway_patch.py
.\.venv\Scripts\python.exe dissertation\scripts\generate_hu_imbalance_histogram.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File dissertation\scripts\build_mean_teacher_tikz_figures.ps1
```

Plot-based figures are exported as individual, title-free panels. Panel letters,
panel descriptions, report-level titles, captions and methodological disclaimers
belong exclusively to the LaTeX `subfigure`/`figure` environments. The only text
retained inside a method schematic is a semantic node or arrow label required to
understand the computation. Assets target the final A4 text width (171.8 mm) with
a minimum final label size near 8 pt.
