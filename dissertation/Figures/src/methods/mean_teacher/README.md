# Mean Teacher TikZ figure source

These sources generate two independent, title-free dissertation panels:

- `mean_teacher_flow.tex`: corrected one-labelled/one-unlabelled training and
  gradient/EMA routing.
- `nnunet_backbone.tex`: the exact Dataset126 six-stage 3-D PlainConvUNet plan.

All report-level titles, panel letters, captions and methodological caveats belong
to the LaTeX `figure` or `subfigure` environment. Text inside these sources is
limited to semantic labels needed to interpret the computation.

The CT volumes and mask, probability and skeleton tiles are deliberately replaceable
placeholders. Their named anchors allow actual same-FOV image tiles to be inserted later
without changing the flow.
The probability-target expression in `mean_teacher_flow.tex` is a single macro and
can be switched to the historical thresholded target after the final experiment is
selected.

The drawings use native PGF/TikZ only and contain no copied third-party artwork.
`mean_teacher_styles.tex` defines their shared perspective feature-volume primitive.
The detailed panel enumerates the exact 11-stage encoder--bottleneck--decoder silhouette,
channels, spatial sizes, anisotropic strides and literal right-angle
skip-to-concatenation routes. The Mean Teacher panel uses five representative volumes as
an explicitly symbolic summary of that same backbone. Consequently, the panels act as two
zoom levels of one system while retaining different levels of methodological detail.

## Building

Both files are `standalone` documents and are *not* `\input` into `main.tex`; the chapter
includes the compiled PDF under `Figures/pdf/methods/mean_teacher/`. Editing a source
therefore has no effect on `main.pdf` until both of these have run, in order:

```powershell
.\dissertation\scripts\build_mean_teacher_tikz_figures.ps1   # compile -> copy into Figures/pdf/ + png/
.\dissertation\scripts\build_dissertation.ps1                # rebuild main.pdf against the new PDFs
```

The `\input{mean_teacher_styles.tex}` on line 6 of each source is deliberately a bare
same-directory path. That is the only form that resolves both for the build script (which
compiles from this directory) and for an editor build such as latex-workshop, which sets the
working directory to the file's own folder. A root-relative path silently breaks the latter.

## Horizontal budget

Both panels are set at `width=\linewidth`, so the rendered type size is fixed by the figure's
*natural width* alone; adding vertical space never makes the labels bigger. Any change that
widens a panel shrinks its text proportionally. In particular, perspective depth costs
horizontal space without carrying information, so it is kept moderate and traded into
front-face width, which is where the cuboid reading comes from.

## Provenance

`nnunet_backbone.tex` is drawn with PlotNeuralNet's own primitives, vendored into
`plotneuralnet_layers.tex`:

- source: https://github.com/HarisIqbal88/PlotNeuralNet (`layers/Box.sty`,
  `layers/RightBandedBox.sty`, `layers/init.tex`)
- licence: MIT, Copyright (c) 2018 HarisIqbal88 — attribution is its only condition
- the panel's structure follows that project's `examples/Unet/Unet.tex`: a stage is a
  `RightBandedBox` whose banded sub-boxes are its individual convolutions, a narrow
  `Box` slab marks each resolution change, and a translucent grey `RightBandedBox` is
  the concatenated encoder tensor.

Only three mechanical changes were made to the vendored code so it can be `\input`
rather than `\usepackage`d; no geometry, colour, anchor or label logic was altered.
The header of `plotneuralnet_layers.tex` records them.

Two things are supplied locally rather than through the pic options. Spatial tensor
sizes are free nodes, because the vendored depth label uses `text width = 14*depth`
and a long size string wraps into an unreadable column on the small deep blocks. The
anisotropic stride notes are free nodes for the same reason — the pic caption uses
`text width = 15*width/scale`, which is far too narrow on a width-1 slab.

`mean_teacher_flow.tex` remains native PGF/TikZ and shares `mean_teacher_styles.tex`;
it vendors nothing.
