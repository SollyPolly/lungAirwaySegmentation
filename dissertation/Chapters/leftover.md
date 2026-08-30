ction.
  \item The largest-connected-component prior: a healthy airway annotation is intended to form
  one tree, so isolated predictions are commonly removed.
  \item Critique it properly. Largest-by-size can select an extra-thoracic false component
  (the CT table is air-density and survives thresholding), and any connected-component filter
  can delete genuine disconnected branches. Trachea seeding fixes the first failure and not
  the second.
  \item Therefore post-processing must be declared and evaluated as part of the method, not
  applied as neutral clean-up. Motivates the native-primary plus declared-sensitivity policy
  used throughout Chapter~\ref{ch:results}.
\end{itemize}

\section{Datasets, domain shift and evaluation}

\begin{itemize}
  \item ATM'22 as the in-domain development benchmark; AeroPath as an external pathology-rich
  test set~\cite{zhang2023atm22,stoverud2023aeropath}.
  \item AeroPath as evidence that training regime matters independently of architecture: its
  patch-wise and full-volume baselines differ modestly in Dice ($84.98\pm3.24$ versus
  $83.88\pm3.51$) but enormously in structure (detected tree length $91.80\pm3.50$ versus
  $48.87\pm11.21$; branch detection $84.67\pm7.11$ versus $34.94\pm7.59$). A compact,
  quotable demonstration that overlap and topology are not interchangeable.
  \item Sources of shift: scanner, reconstruction kernel, voxel spacing, pathology burden,
  annotation policy. Matching the intensity window removes one avoidable mismatch but does not
  make the datasets identically distributed.
  \item Why external evaluation is worth the words: an apparently strong in-domain result can
  depend on acquisition and annotation conventions.
  \item TABLE CANDIDATE (small, one third of a page): benchmark context row set --- EXACT'09,
  ATM'22, AeroPath --- with scans, setting and best reported Dice/TLD/BD. Purely contextual;
  caption must state that these are not head-to-head comparable with this study's protocol.
\end{itemize}

\section{Synthesis and research gap}

\begin{itemize}
  \item Overlap, connectivity, calibre and branch reach are non-equivalent axes, and reported
  SSL gains are routinely entangled with extra optimisation, augmentation and post-processing.
  \item Consistency-based SSL is validated mainly on dense targets; the sparse tubular regime
  has documented failure mechanisms and very few matched-control studies.
  \item Teacher-target construction is treated as an implementation detail in the segmentation
  literature despite converting Mean Teacher into a different algorithm.
  \item \textbf{Gap addressed here:} a faithful continuous-target, geometry-aware Mean Teacher
  on a competitive pipeline, measured against an exactly matched no-consistency control, with
  instrumentation that reports the mechanism rather than only the score, under a
  develop/seal/out-of-distribution evaluation protocol.
\end{itemize}

% =====================================================================================
% ADDED 2026-08-17 (soft-skeleton audit) --- introduction-level points, parked at the end
% as scaffold bullets. These sharpen framing already present above; merge them into
% "Losses, imbalance and topology geometry", "Technical problem: what Dice cannot see"
% and "Contributions" rather than adding a new section. WORD BUDGET: the contribution
% bullet is the one that must survive; the rest are one clause each at most.
% =====================================================================================

\begin{itemize}
  \item \textbf{Say what the centreline objective actually does, at first mention.} It is
  routinely introduced as a topology or connectivity loss. It is not one: the differentiable
  skeleton is an accumulation of morphological opening residuals, which carries no
  connectivity guarantee, and severing a one-voxel branch costs it a single skeleton voxel. What
  it does supply is removal of the $r^2$ area weighting, so a bronchiole carries loss weight
  comparable to a main bronchus. Framing it as a re-weighting device rather than a topological
  constraint is both accurate and better supported by this study's own measurements, where tree
  length and branch detection move while overlap does not.
  \item \textbf{The label budget is not the scarce resource, and the introduction should not
  imply that it is.} A self-configuring pipeline trained on sixteen labelled patients already
  reaches a composite score near $0.90$; what it does not reach is the periphery. The scarcity
  that matters is evidence at the acquisition resolution --- sub-voxel lumina and partial-volume
  walls --- not annotated patients. This reframes the semi-supervised question from ``can
  unlabelled data substitute for labels'' to ``can agreement between two views of the same model
  recover structure neither view resolves'', which is the question the study actually answers.
  \item \textbf{Add to the contributions list:} instrumentation of the consistency target
  itself, rather than of the score alone. Reporting how much teacher evidence a thresholding
  step discards, and how much of it survives skeletonisation, is what converts a small effect
  into a mechanism --- and it is what allows the limitation to be attributed to the target rather
  than to the geometry of the loss. Very few matched-control semi-supervised segmentation studies
  report anything of this kind.
  \item Optional, only if space allows: note early that the two views in a Mean Teacher are
  separated by the perturbation applied \emph{after} the shared augmentation pipeline, not by the
  augmentation itself. It is a small point but it pre-empts the obvious reader objection that the
  training data was already heavily augmented.
\end{itemize}
