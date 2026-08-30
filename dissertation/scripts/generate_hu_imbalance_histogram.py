"""Generate the airway class-imbalance figure: HU distribution by generation depth.

One title-free figure, in the house convention (panel letters and captions live in
the LaTeX ``figure`` environment, never in the artwork): a whole-scan Hounsfield
histogram of every voxel against the annotated airway voxels, with the low-density
window marked and magnified in an inset that splits the foreground into proximal
and distal by generation depth.

Two deliberate choices about the main axes:

*Whole volume, not the lung ROI.*  Restricting the denominator to the lung mask
removes the soft-tissue population almost entirely -- about 1% of in-mask voxels lie
above -100 HU, against 20-27% of the whole scan -- which erases the mode near 0 HU
that a thoracic CT histogram is expected to show.  The whole-scan view is also the
stronger statement of the imbalance: the denominator is every voxel acquired.

*Linear counts, not log.*  On a log axis the foreground would be plainly visible and
the figure would stop making its point.  The inset, magnified in both axes, is what
recovers the detail.

``--extra-panels`` additionally emits the lung-ROI version, an area-normalised
intensity-overlap panel and a branch-share bar chart.

The generation split
--------------------
Branches are parsed with the *exact* ATM'22 reference parser this project already
uses for the branch-detected (BD) metric, imported from
``lung_airway_segmentation.metrics.topology``.  Reusing it rather than
re-implementing means the figure's notion of a "branch" is the same object the
reported BD counts, so the two are directly comparable.  Depth is then a
breadth-first traversal of the parsed branch adjacency graph, rooted at the
branch the parser identifies as the trachea (the largest by volume).

``--proximal-max-generation`` (default 2) sets the line.  Depth <= 2 is
trachea + main bronchi + lobar bronchi; depth >= 3 is segmental bronchi and
everything beyond.  This follows the split used in the airway segmentation
literature, which reports "trachea + main bronchi + lobar bronchus" against
"segmental airways" as its two evaluation categories, and it is the deepest line
that still sits safely above the annotation floor of the dataset.  The clinical
small-airway definition (< 2 mm diameter, roughly generation 8 and beyond) is
NOT usable here: those airways are at or below the resolution limit of the CT and
are largely absent from the reference annotation, so a line drawn there would
describe the annotation protocol rather than the anatomy.

Caveats that belong in the caption, not in a footnote
-----------------------------------------------------
BFS depth is an *approximation* of anatomical generation.  Real airway branching
is asymmetric, so a fixed depth does not correspond to one anatomical level along
every path; the parser's junction-splitting and its single-child merge step can
also shift a branch by one.  What the split is reliable for is the coarse
proximal/distal contrast, and the first three depths recover the expected
anatomy closely (depth 0 = 1 branch, depth 1 = 2, depth 2 = 4-6, against 5 lobar
bronchi anatomically).

Usage, from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\generate_hu_imbalance_histogram.py

Per-case histograms are cached under ``--cache-dir`` as compressed npz, so
re-plotting after a styling change costs seconds rather than the ~20 s per case
the parse takes.  Pass ``--refresh`` to recompute.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# Running a file from ``dissertation/scripts/`` puts that directory, rather than
# the repository root, on sys.path. Add the root so the canonical parser imports.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lung_airway_segmentation.io.nnunet_lungcrop import (  # noqa: E402
    parse_case_intensity_overrides,
    resolve_lung_roi,
)
from lung_airway_segmentation.metrics.topology import (  # noqa: E402
    _branch_adjacency,
    _foreground_slices,
    _largest_connected_component,
    _locate_trachea,
    _nearest_branch_tree_parsing,
    _parent_children_maps,
    _refine_tree_parsing,
    _skeletonize,
    _split_reference_skeleton,
)

DATA_ROOT = ROOT / "data" / "ATM22"
DEFAULT_SPLIT_CONFIG = ROOT / "configs" / "nnunet" / "atm22_split_l20_u240.yaml"
FIGURE_ROOT = ROOT / "dissertation" / "Figures"
DEFAULT_PDF_OUT = FIGURE_ROOT / "pdf" / "background"
DEFAULT_PNG_OUT = FIGURE_ROOT / "png" / "background"
DEFAULT_PROVENANCE_OUT = FIGURE_ROOT / "provenance"
DEFAULT_CACHE_DIR = ROOT / ".codex_tmp" / "hu_imbalance"
# Bumped whenever the set of cached series changes, so a stale cache is never mixed
# with a new one -- the totals vector is positional and would silently misalign.
CACHE_VERSION = "v2"

# 1 HU bins over the full stored CT range. Wide enough that nothing is clipped
# into the end bins in practice, and fine enough to rebin freely for display.
HU_MIN = -1024
HU_MAX = 1024
HU_BINS = HU_MAX - HU_MIN

from figure_theme import (  # noqa: E402
    ARM_COLOUR, INK, MUTED, GREY, apply_theme, broken_axis_marker, finish,
)

BLUE = ARM_COLOUR["mt_soft"]
ORANGE = ARM_COLOUR["mt_hard_f0"]

# One colour per class, used identically in the main axes and the inset, so that a
# colour means the same thing everywhere in the figure. The earlier version reused
# the airway colour for the distal split, which implied the wrong grouping.
#
# The two airway classes are Okabe-Ito, matching the results figures: blue against
# vermillion is the most separable pair in that palette and stays legible under every
# common colour-vision deficiency and in greyscale.
#
# Non-airway keeps its original pastel blue rather than a neutral grey. It shares a
# hue family with the proximal class but not a value -- pastel against saturated --
# and the two are never adjacent in a way that matters: in the main axes the airway
# bars are a sliver against millions of voxels, which is the figure's whole point,
# and the magnified inset contains no non-airway at all.
NON_AIRWAY = "#9ec9e2"
PROXIMAL_COLOUR = ARM_COLOUR["mt_soft"]
DISTAL_COLOUR = ARM_COLOUR["mt_hard_f0"]
# Annotation ink, deliberately not one of the three data colours. Mid-grey rather
# than full ink: this colour also draws the inset leader lines, which cross the
# histogram and start competing with the data if they are as dark as the axes.
MARKER = MUTED

# Upper edge of the magnified window. Distal airway voxels reach a 95th percentile
# near -660 HU, so -600 keeps the whole foreground population inside the box while
# staying well clear of the parenchyma mode.
AIRWAY_WINDOW_MAX = -600
KILO = 1e3

PROXIMAL_LABEL = "Proximal airway"
DISTAL_LABEL = "Distal airway"

# Three panels across the A4 text width (171.8 mm) at 0.32\textwidth each leaves
# 2.17 in per panel, so the panels are AUTHORED at that width. Declaring the usual
# 3.20 in and letting LaTeX scale by 0.68 would shrink an 8 pt label to 5.4 pt, well
# under the minimum this project's figures target. Point sizes below are therefore
# close to their final rendered size.
PANEL_SIZE = (2.32, 2.05)
LABEL_PT = 8.5
TICK_PT = 7.0
LEGEND_PT = 6.8
ANNOTATION_PT = 6.8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG,
        help="Frozen split manifest supplying the cohort and the HU decoding overrides.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["labelled_train", "unlabelled_train"],
        help=(
            "Split keys to aggregate over. Defaults to the 260 training cases: a figure "
            "describing the data belongs on the data the model is allowed to see, not on "
            "the held-out val/test cohorts."
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Explicit ATM case ids, overriding --split.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N cases.")
    parser.add_argument(
        "--proximal-max-generation",
        type=int,
        default=2,
        help="Largest BFS depth counted as proximal (default 2 = trachea+main+lobar).",
    )
    parser.add_argument(
        "--minimum-branch-voxels",
        type=int,
        default=5,
        help="ATM'22 parser minimum branch length; keep at the scorer's default.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh", action="store_true", help="Ignore cached per-case counts.")
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    parser.add_argument("--provenance-output-dir", type=Path, default=DEFAULT_PROVENANCE_OUT)
    parser.add_argument(
        "--stem",
        default="airway_class_imbalance",
        help="Output filename stem; panels get a _a/_b/_c suffix.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Compute and print the aggregate statistics without writing figures.",
    )
    parser.add_argument(
        "--extra-panels",
        action="store_true",
        help="Also emit the lung-ROI, intensity-overlap and branch-share panels.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Cohort and volume loading
# --------------------------------------------------------------------------


def _load_cohort(args: argparse.Namespace) -> tuple[list[str], dict[str, dict]]:
    from lung_airway_segmentation.config import load_yaml_config

    config = load_yaml_config(args.split_config)
    overrides = parse_case_intensity_overrides(config)
    if args.cases:
        case_ids = [f"{int(str(c).removeprefix('ATM_')):03d}" for c in args.cases]
    else:
        splits = config.get("splits", {})
        missing = [key for key in args.split if key not in splits]
        if missing:
            raise SystemExit(
                f"Split(s) {missing} not in {args.split_config}; have {sorted(splits)}."
            )
        case_ids = sorted(
            {f"{int(c):03d}" for key in args.split for c in splits[key]}
        )
    if args.limit is not None:
        case_ids = case_ids[: args.limit]
    return case_ids, overrides


def _load_hu(ct_path: Path, override: dict | None) -> np.ndarray:
    """Load a CT as Hounsfield units, honouring the manifest's stored-value overrides.

    Mirrors ``write_roi_ct``'s decoding contract exactly: a declared override means
    the file carries raw stored values with identity NIfTI scaling, which must be
    decoded as ``stored * scale + offset``. Reading such a file with the ordinary
    scaled accessor yields values around 0-65535 rather than HU, which would put a
    tenth of the cohort's mass in the wrong place with no error raised.
    """
    image = nib.load(str(ct_path))
    proxy = image.dataobj
    if override is None:
        return np.asanyarray(proxy)

    slope = getattr(proxy, "slope", 1.0)
    intercept = getattr(proxy, "inter", 0.0)
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)
    if not np.isclose(slope, 1.0) or not np.isclose(intercept, 0.0):
        raise ValueError(
            f"{ct_path.name}: override decoding requires identity NIfTI scaling, "
            f"got slope={slope:g}, intercept={intercept:g}."
        )
    get_unscaled = getattr(proxy, "get_unscaled", None)
    raw = get_unscaled() if get_unscaled is not None else np.asanyarray(proxy)
    return np.asarray(raw, dtype=np.float32) * np.float32(override["scale"]) + np.float32(
        override["offset"]
    )


# --------------------------------------------------------------------------
# Generation labelling
# --------------------------------------------------------------------------


def branch_generation_labels(
    target_mask: np.ndarray,
    *,
    minimum_branch_voxels: int = 5,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    """Label every reference airway voxel with its BFS depth from the trachea.

    Returns ``(voxel_generation, branch_generation, crop_slices)``:

    ``voxel_generation``
        int16 volume over the cropped foreground bounding box; -1 for background,
        otherwise the depth of the branch owning that voxel.
    ``branch_generation``
        depth of each parsed branch, indexed from 0, so ``branch_generation.size``
        is the branch count the ATM'22 BD metric would report.
    ``crop_slices``
        the bounding box the two arrays live in, relative to the input volume.

    The parse itself is the project's ATM'22 scorer verbatim: split the LCC
    skeleton at junctions, assign every foreground voxel to its nearest skeleton
    branch, then iterate the challenge's refinement (merge multi-parent branches,
    absorb single-child chains) to a fixed point. Only the depth traversal at the
    end is added here, and it re-derives the adjacency after the final refinement
    round so the graph matches the branch labels it indexes.

    The labelled set is therefore NOT identical to the ground truth, in both
    directions. Both were measured across 105 scans and both are immaterial at the
    precision anything here is quoted to, but they are real:

    * Voxels LOST: mean 0.34% of foreground, max 19% on one case. Measured
      decomposition on that worst case (ATM_019), because the obvious explanation is
      the wrong one -- the largest-component step costs only 0.22%, while 19.07% is
      lost to the depth traversal itself, on branches the parser leaves disconnected
      in its adjacency graph even though the mask is spatially connected. Such a
      branch has no path to the trachea and therefore no generation, so it is
      excluded rather than silently binned as proximal. Everything outside that one
      case is under 1%.

      Ground-truth fragmentation is NOT the driver and is mostly a convention: at
      6-connectivity (which the project's post-processing uses) only 3 of 30 GT masks
      are a single component, but the extra pieces are tiny -- median 0.05% of
      foreground, max 0.23% -- and ATM_019 is a single component at 26-connectivity.

      Consequence for wording: the proximal/distal shares are shares of the PARSED
      tree, not of the raw ground truth.
    * Voxels GAINED. ``_largest_connected_component`` finishes with
      ``binary_fill_holes``, so an enclosed cavity would be labelled despite not
      being foreground. On an airway tree this is nearly a no-op -- 1 to 5 voxels in
      200,000, with no effect on any median -- because a tree encloses almost
      nothing.

    Class-balance figures ("airway is X% of the ROI") are computed from the raw
    ground truth instead, which is the right denominator for that claim.
    """
    component = _largest_connected_component(target_mask)
    crop = _foreground_slices(component)
    component = component[crop]
    skeleton = _skeletonize(component)

    split_labels, branch_count = _split_reference_skeleton(skeleton, minimum_branch_voxels)
    if branch_count == 0:
        empty = np.full(component.shape, -1, dtype=np.int16)
        return empty, np.zeros(0, dtype=np.int16), crop

    tree_parsing = _nearest_branch_tree_parsing(split_labels, component)
    while True:
        trachea = _locate_trachea(tree_parsing, branch_count)
        adjacency = _branch_adjacency(tree_parsing, branch_count)
        parent_map, children_map = _parent_children_maps(adjacency, trachea)
        tree_parsing, branch_count, changed = _refine_tree_parsing(
            tree_parsing, parent_map, children_map
        )
        if not changed:
            break

    # Breadth-first depth from the trachea over the settled adjacency.
    generation = np.full(branch_count, -1, dtype=np.int32)
    generation[trachea - 1] = 0
    frontier = [trachea - 1]
    while frontier:
        following: list[int] = []
        for current in frontier:
            for child in np.flatnonzero(adjacency[current]):
                if generation[child] < 0:
                    generation[child] = generation[current] + 1
                    following.append(int(child))
        frontier = following

    # A branch the traversal never reaches cannot be placed on either side of the
    # line. The parse runs on the largest connected component, so this should be
    # empty; leaving such voxels at -1 excludes them rather than silently binning
    # them as proximal.
    lookup = np.full(branch_count + 1, -1, dtype=np.int16)
    lookup[1:] = generation
    voxel_generation = lookup[tree_parsing]
    return voxel_generation, generation.astype(np.int16), crop


# --------------------------------------------------------------------------
# Per-case histogram accumulation
# --------------------------------------------------------------------------


def _histogram(values: np.ndarray) -> np.ndarray:
    """1 HU-wide counts over [HU_MIN, HU_MAX), with out-of-range values clipped in."""
    if values.size == 0:
        return np.zeros(HU_BINS, dtype=np.int64)
    index = np.clip(np.rint(values), HU_MIN, HU_MAX - 1).astype(np.int32) - HU_MIN
    return np.bincount(index.ravel(), minlength=HU_BINS).astype(np.int64)


def _case_counts(
    case_id: str,
    override: dict | None,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    ct_path = DATA_ROOT / "imagesTr" / f"ATM_{case_id}_0000.nii.gz"
    gt_path = DATA_ROOT / "labelsTr" / f"ATM_{case_id}_0000.nii.gz"
    lung_path = DATA_ROOT / "lungTr" / f"ATM_{case_id}_lung.nii.gz"
    for path in (ct_path, gt_path, lung_path):
        if not path.is_file():
            raise SystemExit(f"Missing input for ATM_{case_id}: {path}")

    hu = _load_hu(ct_path, override)
    ground_truth = np.asanyarray(nib.load(str(gt_path)).dataobj) > 0
    # The training ROI, from the same helper the preprocessing uses, so the "roi"
    # series below is exactly the volume the network is given after lung cropping.
    roi_bounds, lung = resolve_lung_roi(nib.load(str(ct_path)), nib.load(str(lung_path)))
    if ground_truth.shape != hu.shape or lung.shape != hu.shape:
        raise SystemExit(
            f"ATM_{case_id}: shape mismatch ct={hu.shape} gt={ground_truth.shape} "
            f"lung={lung.shape}."
        )

    voxel_generation, branch_generation, crop = branch_generation_labels(
        ground_truth, minimum_branch_voxels=args.minimum_branch_voxels
    )
    split = args.proximal_max_generation
    hu_crop = hu[crop]
    proximal_voxels = (voxel_generation >= 0) & (voxel_generation <= split)
    distal_voxels = voxel_generation > split

    # The lung mask is the operative denominator: the pipeline trains on a lung
    # ROI, so this is the imbalance the network actually sees. The whole-volume
    # count is kept alongside it because the published version of this panel uses
    # the uncropped scan, and the two ratios differ by roughly an order of
    # magnitude.
    return {
        "lung": _histogram(hu[lung]),
        "volume": _histogram(hu),
        "roi": _histogram(hu[roi_bounds]),
        "proximal": _histogram(hu_crop[proximal_voxels]),
        "distal": _histogram(hu_crop[distal_voxels]),
        "parenchyma": _histogram(hu[lung & ~ground_truth]),
        "branches_per_generation": np.bincount(
            branch_generation[branch_generation >= 0], minlength=1
        ).astype(np.int64),
        "voxels_per_generation": np.bincount(
            voxel_generation[voxel_generation >= 0].ravel(), minlength=1
        ).astype(np.int64),
        "totals": np.array(
            [
                int(lung.sum()),
                int(hu.size),
                int(hu[roi_bounds].size),
                int(ground_truth.sum()),
                int(proximal_voxels.sum()),
                int(distal_voxels.sum()),
                int((branch_generation >= 0).sum()),
                int(((branch_generation >= 0) & (branch_generation <= split)).sum()),
                int((branch_generation > split).sum()),
            ],
            dtype=np.int64,
        ),
    }


TOTAL_KEYS = (
    "lung_voxels",
    "volume_voxels",
    "roi_voxels",
    "airway_voxels",
    "proximal_voxels",
    "distal_voxels",
    "branches",
    "proximal_branches",
    "distal_branches",
)


def _accumulate(
    case_ids: list[str],
    overrides: dict[str, dict],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    cache_dir = args.cache_dir / CACHE_VERSION / f"gen{args.proximal_max_generation}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, np.ndarray] = {}
    per_case: list[dict] = []

    for position, case_id in enumerate(case_ids, start=1):
        cache_path = cache_dir / f"ATM_{case_id}.npz"
        if cache_path.is_file() and not args.refresh:
            with np.load(cache_path) as handle:
                counts = {key: handle[key] for key in handle.files}
            source = "cache"
        else:
            started = time.time()
            counts = _case_counts(case_id, overrides.get(case_id), args)
            np.savez_compressed(cache_path, **counts)
            source = f"{time.time() - started:.0f}s"
        totals = dict(zip(TOTAL_KEYS, (int(v) for v in counts["totals"])))
        per_case.append({"case": f"ATM_{case_id}", **totals})
        print(
            f"  [{position:>3}/{len(case_ids)}] ATM_{case_id} ({source}) "
            f"airway={totals['airway_voxels']:>7} "
            f"branches={totals['branches']:>4} "
            f"distal_branch_share={totals['distal_branches'] / max(totals['branches'], 1):.3f}"
        )
        for key, value in counts.items():
            if key not in aggregate:
                aggregate[key] = value.astype(np.int64).copy()
            elif value.size > aggregate[key].size:  # ragged per-generation vectors
                padded = value.astype(np.int64).copy()
                padded[: aggregate[key].size] += aggregate[key]
                aggregate[key] = padded
            else:
                aggregate[key][: value.size] += value
    return aggregate, per_case


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _summarise(aggregate: dict[str, np.ndarray], split: int) -> dict:
    totals = dict(zip(TOTAL_KEYS, (int(v) for v in aggregate["totals"])))
    centres = np.arange(HU_MIN, HU_MAX) + 0.5

    def _moments(counts: np.ndarray) -> dict:
        total = int(counts.sum())
        if total == 0:
            return {"voxels": 0, "mean_hu": None, "median_hu": None, "p95_hu": None}
        mean = float((counts * centres).sum() / total)
        cumulative = np.cumsum(counts)
        median = float(centres[int(np.searchsorted(cumulative, total * 0.5))])
        p95 = float(centres[int(np.searchsorted(cumulative, total * 0.95))])
        return {"voxels": total, "mean_hu": mean, "median_hu": median, "p95_hu": p95}

    airway = totals["airway_voxels"]
    branches = totals["branches"]
    return {
        "proximal_max_generation": split,
        "totals": totals,
        # NOT reported against the lung MASK. Only 28-42% of the airway lies inside it:
        # lungmask segments parenchyma, so the trachea, main bronchi and the proximal
        # tree running through the mediastinum are outside. Dividing all airway voxels
        # by the mask would be a numerator its denominator does not contain, which is
        # not a class balance. The two valid denominators are the whole volume and the
        # ROI BOX -- the bounding box plus margins that the pipeline actually crops to,
        # whose 120-voxel superior margin exists precisely to keep the trachea.
        "foreground_fraction_of_volume": totals["airway_voxels"] / max(totals["volume_voxels"], 1),
        "foreground_fraction_of_roi_box": totals["airway_voxels"] / max(totals["roi_voxels"], 1),
        "proximal_voxel_share": totals["proximal_voxels"] / max(airway, 1),
        "distal_voxel_share": totals["distal_voxels"] / max(airway, 1),
        "proximal_branch_share": totals["proximal_branches"] / max(branches, 1),
        "distal_branch_share": totals["distal_branches"] / max(branches, 1),
        "hu": {
            "proximal": _moments(aggregate["proximal"]),
            "distal": _moments(aggregate["distal"]),
            "parenchyma": _moments(aggregate["parenchyma"]),
        },
        "branches_per_generation": [int(v) for v in aggregate["branches_per_generation"]],
        "voxels_per_generation": [int(v) for v in aggregate["voxels_per_generation"]],
    }


def _print_summary(summary: dict) -> None:
    totals = summary["totals"]
    print("\n--- aggregate ---")
    print(
        f"  airway = {summary['foreground_fraction_of_roi_box'] * 100:.3f}% of lung-ROI-box voxels, "
        f"{summary['foreground_fraction_of_volume'] * 100:.3f}% of whole-volume voxels"
    )
    print(
        f"  proximal (gen <= {summary['proximal_max_generation']}): "
        f"{summary['proximal_branch_share'] * 100:5.1f}% of branches, "
        f"{summary['proximal_voxel_share'] * 100:5.1f}% of foreground voxels"
    )
    print(
        f"  distal   (gen >  {summary['proximal_max_generation']}): "
        f"{summary['distal_branch_share'] * 100:5.1f}% of branches, "
        f"{summary['distal_voxel_share'] * 100:5.1f}% of foreground voxels"
    )
    print(
        f"  branches {totals['branches']}, airway voxels {totals['airway_voxels']}, "
        f"lung-ROI voxels {totals['lung_voxels']}"
    )
    for name in ("proximal", "distal", "parenchyma"):
        moments = summary["hu"][name]
        print(
            f"  HU {name:<10} mean {moments['mean_hu']:8.1f}  "
            f"median {moments['median_hu']:8.1f}  p95 {moments['p95_hu']:8.1f}"
        )
    print("\n  generation profile (branches / voxels):")
    for depth, (branch_count, voxel_count) in enumerate(
        zip(summary["branches_per_generation"], summary["voxels_per_generation"])
    ):
        print(f"    gen {depth:>2}: {branch_count:>5} branches   {voxel_count:>9} voxels")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def _rebin(counts: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    usable = (counts.size // width) * width
    binned = counts[:usable].reshape(-1, width).sum(axis=1)
    edges = HU_MIN + np.arange(binned.size + 1) * width
    return edges, binned


def _style(axis: plt.Axes) -> None:
    """Hard left/bottom axes from the shared theme."""
    finish(axis)


def _save(fig: plt.Figure, pdf_dir: Path, png_dir: Path, stem: str) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir / f"{stem}.pdf", facecolor="white")
    fig.savefig(png_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)


def _figure_imbalance(
    aggregate: dict[str, np.ndarray],
    summary: dict,
    case_count: int,
    *,
    denominator: str = "volume",
    inset_rect: tuple[float, float, float, float] = (0.46, 0.50, 0.52, 0.48),
) -> plt.Figure:
    """The headline figure: HU histogram of the training ROI with a magnified airway window.

    The denominator is the LUNG ROI BOX -- the lung-mask bounding box with the
    pipeline's own margins, taken from the same ``resolve_lung_roi`` the
    preprocessing uses. That choice is not cosmetic. Two alternatives were tried:

    * the lung mask, which deletes the soft-tissue population almost entirely (~1% of
      in-mask voxels exceed -100 HU, against 20-27% of the scan) and so has no mode
      near 0 HU at all.
    * the lung ROI box, which keeps both modes but hides the field-of-view air.

    The whole scan wins for a BACKGROUND figure specifically, because the air spike is
    itself part of the argument: it is the single largest population in the volume,
    it is not anatomy, and it is the reason the pipeline crops to a lung ROI before
    training. Removing it would delete the motivation for a step the method depends
    on. (The ROI-box histogram is still computed, and is what the 0.32%-of-ROI figure
    quoted elsewhere refers to.)

    The y-axis is clipped just above the parenchyma mode and the clipped bar is
    labelled with its true height. Unclipped, the air spike is roughly five times
    anything anatomical and flattens both real modes into the baseline.

    The inset is magnified in BOTH axes: it re-plots the boxed HU window on a count
    axis three orders of magnitude finer, which is the only way the airway class is
    visible at all. Within it the foreground is split by generation depth.
    """
    fig, axis = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
    width = 10
    edges, volume = _rebin(aggregate[denominator], width)
    _, proximal = _rebin(aggregate["proximal"], width)
    _, distal = _rebin(aggregate["distal"], width)
    centres = edges[:-1] + width / 2

    # Per scan, so the axis reads like a single-case histogram rather than a total
    # that happens to depend on how many cases were aggregated.
    volume = volume / case_count
    proximal = proximal / case_count
    distal = distal / case_count
    scale = 1e6
    airway = proximal + distal

    # The airway pair is drawn FROM ZERO, in front of the non-airway bars, not stacked
    # on top of them. Stacking is the correct partition arithmetically, but it puts the
    # foreground sliver at the top of a bar millions of voxels high, which reads as the
    # airway having enormous counts and hides where the class actually lives. Drawing
    # from zero also makes the main axes structurally identical to the inset: proximal
    # from the baseline, distal stacked on proximal. Since airway is under 0.1% of the
    # volume, the non-airway bars are visually indistinguishable from "all voxels".
    non_airway = volume - airway
    axis.bar(
        centres, non_airway / scale, width=width, color=NON_AIRWAY,
        label="Non-airway voxels", zorder=2,
    )
    axis.bar(
        centres, proximal / scale, width=width,
        color=PROXIMAL_COLOUR, label=PROXIMAL_LABEL, zorder=3,
    )
    axis.bar(
        centres, distal / scale, width=width, bottom=proximal / scale,
        color=DISTAL_COLOUR, label=DISTAL_LABEL, zorder=4,
    )
    axis.set_xlim(HU_MIN, 400)

    # Clip only when a non-anatomical spike would otherwise set the scale. Whole-volume
    # histograms are dominated by field-of-view air near -1000 HU; ROI histograms are
    # not, and clipping one that does not need it would just leave dead space. The
    # threshold at -950 HU separates that air from the parenchyma mode, and the test
    # below decides for itself which regime it is in.
    anatomical_max = float(volume[centres >= -950].max() / scale)
    peak = float(volume.max() / scale)
    cap = anatomical_max * 1.55 if peak > anatomical_max * 1.6 else peak * 1.08
    axis.set_ylim(0, cap)
    axis.set_xlabel("Hounsfield units", fontsize=LABEL_PT)
    axis.set_ylabel(r"Voxels per scan ($\times 10^{6}$)", fontsize=LABEL_PT)
    _style(axis)

    if peak > cap:
        # The y-axis is cut below the field-of-view air spike, so mark it broken.
        # The annotation says so in words; this says so where the eye looks first.
        broken_axis_marker(axis, at=0.965)
        peak_hu = float(centres[int(volume.argmax())])
        axis.annotate(
            f"air outside the patient,\n{peak:.0f}" + r"$\times 10^{6}$ (axis clipped)",
            xy=(peak_hu + 12, cap * 0.985),
            xytext=(peak_hu + 145, cap * 0.90),
            fontsize=ANNOTATION_PT, color=MUTED, va="top", ha="left",
            arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.7,
                        "shrinkA": 1.0, "shrinkB": 1.0},
        )

    # One legend for the whole figure, outside the axes, so it cannot collide with
    # either the histogram or the inset. The inset therefore carries none of its own.
    # "outside ..." placement is a figure-legend feature; on an Axes it raises.
    fig.legend(
        *axis.get_legend_handles_labels(),
        fontsize=LEGEND_PT, frameon=False, ncol=3, loc="outside upper center",
    )

    # --- magnified airway window -------------------------------------------------
    window = (HU_MIN, AIRWAY_WINDOW_MAX)
    # Placement is per-denominator, because the two histograms are tall in different
    # places: clipped whole-volume leaves the upper right free, while the ROI version
    # has its largest mode near 0 HU and only the middle band free. Passed in rather
    # than guessed, so a layout change is a one-line, reviewable edit.
    inset = axis.inset_axes(list(inset_rect))
    selection = (centres >= window[0]) & (centres <= window[1])
    inset.bar(
        centres[selection], proximal[selection] / KILO, width=width,
        color=PROXIMAL_COLOUR, zorder=3,
    )
    inset.bar(
        centres[selection], distal[selection] / KILO, width=width,
        bottom=proximal[selection] / KILO, color=DISTAL_COLOUR, zorder=3,
    )
    inset.set_xlim(*window)
    inset.set_xlabel("Hounsfield units", fontsize=TICK_PT, labelpad=1.5)
    inset.set_ylabel(r"Voxels per scan ($\times 10^{3}$)", fontsize=TICK_PT, labelpad=3.5)
    # Size only; finish() owns the colours, so the inset axes match the main ones.
    inset.tick_params(labelsize=TICK_PT - 0.5, length=2.5)
    finish(inset)
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)

    # The rectangle marks the HU WINDOW that the inset magnifies. Sized from the
    # airway counts alone it would be ~0.1% of the axis height -- literally a line --
    # because that invisibility is the very thing the figure is about. So it takes a
    # floor of 4% of the axis, and the caption says the box marks a window rather
    # than a count range, which is what it honestly is.
    box_height = max(float(airway[selection].max() / scale) * 1.35, cap * 0.045)
    rectangle, connectors = axis.indicate_inset(
        bounds=(window[0], 0.0, window[1] - window[0], box_height),
        inset_ax=inset,
        edgecolor=MARKER,
        facecolor="none",
        alpha=1.0,
        linewidth=0.9,
        zorder=6,
    )
    # The connectors have to cross the histogram to reach the inset. Drawing them
    # UNDER the bars and half transparent means they read as a leader line and stop
    # slicing through the data; the rectangle itself stays on top and opaque.
    for connector in connectors:
        connector.set_zorder(1)
        connector.set_alpha(0.55)
        connector.set_linewidth(0.8)
    rectangle.set_zorder(6)
    return fig


def _panel_inter_class(aggregate: dict[str, np.ndarray], summary: dict) -> plt.Figure:
    """(a) Lung-ROI voxels against airway voxels, linear counts."""
    fig, axis = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
    width = 10
    edges, lung = _rebin(aggregate["lung"], width)
    _, proximal = _rebin(aggregate["proximal"], width)
    _, distal = _rebin(aggregate["distal"], width)
    centres = edges[:-1] + width / 2
    scale = 1e6

    axis.bar(
        centres, lung / scale, width=width, color=GREY, label="All lung-ROI voxels", zorder=2
    )
    axis.bar(
        centres,
        (proximal + distal) / scale,
        width=width,
        color=ORANGE,
        label="Airway voxels",
        zorder=3,
    )
    axis.set_xlim(HU_MIN, 200)
    axis.set_xlabel("Hounsfield units", fontsize=LABEL_PT)
    axis.set_ylabel(r"Voxels ($\times 10^{6}$)", fontsize=LABEL_PT)
    axis.legend(fontsize=LEGEND_PT, frameon=False, loc="upper left")
    axis.annotate(
        f"airway = {summary['foreground_fraction_of_lung_roi'] * 100:.2f}%\nof lung-ROI voxels",
        xy=(0.97, 0.62),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=ANNOTATION_PT,
        color=INK,
    )
    _style(axis)
    return fig


def _panel_intensity_overlap(aggregate: dict[str, np.ndarray]) -> plt.Figure:
    """(b) Area-normalised HU densities: parenchyma, proximal, distal."""
    fig, axis = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
    width = 10
    series = (
        ("Lung parenchyma", aggregate["parenchyma"], GREY, "-"),
        (PROXIMAL_LABEL, aggregate["proximal"], BLUE, "-"),
        (DISTAL_LABEL, aggregate["distal"], ORANGE, "-"),
    )
    for label, counts, colour, style in series:
        edges, binned = _rebin(counts, width)
        centres = edges[:-1] + width / 2
        total = binned.sum()
        if total == 0:
            continue
        axis.plot(
            centres,
            binned / total,
            color=colour,
            linestyle=style,
            linewidth=1.4,
            label=label,
            zorder=3,
        )
    # The left-most bin carries the scanner's clamp pile-up at -1024 HU, which is
    # genuine data rather than a binning artefact; starting the axis exactly there
    # keeps it at the boundary instead of showing it as an interior spike.
    axis.set_xlim(HU_MIN, 200)
    axis.set_yscale("log")
    axis.set_ylim(1e-5, 1.0)
    axis.set_xlabel("Hounsfield units", fontsize=LABEL_PT)
    axis.set_ylabel("Fraction of class voxels", fontsize=LABEL_PT)
    axis.legend(fontsize=LEGEND_PT, frameon=False, loc="upper left")
    _style(axis)
    return fig


def _panel_intra_class(summary: dict) -> plt.Figure:
    """(c) Share of branches against share of foreground voxels."""
    fig, axis = plt.subplots(figsize=PANEL_SIZE, layout="constrained")
    positions = np.array([0.0, 1.0])
    bar_width = 0.36
    proximal = np.array([summary["proximal_branch_share"], summary["proximal_voxel_share"]]) * 100
    distal = np.array([summary["distal_branch_share"], summary["distal_voxel_share"]]) * 100

    axis.bar(
        positions - bar_width / 2, proximal, bar_width, color=BLUE, label=PROXIMAL_LABEL, zorder=3
    )
    axis.bar(
        positions + bar_width / 2, distal, bar_width, color=ORANGE, label=DISTAL_LABEL, zorder=3
    )
    for position, (left, right) in zip(positions, zip(proximal, distal)):
        axis.text(
            position - bar_width / 2, left + 2, f"{left:.0f}", ha="center", fontsize=ANNOTATION_PT, color=INK
        )
        axis.text(
            position + bar_width / 2,
            right + 2,
            f"{right:.0f}",
            ha="center",
            fontsize=ANNOTATION_PT,
            color=INK,
        )
    axis.set_xticks(positions)
    # Kept short: at panel width the fuller "Share of parsed branches" wording runs
    # into its neighbour. The caption carries the "share of" framing.
    axis.set_xticklabels(["Parsed\nbranches", "Foreground\nvoxels"], fontsize=TICK_PT)
    # Headroom for the legend above the tallest bar (~97) plus its value label.
    axis.set_ylim(0, 132)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.set_ylabel("Per cent", fontsize=LABEL_PT)
    axis.legend(fontsize=LEGEND_PT, frameon=False, loc="upper center", ncol=2, borderaxespad=0.2)
    _style(axis)
    return fig


def _write_caption_macros(summary: dict, case_count: int, destination: Path) -> Path:
    """Emit the caption's numbers as LaTeX macros.

    The caption quotes seven figures that all come from this computation. Typing them
    into the .tex by hand means they silently go stale the moment the cohort or the
    generation split changes -- and a caption disagreeing with its own figure is the
    kind of error nobody catches in proofreading. Generating them here makes that
    impossible: re-running the script updates plot and prose together.
    """
    hu = summary["hu"]
    per_generation = summary["branches_per_generation"]

    def _per_case(depth: int) -> str:
        value = per_generation[depth] / case_count if depth < len(per_generation) else 0.0
        return f"{value:.1f}"

    macros = {
        "airwayNcases": str(case_count),
        "airwayFgRoi": f"{summary['foreground_fraction_of_roi_box'] * 100:.2f}",
        "airwayFgVolume": f"{summary['foreground_fraction_of_volume'] * 100:.3f}",
        "airwayProxBranchShare": f"{summary['proximal_branch_share'] * 100:.1f}",
        "airwayProxVoxelShare": f"{summary['proximal_voxel_share'] * 100:.1f}",
        "airwayProxMedianHU": f"{hu['proximal']['median_hu']:.0f}",
        "airwayDistMedianHU": f"{hu['distal']['median_hu']:.0f}",
        "airwayParenMedianHU": f"{hu['parenchyma']['median_hu']:.0f}",
        "airwayGenZeroPerCase": _per_case(0),
        "airwayGenOnePerCase": _per_case(1),
        "airwayGenTwoPerCase": _per_case(2),
        "airwayProximalMaxGen": str(summary["proximal_max_generation"]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% GENERATED by dissertation/scripts/generate_hu_imbalance_histogram.py.",
        "% Do not edit: re-run the script instead, so the caption and the plot stay",
        "% in agreement. Negative HU values are wrapped for math mode by the caller.",
    ]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _write_provenance(
    args: argparse.Namespace,
    case_ids: list[str],
    per_case: list[dict],
    summary: dict,
) -> Path:
    args.provenance_output_dir.mkdir(parents=True, exist_ok=True)
    output = args.provenance_output_dir / f"{args.stem}.json"
    output.write_text(
        json.dumps(
            {
                "figure": args.stem,
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
                "split_config": str(args.split_config.relative_to(ROOT)).replace("\\", "/"),
                "splits": None if args.cases else list(args.split),
                "cases": [f"ATM_{case_id}" for case_id in case_ids],
                "branch_parser": "lung_airway_segmentation.metrics.topology (ATM'22 reference)",
                "minimum_branch_voxels": args.minimum_branch_voxels,
                "proximal_max_generation": args.proximal_max_generation,
                "hu_bin_width": 1,
                "hu_range": [HU_MIN, HU_MAX],
                "summary": summary,
                "per_case": per_case,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    apply_theme()
    args = _parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    case_ids, overrides = _load_cohort(args)
    applied = sorted(set(case_ids) & set(overrides))
    print(f"Cohort: {len(case_ids)} cases; HU overrides applied to {len(applied)}: {applied}")

    aggregate, per_case = _accumulate(case_ids, overrides, args)
    summary = _summarise(aggregate, args.proximal_max_generation)
    _print_summary(summary)
    if args.stats_only:
        return

    # Both denominators, so the whole-scan and lung-ROI framings can be compared side
    # by side before one is chosen.
    for suffix, denominator, inset_rect in (
        ("", "volume", (0.46, 0.50, 0.52, 0.48)),
        ("_roi", "roi", (0.20, 0.44, 0.42, 0.54)),
    ):
        _save(
            _figure_imbalance(
                aggregate, summary, len(case_ids),
                denominator=denominator, inset_rect=inset_rect,
            ),
            args.pdf_output_dir,
            args.png_output_dir,
            f"{args.stem}{suffix}",
        )
    if args.extra_panels:
        # Kept, not deleted: the intensity-overlap and branch-share panels answer
        # different questions from the headline histogram and the statistics behind
        # them are quoted in the text, so they stay one flag away.
        # The lung-MASK panel is deliberately absent; see the note in _summarise. The
        # intensity panel below still uses the mask, but only to describe parenchyma
        # HU, which is a legitimate use of it -- it is not used as a denominator.
        for suffix, figure in (
            ("_intensity", _panel_intensity_overlap(aggregate)),
            ("_branchshare", _panel_intra_class(summary)),
        ):
            _save(figure, args.pdf_output_dir, args.png_output_dir, f"{args.stem}{suffix}")
    provenance = _write_provenance(args, case_ids, per_case, summary)
    macros = _write_caption_macros(summary, len(case_ids), FIGURE_ROOT / f"{args.stem}_numbers.tex")
    print(
        f"\nWrote panels to {args.pdf_output_dir.resolve()}, provenance to {provenance.name}, "
        f"caption macros to {macros.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
