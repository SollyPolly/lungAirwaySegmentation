"""Generate the opening figure of the Introduction: the airway tree in thoracic CT.

Two panels, emitted as separate title-free PDFs in the house convention (panel
letters and captions live in the LaTeX ``figure`` environment, never in the
artwork):

``airway_overview_slice``
    One coronal CT slice at lung window with the reference airway overlaid, plus
    two insets of equal *physical* size -- one on the trachea, one on a distal
    branch.  Equal physical size is the whole point: the calibre contrast is the
    figure's argument, so a reader must not be able to attribute it to different
    zoom factors.

``airway_overview_tree_{plain,depth,gen}``
    A 3D render of the same reference tree, in three colourings so the writer can
    choose:  ``plain`` is a single anatomical tone;  ``depth`` uses the *same two
    colours* as the class-imbalance figure, so proximal/distal means one thing
    across the document;  ``gen`` is a continuous ramp over BFS depth.

Where the 3D surface comes from
-------------------------------
By default the tree is ITK-SNAP's own exported surface (``Segmentation > Export
as Surface Mesh``), read from ``DEFAULT_MESH`` if that file is present.  ITK-SNAP
writes legacy ASCII VTK PolyData *version 5.1*, as triangle strips with
per-vertex NORMALS and a header declaring its coordinate space; all three of
those are handled, and the RAS/LPS convention is resolved by checking which one
actually lands inside the annotation rather than by trusting the header.  An STL
export is accepted too.

Rendering the exported mesh here, rather than screenshotting the ITK-SNAP window,
buys print resolution (a screenshot is capped at the on-screen panel size), a
white or transparent background, and the proximal/distal colour variants.  Pass
``--no-mesh`` to ignore the export and extract the surface internally instead.

How the 3D render is made
-------------------------
``--render-mode surface`` (used when no mesh is supplied) is an isosurface render
with no VTK dependency, built on ``skimage.measure.marching_cubes``:

1.  Build a *signed distance field* over the cropped mask and smooth it lightly.
    Not a blurred binary mask: thresholding a blurred binary at 0.5 erases a
    one-voxel tube outright, and 41% of this tree's voxels sit at a local radius
    of 1 mm or less.
2.  March an isosurface at zero, then apply Taubin (lambda/mu) smoothing to the
    MESH.  Field smoothing is capped by connectivity -- past ~0.35 mm it pinches
    thin branches off -- whereas moving vertices cannot change topology, and the
    alternating sign pair does not shrink the surface.  The run prints voxel
    retention and a before/after component count as the check.
3.  Sample the mesh (vertices, face centroids, edge midpoints) and carry true
    area-weighted vertex normals with each sample.
4.  Rotate points and normals, then paint into a depth buffer back-to-front, so
    the nearest surface wins every pixel.  This is an exact z-buffer, not the
    average-z sort ``mplot3d`` uses -- an airway tree self-occludes constantly and
    the painter's-algorithm artefacts of a ``Poly3DCollection`` render land
    exactly where the eye looks.
5.  Shade from the STORED normals.  Deriving normals from the gradient of the
    depth map instead reproduces whatever steps the depth map has, which
    re-creates the blockiness the isosurface was extracted to remove.
6.  Render supersampled and box-downsample; that is the anti-aliasing.

``--render-mode voxel`` keeps the older path, which splats the raw voxel grid.
It is a faithful picture of the data and looks blocky for a real reason: at
0.26 mm/px one 0.82 mm voxel is six pixels across, so the stair steps are
resolved rather than hidden.  Raising the resolution sharpens them.  Use it only
to make the point that the annotation is a voxel grid.

Usage, from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\generate_intro_airway_overview.py
    .venv\\Scripts\\python.exe dissertation\\scripts\\generate_intro_airway_overview.py \\
        --case 023 --azimuth -35 --elevation 12

The branch depths are the project's ATM'22 reference parse, imported from the
class-imbalance script so the two figures cannot disagree about what a branch is.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy import ndimage as ndi

# Running a file from ``dissertation/scripts/`` puts that directory, rather than
# the repository root, on sys.path. Add the root so the project package imports.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_hu_imbalance_histogram import (  # noqa: E402  (sibling script)
    _load_hu,
    branch_generation_labels,
)

from lung_airway_segmentation.io.nnunet_lungcrop import (  # noqa: E402
    parse_case_intensity_overrides,
)

DATA_ROOT = ROOT / "data" / "ATM22"
DEFAULT_SPLIT_CONFIG = ROOT / "configs" / "nnunet" / "atm22_split_l20_u240.yaml"
FIGURE_ROOT = ROOT / "dissertation" / "Figures"
DEFAULT_PDF_OUT = FIGURE_ROOT / "pdf" / "intro"
DEFAULT_PNG_OUT = FIGURE_ROOT / "png" / "intro"
DEFAULT_PROVENANCE_OUT = FIGURE_ROOT / "provenance"
# ITK-SNAP's own surface, exported via Segmentation > Export as Surface Mesh.
# Lives under Figures/src/ with the other figure SOURCES, not beside the rendered
# outputs. Used automatically when present so that re-running the script cannot
# silently fall back to the internally extracted isosurface, which looks different.
DEFAULT_MESH = FIGURE_ROOT / "src" / "intro" / "ATM_001_3D_mesh_figure_introduction.vtk"

# Shared with generate_hu_imbalance_histogram.py. A colour must mean the same
# thing in every figure of the document, so these are copied deliberately rather
# than re-chosen.
PROXIMAL_COLOUR = "#0e7490"
DISTAL_COLOUR = "#f97316"
# Single-tone render: deliberately NOT one of the two above, so the plain variant
# cannot be misread as carrying a proximal/distal encoding.
PLAIN_COLOUR = "#5b7f99"
OVERLAY_COLOUR = "#f97316"
# Inset boxes, leader lines and scale bars.
#
# Dark red is the deliberate choice: higher-chroma alternatives measure better in
# isolation but read as garish over a greyscale CT. Measured worst-case CIE76 dE
# against the orange overlay and teal proximal class, over normal vision and all
# three dichromacies:
#   dark red   #a4161a   C* 66.1   worst dE 26.0   <- chosen
#   magenta    #E7298A   C* 75.3   worst dE  5.6   collapses onto teal under
#                                                  protanopia; a trap by eye
#   purple     #9333EA   C* 102.5  worst dE 48.5   measures best, looks wrong
#
# Red alone is the weakest of these on separation, and its relative luminance
# (0.085) sits close to proximal teal's (0.146), so it does not survive greyscale
# unaided. The achromatic CASING is what makes it safe: a wider white outline
# under every annotation stroke. Being hue-free it is immune to colour-vision
# deficiency, and it guarantees a luminance edge against dark lung, bright
# mediastinum and the airway fills alike. Colour and casing are a pair -- remove
# the casing and this colour choice no longer holds up.
ANNOTATION_COLOUR = "#a4161a"
ANNOTATION_CASING = "#ffffff"
# Millimetres per scale bar. One length inside both insets, which are at equal
# magnification, so the two bars render identically and demonstrate that.
INSET_SCALE_BAR_MM = 5.0
PANEL_SCALE_BAR_MM = 50.0
# Resolution the slice panel is written at. Named rather than inlined so the
# pixel nudge below can be converted into data units.
SLICE_PANEL_DPI = 400
# The inset scale bar sits closer to its frame than the panel bar does to the
# figure edge, because the inset is small. Nudge it clear, in output pixels.
INSET_SCALE_BAR_NUDGE_PX = 3.0
INSET_FRACTION = 0.30

# Standard lung window (level -600, width 1500).
LUNG_WINDOW = (-1350.0, 150.0)

# Panels are authored at their final width: 0.48\textwidth of the A4 text block
# (171.8 mm) is 82.5 mm, i.e. 3.25 in. Authoring at final size means an 8 pt label
# is 8 pt on the page rather than whatever LaTeX's scale factor makes of it.
PANEL_WIDTH_IN = 3.25
INSET_BOX_MM = 30.0


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="001",
        help="ATM case id (zero-padded, e.g. 001). Must have a reference label.",
    )
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument(
        "--azimuth",
        type=float,
        default=-30.0,
        help="Degrees of rotation about the superior-inferior axis. 0 is a straight "
        "anterior view; a small negative value separates the two main bronchi.",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=10.0,
        help="Degrees the camera is raised above the axial plane.",
    )
    parser.add_argument(
        "--px-mm",
        type=float,
        default=0.26,
        help="Millimetres per pixel of the FINAL 3D render, before supersampling.",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=2,
        help="Render at this factor and box-downsample. This is the anti-aliasing.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Render an externally authored surface instead of extracting one: an "
        "STL from ITK-SNAP's Segmentation > Export as Surface Mesh. Overrides "
        "--render-mode. The mesh is matched to the case by physical coordinates, "
        "so it must come from the same scan as --case.",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help=f"Ignore {DEFAULT_MESH.name} and extract the isosurface instead.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("surface", "voxel"),
        default="surface",
        help="'surface' marches an isosurface and shades it with true normals; "
        "'voxel' splats the raw voxel grid, which draws visible stair steps.",
    )
    parser.add_argument(
        "--smooth-mm",
        type=float,
        default=0.25,
        help="Gaussian sigma, in millimetres, applied to the signed distance field "
        "before the isosurface is extracted. This one is CAPPED by connectivity -- "
        "past ~0.35 mm it pinches thin branches off (the run prints the check). "
        "Prefer --mesh-smooth, which cannot change topology.",
    )
    parser.add_argument(
        "--mesh-smooth",
        type=int,
        default=None,
        help="Taubin mesh-smoothing iterations. Defaults to 16 for an extracted "
        "isosurface and 0 for an imported --mesh, since an imported mesh already "
        "carries whatever smoothing its author applied. Safe to raise: moving "
        "vertices cannot disconnect the tree, and the lambda/mu pair does not "
        "shrink it.",
    )
    parser.add_argument(
        "--proximal-max-generation",
        type=int,
        default=2,
        help="Largest BFS depth counted as proximal; keep aligned with the "
        "class-imbalance figure (default 2 = trachea + main + lobar).",
    )
    parser.add_argument(
        "--minimum-branch-voxels",
        type=int,
        default=5,
        help="ATM'22 parser minimum branch length; keep at the scorer's default.",
    )
    parser.add_argument("--pdf-output-dir", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=DEFAULT_PNG_OUT)
    parser.add_argument("--provenance-output-dir", type=Path, default=DEFAULT_PROVENANCE_OUT)
    parser.add_argument(
        "--skip-slice", action="store_true", help="Only render the 3D panels."
    )
    parser.add_argument(
        "--skip-tree", action="store_true", help="Only render the CT slice panel."
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Loading, in RAS+ so that "anterior" and "superior" mean what they say
# --------------------------------------------------------------------------
def _ras_affine(array: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """The RAS+ affine of the reoriented volume, needed to place an external mesh."""
    return np.asarray(
        nib.as_closest_canonical(nib.Nifti1Image(array, affine)).affine, dtype=np.float64
    )


def _to_ras(array: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorient to RAS+ and return the array with its (permuted) voxel sizes.

    Working in a known orientation is not cosmetic here: the coronal slice, the
    camera azimuth and the "superior at the top" convention are all statements
    about anatomical axes, and ATM'22 files are stored LPS.
    """
    image = nib.as_closest_canonical(nib.Nifti1Image(array, affine))
    return np.asanyarray(image.dataobj), np.asarray(image.header.get_zooms(), dtype=float)


def _load_case(case_id: str, overrides: dict) -> dict:
    ct_path = DATA_ROOT / "imagesTr" / f"ATM_{case_id}_0000.nii.gz"
    label_path = DATA_ROOT / "labelsTr" / f"ATM_{case_id}_0000.nii.gz"
    for path in (ct_path, label_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}")

    ct_image = nib.load(str(ct_path))
    hu = np.asarray(_load_hu(ct_path, overrides.get(case_id)), dtype=np.float32)
    ct, zooms = _to_ras(hu, ct_image.affine)

    label_image = nib.load(str(label_path))
    mask_native = np.asanyarray(label_image.dataobj) > 0
    mask, _ = _to_ras(mask_native.astype(np.uint8), label_image.affine)
    mask = mask.astype(bool)

    lung = None
    lung_path = DATA_ROOT / "lungTr" / f"ATM_{case_id}_lung.nii.gz"
    if lung_path.exists():
        lung_image = nib.load(str(lung_path))
        lung_ras, _ = _to_ras(
            (np.asanyarray(lung_image.dataobj) > 0).astype(np.uint8), lung_image.affine
        )
        lung = lung_ras.astype(bool)

    # Branch depth from the project's ATM'22 reference parse, restored to full
    # volume coordinates (the parser works on a cropped bounding box).
    cropped_generation, branch_generation, crop = branch_generation_labels(
        mask, minimum_branch_voxels=5
    )
    generation = np.full(mask.shape, -1, dtype=np.int16)
    generation[crop] = cropped_generation

    return {
        "case_id": case_id,
        "ct": ct,
        "mask": mask,
        "lung": lung,
        "generation": generation,
        "zooms": zooms,
        "ras_affine": _ras_affine(mask_native.astype(np.uint8), label_image.affine),
        "branch_count": int(branch_generation.size),
        "max_generation": int(branch_generation.max()) if branch_generation.size else 0,
    }


# --------------------------------------------------------------------------
# Panel B: z-buffered point-splat render of the tree
# --------------------------------------------------------------------------
def _rotation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Camera rotation for points ordered (R, A, S), i.e. right/anterior/superior.

    Azimuth turns the patient about the superior-inferior axis; elevation then
    raises the camera. Applied in that order, so elevation is about the screen's
    horizontal axis rather than the patient's left-right axis.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    about_s = np.array(
        [[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]]
    )
    about_screen_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(el), -np.sin(el)], [0.0, np.sin(el), np.cos(el)]]
    )
    return about_screen_x @ about_s


def _sample_generations(index: np.ndarray, generation: np.ndarray) -> np.ndarray:
    """Branch depth for surface samples given their fractional voxel indices.

    A nearest lookup, not interpolation: generation is a class label, not a
    quantity to average. Isosurface vertices sit ON the boundary, so rounding
    lands a large fraction of them in unlabelled background and the panel comes
    out speckled; each sample is therefore snapped outward through the
    27-neighbourhood, nearest offset first, until a labelled voxel is found.
    """
    rounded = np.rint(index).astype(np.int32)
    for axis in range(3):
        np.clip(rounded[:, axis], 0, generation.shape[axis] - 1, out=rounded[:, axis])

    offsets = np.array(
        [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)],
        dtype=np.int32,
    )
    offsets = offsets[np.argsort((offsets**2).sum(axis=1))]
    sampled = np.full(rounded.shape[0], -1, dtype=np.int16)
    for offset in offsets:
        pending = np.flatnonzero(sampled < 0)
        if pending.size == 0:
            break
        probe = rounded[pending] + offset
        for axis in range(3):
            np.clip(probe[:, axis], 0, generation.shape[axis] - 1, out=probe[:, axis])
        sampled[pending] = generation[probe[:, 0], probe[:, 1], probe[:, 2]]
    return sampled


def _densify(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices plus face centroids and edge midpoints, with their normals.

    Vertices alone sit roughly one voxel apart, which would need a splat radius
    large enough to round the silhouette off again.
    """
    triangle = vertices[faces]
    triangle_normals = normals[faces]
    points = [vertices, triangle.mean(axis=1)]
    point_normals = [normals, triangle_normals.mean(axis=1)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        points.append(0.5 * (triangle[:, a] + triangle[:, b]))
        point_normals.append(0.5 * (triangle_normals[:, a] + triangle_normals[:, b]))
    stacked = np.concatenate(points).astype(np.float32)
    stacked_normals = np.concatenate(point_normals).astype(np.float32)
    stacked_normals /= np.maximum(
        np.linalg.norm(stacked_normals, axis=1, keepdims=True), 1e-6
    )
    return stacked, stacked_normals


def _mesh_points(
    mesh_path: Path, case: dict, *, taubin_iterations: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Surface samples from an externally authored mesh (ITK-SNAP's VTK or STL)."""
    if mesh_path.suffix.lower() == ".vtk":
        vertices, faces, supplied_normals = _load_vtk_polydata(mesh_path)
    else:
        vertices, faces = _load_stl(mesh_path)
        supplied_normals = None

    render_vertices, _, convention = _mesh_to_voxel_frame(vertices, case)

    stats = {
        "mesh_file": mesh_path.name,
        "mesh_convention": convention,
        "mesh_normals": "file" if supplied_normals is not None else "recomputed",
        "surface_vertices": int(vertices.shape[0]),
        "surface_faces": int(faces.shape[0]),
    }

    if taubin_iterations > 0:
        # Smoothing invalidates the supplied normals, so they have to be rebuilt.
        render_vertices = _taubin_smooth(
            render_vertices, faces, iterations=taubin_iterations
        )
        supplied_normals = None
        stats["taubin_iterations"] = int(taubin_iterations)
        stats["mesh_normals"] = "recomputed"

    if supplied_normals is not None:
        # The mesh was rotated only by axis flips into the render frame, so the
        # file's normals need the same flips -- recovered from the transform that
        # _mesh_to_voxel_frame chose rather than assumed.
        sign = np.array([-1.0, -1.0, 1.0], np.float32) if convention == "LPS" else np.ones(3, np.float32)
        axis_scale = np.sign(np.diag(case["ras_affine"])[:3]).astype(np.float32)
        normals = supplied_normals * sign * axis_scale
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    else:
        normals = _vertex_normals(render_vertices, faces)

    # A wrong global sign lights the inside of the tree and reads as a
    # photographic negative, so it is checked rather than trusted.
    outward = render_vertices - render_vertices.mean(axis=0)
    if float(np.einsum("ij,ij->", normals, outward)) < 0:
        normals = -normals

    points, point_normals = _densify(render_vertices, faces, normals)
    sample_index = points / case["zooms"].astype(np.float32)
    sample_generation = _sample_generations(sample_index, case["generation"])
    stats["surface_samples"] = int(points.shape[0])
    stats["samples_without_generation"] = int((sample_generation < 0).sum())
    return points, point_normals, sample_generation, stats


def _vtk_tokens(text: str, start: int, count: int, dtype) -> np.ndarray:
    """Read exactly ``count`` whitespace-separated numbers starting at ``start``.

    ``str.split`` with a maxsplit stops once it has what it needs, so this does
    not tokenise the remaining megabytes of the file for every section.
    """
    values = text[start:].split(None, count)[:count]
    if len(values) < count:
        raise SystemExit(f"Truncated VTK section: wanted {count} values, found {len(values)}.")
    return np.asarray(values, dtype=dtype)


def _strips_to_triangles(connectivity: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Expand VTK triangle strips into independent triangles.

    Winding alternates along a strip. It is preserved here for tidiness, but it
    does not matter for this figure: ITK-SNAP writes explicit per-vertex NORMALS
    and those are used in preference to anything recomputed from the topology.
    """
    triangles: list[np.ndarray] = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        strip = connectivity[start:stop]
        if strip.size < 3:
            continue
        first, second, third = strip[:-2], strip[1:-1], strip[2:]
        block = np.column_stack([first, second, third])
        block[1::2] = block[1::2][:, [1, 0, 2]]
        triangles.append(block)
    if not triangles:
        raise SystemExit("VTK file contained no triangles.")
    return np.concatenate(triangles).astype(np.int64)


def _load_vtk_polydata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read legacy ASCII VTK PolyData, including the VTK 5.1 cell layout.

    ITK-SNAP's ``Export as Surface Mesh`` writes version 5.1, which replaced the
    old inline ``<count> i0 i1 i2`` cell records with separate ``OFFSETS`` and
    ``CONNECTIVITY`` arrays; a reader written against the pre-5.1 spec silently
    misparses it. Both layouts are handled. Cells arrive as TRIANGLE_STRIPS
    rather than POLYGONS, and POINT_DATA carries per-vertex NORMALS.
    """
    import re

    text = path.read_text(errors="ignore")
    if "POLYDATA" not in text:
        raise SystemExit(f"{path} is not a VTK PolyData file.")
    if re.search(r"^BINARY\s*$", text[:4096], re.M):
        raise SystemExit(
            f"{path} is binary VTK, which this reader does not handle. Re-export as "
            "ASCII, or export STL instead."
        )

    match = re.search(r"^POINTS\s+(\d+)\s+\w+\s*\n", text, re.M)
    if match is None:
        raise SystemExit(f"{path} has no POINTS section.")
    point_count = int(match.group(1))
    vertices = _vtk_tokens(text, match.end(), 3 * point_count, np.float32).reshape(-1, 3)

    match = re.search(r"^(POLYGONS|TRIANGLE_STRIPS)\s+(\d+)\s+(\d+)\s*\n", text, re.M)
    if match is None:
        raise SystemExit(f"{path} has no POLYGONS or TRIANGLE_STRIPS section.")
    kind, cell_count, total = match.group(1), int(match.group(2)), int(match.group(3))

    offsets_match = re.compile(r"\s*OFFSETS\s+\S+\s*\n").match(text, match.end())
    if offsets_match is not None:
        # VTK 5.1 redefined the header: the first number is the length of the
        # OFFSETS array (cells + 1), NOT the cell count as in earlier versions.
        # Reading it as a cell count overruns into the CONNECTIVITY keyword.
        offsets = _vtk_tokens(text, offsets_match.end(), cell_count, np.int64)
        connectivity_match = re.compile(r"\s*CONNECTIVITY\s+\S+\s*\n").search(
            text, offsets_match.end()
        )
        if connectivity_match is None:
            raise SystemExit(f"{path}: OFFSETS present but CONNECTIVITY missing.")
        connectivity = _vtk_tokens(text, connectivity_match.end(), total, np.int64)
        if int(offsets[-1]) != total:
            raise SystemExit(
                f"{path}: OFFSETS ends at {int(offsets[-1])} but CONNECTIVITY holds "
                f"{total} entries. The cell arrays do not agree."
            )
    else:  # pre-5.1: <count> i0 i1 ... repeated
        flat = _vtk_tokens(text, match.end(), total, np.int64)
        offsets = np.zeros(cell_count + 1, dtype=np.int64)
        pieces: list[np.ndarray] = []
        cursor = 0
        for cell in range(cell_count):
            size = int(flat[cursor])
            pieces.append(flat[cursor + 1 : cursor + 1 + size])
            offsets[cell + 1] = offsets[cell] + size
            cursor += 1 + size
        connectivity = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)

    faces = (
        _strips_to_triangles(connectivity, offsets)
        if kind == "TRIANGLE_STRIPS"
        else connectivity.reshape(-1, 3)
    )

    normals = None
    normals_match = re.search(r"^NORMALS\s+\S+\s+\w+\s*\n", text, re.M)
    if normals_match is not None:
        normals = _vtk_tokens(
            text, normals_match.end(), 3 * point_count, np.float32
        ).reshape(-1, 3)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    return vertices, faces, normals


def _load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read binary or ASCII STL, weld duplicate vertices, return (vertices, faces).

    STL is a triangle soup: every triangle carries its own three vertex copies and
    a single facet normal. Rendering that directly gives flat-shaded facets, which
    is the look the isosurface was meant to avoid, so the vertices are welded on a
    rounded coordinate key and per-vertex normals are recomputed from the welded
    topology afterwards.
    """
    raw = path.read_bytes()
    record = np.dtype(
        [("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    expected = 0
    if len(raw) >= 84:
        expected = int.from_bytes(raw[80:84], "little")

    if len(raw) == 84 + expected * record.itemsize and expected > 0:
        triangles = np.frombuffer(raw, dtype=record, count=expected, offset=84)["vertices"]
    else:  # ASCII STL (vtkSTLWriter's default file type)
        import re

        text = raw.decode("ascii", errors="ignore")
        found = re.findall(
            r"vertex\s+(\S+)\s+(\S+)\s+(\S+)",
            text,
        )
        if not found:
            raise SystemExit(f"{path} is neither a binary nor an ASCII STL.")
        triangles = np.asarray(found, dtype=np.float32).reshape(-1, 3, 3)

    flat = np.ascontiguousarray(triangles.reshape(-1, 3), dtype=np.float32)
    # Weld on a 1 um key. STL stores float32, so bit-identical duplicates are the
    # norm and an exact key would already work; rounding guards against writers
    # that round-trip through text.
    key = np.rint(flat.astype(np.float64) / 1e-3).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    vertices = flat[first]
    faces = inverse.reshape(-1, 3).astype(np.int64)
    # Degenerate triangles survive welding and produce zero-area normals.
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return vertices, faces[keep]


def _mesh_to_voxel_frame(
    vertices: np.ndarray, case: dict
) -> tuple[np.ndarray, np.ndarray, str]:
    """Map mesh vertices into the same frame the voxel renderer uses.

    ITK-SNAP writes the mesh in the image's physical space, but whether that space
    is RAS or LPS depends on the reader path, and guessing wrong mirrors the
    patient left-right -- an error that looks entirely plausible in a picture of a
    roughly symmetric tree, which is exactly why it is resolved by measurement
    rather than assumption. Both conventions are tried and the one that actually
    lands inside the annotation is kept.
    """
    inverse_affine = np.linalg.inv(case["ras_affine"])
    homogeneous = np.column_stack([vertices, np.ones(len(vertices), dtype=np.float64)])
    shape = np.asarray(case["mask"].shape)

    best = None
    for name, flip in (("RAS", np.array([1.0, 1.0, 1.0])), ("LPS", np.array([-1.0, -1.0, 1.0]))):
        index = (homogeneous * np.append(flip, 1.0)) @ inverse_affine.T
        index = index[:, :3]
        inside = np.all((index >= -2) & (index <= shape + 1), axis=1).mean()
        if best is None or inside > best[0]:
            best = (inside, index, name)

    inside, index, name = best
    if inside < 0.9:
        raise SystemExit(
            f"Only {inside:.1%} of mesh vertices land inside the image under either "
            "RAS or LPS. The mesh does not match this case."
        )
    return index.astype(np.float32) * case["zooms"].astype(np.float32), index, name


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals. Not normalising the face normal is the
    weighting: a triangle's cross product already has twice its area."""
    triangle = vertices[faces]
    face_normal = np.cross(
        triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    )
    flat = faces.ravel()
    normals = np.zeros_like(vertices)
    for axis in range(3):
        normals[:, axis] = np.bincount(
            flat, weights=np.repeat(face_normal[:, axis], 3), minlength=vertices.shape[0]
        )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    return normals.astype(np.float32)


def _taubin_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int,
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """Taubin lambda/mu mesh smoothing over the uniform (umbrella) Laplacian.

    ``mu`` slightly more negative than ``lam`` is the whole trick: the shrinking
    pass and the inflating pass cancel to first order, so the surface loses its
    high-frequency stair steps without losing volume. A plain Laplacian would
    smooth just as well and shrink the mesh toward its skeleton, which on an
    airway tree means the one-voxel branches vanish first.
    """
    import scipy.sparse as sp

    count = vertices.shape[0]
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    adjacency = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(count, count),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    average = sp.diags((1.0 / degree).astype(np.float32)) @ adjacency

    smoothed = vertices.astype(np.float32)
    for step in range(iterations):
        weight = lam if step % 2 == 0 else mu
        smoothed = smoothed + weight * (average @ smoothed - smoothed)
    return smoothed.astype(np.float32)


def _surface_points(
    mask: np.ndarray,
    generation: np.ndarray,
    zooms: np.ndarray,
    *,
    smooth_mm: float,
    taubin_iterations: int,
    clip_mm: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Isosurface samples with true interpolated normals, ITK-SNAP style.

    Splatting voxels draws the voxel grid, which is what makes a naive render look
    blocky: at 0.26 mm/px one 0.82 mm voxel is six pixels across, so the stair
    steps are resolved rather than hidden. More pixels make them sharper, not
    smoother. The fix is to stop drawing cubes and draw an isosurface instead.

    The scalar field is a *signed distance*, not a blurred binary mask. This
    matters more here than anywhere else in the document: blurring a binary mask
    and thresholding at 0.5 erases a one-voxel-diameter tube outright, and 41% of
    this tree's voxels sit at a local radius of 1 mm or less. A signed distance
    field keeps its zero crossing at the true boundary under smoothing, so thin
    branches survive. The retention check returned in the stats dictionary exists
    to prove that on every run rather than assume it.

    Returns ``(points_mm, normals, generations, stats)``. Points are the marching
    cubes vertices plus per-face centroids and edge midpoints, which quadruples
    the sample density so the splat covers without needing a large radius.
    """
    from skimage.measure import marching_cubes

    crop = tuple(
        slice(max(0, int(s.start) - 4), min(int(dim), int(s.stop) + 4))
        for s, dim in zip(ndi.find_objects(mask.astype(np.uint8))[0], mask.shape)
    )
    local = mask[crop]

    inside = ndi.distance_transform_edt(local, sampling=zooms)
    outside = ndi.distance_transform_edt(~local, sampling=zooms)
    field = np.clip(inside - outside, -clip_mm, clip_mm).astype(np.float32)
    if smooth_mm > 0:
        field = ndi.gaussian_filter(field, sigma=smooth_mm / zooms, mode="nearest")

    retained = field > 0
    stats = {
        "voxels_before_smoothing": int(local.sum()),
        "voxels_inside_isosurface": int((retained & local).sum()),
        "voxel_retention": round(float((retained & local).sum() / max(1, local.sum())), 4),
        "components_before": int(ndi.label(local, np.ones((3, 3, 3)))[1]),
        "components_after": int(ndi.label(retained, np.ones((3, 3, 3)))[1]),
    }

    vertices, faces, normals, _ = marching_cubes(
        field, level=0.0, spacing=tuple(float(z) for z in zooms)
    )
    vertices = vertices.astype(np.float32)
    normals = normals.astype(np.float32)

    # Taubin smoothing of the MESH, not the field. Field smoothing is capped by
    # connectivity -- past about 0.35 mm on this cohort it pinches thin branches
    # off, and a fragmented tree is the one artefact this document cannot show.
    # Moving vertices cannot change topology, so the residual stair-stepping from
    # slice-by-slice annotation can be removed without that risk. The alternating
    # positive/negative pass is what stops it shrinking the tree, which a plain
    # Laplacian would do -- and shrinkage would eat the distal branches first.
    if taubin_iterations > 0:
        vertices = _taubin_smooth(vertices, faces, iterations=taubin_iterations)
        smoothed_normals = _vertex_normals(vertices, faces)
        if float(np.einsum("ij,ij->", smoothed_normals, normals)) < 0:
            smoothed_normals = -smoothed_normals
        normals = smoothed_normals
        stats["taubin_iterations"] = int(taubin_iterations)

    points, point_normals = _densify(vertices, faces, normals)

    # Branch depth of the voxel each sample falls in. Nearest lookup rather than
    # interpolation: generation is a class label, not a quantity to average.
    #
    # Isosurface vertices sit ON the boundary, so rounding lands a large fraction
    # of them in unlabelled background and the panel comes out speckled. Snap each
    # sample outward through the 27-neighbourhood, nearest offset first, until a
    # labelled voxel is found.
    sample_generation = _sample_generations(points / zooms, generation[crop])
    stats["samples_without_generation"] = int((sample_generation < 0).sum())

    stats["surface_vertices"] = int(vertices.shape[0])
    stats["surface_faces"] = int(faces.shape[0])
    stats["surface_samples"] = int(points.shape[0])
    return points, point_normals, sample_generation.astype(np.int16), stats


def _project_surface(
    points: np.ndarray,
    normals: np.ndarray,
    generation: np.ndarray,
    *,
    azimuth: float,
    elevation: float,
    px_mm: float,
    splat_radius_px: int,
    margin_px: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-buffer the isosurface samples, carrying their normals into the buffer.

    Shading from stored normals rather than from the gradient of the depth map is
    the second half of the fix: a depth-gradient normal reproduces whatever steps
    the depth map has, so it re-creates the blockiness it was meant to hide.
    """
    rotation = _rotation(azimuth, elevation).astype(np.float32)
    centred = points - points.mean(axis=0)
    rotated = centred @ rotation.T
    rotated_normals = normals @ rotation.T

    screen_x, depth, screen_y = rotated[:, 0], rotated[:, 1], rotated[:, 2]
    u = (screen_x.max() - screen_x) / px_mm + margin_px
    v = (screen_y.max() - screen_y) / px_mm + margin_px
    height = int(np.ceil(v.max())) + margin_px
    width = int(np.ceil(u.max())) + margin_px

    # Back faces carry no information and only fight the front surface for pixels.
    front = rotated_normals[:, 1] > -0.05
    if front.sum() > 1000:
        depth, u, v = depth[front], u[front], v[front]
        rotated_normals = rotated_normals[front]
        generation = generation[front]

    order = np.argsort(depth)
    rows = np.clip(v[order].astype(np.int32), 0, height - 1)
    cols = np.clip(u[order].astype(np.int32), 0, width - 1)

    depth_image = np.full((height, width), -np.inf, dtype=np.float32)
    depth_image[rows, cols] = depth[order]
    normal_image = np.zeros((height, width, 3), dtype=np.float32)
    normal_image[rows, cols] = rotated_normals[order]
    generation_image = np.full((height, width), -1, dtype=np.int16)
    generation_image[rows, cols] = generation[order]

    grid = np.ogrid[-splat_radius_px : splat_radius_px + 1, -splat_radius_px : splat_radius_px + 1]
    disc = (grid[0] ** 2 + grid[1] ** 2) <= splat_radius_px**2
    filled = ndi.maximum_filter(depth_image, footprint=disc, mode="nearest")

    painted = np.isfinite(depth_image)
    _, nearest = ndi.distance_transform_edt(~painted, return_indices=True)
    normal_image = normal_image[nearest[0], nearest[1]]
    generation_image = generation_image[nearest[0], nearest[1]]

    covered = np.isfinite(filled)
    filled[~covered] = np.inf
    generation_image[~covered] = -1
    # Mild blur removes the faceting the nearest-neighbour fill leaves behind.
    normal_image = ndi.gaussian_filter(normal_image, sigma=(1.0, 1.0, 0))
    normal_image /= np.maximum(np.linalg.norm(normal_image, axis=-1, keepdims=True), 1e-6)
    return filled, generation_image, normal_image


def _shade_normals(
    normal: np.ndarray, depth: np.ndarray, *, specular_strength: float = 0.28
) -> tuple[np.ndarray, np.ndarray]:
    """Blinn-Phong shading from stored surface normals, plus a depth cue.

    Returns ``(shade, specular)``: the first multiplies the surface colour, the
    second is added on top as white highlight. Diffuse-only shading is what makes
    a render look like flat plastic -- VTK, and therefore ITK-SNAP, lights with a
    specular term, and on a tube-shaped object the highlight running along each
    branch is most of what communicates that it is round.
    """
    light = np.array([-0.45, 0.55, 0.70], dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip(normal @ light, 0.0, 1.0)

    ambient = 0.34
    shade = ambient + (1.0 - ambient) * lambert

    # Camera looks along -y (it sits at +y), so the view vector is +y.
    view = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    half = light + view
    half /= np.linalg.norm(half)
    specular = specular_strength * np.clip(normal @ half, 0.0, 1.0) ** 24.0

    covered = np.isfinite(depth)
    finite = depth[covered]
    if finite.size and np.ptp(finite) > 0:
        relative = np.zeros_like(depth, dtype=np.float32)
        relative[covered] = (depth[covered] - finite.min()) / np.ptp(finite)
        cue = 0.78 + 0.22 * relative
        shade = shade * cue
        specular = specular * cue
    return (
        np.clip(shade, 0.0, 1.0).astype(np.float32),
        np.clip(specular, 0.0, 1.0).astype(np.float32),
    )


def _project(
    mask: np.ndarray,
    generation: np.ndarray,
    zooms: np.ndarray,
    *,
    azimuth: float,
    elevation: float,
    px_mm: float,
    margin_px: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth_mm, generation) images; +inf / -1 where nothing was hit.

    Depth increases toward the camera, so the nearest surface has the LARGEST
    value. Painting in ascending depth order therefore leaves the nearest write
    last, which is an exact z-buffer.
    """
    shell = mask & ~ndi.binary_erosion(mask)
    index = np.nonzero(shell)
    points = np.column_stack(index).astype(np.float32) * zooms.astype(np.float32)
    points -= points.mean(axis=0)
    points = points @ _rotation(azimuth, elevation).T.astype(np.float32)

    screen_x, depth, screen_y = points[:, 0], points[:, 1], points[:, 2]
    # Patient's right (+R) is drawn on the viewer's LEFT, as when facing someone;
    # superior (+S) is drawn at the top, so both axes are negated.
    u = (screen_x.max() - screen_x) / px_mm + margin_px
    v = (screen_y.max() - screen_y) / px_mm + margin_px
    height = int(np.ceil(v.max())) + margin_px
    width = int(np.ceil(u.max())) + margin_px

    order = np.argsort(depth)  # far first; the nearest write lands last
    rows = np.clip(v[order].astype(np.int32), 0, height - 1)
    cols = np.clip(u[order].astype(np.int32), 0, width - 1)

    depth_image = np.full((height, width), -np.inf, dtype=np.float32)
    depth_image[rows, cols] = depth[order]
    generation_image = np.full((height, width), -1, dtype=np.int16)
    generation_image[rows, cols] = generation[index][order]

    # One voxel projects to roughly this many pixels; expand each splat to that
    # size. A maximum filter over the depth is a disc splat done in image space
    # (the nearest surface still wins, because "nearest" is now the maximum).
    radius = max(1, int(np.ceil(0.62 * float(zooms.max()) / px_mm)))
    grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disc = (grid[0] ** 2 + grid[1] ** 2) <= radius**2
    filled = ndi.maximum_filter(depth_image, footprint=disc, mode="nearest")

    # Colour follows the pixel the depth came from: fill each newly covered pixel
    # from its nearest painted neighbour rather than interpolating a class label.
    painted = np.isfinite(depth_image)
    _, nearest = ndi.distance_transform_edt(~painted, return_indices=True)
    generation_image = generation_image[nearest[0], nearest[1]]

    covered = np.isfinite(filled)
    filled[~covered] = np.inf
    generation_image[~covered] = -1
    return filled, generation_image


def _shade(depth: np.ndarray, px_mm: float) -> np.ndarray:
    """Lambertian shading from the gradient of the depth map, in [0, 1].

    The depth map *is* a height field over the visible surface, so its gradient
    gives a usable normal without ever building a mesh. Silhouette edges produce
    near-infinite gradients, hence the blur and the clip.
    """
    covered = np.isfinite(depth)
    if not covered.any():
        return np.zeros_like(depth, dtype=np.float32)

    height = np.where(covered, depth, np.nan).astype(np.float32)
    # Fill the background with the nearest surface height before blurring, so the
    # silhouette does not shade against an artificial cliff.
    _, nearest = ndi.distance_transform_edt(~covered, return_indices=True)
    height = height[nearest[0], nearest[1]]
    height = ndi.gaussian_filter(height, sigma=1.6)

    d_v, d_u = np.gradient(height, px_mm)
    limit = 3.0
    d_u = np.clip(d_u, -limit, limit)
    d_v = np.clip(d_v, -limit, limit)
    # v runs downward on screen, so its gradient sign is flipped to give an
    # outward normal in a right-handed screen frame.
    normal = np.stack([-d_u, d_v, np.ones_like(d_u)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    light = np.array([-0.45, 0.55, 0.70], dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip(normal @ light, 0.0, 1.0)

    ambient = 0.42
    shade = ambient + (1.0 - ambient) * lambert
    # Mild depth cue: the far side of the tree recedes rather than competing with
    # the near side for attention.
    finite = depth[covered]
    if finite.size and np.ptp(finite) > 0:
        span = np.ptp(finite)
        relative = np.zeros_like(depth, dtype=np.float32)
        relative[covered] = (depth[covered] - finite.min()) / span
        shade *= 0.72 + 0.28 * relative
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def _colourise(
    depth: np.ndarray,
    generation: np.ndarray,
    shade: np.ndarray,
    *,
    mode: str,
    proximal_max_generation: int,
    max_generation: int,
    specular: np.ndarray | None = None,
) -> np.ndarray:
    covered = np.isfinite(depth)
    base = np.zeros(depth.shape + (3,), dtype=np.float32)

    if mode == "plain":
        base[...] = np.asarray(to_rgb(PLAIN_COLOUR), dtype=np.float32)
    elif mode == "depth":
        proximal = np.asarray(to_rgb(PROXIMAL_COLOUR), dtype=np.float32)
        distal = np.asarray(to_rgb(DISTAL_COLOUR), dtype=np.float32)
        is_distal = generation > proximal_max_generation
        base[...] = proximal
        base[is_distal] = distal
        # Unreachable branches keep the plain tone rather than being silently
        # counted as proximal, which is the same convention as the parser.
        base[generation < 0] = np.asarray(to_rgb(PLAIN_COLOUR), dtype=np.float32)
    elif mode == "gen":
        ramp = LinearSegmentedColormap.from_list(
            "airway_depth", [PROXIMAL_COLOUR, "#3f8f8f", "#c9a227", DISTAL_COLOUR]
        )
        top = max(1, max_generation)
        fraction = np.clip(generation.astype(np.float32) / top, 0.0, 1.0)
        base = ramp(fraction)[..., :3].astype(np.float32)
        base[generation < 0] = np.asarray(to_rgb(PLAIN_COLOUR), dtype=np.float32)
    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"Unknown colouring mode {mode!r}")

    rgba = np.zeros(depth.shape + (4,), dtype=np.float32)
    lit = base * shade[..., None]
    if specular is not None:
        lit = lit + specular[..., None]
    rgba[..., :3] = np.clip(lit, 0.0, 1.0)
    rgba[..., 3] = covered.astype(np.float32)
    return rgba


def _downsample(rgba: np.ndarray, factor: int) -> np.ndarray:
    """Box-downsample premultiplied RGBA, which is the anti-aliasing step."""
    if factor <= 1:
        return rgba
    height = (rgba.shape[0] // factor) * factor
    width = (rgba.shape[1] // factor) * factor
    block = rgba[:height, :width].reshape(
        height // factor, factor, width // factor, factor, 4
    )
    premultiplied = block[..., :3] * block[..., 3:4]
    alpha = block[..., 3:4].mean(axis=(1, 3))
    colour = premultiplied.mean(axis=(1, 3))
    out = np.zeros(alpha.shape[:2] + (4,), dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[..., :3] = np.where(alpha > 0, colour / np.maximum(alpha, 1e-6), 0.0)
    out[..., 3] = alpha[..., 0]
    return np.clip(out, 0.0, 1.0)


def _save_image(rgba: np.ndarray, pdf_dir: Path, png_dir: Path, stem: str) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    height, width = rgba.shape[:2]
    figure = plt.figure(figsize=(PANEL_WIDTH_IN, PANEL_WIDTH_IN * height / width))
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_axis_off()
    axes.imshow(rgba, interpolation="none")
    dots_per_inch = width / PANEL_WIDTH_IN
    for directory, suffix in ((pdf_dir, "pdf"), (png_dir, "png")):
        figure.savefig(
            directory / f"{stem}.{suffix}",
            dpi=dots_per_inch,
            transparent=True,
            pad_inches=0,
        )
    plt.close(figure)


def render_tree_panels(case: dict, args: argparse.Namespace) -> dict:
    supersample = max(1, args.supersample)
    px_mm = args.px_mm / supersample
    surface_stats: dict = {}
    mesh_smooth = args.mesh_smooth
    if mesh_smooth is None:
        mesh_smooth = 0 if args.mesh is not None else 16

    if args.mesh is not None:
        points, normals, sample_generation, surface_stats = _mesh_points(
            args.mesh, case, taubin_iterations=mesh_smooth
        )
        print(
            f"  {args.mesh.name}: {surface_stats['surface_faces']:,} faces, "
            f"{surface_stats['mesh_convention']} physical space",
            flush=True,
        )
        depth, generation, normal = _project_surface(
            points,
            normals,
            sample_generation,
            azimuth=args.azimuth,
            elevation=args.elevation,
            px_mm=px_mm,
            splat_radius_px=max(1, int(round(0.35 * float(case["zooms"].max()) / px_mm))),
        )
        shade, specular = _shade_normals(normal, depth)
    elif args.render_mode == "surface":
        points, normals, sample_generation, surface_stats = _surface_points(
            case["mask"],
            case["generation"],
            case["zooms"],
            smooth_mm=args.smooth_mm,
            taubin_iterations=mesh_smooth,
        )
        print(
            f"  isosurface: {surface_stats['surface_faces']:,} faces, "
            f"{surface_stats['voxel_retention'] * 100:.2f}% of airway voxels retained, "
            f"components {surface_stats['components_before']} -> "
            f"{surface_stats['components_after']}",
            flush=True,
        )
        # Smoothing that fragments the tree is disqualifying in a document whose
        # subject is distal branch completeness: the figure would show a broken
        # tree that the annotation does not contain. Measured on ATM_001, the
        # break happens between sigma 0.35 and 0.45 mm.
        if (
            surface_stats["components_after"] > surface_stats["components_before"]
            or surface_stats["voxel_retention"] < 0.995
        ):
            print(
                f"  WARNING: --smooth-mm {args.smooth_mm} pinches the tree apart "
                f"(retention {surface_stats['voxel_retention']:.4f}, components "
                f"{surface_stats['components_before']} -> "
                f"{surface_stats['components_after']}). Lower it.",
                flush=True,
            )
        depth, generation, normal = _project_surface(
            points,
            normals,
            sample_generation,
            azimuth=args.azimuth,
            elevation=args.elevation,
            px_mm=px_mm,
            splat_radius_px=max(1, int(round(0.35 * float(case["zooms"].max()) / px_mm))),
        )
        shade, specular = _shade_normals(normal, depth)
    else:
        depth, generation = _project(
            case["mask"],
            case["generation"],
            case["zooms"],
            azimuth=args.azimuth,
            elevation=args.elevation,
            px_mm=px_mm,
        )
        shade = _shade(depth, px_mm)
        specular = None

    for mode, stem in (
        ("plain", "airway_overview_tree_plain"),
        ("depth", "airway_overview_tree_depth"),
        ("gen", "airway_overview_tree_gen"),
    ):
        rgba = _colourise(
            depth,
            generation,
            shade,
            mode=mode,
            proximal_max_generation=args.proximal_max_generation,
            max_generation=case["max_generation"],
            specular=specular,
        )
        _save_image(_downsample(rgba, supersample), args.pdf_output_dir, args.png_output_dir, stem)

    return {
        "render_pixels": [int(depth.shape[0]), int(depth.shape[1])],
        "supersample": supersample,
        "px_mm_final": args.px_mm,
        "azimuth_deg": args.azimuth,
        "elevation_deg": args.elevation,
        # An imported mesh brings its own surface, so the extraction settings did
        # not run and must not be recorded as though they had.
        "render_mode": "mesh" if args.mesh is not None else args.render_mode,
        "smooth_mm": (
            args.smooth_mm
            if args.mesh is None and args.render_mode == "surface"
            else None
        ),
        **surface_stats,
    }


# --------------------------------------------------------------------------
# Panel A: coronal CT slice with equal-magnification insets
# --------------------------------------------------------------------------
def _choose_coronal_slice(mask: np.ndarray, generation: np.ndarray) -> int:
    """Anterior-posterior index of the slice carrying the most proximal airway.

    Maximising *proximal* rather than total airway lands the slice on the trachea
    and carina, which is the view a reader recognises. Maximising total airway
    tends to pick a posterior slice dense with small branches.
    """
    proximal = mask & (generation >= 0) & (generation <= 2)
    counts = proximal.sum(axis=(0, 2))
    if counts.max() == 0:
        counts = mask.sum(axis=(0, 2))
    return int(np.argmax(counts))


def _slice_arrays(case: dict, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coronal slice as displayed: rows superior->inferior, columns patient right->left."""
    ct = case["ct"][:, index, :]
    mask = case["mask"][:, index, :]
    generation = case["generation"][:, index, :]
    # Array axes are (R, S); display wants (S descending, R descending) so that
    # superior is at the top and the patient's right is on the image's left.
    transform = lambda a: a.T[::-1, ::-1]  # noqa: E731
    return transform(ct), transform(mask), transform(generation)


def _inset_centres(
    mask: np.ndarray,
    generation: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    row_mm: float,
    col_mm: float,
    max_distal_radius_mm: float = 1.6,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Millimetre centres for the trachea inset and the distal-branch inset.

    The distal candidate is the largest *in-plane connected component* of distal
    voxels, not simply the deepest or the most peripheral voxel. A coronal slice
    cuts most distal branches transversely, leaving two- or three-voxel specks
    that illustrate nothing; the largest component is a branch that happens to run
    within the plane, which is the one a reader can actually see tapering.

    It is additionally constrained to be *thin*: a component whose in-plane radius
    exceeds ``max_distal_radius_mm`` is rejected however deep the parser calls it.
    BFS depth lags anatomy on an asymmetric tree, so "generation >= 5" alone can
    return a six-millimetre lobar bronchus -- which would quietly destroy the only
    claim this panel makes.
    """
    left, right, bottom, top = extent
    height, width = mask.shape

    def _to_mm(row: float, col: float) -> tuple[float, float]:
        return (
            left + (col + 0.5) / width * (right - left),
            top + (row + 0.5) / height * (bottom - top),
        )

    trachea = None
    proximal = mask & (generation == 0)
    if proximal.any():
        rows, cols = np.nonzero(proximal)
        trachea = _to_mm(rows.mean(), cols.mean())

    # In-plane radius of every airway pixel, in millimetres.
    radius = ndi.distance_transform_edt(mask, sampling=(row_mm, col_mm))

    distal = None
    for floor, radius_limit in ((5, max_distal_radius_mm), (4, max_distal_radius_mm), (4, 2.4)):
        candidate = mask & (generation >= floor)
        if candidate.sum() < 4:
            continue
        components, count = ndi.label(candidate, structure=np.ones((3, 3)))
        if count == 0:
            continue
        best_label, best_size = 0, 0
        for label in range(1, count + 1):
            selected = components == label
            size = int(selected.sum())
            if size < 8 or radius[selected].max() > radius_limit:
                continue
            if size > best_size:
                best_label, best_size = label, size
        if best_label:
            rows, cols = np.nonzero(components == best_label)
            distal = _to_mm(float(rows.mean()), float(cols.mean()))
            break
    return trachea, distal


def _assign_corners(
    targets: list[tuple[float, float] | None],
    *,
    left_mm: float,
    right_mm: float,
    top_mm: float,
    bottom_mm: float,
    inset_fraction: float,
    aspect: float,
    keepout: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, float, float, float] | None]:
    """Place each inset in the free corner nearest its subject.

    Nearest-corner placement is what keeps the leader lines short. Fixed corners
    produced a leader that crossed the entire slice, which is worse than no
    indicator at all.

    The height fraction is derived from the panel aspect so the inset axes is
    SQUARE on the page. Without this, ``imshow``'s equal aspect silently widens
    the data range of a non-square axes, and the indicator rectangle then stops
    matching the magnified content -- and the two insets stop being comparable,
    which is the one property this panel must have.
    """
    height_fraction = inset_fraction / aspect
    margin = 0.022
    # Keyed (row, column) with row 0 at the TOP, matching how the targets below
    # are measured. Values are the axes-fraction origin matplotlib wants, whose
    # y runs upward -- hence the flip on the top row.
    corners = {
        (0, 0): (margin, 1.0 - margin - height_fraction),
        (0, 1): (1.0 - margin - inset_fraction, 1.0 - margin - height_fraction),
        (1, 0): (margin, margin),
        (1, 1): (1.0 - margin - inset_fraction, margin),
    }

    def _centre_of(key: tuple[int, int]) -> tuple[float, float]:
        column_centre = margin + inset_fraction / 2 if key[1] == 0 else 1.0 - margin - inset_fraction / 2
        row_centre = margin + height_fraction / 2 if key[0] == 0 else 1.0 - margin - height_fraction / 2
        return column_centre, row_centre

    positions: list[tuple[float, float] | None] = []
    for centre in targets:
        if centre is None:
            positions.append(None)
            continue
        positions.append(
            (
                (centre[0] - left_mm) / (right_mm - left_mm),
                (centre[1] - top_mm) / (bottom_mm - top_mm),
            )
        )

    def _covers(key: tuple[int, int], point: tuple[float, float]) -> bool:
        """Does the inset at ``key`` sit over ``point``'s indicator box?

        An inset that lands on its own target hides the very box it is magnifying,
        and the leader lines vanish underneath it -- which reads as a missing
        annotation rather than as a mistake, so it has to be excluded outright.
        """
        x0, y0 = corners[key]
        left = x0 - keepout[0]
        right = x0 + inset_fraction + keepout[0]
        # corners' y is measured upward; the targets are measured downward.
        top = (1.0 - y0 - height_fraction) - keepout[1]
        bottom = (1.0 - y0) + keepout[1]
        return left <= point[0] <= right and top <= point[1] <= bottom

    present = [i for i, position in enumerate(positions) if position is not None]
    keys = list(corners)
    best_choice, best_score = None, None
    for assignment in itertools.permutations(keys, len(present)):
        if any(
            _covers(key, positions[j])
            for key in assignment
            for j in present
        ):
            continue
        leaders = [(positions[i], _centre_of(key)) for i, key in zip(present, assignment)]
        # Crossing leaders are the artefact this function exists to prevent, so
        # they are rejected outright rather than merely penalised.
        if len(leaders) == 2 and _segments_cross(*leaders[0], *leaders[1]):
            continue
        score = sum(
            float(np.hypot(a[0] - b[0], a[1] - b[1])) for a, b in leaders
        )
        # Prefer diagonally opposite corners. Two insets sharing a row bury both
        # lung apices and sharing a column buries one lung; the diagonal is the
        # only arrangement that leaves the slice mostly visible. Weighted as a
        # strong nudge rather than a rule, so a badly stretched leader can still
        # win if the diagonal is genuinely worse.
        if len({key[0] for key in assignment}) == 1:
            score += 0.30
        if len({key[1] for key in assignment}) == 1:
            score += 0.30
        if best_score is None or score < best_score:
            best_choice, best_score = assignment, score

    if best_choice is None:  # every arrangement crossed; fall back to nearest
        best_choice = tuple(
            min(keys, key=lambda key: float(np.hypot(*np.subtract(positions[i], _centre_of(key)))))
            for i in present
        )

    placements: list[tuple[float, float, float, float] | None] = [None] * len(targets)
    for index, key in zip(present, best_choice):
        x0, y0 = corners[key]
        placements[index] = (x0, y0, inset_fraction, height_fraction)
    return placements


def _segments_cross(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    """True if segments a1-a2 and b1-b2 properly intersect."""

    def orientation(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = orientation(b1, b2, a1), orientation(b1, b2, a2)
    d3, d4 = orientation(a1, a2, b1), orientation(a1, a2, b2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


# Display-space corners, y measured UPWARD: (left/right, bottom/top).
_CORNERS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


def _leader_corners(
    target: tuple[float, float], inset_centre: tuple[float, float]
) -> list[tuple[int, int]]:
    """The two corners whose leaders form a funnel rather than a cross.

    Both arguments are y-UPWARD display fractions. The correct pair is always the
    corners extreme along the direction *perpendicular* to the line joining the
    two boxes -- the outer tangents of the pair.
    """
    dx = inset_centre[0] - target[0]
    dy = inset_centre[1] - target[1]
    perpendicular = (-dy, dx)
    projected = {
        index: (2 * corner[0] - 1) * perpendicular[0] + (2 * corner[1] - 1) * perpendicular[1]
        for index, corner in _CORNERS.items()
    }
    chosen = (max(projected, key=projected.get), min(projected, key=projected.get))
    return [_CORNERS[index] for index in chosen]


def _cased(linewidth: float) -> list:
    """Path effect giving a stroke an achromatic outline.

    This is what makes the dark-red annotation safe: the coloured core carries
    the hue, the white casing guarantees a luminance edge whatever it crosses.
    """
    from matplotlib import patheffects

    return [
        patheffects.withStroke(linewidth=linewidth + 1.6, foreground=ANNOTATION_CASING),
    ]


def _measure_diameter_mm(
    mask: np.ndarray,
    centre: tuple[float, float],
    half_mm: float,
    extent: tuple[float, float, float, float],
    *,
    row_mm: float,
    col_mm: float,
) -> float | None:
    """Largest inscribed airway diameter within an inset box, in millimetres.

    The maximum of the in-plane Euclidean distance transform is the radius of the
    largest circle that fits inside the cross-section. For a cylinder cut at any
    obliquity the in-plane section is an ellipse whose semi-MINOR axis equals the
    cylinder radius, and the inscribed circle is bounded by that same semi-minor
    axis. The measurement is therefore the true local radius, not an inflated
    oblique width, which is why it is safe to quote as a diameter.
    """
    left, right, bottom, top = extent
    height, width = mask.shape
    col_lo = int(np.floor((centre[0] - half_mm - left) / (right - left) * width))
    col_hi = int(np.ceil((centre[0] + half_mm - left) / (right - left) * width))
    row_lo = int(np.floor((centre[1] - half_mm - top) / (bottom - top) * height))
    row_hi = int(np.ceil((centre[1] + half_mm - top) / (bottom - top) * height))
    col_lo, col_hi = max(0, col_lo), min(width, col_hi)
    row_lo, row_hi = max(0, row_lo), min(height, row_hi)
    window = mask[row_lo:row_hi, col_lo:col_hi]
    if not window.any():
        return None
    # Pad so a branch touching the window edge is not given an infinite radius by
    # the transform running out of background to measure against.
    padded = np.pad(window, 1, mode="constant", constant_values=False)
    radius = ndi.distance_transform_edt(padded, sampling=(row_mm, col_mm))
    return float(2.0 * radius.max())


def _draw_scale_bar(
    axes,
    length_mm: float,
    *,
    colour: str,
    linewidth: float = 2.0,
    x_fraction: float = 0.06,
) -> None:
    """Draw a scale bar in the lower-left of an axes, in data (millimetre) units."""
    x0, x1 = sorted(axes.get_xlim())
    y0, y1 = sorted(axes.get_ylim())
    span_x, span_y = x1 - x0, y1 - y0
    start_x = x0 + x_fraction * span_x
    base_y = y1 - 0.075 * span_y
    axes.plot(
        [start_x, start_x + length_mm],
        [base_y, base_y],
        color=colour,
        linewidth=linewidth,
        solid_capstyle="butt",
        path_effects=_cased(linewidth),
        zorder=6,
    )
    # Left-aligned to the bar's own start rather than centred on it. The label is
    # wider than a short bar, so centring makes the text overhang to the left and
    # crowd the frame even when the bar itself is clear of it.
    axes.annotate(
        f"{length_mm:g} mm",
        xy=(start_x, base_y - 0.02 * span_y),
        ha="left",
        va="bottom",
        color=colour,
        fontsize=6.5,
        path_effects=_cased(0.6),
        zorder=6,
    )


def _frame_axes(axes, colour: str, linewidth: float = 1.0) -> None:
    """Outline an axes with a single cased rectangle instead of four spines.

    Matplotlib draws the four spines as SEPARATE artists, so a path effect gives
    each side its own casing. Where two sides meet, one side's white casing is
    painted over the neighbour's coloured core, and the corners come apart into
    four independently haloed bars rather than reading as concentric squares.
    One closed rectangle path is stroked once, so the casing and the core are
    each continuous and the corners join properly.
    """
    from matplotlib.patches import Rectangle

    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=axes.transAxes,
            fill=False,
            edgecolor=colour,
            linewidth=linewidth,
            path_effects=_cased(linewidth),
            clip_on=False,
            zorder=6,
        )
    )


def _draw_inset_leaders(
    axes,
    inset,
    centre: tuple[float, float],
    half: float,
    corners: list[tuple[int, int]],
    colour: str,
) -> None:
    """Draw the indicator box and its two leader lines explicitly.

    ``indicate_inset_zoom`` is not used: it flips which rectangle corner each
    connector attaches to when the parent y-axis is inverted (which it is here,
    because the slice is drawn with superior at the top), so its "matching
    corners" become a twist and the leaders cross. Anchoring the inset end in
    ``transAxes`` -- which is display-based and therefore immune to data
    inversion -- and the box end in data coordinates removes the ambiguity.
    """
    from matplotlib.patches import ConnectionPatch, Rectangle

    axes.add_patch(
        Rectangle(
            (centre[0] - half, centre[1] - half),
            2 * half,
            2 * half,
            fill=False,
            edgecolor=colour,
            linewidth=1.0,
            path_effects=_cased(1.0),
            zorder=5,
        )
    )
    for horizontal, vertical in corners:
        box_x = centre[0] + (2 * horizontal - 1) * half
        # The data axis runs downward, so the display-top corner is the smaller
        # data value.
        box_y = centre[1] - (2 * vertical - 1) * half
        axes.add_patch(
            ConnectionPatch(
                xyA=(horizontal, vertical),
                coordsA=inset.transAxes,
                xyB=(box_x, box_y),
                coordsB=axes.transData,
                color=colour,
                linewidth=0.8,
                path_effects=_cased(0.8),
                zorder=5,
            )
        )


def _overlay_rgba(
    mask: np.ndarray,
    generation: np.ndarray,
    *,
    mode: str,
    proximal_max_generation: int,
    alpha: float,
) -> np.ndarray:
    overlay = np.zeros(mask.shape + (4,), dtype=np.float32)
    if mode == "depth":
        overlay[..., :3] = np.asarray(to_rgb(PROXIMAL_COLOUR), dtype=np.float32)
        distal = generation > proximal_max_generation
        overlay[distal, :3] = np.asarray(to_rgb(DISTAL_COLOUR), dtype=np.float32)
    else:
        overlay[..., :3] = np.asarray(to_rgb(OVERLAY_COLOUR), dtype=np.float32)
    overlay[..., 3] = mask.astype(np.float32) * alpha
    return overlay


def _draw_slice(axes, ct: np.ndarray, overlay: np.ndarray, extent, *, alpha_scale=1.0) -> None:
    axes.imshow(
        ct,
        cmap="gray",
        vmin=LUNG_WINDOW[0],
        vmax=LUNG_WINDOW[1],
        extent=extent,
        origin="upper",
        interpolation="bilinear",
    )
    if alpha_scale != 1.0:
        overlay = overlay.copy()
        overlay[..., 3] *= alpha_scale
    axes.imshow(overlay, extent=extent, origin="upper", interpolation="nearest")


def render_slice_panel(case: dict, args: argparse.Namespace) -> dict:
    index = _choose_coronal_slice(case["mask"], case["generation"])
    ct, mask, generation = _slice_arrays(case, index)
    zooms = case["zooms"]
    # Display axes are (S, R): row pitch is the S zoom, column pitch the R zoom.
    height_mm = ct.shape[0] * zooms[2]
    width_mm = ct.shape[1] * zooms[0]
    extent = (0.0, width_mm, height_mm, 0.0)

    # Crop to the lungs rather than the body: the arms and the chest wall carry no
    # information here and cost half the panel width. The airway is unioned in so
    # that the extrathoracic trachea, which rises above the lung apices, survives.
    lung = case["lung"][:, index, :] if case.get("lung") is not None else None
    region = mask if lung is None else (lung.T[::-1, ::-1] | mask)
    rows, cols = np.nonzero(region)
    pad_mm = 12.0
    top_mm = max(0.0, rows.min() * zooms[2] - pad_mm)
    bottom_mm = min(height_mm, (rows.max() + 1) * zooms[2] + pad_mm)
    left_mm = max(0.0, cols.min() * zooms[0] - pad_mm)
    right_mm = min(width_mm, (cols.max() + 1) * zooms[0] + pad_mm)

    trachea, distal = _inset_centres(
        mask, generation, extent, row_mm=float(zooms[2]), col_mm=float(zooms[0])
    )
    half = INSET_BOX_MM / 2.0
    # The calibre contrast is the panel's whole argument, so it is measured from
    # the voxel spacing rather than left to the reader's eye and the scale bar.
    diameters = {
        name: (
            None
            if centre is None
            else _measure_diameter_mm(
                mask, centre, half, extent, row_mm=float(zooms[2]), col_mm=float(zooms[0])
            )
        )
        for name, centre in (("trachea", trachea), ("distal", distal))
    }
    aspect = (bottom_mm - top_mm) / (right_mm - left_mm)
    placements = _assign_corners(
        [trachea, distal],
        left_mm=left_mm,
        right_mm=right_mm,
        top_mm=top_mm,
        bottom_mm=bottom_mm,
        inset_fraction=INSET_FRACTION,
        aspect=aspect,
        # Half the indicator box, so "covers the target" means covers the box.
        keepout=(
            0.5 * INSET_BOX_MM / (right_mm - left_mm),
            0.5 * INSET_BOX_MM / (bottom_mm - top_mm),
        ),
    )
    targets = list(zip((trachea, distal), placements))

    written = []
    # Four variants: the two colourings, each with and without the magnified
    # insets. The inset-free pair exists because the opening figure's job is
    # orientation; the calibre argument the insets make is already carried,
    # measured rather than illustrated, by the class-imbalance figure.
    for mode, with_insets, stem in (
        ("plain", True, "airway_overview_slice"),
        ("depth", True, "airway_overview_slice_depth"),
        ("plain", False, "airway_overview_slice_noinset"),
        ("depth", False, "airway_overview_slice_noinset_depth"),
    ):
        overlay = _overlay_rgba(
            mask,
            generation,
            mode=mode,
            proximal_max_generation=args.proximal_max_generation,
            alpha=0.85,
        )
        figure = plt.figure(figsize=(PANEL_WIDTH_IN, PANEL_WIDTH_IN * aspect))
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_axis_off()
        _draw_slice(axes, ct, overlay, extent)
        axes.set_xlim(left_mm, right_mm)
        axes.set_ylim(bottom_mm, top_mm)
        _draw_scale_bar(axes, PANEL_SCALE_BAR_MM, colour=ANNOTATION_COLOUR)

        for centre, placement in targets if with_insets else ():
            if centre is None or placement is None:
                continue
            inset = axes.inset_axes(
                placement,
                xlim=(centre[0] - half, centre[0] + half),
                ylim=(centre[1] + half, centre[1] - half),
                xticks=[],
                yticks=[],
            )
            # Weaker overlay inside the insets: the point of the magnification is
            # the CT evidence under the label -- a sharp wall proximally, a blurred
            # one distally -- and an opaque mask would hide exactly that.
            _draw_slice(inset, ct, overlay, extent, alpha_scale=0.6)
            # Convert the pixel nudge into a fraction of the inset's own width.
            inset_px = PANEL_WIDTH_IN * SLICE_PANEL_DPI * INSET_FRACTION
            _draw_scale_bar(
                inset,
                INSET_SCALE_BAR_MM,
                colour=ANNOTATION_COLOUR,
                linewidth=1.8,
                x_fraction=0.06 + INSET_SCALE_BAR_NUDGE_PX / inset_px,
            )
            _frame_axes(inset, ANNOTATION_COLOUR, linewidth=1.0)
            _draw_inset_leaders(
                axes,
                inset,
                centre,
                half,
                _leader_corners(
                    (
                        (centre[0] - left_mm) / (right_mm - left_mm),
                        1.0 - (centre[1] - top_mm) / (bottom_mm - top_mm),
                    ),
                    (placement[0] + placement[2] / 2, placement[1] + placement[3] / 2),
                ),
                ANNOTATION_COLOUR,
            )

        args.pdf_output_dir.mkdir(parents=True, exist_ok=True)
        args.png_output_dir.mkdir(parents=True, exist_ok=True)
        for directory, suffix in ((args.pdf_output_dir, "pdf"), (args.png_output_dir, "png")):
            figure.savefig(
                directory / f"{stem}.{suffix}", dpi=SLICE_PANEL_DPI, transparent=True, pad_inches=0
            )
        plt.close(figure)
        written.append(stem)

    return {
        "coronal_index_ras": int(index),
        "inset_box_mm": INSET_BOX_MM,
        "trachea_inset_centre_mm": list(trachea) if trachea else None,
        "distal_inset_centre_mm": list(distal) if distal else None,
        "airway_voxels_in_slice": int(mask.sum()),
        "crop_extent_mm": {
            "left": round(float(left_mm), 2),
            "right": round(float(right_mm), 2),
            "top": round(float(top_mm), 2),
            "bottom": round(float(bottom_mm), 2),
            "width": round(float(right_mm - left_mm), 2),
            "height": round(float(bottom_mm - top_mm), 2),
        },
        "inset_scale_bar_mm": INSET_SCALE_BAR_MM,
        "panel_scale_bar_mm": PANEL_SCALE_BAR_MM,
        "measured_diameter_mm": diameters,
        "stems": written,
    }


# --------------------------------------------------------------------------
def _write_caption_macros(provenance: dict, destination: Path) -> Path | None:
    """Emit the measured calibres as LaTeX macros.

    Quoting these in the caption rather than typing them means the prose cannot
    drift from the panel when the case or the inset selection changes -- the same
    contract the class-imbalance figure uses.
    """
    slice_panel = provenance.get("slice_panel")
    if not slice_panel:
        return None
    diameters = slice_panel.get("measured_diameter_mm") or {}
    trachea, distal = diameters.get("trachea"), diameters.get("distal")
    if trachea is None or distal is None:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% GENERATED by dissertation/scripts/generate_intro_airway_overview.py.",
        "% Do not edit: re-run the script instead, so the caption and the panel stay",
        "% in agreement. Each diameter is the largest inscribed in-plane circle within",
        "% that inset box, which equals the true local calibre for a tube cut at any",
        "% obliquity rather than an inflated oblique width.",
        rf"\newcommand{{\airwayTracheaDiameter}}{{{trachea:.0f}}}",
        rf"\newcommand{{\airwayDistalDiameter}}{{{distal:.1f}}}",
        rf"\newcommand{{\airwayInsetBox}}{{{slice_panel['inset_box_mm']:.0f}}}",
        rf"\newcommand{{\airwayInsetScaleBar}}{{{slice_panel['inset_scale_bar_mm']:.0f}}}",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    args = _parse_args()
    from lung_airway_segmentation.config import load_yaml_config

    overrides = parse_case_intensity_overrides(load_yaml_config(args.split_config))
    case_id = f"{int(str(args.case).removeprefix('ATM_')):03d}"

    if args.mesh is None and not args.no_mesh and case_id == "001" and DEFAULT_MESH.exists():
        args.mesh = DEFAULT_MESH
        print(f"Using exported mesh {DEFAULT_MESH.name} (--no-mesh to override).")

    print(f"Loading ATM_{case_id} and parsing branch depths ...", flush=True)
    case = _load_case(case_id, overrides)
    print(
        f"  {int(case['mask'].sum()):,} airway voxels, {case['branch_count']} parsed "
        f"branches, max depth {case['max_generation']}",
        flush=True,
    )

    provenance = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "case": f"ATM_{case_id}",
        "voxel_size_mm_ras": [round(float(z), 4) for z in case["zooms"]],
        "airway_voxels": int(case["mask"].sum()),
        "parsed_branches": case["branch_count"],
        "max_branch_depth": case["max_generation"],
        "proximal_max_generation": args.proximal_max_generation,
    }

    if not args.skip_slice:
        print("Rendering the CT slice panel ...", flush=True)
        provenance["slice_panel"] = render_slice_panel(case, args)
    if not args.skip_tree:
        print("Rendering the 3D tree panels ...", flush=True)
        provenance["tree_panels"] = render_tree_panels(case, args)

    args.provenance_output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.provenance_output_dir / "airway_overview.json"
    destination.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")

    macros = _write_caption_macros(provenance, FIGURE_ROOT / "airway_overview_numbers.tex")
    if macros is not None:
        print(f"Wrote {macros}")
    print(f"Figures in {args.pdf_output_dir}")


if __name__ == "__main__":
    main()
