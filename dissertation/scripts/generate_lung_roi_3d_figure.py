"""The airway region of interest as a three-dimensional scene, not two coronal panels.

A companion to ``generate_lung_roi_figure.py``, which is left in place and still produces
Figure~\\ref{fig:lung-roi}. The flat version answers "where is the box" correctly but has
to argue about a three-dimensional crop with a single projection: the anterior-posterior
margin is invisible in a coronal view, so a reader has to take two of the six faces on
trust. This one draws the box as a box.

What is in the scene
--------------------
Three orthogonal CT planes, textured from the volume itself and placed on the far side of
the anatomy so they act as backdrops rather than cutting the tree in half; the lung mask
as a translucent surface; the reference airway tree as a shaded isosurface; and the two
wireframes the figure exists for -- the lung mask's own bounding box and the retained ROI.
Both boxes are drawn solid all the way round, over the scene: their depth is carried by
the perspective and by what they enclose, and a wireframe that is half solid and half
dashed spends the dash on occlusion when the flat version of this figure spends it on
telling the two boxes apart. The dash is spent instead on the ROI's cross-section in each
CT plane, which is the one line in the figure that lands on the seam in the staged texture
-- the solid box cannot, because its near faces stand up to 166 mm in front of the plane.

Nothing is redrawn or idealised. The planes are resampled out of the CT, panel (b) is the
staged file the network is actually given, read off disk, and both boxes are recomputed
from the lung mask and checked against the staging manifest before anything is rendered.

How it is rendered
------------------
No VTK, no ``mplot3d``. The surfaces reuse the exact z-buffer of
``generate_intro_airway_overview`` -- a signed distance field, marching cubes, Taubin mesh
smoothing, point splatting with stored normals and Blinn-Phong shading -- through the
shared-frame projector of ``render_tree``, so every layer of every panel lands in one
orthographic frame at one scale. ``mplot3d`` sorts whole polygons by mean depth, and an
airway tree self-occludes constantly; the painter's-algorithm artefacts of a
``Poly3DCollection`` land exactly where the eye looks.

The CT planes are ray-cast rather than splatted. The projection is orthographic, so for a
plane at a fixed anatomical coordinate every screen pixel has a closed-form intersection;
each pixel is solved for it and the volume sampled there by bilinear interpolation. That
gives an exact depth for the plane, which is what lets the tree, the lung and the box
wireframes be composited against it by depth test rather than by drawing order.

Wireframes and text are matplotlib vectors in the render's own pixel coordinates, so they
stay sharp in the PDF at any print size while still being occluded by the raster scene.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\generate_lung_roi_3d_figure.py

Draft quickly with ``--px-mm 0.9 --supersample 1``; the default is print resolution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from figure_theme import ANNOTATION_RED, MUTED, apply_theme, panel_label  # noqa: E402
from generate_intro_airway_overview import _downsample, _shade_normals  # noqa: E402
from render_tree import Camera, build_frame, project, surface_samples  # noqa: E402

DEFAULT_CASE = "ATM_016"
DEFAULT_MANIFEST = (
    ROOT / "data" / "nnunet" / "predict_in" / "val_lungroi_m8_s120" / "lung_crop_manifest.json"
)
ATM_ROOT = ROOT / "data" / "ATM22"
PDF_OUT = ROOT / "dissertation" / "Figures" / "pdf" / "methods" / "lungroi"
PNG_OUT = ROOT / "dissertation" / "Figures" / "png" / "methods" / "lungroi"

# Lung window, as in the flat version of this figure: wide enough that the mediastinum
# stays bright against the air column.
WINDOW = (-1350.0, 150.0)
# NOT the render family's reference tone (render_tree.REFERENCE_PALETTE, #5B7F99). That one
# is a slate blue chosen to sit quietly against white, and here it sits against a lung field
# at lung window -- a dark grey of about the same luminance -- so the tree disappeared into
# its own background. Okabe-Ito sky blue is two stops lighter, survives the 0.34 ambient
# floor of the shader, and separates from ANNOTATION_RED under all three dichromacies, which
# matters because the red wireframe is the only other saturated thing in the frame.
TREE_COLOUR = "#56B4E9"
# Deliberately desaturated: the lung is the container, not the subject.
LUNG_COLOUR = "#AFC3D0"
LUNG_ALPHA = 0.16
# Grazing-angle boost. A flat alpha makes a translucent surface read as fog; the rim term
# is what makes it read as a surface with a shape. Kept low: past about 0.25 the rim
# closes into a white shell and the CT plane behind the lung stops being readable, which
# would cost the figure the anatomy it is drawn on.
LUNG_RIM = 0.26
# Dark enough to survive being printed and reduced to \linewidth. The wireframe crosses
# black air, mid-grey lung and white mediastinum in one edge, so no single ink is legible
# along its whole length; it is drawn with a thin white casing instead, which is the same
# trick the annotations use.
LUNG_BOX_COLOUR = "#334155"
LUNG_BOX_CASING = [path_effects.withStroke(linewidth=2.0, foreground="white")]
PLANE_EDGE_COLOUR = "#B9C2CB"
# Larger than the shared ANNOTATION_PT: two panels side by side on a 6.9 in figure are
# aspect-limited, so legibility on the page has to come from type size.
OVERLAY_PT = 8.0
HALO = [path_effects.withStroke(linewidth=1.7, foreground="white")]

# Blank band left above the scene, as a fraction of its height, for the callout.
HEADROOM_FRACTION = 0.11

# Depth tolerance of the hidden-line test, in millimetres. A box face that lies exactly on
# a CT plane would otherwise z-fight along its whole length.
OCCLUSION_TOL_MM = 1.0

CORNER_BITS = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
BOX_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if sum(a != b for a, b in zip(CORNER_BITS[i], CORNER_BITS[j])) == 1
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--azimuth",
        type=float,
        default=-35.0,
        help="Degrees about the superior-inferior axis. 0 is a straight anterior view; a "
        "negative value opens the angle between the two main bronchi.",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-14.0,
        help="Degrees the camera is raised above the axial plane. Negative raises it in "
        "this convention, which puts the axial plane underneath the lungs.",
    )
    parser.add_argument("--px-mm", type=float, default=0.42,
                        help="Millimetres per pixel of the final render.")
    parser.add_argument("--supersample", type=int, default=2,
                        help="Render at this factor and box-downsample; the anti-aliasing.")
    parser.add_argument(
        "--plane-quantile",
        type=float,
        default=0.05,
        help="Where each CT plane sits, as a quantile of the lung mask's extent along that "
        "axis, measured from the side away from the camera; 0.5 would cut straight through "
        "the middle. The default is shallow on purpose: at 0.12 the axial plane rises far "
        "enough to hide the lower lobes and the tree loses its inferior third.",
    )
    parser.add_argument("--smooth-mm", type=float, default=0.25,
                        help="Signed-distance smoothing before the airway isosurface.")
    parser.add_argument("--taubin", type=int, default=16,
                        help="Taubin mesh-smoothing iterations for the airway surface.")
    parser.add_argument("--lung-stride", type=int, default=3,
                        help="Subsampling factor of the lung mask before its surface is "
                        "extracted. The lung is centimetres thick; the tree is not.")
    parser.add_argument("--no-lung-surface", action="store_true")
    parser.add_argument("--no-lung-box", action="store_true")
    parser.add_argument("--panel-a-only", action="store_true")
    parser.add_argument("--stem", default="lung_roi_bbox_3d")
    parser.add_argument("--pdf-output-dir", type=Path, default=PDF_OUT)
    parser.add_argument("--png-output-dir", type=Path, default=PNG_OUT)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Loading, in RAS+ so that "superior" and "anterior" mean what they say
# --------------------------------------------------------------------------
def _orientation(image) -> np.ndarray:
    return nib.orientations.io_orientation(image.affine)


def _to_ras(image) -> tuple[np.ndarray, np.ndarray]:
    """Reorient to RAS+ and permute the voxel sizes with it.

    ATM'22 is stored LPS, so this is not cosmetic: the superior margin, the camera
    elevation and the plane placement are all statements about anatomical axes.
    """
    ornt = _orientation(image)
    array = nib.apply_orientation(np.asanyarray(image.dataobj), ornt)
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)[np.argsort(ornt[:, 0])]
    return array, zooms


def _box_to_ras(box, shape, ornt) -> list[tuple[int, int]]:
    """Carry a half-open per-axis index box through the same reorientation."""
    out: list[tuple[int, int]] = [(0, 0)] * 3
    for axis, (start, stop) in enumerate(box):
        target, flip = int(ornt[axis, 0]), int(ornt[axis, 1])
        if flip == -1:
            start, stop = shape[axis] - stop, shape[axis] - start
        out[target] = (int(start), int(stop))
    return out


def _bounding_box(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open per-axis extent of the mask."""
    box = []
    for axis in range(3):
        others = tuple(other for other in range(3) if other != axis)
        occupied = np.flatnonzero(np.any(mask, axis=others))
        box.append((int(occupied[0]), int(occupied[-1]) + 1))
    return box


def _box_mm(box, zooms) -> list[tuple[float, float]]:
    return [(start * zooms[axis], stop * zooms[axis]) for axis, (start, stop) in enumerate(box)]


def _corners(box_mm) -> np.ndarray:
    return np.array(
        [[box_mm[axis][bit] for axis, bit in enumerate(bits)] for bits in CORNER_BITS],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------
# The CT planes: an orthographic ray cast, one closed-form solve per pixel
# --------------------------------------------------------------------------
def _screen_grid(frame) -> tuple[np.ndarray, np.ndarray]:
    columns = np.arange(frame.width, dtype=np.float32) + 0.5
    rows = np.arange(frame.height, dtype=np.float32) + 0.5
    screen_x = frame.u_reference - (columns - frame.margin_px) * frame.px_mm
    screen_y = frame.v_reference - (rows - frame.margin_px) * frame.px_mm
    return np.meshgrid(screen_x.astype(np.float32), screen_y.astype(np.float32))


def _plane_layer(volume, frame, screen, axis: int, index: int, zooms) -> tuple | None:
    """Depth and sampled HU of one anatomical plane, per pixel.

    Under an orthographic camera the ray through a pixel meets the plane exactly once and
    the parameter of that meeting IS the depth in the shared frame, so no splatting and no
    tolerance are involved: the plane composites against the surfaces by depth test.
    """
    screen_x, screen_y = screen
    rotation, origin = frame.rotation, frame.origin_mm
    along_view = float(rotation[1, axis])
    if abs(along_view) < 1e-4:  # edge-on; it would be a line, not a plane
        return None

    position = (index + 0.5) * zooms[axis]
    depth = (
        position - origin[axis] - rotation[0, axis] * screen_x - rotation[2, axis] * screen_y
    ) / along_view

    inside = np.ones(depth.shape, dtype=bool)
    coordinates = []
    for other in (a for a in range(3) if a != axis):
        world = (
            origin[other]
            + rotation[0, other] * screen_x
            + rotation[1, other] * depth
            + rotation[2, other] * screen_y
        )
        index_along = world / zooms[other]
        inside &= (index_along >= 0.0) & (index_along <= volume.shape[other] - 1)
        coordinates.append(index_along)

    plane = np.take(volume, index, axis=axis).astype(np.float32)
    value = ndi.map_coordinates(plane, np.stack(coordinates), order=1, mode="nearest")
    return np.where(inside, depth, -np.inf).astype(np.float32), value.astype(np.float32)


def _plane_positions(lung: np.ndarray, frame, quantile: float) -> dict[int, int]:
    """One index per axis, at ``quantile`` of the lung's extent from the far side.

    Placing the planes at the volume's own faces would put three slices of air behind the
    anatomy; placing them through the middle would cut the tree in half. A shallow
    quantile of the lung's own extent gives a plane that still carries anatomy and still
    sits behind almost all of the tree.
    """
    positions: dict[int, int] = {}
    for axis in range(3):
        others = tuple(other for other in range(3) if other != axis)
        counts = lung.sum(axis=others).astype(np.float64)
        cumulative = np.cumsum(counts) / counts.sum()
        # The camera sits along +rotation[1]; the far side of the scene is the other one.
        from_low = frame.rotation[1, axis] > 0
        target = quantile if from_low else 1.0 - quantile
        positions[axis] = int(np.searchsorted(cumulative, target))
    return positions


# --------------------------------------------------------------------------
# Compositing
# --------------------------------------------------------------------------
def _surface_layer(mask, zooms, frame, *, smooth_mm, taubin, px_mm, pad=4):
    """Project one binary mask as a shaded isosurface into the shared frame."""
    located = ndi.find_objects(mask.astype(np.uint8))[0]
    crop = tuple(
        slice(max(0, int(s.start) - pad), min(int(dim), int(s.stop) + pad))
        for s, dim in zip(located, mask.shape)
    )
    points, normals, classes, stats = surface_samples(
        mask.astype(np.int16), zooms, crop, smooth_mm=smooth_mm, taubin_iterations=taubin
    )
    origin = np.array([crop[axis].start * zooms[axis] for axis in range(3)], dtype=np.float32)
    points = points + origin
    depth, _, normal = project(
        points,
        normals,
        classes,
        frame,
        splat_radius_px=max(1, int(round(0.45 * float(np.max(zooms)) / px_mm))),
    )
    shade, specular = _shade_normals(normal, depth)
    return depth, shade, specular, normal, stats


def _render_scene(volume, tree_layer, lung_layer, frame, screen, planes, zooms):
    """Planes, then the tree, then the lung as glass over whatever survived."""
    height, width = frame.height, frame.width
    scene_depth = np.full((height, width), -np.inf, dtype=np.float32)
    grey = np.zeros((height, width), dtype=np.float32)

    for axis, index in planes.items():
        layer = _plane_layer(volume, frame, screen, axis, index, zooms)
        if layer is None:
            continue
        depth, value = layer
        nearer = depth > scene_depth
        scene_depth = np.where(nearer, depth, scene_depth)
        grey = np.where(nearer, np.clip((value - WINDOW[0]) / (WINDOW[1] - WINDOW[0]), 0, 1), grey)

    rgb = np.repeat(grey[..., None], 3, axis=2)
    covered = np.isfinite(scene_depth)

    depth, shade, specular, _, _ = tree_layer
    front = np.isfinite(depth) & (depth > scene_depth)
    lit = np.clip(
        np.asarray(to_rgb(TREE_COLOUR), dtype=np.float32) * shade[..., None]
        + specular[..., None],
        0.0,
        1.0,
    )
    rgb = np.where(front[..., None], lit, rgb)
    scene_depth = np.where(front, depth, scene_depth)
    covered |= front
    # Where a plane outline is cut off. Tested against the opaque scene only: the lung is
    # glass, and glass does not end a line.
    opaque_depth = scene_depth.copy()

    if lung_layer is not None:
        depth, shade, specular, normal, _ = lung_layer
        visible = np.isfinite(depth) & (depth > scene_depth)
        lit = np.clip(
            np.asarray(to_rgb(LUNG_COLOUR), dtype=np.float32) * shade[..., None]
            + 0.25 * specular[..., None],
            0.0,
            1.0,
        )
        # Grazing incidence: the view vector is +y in the rotated frame.
        rim = np.clip(1.0 - np.abs(normal[..., 1]), 0.0, 1.0) ** 2
        alpha = np.clip(LUNG_ALPHA + LUNG_RIM * rim, 0.0, 1.0)[..., None] * visible[..., None]
        rgb = np.where(covered[..., None], rgb * (1.0 - alpha) + lit * alpha,
                       np.where(visible[..., None], lit, rgb))
        covered |= visible
        scene_depth = np.where(visible, depth, scene_depth)

    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(rgb, 0.0, 1.0)
    rgba[..., 3] = covered.astype(np.float32)
    return rgba, opaque_depth


# --------------------------------------------------------------------------
# Vector overlay: solid wireframes, and plane outlines clipped by the scene
# --------------------------------------------------------------------------
def _to_screen(points: np.ndarray, frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotated = (np.asarray(points, dtype=np.float32) - frame.origin_mm) @ frame.rotation.T
    column = (frame.u_reference - rotated[:, 0]) / frame.px_mm + frame.margin_px
    row = (frame.v_reference - rotated[:, 2]) / frame.px_mm + frame.margin_px
    return column, row, rotated[:, 1]


def _visible_runs(start, stop, frame, depth_image, samples=None):
    """Split one 3D segment into runs the scene hides and runs it does not.

    Used for the plane outlines, which stop where another plane is in front of them; the
    box wireframes are drawn whole and do not go through here.
    """
    length = float(np.linalg.norm(stop - start)) / frame.px_mm
    count = int(max(8, samples or length))
    fractions = np.linspace(0.0, 1.0, count)
    points = start[None, :] + fractions[:, None] * (stop - start)[None, :]
    column, row, depth = _to_screen(points, frame)

    rows = np.clip(np.rint(row).astype(int), 0, frame.height - 1)
    columns = np.clip(np.rint(column).astype(int), 0, frame.width - 1)
    scene = depth_image[rows, columns]
    visible = ~np.isfinite(scene) | (depth >= scene - OCCLUSION_TOL_MM)

    # Consecutive samples of one verdict are merged into a single straight run rather than
    # emitted one per sample. Not only for the file size: a cased line drawn as a thousand
    # pixel-long segments has each segment's white casing painted over its neighbour's
    # core, and the whole edge comes out white.
    span = visible[:-1] & visible[1:]  # one verdict per segment, count - 1 of them
    breaks = [0, *(1 + np.flatnonzero(span[1:] != span[:-1])), span.size]
    shown, hidden = [], []
    for start, stop in zip(breaks[:-1], breaks[1:]):
        # Segments start .. stop-1 share a verdict, so they span samples start .. stop.
        run = [(column[start], row[start]), (column[stop], row[stop])]
        (shown if span[start] else hidden).append(run)
    return shown, hidden


def _draw_wireframe(axis, corners, frame, *, colour, width, zorder=5, casing=None):
    """All twelve edges, solid, over the scene.

    No hidden-line pass. A box drawn half solid and half dashed asks the reader to hold two
    encodings at once -- dash for occlusion here, dash for "this is the lung box" in the
    flat version of the figure -- and the depth of a wireframe is already carried by its
    perspective and by what it encloses. Each edge is straight under an orthographic
    camera, so one two-point segment per edge is exact.
    """
    segments = []
    for start, stop in BOX_EDGES:
        column, row, _ = _to_screen(corners[[start, stop]], frame)
        segments.append(list(zip(column, row)))
    axis.add_collection(
        LineCollection(segments, colors=colour, linewidths=width, zorder=zorder,
                       capstyle="round", path_effects=casing)
    )


def _plane_rectangle(plane_axis: int, position: float, extent) -> list[np.ndarray]:
    """The four corners of a rectangle lying in one anatomical plane, in order."""
    others = [a for a in range(3) if a != plane_axis]
    corners = []
    for first, second in ((0, 0), (0, 1), (1, 1), (1, 0)):
        point = np.zeros(3, dtype=np.float32)
        point[plane_axis] = position
        point[others[0]] = extent[others[0]][first]
        point[others[1]] = extent[others[1]][second]
        corners.append(point)
    return corners


def _draw_on_planes(axis, frame, depth_image, rectangles, *, colour, width, zorder,
                    linestyle="solid"):
    """Rectangles that lie in the CT planes, clipped where the scene covers them.

    These are the one family of lines that DOES get a hidden-line test. An edge lying in a
    plane is only meaningful where that plane can be seen, so a run of it behind another
    plane is not a depth cue to be drawn faintly -- it is a line that is not there.
    """
    segments = []
    for rectangle in rectangles:
        for corner in range(4):
            shown, _ = _visible_runs(rectangle[corner], rectangle[(corner + 1) % 4], frame,
                                     depth_image)
            segments.extend(shown)
    axis.add_collection(
        LineCollection(segments, colors=colour, linewidths=width, zorder=zorder,
                       linestyles=linestyle)
    )


def _draw_plane_edges(axis, volume, frame, depth_image, planes, zooms):
    """A thin outline on each CT plane, so the three of them read as three surfaces."""
    volume_mm = [(0.0, (volume.shape[axis_index] - 1) * zooms[axis_index]) for axis_index in range(3)]
    rectangles = [
        _plane_rectangle(plane_axis, (index + 0.5) * zooms[plane_axis], volume_mm)
        for plane_axis, index in planes.items()
    ]
    _draw_on_planes(axis, frame, depth_image, rectangles, colour=PLANE_EDGE_COLOUR,
                    width=0.6, zorder=4)


def _draw_plane_intersections(axis, frame, depth_image, planes, zooms, box_mm):
    """Where the region of interest cuts each CT plane, dotted.

    Without this the figure invites one wrong reading. On a plane of the staged volume the
    data stops exactly on the region's limits, but the box is drawn in three dimensions and
    its near faces sit up to 166 mm in front of the plane -- about a sixth of the panel
    width at this camera. The box therefore projects OUTSIDE the seam in the texture, and
    the two together look like a box that does not fit its own crop. The dotted rectangle
    is the box's actual cross-section in that plane; it lands on the seam, and the offset
    between it and the solid box is then legible as the parallax it is.
    """
    rectangles = [
        _plane_rectangle(plane_axis, (index + 0.5) * zooms[plane_axis], box_mm)
        for plane_axis, index in planes.items()
    ]
    _draw_on_planes(axis, frame, depth_image, rectangles, colour=ANNOTATION_RED, width=0.9,
                    zorder=5.5, linestyle=(0, (1.1, 1.7)))


def _content_crop(alphas, overlays, frame, pad_px: int = 14):
    """The drawn extent, in render pixels, so the panel is not sized by empty air.

    The frame is sized on the acquisition volume, whose silhouette is a hexagon; three
    planes and a box do not fill it, and the uncovered corners are a quarter of the page
    at no information per square inch. Cropping after rendering rather than shrinking the
    frame keeps every layer in the coordinates it was projected into.
    """
    rows, columns = [], []
    for alpha in alphas:
        painted = np.argwhere(alpha > 0)
        if painted.size:
            rows.extend([painted[:, 0].min(), painted[:, 0].max()])
            columns.extend([painted[:, 1].min(), painted[:, 1].max()])
    scale = frame.height / alphas[0].shape[0]
    rows = [value * scale for value in rows]
    columns = [value * scale for value in columns]
    for points in overlays:
        column, row, _ = _to_screen(points, frame)
        rows.extend([row.min(), row.max()])
        columns.extend([column.min(), column.max()])
    left = max(0.0, min(columns) - pad_px)
    right = min(float(frame.width), max(columns) + pad_px)
    top = max(0.0, min(rows) - pad_px)
    bottom = min(float(frame.height), max(rows) + pad_px)
    # Headroom for the superior-extension callout, which is hung off the panel's top
    # corner. Without it the label sits on the box's top face, and the top face of a box
    # drawn in projection is a band a hundred pixels deep -- there is nowhere inside the
    # panel for the text to go. The limits may leave the rendered image; that is only
    # page white, and it costs nothing but a slightly smaller scene.
    top -= HEADROOM_FRACTION * (bottom - top)
    return left, right, top, bottom


def _nearest_top_corner(box_mm, frame) -> tuple[float, float]:
    """The (x, y) top corner of a box that faces the camera, for hanging text on."""
    best, best_depth = (box_mm[0][0], box_mm[1][0]), -np.inf
    for x in box_mm[0]:
        for y in box_mm[1]:
            _, _, depth = _to_screen(np.array([[x, y, box_mm[2][1]]]), frame)
            if depth[0] > best_depth:
                best, best_depth = (x, y), float(depth[0])
    return best


# --------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    apply_theme()

    manifest = json.loads(args.manifest.read_text())
    entry = manifest["cases"][args.case]
    margin = int(manifest["margin_voxels"])
    superior_request = int(manifest["superior_margin_voxels"])

    ct_image = nib.load(entry["ct"])
    lung_image = nib.load(entry["lung"])
    label_path = ATM_ROOT / "labelsTr" / f"{args.case}_0000.nii.gz"
    if not label_path.exists():
        raise SystemExit(f"Missing reference annotation {label_path}")

    ct, zooms = _to_ras(ct_image)
    lung_raw, _ = _to_ras(lung_image)
    lung = lung_raw > 0
    tree_raw, _ = _to_ras(nib.load(str(label_path)))
    tree = tree_raw > 0

    ornt = _orientation(ct_image)
    roi = _box_to_ras([(int(lo), int(hi)) for lo, hi in entry["bbox"]], ct_image.shape, ornt)
    lung_box = _bounding_box(lung)

    # The manifest is a record of a computation; recompute the box from the mask and the
    # stated margins and require the two to agree, so the figure can never quietly
    # disagree with the staging it claims to illustrate. Axis 2 is superior in RAS, which
    # is the axis the superior margin is asymmetric on.
    for axis, (start, stop) in enumerate(lung_box):
        upper = superior_request if axis == 2 else margin
        expected = (max(0, start - margin), min(ct.shape[axis], stop + upper))
        if expected != roi[axis]:
            raise SystemExit(
                f"{args.case}: recomputed ROI on RAS axis {axis} is {expected}, "
                f"manifest says {roi[axis]}"
            )
    superior_applied = roi[2][1] - lung_box[2][1]

    roi_mm = _box_mm(roi, zooms)
    lung_box_mm = _box_mm(lung_box, zooms)

    supersample = max(1, args.supersample)
    px_mm = args.px_mm / supersample
    camera = Camera(azimuth=args.azimuth, elevation=args.elevation, px_mm=px_mm,
                    supersample=supersample, smooth_mm=args.smooth_mm,
                    taubin_iterations=args.taubin)

    # One frame for both panels and every layer: the whole acquisition volume, so the ROI
    # box is drawn inside the field of view it was cut from rather than filling it.
    volume_corners = _corners([(0.0, (ct.shape[axis] - 1) * zooms[axis]) for axis in range(3)])
    frame = build_frame([volume_corners, _corners(roi_mm)], camera=camera, px_mm=px_mm,
                        margin_px=16)
    print(f"  frame {frame.width} x {frame.height} px at {px_mm:.3f} mm/px", flush=True)

    screen = _screen_grid(frame)
    planes = _plane_positions(lung, frame, args.plane_quantile)
    print("  planes at RAS indices " + ", ".join(f"{a}:{i}" for a, i in sorted(planes.items())),
          flush=True)

    tree_layer = _surface_layer(tree, zooms, frame, smooth_mm=args.smooth_mm,
                                taubin=args.taubin, px_mm=px_mm)
    print(f"  airway surface: {tree_layer[4]['surface_samples']:,} samples, "
          f"{tree_layer[4]['voxel_retention'] * 100:.2f}% of airway voxels retained, "
          f"components {tree_layer[4]['components_before']} -> "
          f"{tree_layer[4]['components_after']}", flush=True)

    lung_layer = None
    if not args.no_lung_surface:
        stride = max(1, args.lung_stride)
        lung_layer = _surface_layer(
            lung[::stride, ::stride, ::stride], zooms * stride, frame,
            smooth_mm=args.smooth_mm * stride, taubin=args.taubin, px_mm=px_mm,
        )
        print(f"  lung surface: {lung_layer[4]['surface_samples']:,} samples at stride "
              f"{stride}", flush=True)

    panels = [("a", ct, f"{args.case}: acquisition volume, "
               f"{' × '.join(str(size) for size in ct_image.shape)} voxels")]
    if not args.panel_a_only:
        staged, _ = _to_ras(nib.load(str(ROOT / entry["output"])))
        panels.append(
            ("b", staged, "staged input: outside the ROI set to 0 HU, "
             f"{100 * entry['roi_fraction']:.1f}% retained")
        )

    renders = []
    for letter, volume, title in panels:
        rgba, opaque = _render_scene(volume, tree_layer, lung_layer, frame, screen, planes, zooms)
        renders.append((letter, volume, title, _downsample(rgba, supersample), opaque))
        print(f"  panel ({letter}) composited", flush=True)

    left, right, top, bottom = _content_crop(
        [render[3][..., 3] for render in renders],
        [_corners(roi_mm), _corners(lung_box_mm)],
        frame,
    )
    aspect = (bottom - top) / (right - left)
    figure, axes = plt.subplots(1, len(panels), figsize=(6.9, 6.9 * aspect / len(panels) + 0.55),
                                layout="constrained")
    axes = np.atleast_1d(axes)

    for axis, (letter, volume, title, rgba, opaque) in zip(axes, renders):
        axis.imshow(rgba, extent=(0, frame.width, frame.height, 0),
                    interpolation="none", zorder=1)
        axis.set_xlim(left, right)
        axis.set_ylim(bottom, top)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

        _draw_plane_edges(axis, volume, frame, opaque, planes, zooms)
        _draw_plane_intersections(axis, frame, opaque, planes, zooms, roi_mm)
        if letter == "a" and not args.no_lung_box:
            _draw_wireframe(axis, _corners(lung_box_mm), frame,
                            colour=LUNG_BOX_COLOUR, width=1.0, casing=LUNG_BOX_CASING)
        _draw_wireframe(axis, _corners(roi_mm), frame, colour=ANNOTATION_RED,
                        width=1.5, zorder=6)

        if letter == "a":
            # Stood off the corner rather than on it: on the corner itself the double
            # arrow lands exactly along the box's own vertical edge and the two read as
            # one long line.
            corner_x, corner_y = _nearest_top_corner(roi_mm, frame)
            centre = [0.5 * (low + high) for low, high in roi_mm]
            stand_off = 16.0
            corner_x += np.sign(corner_x - centre[0]) * stand_off
            corner_y += np.sign(corner_y - centre[1]) * stand_off
            ends = np.array([[corner_x, corner_y, lung_box_mm[2][1]],
                             [corner_x, corner_y, roi_mm[2][1]]], dtype=np.float32)
            columns, rows, _ = _to_screen(ends, frame)
            axis.add_patch(FancyArrowPatch((columns[0], rows[0]), (columns[1], rows[1]),
                                           arrowstyle="<->", mutation_scale=7,
                                           color=ANNOTATION_RED, linewidth=1.1, zorder=7))
            # Hung off the panel's own corner rather than off the arrow: the top face of
            # the box projects to a band a hundred pixels deep, so any offset small
            # enough to keep the text near the arrow lands it on an edge.
            axis.annotate(
                # Spelled out rather than "+71 of 120": the short form reads as a fraction
                # of something, and what it actually says is that the scan ended first.
                f"superior extension:\n$+${superior_applied} voxels available "
                f"($+${superior_request} max)",
                xy=(columns.mean(), rows.mean()), xycoords="data",
                xytext=(0.99, 0.99), textcoords="axes fraction",
                fontsize=OVERLAY_PT, color=ANNOTATION_RED, ha="right", va="top",
                zorder=8, path_effects=HALO, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=ANNOTATION_RED, linewidth=0.6,
                                shrinkA=2, shrinkB=4, path_effects=HALO),
            )

        axis.set_title(title, fontsize=OVERLAY_PT, color=MUTED)
        panel_label(axis, letter, x=-0.02)

    keys = [
        plt.Line2D([], [], color=TREE_COLOUR, linewidth=2.4, label="reference airway tree"),
        plt.Line2D([], [], color=LUNG_COLOUR, linewidth=3.2, label="lung mask"),
        plt.Line2D([], [], color=LUNG_BOX_COLOUR, linewidth=1.0,
                   label="lung-mask bounding box"),
        plt.Line2D([], [], color=ANNOTATION_RED, linewidth=1.5,
                   label=f"retained ROI ($+${margin} voxels, $+${superior_request} superior)"),
        plt.Line2D([], [], color=ANNOTATION_RED, linewidth=0.9, linestyle=(0, (1.1, 1.7)),
                   label="ROI where it cuts each plane"),
    ]
    # Three columns over two rows rather than one row of five: at 8 pt the five labels are
    # wider than the 6.9 in figure, and a legend that overruns its figure is scaled down
    # with it when LaTeX fits the graphic to \linewidth.
    figure.legend(handles=keys, loc="outside lower center", ncol=3, fontsize=OVERLAY_PT,
                  frameon=False, handletextpad=0.5, columnspacing=1.6)

    args.pdf_output_dir.mkdir(parents=True, exist_ok=True)
    args.png_output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.pdf_output_dir / f"{args.stem}.pdf", facecolor="white")
    figure.savefig(args.png_output_dir / f"{args.stem}.png", dpi=300, facecolor="white")
    plt.close(figure)
    print(f"  lung bbox {lung_box}  ROI {roi} (RAS)  superior applied {superior_applied}")
    print("  wrote", args.pdf_output_dir / f"{args.stem}.pdf")


if __name__ == "__main__":
    main()
