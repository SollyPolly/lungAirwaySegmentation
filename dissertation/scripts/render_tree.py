"""Render an airway tree in 3D, coloured by a per-voxel class rather than by arm.

The Introduction's tree panel already solved the hard part of this: an isosurface over a
signed distance field, Taubin-smoothed, splatted into an exact z-buffer and shaded from
stored normals. That machinery is imported from ``generate_intro_airway_overview`` rather
than re-derived, so both figures draw the same anatomy the same way.

Two things it does not do are added here, and both are correctness requirements rather
than decoration:

**A shared screen frame.** ``_project_surface`` there centres each render on its own point
cloud and sizes the canvas to fit. That is right for a single panel and wrong for a
comparison: two arms would be drawn at two scales, and a reader would read the difference
in framing as a difference in anatomy. Every panel of one figure is therefore projected in
one frame, computed from the union of the point clouds it will hold, so the trees overlay
each other pixel for pixel and only the colouring changes.

**An explicit crop.** The surface extractor there crops to the mask's own bounding box, so
the millimetre coordinates it returns are relative to that box. Passing an explicit crop,
shared across panels, is what makes the point clouds comparable in the first place.

Colour convention
-----------------
Renders use a voxel-class palette that is deliberately NOT the arm palette of
``figure_theme``. In a plot, colour means "which arm"; in a render, colour means "how this
voxel stands against the reference", and the two encodings must not be confusable. Within
the render family the meaning is fixed:

    grey        agreement, or a voxel no arm changed -- the reference tone
    red         a reference branch the model did not produce
    cyan        a predicted voxel with no reference support
    teal        a change in the treatment's favour
    orange      a change against it

Greys carry what did not change and colour carries what did, which is the same rule the
statistical figures follow. The palette is Paul Tol's vibrant scheme, which holds up under
all three dichromacies; grey against red against cyan is the widest separation available
for the three-class case that matters most.
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from generate_intro_airway_overview import (  # noqa: E402  (sibling script)
    _densify,
    _downsample,
    _rotation,
    _sample_generations,
    _shade_normals,
    _taubin_smooth,
    _vertex_normals,
)

from figure_theme import INK, LABEL_PT, LEGEND_PT, MUTED, TICK_PT  # noqa: E402

ATM_ROOT = ROOT / "data" / "ATM22"
AEROPATH_ROOT = ROOT / "data" / "AeroPath_atm_layout"

# --------------------------------------------------------------------------
# Class palettes. Keys are the integer labels a class volume carries.
# --------------------------------------------------------------------------
# Saturated Okabe--Ito/Tol hues.  The earlier pale agreement and missed-tree tones
# were legible as legend swatches but washed into the white page once a specular
# highlight was added to a thin surface.  These darker bases retain their identity
# after lighting and remain separable under the common colour-vision deficiencies.
GREY = "#8A9099"
DARK_GREY = "#4B5563"
PALE_RED = "#CC79A7"
RED = "#D55E00"
CYAN = "#0072B2"
TEAL = "#009E73"
ORANGE = "#E69F00"

TP, FP, FN = 1, 2, 3
ERROR_PALETTE = {
    TP: (GREY, "True positive"),
    FP: (CYAN, "False positive"),
    FN: (RED, "False negative"),
}

BOTH_FOUND, RECOVERED, LOST, MISSED_BOTH, EITHER_FP = 1, 2, 3, 4, 5
RECOVERY_PALETTE = {
    BOTH_FOUND: (DARK_GREY, "Found by both arms"),
    RECOVERED: (TEAL, "Recovered by the Mean Teacher"),
    LOST: (ORANGE, "Lost by the Mean Teacher"),
    MISSED_BOTH: (PALE_RED, "Missed by both arms"),
    EITHER_FP: (CYAN, "False positive of either arm"),
}

KEPT, REMOVED_FP, REMOVED_TP = 1, 2, 3
POSTPROCESS_PALETTE = {
    KEPT: (GREY, "Kept by the filter"),
    REMOVED_FP: (TEAL, "Removed, false positive"),
    REMOVED_TP: (RED, "Removed, true airway"),
}

REFERENCE = 1
REFERENCE_PALETTE = {REFERENCE: ("#1F6D8A", "Reference annotation")}

# Width of one character at the caption and legend point size, in inches. Used only to
# decide how wide a page has to be before anything is drawn, so an approximation from
# the average glyph width of a sans face is sufficient.
_CHAR_IN = 0.052
# The same, at the panel-title point size.
_TITLE_CHAR_IN = 0.060

# Authored at final width, as the other figures are: 0.32\textwidth of the A4 text block
# is 55 mm, which is the natural size for a three-panel row.
PANEL_WIDTH_IN = 2.2
# The A4 text block, 171.8 mm. A figure never exceeds it: LaTeX would scale the whole
# thing down to fit, and an 8.5 pt panel title authored at 2.2 in a panel would land at
# about 6 pt on the page. Panels shrink instead, so the type stays the size it was set.
TEXT_WIDTH_IN = 6.76


@dataclass(frozen=True)
class Panel:
    """One render: a class volume, its palette, and the caption strip under it."""

    title: str
    labels: np.ndarray               # int16, -1 background, >=1 class ids
    palette: dict[int, tuple[str, str]]
    subtitle: str = ""


@dataclass(frozen=True)
class Camera:
    azimuth: float = -30.0
    elevation: float = 10.0
    px_mm: float = 0.32
    supersample: int = 2
    smooth_mm: float = 0.25
    taubin_iterations: int = 16


# --------------------------------------------------------------------------
# Loading, in RAS+ so that "superior" and "anterior" mean what they say
# --------------------------------------------------------------------------
def _to_ras(array: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(nib.Nifti1Image(array, affine))
    return np.asanyarray(image.dataobj), np.asarray(image.header.get_zooms(), dtype=float)


def load_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A binary mask in its STORED orientation, with the file affine.

    Deliberately not reoriented here. Class volumes are built in the orientation the
    scorer worked in, because ``skimage``'s three-dimensional thinning is not invariant
    to an axis permutation: skeletonising a reoriented mask moves a handful of centreline
    voxels, and the change panel would then imply a tree-length difference a percent or so
    away from the one the results table reports. The finished class volume is reoriented
    for the camera instead, which is a lossless relabelling of the same voxels.
    """
    image = nib.load(str(path))
    mask = np.asarray(image.dataobj, dtype=np.uint8) > 0
    return mask, np.asarray(image.affine, dtype=np.float64)


def reference_path(case_id: str, cohort: str) -> Path:
    """The annotation for a case, in whichever layout its cohort uses.

    AeroPath is read from the restaged ATM layout when it is present, because that is what
    the predictions were produced from and its masks are already binarised; the published
    release directory is the fallback.
    """
    padded = str(case_id).zfill(3)
    if cohort == "ood":
        number = int(str(case_id).lstrip("0") or "0")
        release = number - 900 if number > 900 else number
        candidates = (
            AEROPATH_ROOT / "labelsTr" / f"ATM_{padded}.nii.gz",
            AEROPATH_ROOT / "labelsTr" / f"ATM_{padded}_0000.nii.gz",
            ROOT / "data" / "AeroPath" / str(release) / f"{release}_CT_HR_label_airways.nii.gz",
        )
    else:
        candidates = (
            ATM_ROOT / "labelsTr" / f"ATM_{padded}_0000.nii.gz",
            ATM_ROOT / "labelsTr" / f"ATM_{padded}.nii.gz",
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    listed = "\n".join(f"  - {c}" for c in candidates)
    raise SystemExit(f"No reference annotation for case {case_id}. Tried:\n{listed}")


def prediction_path(directory: Path, case_id: str) -> Path:
    padded = str(case_id).zfill(3)
    candidate = directory / f"ATM_{padded}.nii.gz"
    if not candidate.is_file():
        raise SystemExit(f"No prediction for case {case_id} in {directory}")
    return candidate


def _agree(masks: dict[str, np.ndarray]) -> None:
    shapes = {name: mask.shape for name, mask in masks.items()}
    if len(set(shapes.values())) > 1:
        raise SystemExit(f"Masks are on different grids and cannot be compared: {shapes}")


def load_case(
    case_id: str,
    cohort: str,
    prediction_dirs: dict[str, Path],
) -> dict:
    """Reference and one prediction per named arm, on one grid in stored orientation."""
    reference, affine = load_mask(reference_path(case_id, cohort))
    masks = {"reference": reference}
    for name, directory in prediction_dirs.items():
        mask, mask_affine = load_mask(prediction_path(directory, case_id))
        if not np.allclose(mask_affine, affine, atol=1e-4, rtol=1e-5):
            raise SystemExit(
                f"{name} for case {case_id} is not on the reference grid.\n"
                f"prediction affine:\n{mask_affine}\nreference affine:\n{affine}"
            )
        masks[name] = mask
    _agree(masks)
    return {"case_id": str(case_id), "cohort": cohort, "affine": affine, "masks": masks}


# --------------------------------------------------------------------------
# Class volumes. Background is -1, never 0: the surface sampler snaps a sample
# outward until it lands on a NON-NEGATIVE voxel, so a zero background would
# absorb boundary samples and speckle the render with an invented class.
# --------------------------------------------------------------------------
def _blank(shape) -> np.ndarray:
    return np.full(shape, -1, dtype=np.int16)


def error_classes(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    labels = _blank(prediction.shape)
    labels[prediction & reference] = TP
    labels[prediction & ~reference] = FP
    labels[~prediction & reference] = FN
    return labels


def reference_classes(reference: np.ndarray) -> np.ndarray:
    labels = _blank(reference.shape)
    labels[reference] = REFERENCE
    return labels


def reference_centreline(reference: np.ndarray) -> np.ndarray:
    """The reference skeleton the scorer measures tree length against.

    Imported from the scorer rather than re-implemented: tree length detected is the
    fraction of THIS skeleton a prediction covers, so a figure that skeletonised the
    reference any other way would be colouring a different quantity from the one the
    table reports.
    """
    from lung_airway_segmentation.metrics.external_masks import gt_centerline

    slices, skeleton, _ = gt_centerline(reference)
    full = np.zeros(reference.shape, dtype=bool)
    full[slices] = skeleton
    return full


def recovery_classes(
    control: np.ndarray,
    treatment: np.ndarray,
    reference: np.ndarray,
    *,
    basis: str = "centreline",
    dilate: int = 1,
    change_dilate: int = 2,
    include_false_positives: bool = False,
) -> tuple[np.ndarray, dict]:
    """Where the treatment changed the outcome, against the reference tree.

    ``basis="centreline"`` classifies the reference SKELETON, which is what tree length
    detected counts: the recovered and lost counts returned here divide by the skeleton
    size to give exactly the paired difference the results table reports. The voxel basis
    instead classifies every reference voxel, which is dominated by one-voxel disagreement
    along the walls of large airways and therefore says almost nothing about the branches
    the treatment is claimed to add.

    A skeleton is one voxel wide and would render as a thread, so it is dilated for
    display. Dilation happens AFTER counting and in ascending order of interest, so a
    recovered branch cannot be painted over by its neighbours and no count is inflated.
    The two changed classes are dilated further than the unchanged ones, for the same
    reason a scatter plot draws its highlighted points larger: on a typical case a
    hundred-odd centreline voxels move out of five thousand, and at print size a
    single-voxel thread of them is invisible. The counts on the panel are the true ones,
    and the caption is expected to say that the changed classes are drawn thicker.

    False positives are excluded by default: the claim is about reference branches
    recovered, and both arms' spurious voxels would put a fifth class on a surface that
    already carries four.
    """
    base = reference_centreline(reference) if basis == "centreline" else reference
    members = {
        BOTH_FOUND: base & control & treatment,
        MISSED_BOTH: base & ~control & ~treatment,
        LOST: base & control & ~treatment,
        RECOVERED: base & ~control & treatment,
    }
    if include_false_positives:
        members[EITHER_FP] = ~reference & (control | treatment)

    total = int(base.sum())
    counts = {RECOVERY_PALETTE[k][1]: int(v.sum()) for k, v in members.items()}
    counts["basis"] = basis
    counts["reference_total"] = total
    counts["implied_delta"] = (
        (counts[RECOVERY_PALETTE[RECOVERED][1]] - counts[RECOVERY_PALETTE[LOST][1]]) / total
        if total
        else float("nan")
    )

    labels = _blank(reference.shape)
    # Painted least-interesting first, so a dilated neighbour never buries a change.
    for value in (EITHER_FP, BOTH_FOUND, MISSED_BOTH, LOST, RECOVERED):
        if value not in members:
            continue
        member = members[value]
        iterations = change_dilate if value in (RECOVERED, LOST) else dilate
        if basis == "centreline" and iterations > 0 and member.any():
            member = ndi.binary_dilation(member, iterations=int(iterations))
        labels[member] = value
    return labels, counts


def postprocess_classes(
    raw: np.ndarray,
    filtered: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """What the trachea-seeded component filter took away, and whether it should have."""
    removed = raw & ~filtered
    labels = _blank(raw.shape)
    labels[filtered] = KEPT
    labels[removed & ~reference] = REMOVED_FP
    labels[removed & reference] = REMOVED_TP
    return labels


def class_counts(labels: np.ndarray, palette: dict[int, tuple[str, str]]) -> dict[str, int]:
    return {palette[k][1]: int((labels == k).sum()) for k in palette}


# --------------------------------------------------------------------------
# Surface extraction and projection, in one frame shared by every panel
# --------------------------------------------------------------------------
def shared_crop(volumes: list[np.ndarray], padding: int = 4) -> tuple[slice, slice, slice]:
    """One bounding box over every panel, so their millimetre frames coincide."""
    union = np.zeros(volumes[0].shape, dtype=bool)
    for labels in volumes:
        union |= labels > 0
    found = ndi.find_objects(union.astype(np.uint8))
    if not found:
        raise SystemExit("Every class volume is empty; nothing to render.")
    return tuple(
        slice(max(0, int(s.start) - padding), min(int(dim), int(s.stop) + padding))
        for s, dim in zip(found[0], union.shape)
    )


def surface_samples(
    labels: np.ndarray,
    zooms: np.ndarray,
    crop: tuple[slice, slice, slice],
    *,
    smooth_mm: float,
    taubin_iterations: int,
    clip_mm: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Isosurface samples of ``labels > 0``, each carrying its class.

    Adapted from ``generate_intro_airway_overview._surface_points``; the difference is
    that the crop is supplied rather than derived, which is what lets several panels share
    one millimetre frame. Coordinates are returned relative to the CROP ORIGIN, so a
    caller comparing panels must give them all the same crop.

    The scalar field is a signed distance, not a blurred binary mask: thresholding a
    blurred binary at a half erases a one-voxel tube outright, and the distal branches
    this document is about are exactly those tubes.
    """
    from skimage.measure import marching_cubes

    local_labels = labels[crop]
    local = local_labels > 0
    if not local.any():
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0, 3), np.float32),
            np.zeros(0, np.int16),
            {"surface_samples": 0},
        )

    inside = ndi.distance_transform_edt(local, sampling=zooms)
    outside = ndi.distance_transform_edt(~local, sampling=zooms)
    field = np.clip(inside - outside, -clip_mm, clip_mm).astype(np.float32)
    if smooth_mm > 0:
        field = ndi.gaussian_filter(field, sigma=smooth_mm / zooms, mode="nearest")

    retained = field > 0
    stats = {
        "voxels": int(local.sum()),
        "voxel_retention": round(float((retained & local).sum() / max(1, local.sum())), 4),
        "components_before": int(ndi.label(local, np.ones((3, 3, 3)))[1]),
        "components_after": int(ndi.label(retained, np.ones((3, 3, 3)))[1]),
    }

    vertices, faces, normals, _ = marching_cubes(
        field, level=0.0, spacing=tuple(float(z) for z in zooms)
    )
    vertices = vertices.astype(np.float32)
    normals = normals.astype(np.float32)
    if taubin_iterations > 0:
        vertices = _taubin_smooth(vertices, faces, iterations=taubin_iterations)
        smoothed = _vertex_normals(vertices, faces)
        if float(np.einsum("ij,ij->", smoothed, normals)) < 0:
            smoothed = -smoothed
        normals = smoothed

    points, point_normals = _densify(vertices, faces, normals)
    # Class of the voxel each sample falls in. Isosurface vertices sit ON the boundary,
    # so a rounded lookup lands many of them in background; the helper snaps each sample
    # outward through the 27-neighbourhood until it finds a labelled voxel.
    sample_class = _sample_generations(points / zooms, local_labels.astype(np.int16))
    stats["surface_samples"] = int(points.shape[0])
    stats["samples_without_class"] = int((sample_class < 0).sum())
    return points, point_normals, sample_class.astype(np.int16), stats


@dataclass(frozen=True)
class Frame:
    """The screen frame every panel of one figure is projected into."""

    rotation: np.ndarray
    origin_mm: np.ndarray
    u_reference: float
    v_reference: float
    height: int
    width: int
    px_mm: float
    margin_px: int = 12


def build_frame(
    point_clouds: list[np.ndarray],
    *,
    camera: Camera,
    px_mm: float,
    margin_px: int = 12,
) -> Frame:
    """Size one canvas around every panel's points, at one scale and one centre."""
    stacked = np.concatenate([p for p in point_clouds if len(p)], axis=0)
    rotation = _rotation(camera.azimuth, camera.elevation).astype(np.float32)
    origin = stacked.mean(axis=0)
    rotated = (stacked - origin) @ rotation.T
    screen_x, screen_y = rotated[:, 0], rotated[:, 2]
    u_reference = float(screen_x.max())
    v_reference = float(screen_y.max())
    width = int(np.ceil((u_reference - float(screen_x.min())) / px_mm)) + 2 * margin_px
    height = int(np.ceil((v_reference - float(screen_y.min())) / px_mm)) + 2 * margin_px
    return Frame(
        rotation=rotation,
        origin_mm=origin.astype(np.float32),
        u_reference=u_reference,
        v_reference=v_reference,
        height=height,
        width=width,
        px_mm=px_mm,
        margin_px=margin_px,
    )


def project(
    points: np.ndarray,
    normals: np.ndarray,
    classes: np.ndarray,
    frame: Frame,
    *,
    splat_radius_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-buffer the samples into ``frame``, carrying their normals and classes.

    Depth increases toward the camera, so painting in ascending depth order leaves the
    nearest write last: an exact z-buffer, not the average-z sort a ``Poly3DCollection``
    would give. An airway tree self-occludes constantly, and painter's-algorithm artefacts
    land exactly where the eye looks.
    """
    rotated = (points - frame.origin_mm) @ frame.rotation.T
    rotated_normals = normals @ frame.rotation.T

    screen_x, depth, screen_y = rotated[:, 0], rotated[:, 1], rotated[:, 2]
    # Patient right is drawn on the viewer's left, as when facing someone, and superior
    # at the top; both axes are therefore measured back from the frame's reference corner.
    u = (frame.u_reference - screen_x) / frame.px_mm + frame.margin_px
    v = (frame.v_reference - screen_y) / frame.px_mm + frame.margin_px

    # Back faces carry no information and only fight the front surface for pixels.
    front = rotated_normals[:, 1] > -0.05
    if front.sum() > 1000:
        depth, u, v = depth[front], u[front], v[front]
        rotated_normals = rotated_normals[front]
        classes = classes[front]

    order = np.argsort(depth)
    rows = np.clip(v[order].astype(np.int32), 0, frame.height - 1)
    cols = np.clip(u[order].astype(np.int32), 0, frame.width - 1)

    depth_image = np.full((frame.height, frame.width), -np.inf, dtype=np.float32)
    depth_image[rows, cols] = depth[order]
    normal_image = np.zeros((frame.height, frame.width, 3), dtype=np.float32)
    normal_image[rows, cols] = rotated_normals[order]
    class_image = np.full((frame.height, frame.width), -1, dtype=np.int16)
    class_image[rows, cols] = classes[order]

    radius = max(1, int(splat_radius_px))
    grid = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    disc = (grid[0] ** 2 + grid[1] ** 2) <= radius**2
    filled = ndi.maximum_filter(depth_image, footprint=disc, mode="nearest")

    painted = np.isfinite(depth_image)
    _, nearest = ndi.distance_transform_edt(~painted, return_indices=True)
    normal_image = normal_image[nearest[0], nearest[1]]
    class_image = class_image[nearest[0], nearest[1]]

    covered = np.isfinite(filled)
    filled[~covered] = np.inf
    class_image[~covered] = -1
    # Mild blur removes the faceting the nearest-neighbour fill leaves behind.
    normal_image = ndi.gaussian_filter(normal_image, sigma=(1.0, 1.0, 0))
    normal_image /= np.maximum(np.linalg.norm(normal_image, axis=-1, keepdims=True), 1e-6)
    return filled, class_image, normal_image


def colourise(
    depth: np.ndarray,
    classes: np.ndarray,
    shade: np.ndarray,
    specular: np.ndarray,
    palette: dict[int, tuple[str, str]],
) -> np.ndarray:
    covered = np.isfinite(depth)
    base = np.zeros(depth.shape + (3,), dtype=np.float32)
    base[...] = np.asarray(to_rgb(GREY), dtype=np.float32)
    for value, (colour, _) in palette.items():
        base[classes == value] = np.asarray(to_rgb(colour), dtype=np.float32)

    rgba = np.zeros(depth.shape + (4,), dtype=np.float32)
    rgba[..., :3] = np.clip(base * shade[..., None] + specular[..., None], 0.0, 1.0)
    rgba[..., 3] = covered.astype(np.float32)
    return rgba


def render_panels(
    panels: list[Panel],
    affine: np.ndarray,
    *,
    camera: Camera = Camera(),
    annotations: dict[str, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Render every panel into one shared frame. Returns the images and a stats record.

    Class volumes arrive in the orientation their masks were stored in and are reoriented
    to RAS+ here. Reorientation is a permutation and a flip of the voxel grid, so it moves
    no label; working in a known orientation is what lets the camera's azimuth and "superior
    at the top" mean what they say when the cohort files are stored LPS.
    """
    supersample = max(1, camera.supersample)
    px_mm = camera.px_mm / supersample

    oriented, zooms = [], None
    for panel in panels:
        labels, panel_zooms = _to_ras(np.ascontiguousarray(panel.labels), affine)
        oriented.append(np.asarray(labels, dtype=np.int16))
        zooms = np.asarray(panel_zooms, dtype=float)

    crop = shared_crop(oriented)
    samples, stats = [], {}
    for panel, labels in zip(panels, oriented):
        points, normals, classes, panel_stats = surface_samples(
            labels,
            zooms,
            crop,
            smooth_mm=camera.smooth_mm,
            taubin_iterations=camera.taubin_iterations,
        )
        if not len(points):
            raise SystemExit(f"Panel {panel.title!r} has no foreground to render.")
        samples.append((points, normals, classes))
        stats[panel.title] = panel_stats

    frame = build_frame([s[0] for s in samples], camera=camera, px_mm=px_mm)
    splat_radius = max(1, int(round(0.35 * float(np.max(zooms)) / px_mm)))

    # Project sparse feature masks into the same frame as the surfaces.  The caller can
    # use these bounds for objective A/B callouts without choosing boxes by eye.  Masks
    # are reoriented by the identical RAS+ transform used for the class volumes.
    annotation_boxes: dict[str, list[float]] = {}
    for name, mask in (annotations or {}).items():
        oriented_mask, annotation_zooms = _to_ras(
            np.ascontiguousarray(mask, dtype=np.uint8), affine
        )
        if not np.allclose(annotation_zooms, zooms):
            raise SystemExit(f"Annotation {name!r} does not share the panel geometry.")
        coordinates = np.argwhere(oriented_mask > 0)
        if not len(coordinates):
            continue
        crop_start = np.asarray([s.start for s in crop], dtype=np.float32)
        points = (coordinates.astype(np.float32) - crop_start) * zooms
        rotated = (points - frame.origin_mm) @ frame.rotation.T
        u = (frame.u_reference - rotated[:, 0]) / frame.px_mm + frame.margin_px
        v = (frame.v_reference - rotated[:, 2]) / frame.px_mm + frame.margin_px
        annotation_boxes[name] = [
            float(np.clip(u.min() / supersample, 0, frame.width / supersample - 1)),
            float(np.clip(v.min() / supersample, 0, frame.height / supersample - 1)),
            float(np.clip(u.max() / supersample, 0, frame.width / supersample - 1)),
            float(np.clip(v.max() / supersample, 0, frame.height / supersample - 1)),
        ]

    images = []
    for panel, (points, normals, classes) in zip(panels, samples):
        depth, class_image, normal_image = project(
            points, normals, classes, frame, splat_radius_px=splat_radius
        )
        # A restrained highlight preserves the class hue on one-voxel distal branches;
        # the old default was tuned for a single-colour anatomical render and bleached
        # precisely the small error regions these figures need to expose.
        shade, specular = _shade_normals(normal_image, depth, specular_strength=0.10)
        rgba = colourise(depth, class_image, shade, specular, panel.palette)
        images.append(_downsample(rgba, supersample))

    stats["frame"] = {
        "render_pixels": [frame.height, frame.width],
        "px_mm_final": camera.px_mm,
        "supersample": supersample,
        "azimuth_deg": camera.azimuth,
        "elevation_deg": camera.elevation,
        "smooth_mm": camera.smooth_mm,
        "taubin_iterations": camera.taubin_iterations,
        "splat_radius_px": splat_radius,
    }
    if annotation_boxes:
        stats["annotations"] = annotation_boxes
    return images, stats


# --------------------------------------------------------------------------
# Composition: panels in a row, one shared legend, saved as PDF and PNG
# --------------------------------------------------------------------------
def compose(
    images: list[np.ndarray],
    panels: list[Panel],
    *,
    pdf_dir: Path,
    png_dir: Path,
    stem: str,
    legend: list[tuple[str, str]] | dict[int, tuple[str, str]] | None = None,
    legend_columns: int = 4,
    caption: str = "",
    columns: int | None = None,
    overlays: dict[int, list[dict]] | None = None,
    row_headers: dict[int, tuple[str, str]] | None = None,
    row_gaps_in: dict[int, float] | None = None,
) -> Path:
    """Lay the renders out with their titles, one legend and an optional footnote.

    Titles and legend live in the artwork here, unlike the Introduction's bare panels.
    A comparison figure is only readable if the label sits against the render it names,
    and a LaTeX subcaption cannot do that for a class colour.
    """
    # A legend is an ORDERED LIST of (colour, name), not a class-id map: one figure can
    # show two palettes whose class ids happen to collide, and keying the legend by id
    # would silently drop an entry that the render still contains.
    entries = list(legend.values()) if isinstance(legend, dict) else list(legend or [])
    legend_rows = int(np.ceil(len(entries) / legend_columns)) if entries else 0

    count = len(images)
    columns = columns or count
    rows = int(np.ceil(count / columns))
    # Panels of one comparison share a frame and therefore an aspect ratio, but a caller
    # may also compose two patients rendered separately. Each axes is given its OWN
    # image's aspect ratio: imshow keeps the pixel aspect equal by shrinking the axes box
    # it was handed, and a shrunken box drags its title down with it, so panels sized to a
    # common height would come out with their titles at different heights.
    aspects = [image.shape[0] / image.shape[1] for image in images]
    # Panels give way to the text block rather than the other way round, so a four-panel
    # row is authored at 1.69 in a panel and included at \linewidth with no scaling.
    panel_in = min(PANEL_WIDTH_IN, TEXT_WIDTH_IN / columns)

    # Titles and metric strips are wrapped to the panel they belong to. A panel width that
    # depends on the column count means no caller can know how long a title may safely be,
    # and a title that overruns is not merely ugly: it is clipped at the figure's bounding
    # box, so the reader loses the end of the words rather than seeing them overlap.
    titles = [textwrap.wrap(p.title, width=max(8, int(panel_in / _TITLE_CHAR_IN)))
              for p in panels]
    subtitles = [
        [line for raw in p.subtitle.splitlines()
         for line in (textwrap.wrap(raw, width=max(8, int(panel_in / _CHAR_IN))) or [""])]
        for p in panels
    ]
    row_aspects = [
        max(aspects[row * columns:min((row + 1) * columns, count)])
        for row in range(rows)
    ]
    row_title_in = [
        0.155 * max(len(t) for t in titles[row * columns:min((row + 1) * columns, count)])
        + 0.06
        for row in range(rows)
    ]
    row_subtitle_in = [
        0.14 * max(len(s) for s in subtitles[row * columns:min((row + 1) * columns, count)])
        for row in range(rows)
    ]
    headers = row_headers or {}
    gaps = row_gaps_in or {}
    # A spanning row title is visually independent of the four panel titles below it.
    # The extra gap before a later diagnostic row prevents the preceding images from
    # being mistaken for the panels named by that title.
    row_prefix_in = [
        float(gaps.get(row, 0.0)) + (0.25 if row in headers else 0.0)
        for row in range(rows)
    ]
    legend_in = 0.19 * legend_rows + 0.08 if legend_rows else 0.0
    row_cell_in = [
        prefix + title + panel_in * aspect + subtitle
        for prefix, title, aspect, subtitle in zip(
            row_prefix_in, row_title_in, row_aspects, row_subtitle_in
        )
    ]

    # A legend entry cannot be broken mid-phrase, so on a one-panel figure the legend, not
    # the render, decides how wide the page has to be. The caption can be broken, so it is
    # wrapped to whatever width the panels and legend have already settled on rather than
    # widening the figure further.
    block_in = panel_in * columns
    legend_width_in = (
        legend_columns * (max(len(name) for _, name in entries) * _CHAR_IN + 0.26)
        if entries else 0.0
    )
    figure_width = max(block_in, legend_width_in)
    caption_lines = (
        textwrap.wrap(caption, width=max(20, int(figure_width / _CHAR_IN))) if caption else []
    )
    caption_in = 0.15 * len(caption_lines) + (0.05 if caption_lines else 0.0)
    figure_height = sum(row_cell_in) + legend_in + caption_in
    # Panels stay centred when the legend has widened the page.
    margin = (figure_width - block_in) / 2.0 / figure_width

    figure = plt.figure(figsize=(figure_width, figure_height))
    for row, (header, colour) in headers.items():
        if row < 0 or row >= rows:
            continue
        band_top = 1.0 - sum(row_cell_in[:row]) / figure_height
        gap = float(gaps.get(row, 0.0))
        header_top = band_top - (gap + 0.015) / figure_height
        figure.text(
            0.5, header_top, header, ha="center", va="top",
            fontsize=LABEL_PT, fontweight="bold", color=colour,
        )
        rule_y = band_top - (gap + 0.185) / figure_height
        figure.add_artist(plt.Line2D(
            [margin + 0.01, 1.0 - margin - 0.01], [rule_y, rule_y],
            transform=figure.transFigure, color=colour, linewidth=0.85,
            alpha=0.75,
        ))

    for index, (image, panel) in enumerate(zip(images, panels)):
        row, column = divmod(index, columns)
        band_top = 1.0 - sum(row_cell_in[:row]) / figure_height
        content_top = band_top - row_prefix_in[row] / figure_height
        height = panel_in * aspects[index] / figure_height
        axes = figure.add_axes((
            margin + (column + 0.01) * panel_in / figure_width,
            content_top - row_title_in[row] / figure_height - height,
            0.98 * panel_in / figure_width,
            height,
        ))
        axes.set_axis_off()
        axes.imshow(image, interpolation="none")
        for overlay in (overlays or {}).get(index, []):
            left, top, right, bottom = overlay["bbox"]
            # A little context around the projected feature makes the box readable on
            # the whole tree while leaving the zoom itself defined by physical padding.
            pad = max(8.0, 0.022 * max(image.shape[:2]))
            left = max(-0.5, left - pad)
            top = max(-0.5, top - pad)
            right = min(image.shape[1] - 0.5, right + pad)
            bottom = min(image.shape[0] - 0.5, bottom + pad)
            colour = overlay.get("colour", INK)
            axes.add_patch(Rectangle(
                (left, top), right - left, bottom - top,
                fill=False, edgecolor=colour, linewidth=1.35,
            ))
            axes.text(
                left + 2.5, top + 2.5, overlay.get("label", ""),
                ha="left", va="top", fontsize=LEGEND_PT, fontweight="bold",
                color="white",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": colour,
                      "edgecolor": "white", "linewidth": 0.45},
            )
        axes.set_title(chr(10).join(titles[index]), fontsize=LABEL_PT, color=INK, pad=3.0)
        if subtitles[index]:
            # Anchored to the CELL, not to the axes: two panels of different heights would
            # otherwise print their metric strips at two different heights.
            figure.text(
                margin + (column + 0.5) * panel_in / figure_width,
                content_top - (
                    row_title_in[row] + panel_in * row_aspects[row] + 0.04
                ) / figure_height,
                chr(10).join(subtitles[index]), ha="center", va="top",
                fontsize=TICK_PT, color=MUTED,
            )

    if entries:
        handles = [
            plt.Line2D([], [], marker="s", linestyle="none", markersize=5.4,
                       markerfacecolor=colour, markeredgecolor="none", label=name)
            for colour, name in entries
        ]
        figure.legend(handles=handles, loc="lower center",
                      ncol=min(len(handles), legend_columns),
                      fontsize=LEGEND_PT, frameon=False, handletextpad=0.35,
                      columnspacing=1.4,
                      bbox_to_anchor=(0.5, caption_in / figure_height))
    if caption_lines:
        figure.text(0.5, 0.004, chr(10).join(caption_lines), ha="center",
                    va="bottom", fontsize=LEGEND_PT, color=MUTED)

    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    destination = pdf_dir / f"{stem}.pdf"
    figure.savefig(destination, facecolor="white", dpi=600)
    figure.savefig(png_dir / f"{stem}.png", facecolor="white", dpi=400)
    plt.close(figure)
    # The width LaTeX must include it at. Anything else rescales the artwork and with it
    # every point size in the panel titles, subtitles, legend and caption strip.
    print(f"    include at width={figure_width / TEXT_WIDTH_IN:.2f}\\linewidth "
          f"({figure_width:.2f} in natural)")
    return destination
