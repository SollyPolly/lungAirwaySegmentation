"""Measure what skeletonising at 2x resolution changes in the soft-clDice consistency term.

Motivation
----------
The soft skeleton in ``_soft_skeleton3d`` accumulates ``relu(x - open(x))`` and its
erosion is a 7-voxel cross.  For any structure at most two voxels thick the cross
erosion returns zero, so ``open(x) == 0``, ``relu(x - open(x)) == x``, and the
"skeleton" IS the object.  The proposed repair is to upsample the probability map
2x, skeletonise there with twice the iterations to cover the same physical radius,
and evaluate the clDice terms at that resolution.

This script decides whether that repair could increase the distal pull of the
consistency term, *without training anything*.

Part of the answer is analytic.  The topology-sensitivity term is

    t_sens = sum(skel_t * pred) / sum(skel_t)

whose derivative with respect to a student voxel is ``skel_t / sum(skel_t)``: through
this term alone, a distal branch's pull is exactly its share of the TEACHER's total
soft-skeleton mass.  That is why the bucketed skeleton mass share below is reported.

It is not the whole gradient.  The loss is ``1 - F_beta(t_prec, t_sens)``, and t_prec
depends on the STUDENT's skeleton, which 2x skeletonisation also thins.  The F_beta
combination then weights each term by the other.  So the bucketed share settles the
t_sens channel only, and the amputation and measured-gradient sections exist to
capture the rest.  The two can disagree in sign, and if they do, the measured
gradient is the quantity that governs training.

What is measured
----------------
1. ``--synthetic-check``: ideal tubes of 1..6 voxels thickness confirm the
   mechanism in isolation (does the degeneracy exist, does 2x remove it).
2. Real patches drawn from exported EMA-teacher probability maps, sized and
   sampled to match what the trainer's unlabelled stream actually sees:
     a. annihilation census - share of teacher foreground the cross erosion kills;
     b. soft-skeleton mass share by GT centreline radius, at 1x and 2x, on
        identical radius buckets (the decision quantity above);
     c. an amputation counterfactual - delete the distal tree from a surrogate
        student and measure how much more the clDice terms punish it at 2x;
     d. the measured gradient on distal versus proximal voxels, which folds in
        the t_prec term that (b) alone does not capture.
3. Feasibility: wall time and peak memory for the skeleton loop with and without
   gradient checkpointing, at 1x/10 iterations and 2x/20 iterations.

The morphology and the loss algebra are imported from the trainers, never
reimplemented, so a change to the pinned trainer changes this measurement too.

Results are written after every case, so a failure late in the run keeps the cases
that already finished.

Usage, from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\measure_soft_skeleton_scale.py

Inputs are the probability maps exported by::

    nnUNetv2_predict -i data/skeleton_scale_probe/predict_in \\
        -o data/skeleton_scale_probe/teacher_probabilities \\
        -d 126 -c 3d_fullres -f 0 -chk checkpoint_final.pth --save_probabilities \\
        -tr nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# The checkpointed 2x backward peaks near the ceiling of a small card, where a
# fragmented allocator can spend minutes reshuffling blocks. What actually fixes that
# here is releasing the cache between patches and probes (see ``_release_gpu``) and
# sharing one teacher skeleton per patch instead of recomputing it per amputation cut.
# This hint is a no-op on Windows, which does not support expandable segments, but
# helps on Linux; it must be set before the first CUDA allocation either way.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Running a file from ``dissertation/scripts/`` puts that directory, rather than the
# repository root, on sys.path. Add the root so the canonical trainer implementation
# imports.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lung_airway_segmentation.metrics.topology import _skeletonize
from nnunet_trainers.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (
    _soft_erode3d,
    _soft_open3d,
    _soft_skeleton3d,
)
from nnunet_trainers.nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring import (  # noqa: E501
    _cldice_from_skeletons,
)


PROBE_ROOT = ROOT / "data" / "skeleton_scale_probe"
DEFAULT_PROBABILITY_DIR = PROBE_ROOT / "teacher_probabilities"
DEFAULT_GROUND_TRUTH_DIR = ROOT / "data" / "ATM22" / "labelsTr"
DEFAULT_OUTPUT_DIR = PROBE_ROOT / "results"

# Trainer defaults for the reported soft-clDice arm. ``cldice_iters`` is scaled
# with the resolution so both scales cover the same physical radius.
BASE_ITERATIONS = 10
BASE_BETA = 1.0

# GT centreline radius buckets, in 1x INDEX units, so the same boundaries apply at
# both scales.
#
# A Euclidean distance transform of a binary mask is at least 1.0 everywhere inside
# it, so there is no sub-1.0 bucket: a centreline value of exactly 1.0 already means
# the structure is at most two voxels thick, which is the degeneracy zone. In general
# a centreline radius r corresponds to a thickness of about 2r-1 to 2r voxels, so the
# bucket labels state the thickness the bucket describes.
RADIUS_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("thick<=2", 1.0, 1.5),
    ("thick3-4", 1.5, 2.5),
    ("thick5-6", 2.5, 3.5),
    ("thick7-10", 3.5, 5.5),
    ("thick11-16", 5.5, 8.5),
    ("thick>16", 8.5, float("inf")),
)

# Radii below which the surrogate student's tree is amputated, in 1x index units.
# 1.5 removes only the degeneracy zone (at most two voxels thick); the larger cuts
# remove progressively more of the tree.
AMPUTATION_RADII: tuple[float, ...] = (1.5, 2.5, 3.5, 5.5)

# Halo around the GT bounding box for the case-level distance transforms, so the
# centreline near the box face is not a boundary artefact.
RADIUS_MARGIN_VOXELS = 32

GRADIENT_AMPUTATION_RADIUS = 1.5

# (1x, Nx) pair compared in every table. Set from --upsample-scale; 2 is the arm
# that was measured, 3 and above are exploratory and cost N**3 the memory.
SCALES: tuple[int, int] = (1, 2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probability-dir", type=Path, default=DEFAULT_PROBABILITY_DIR)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Case ids such as ATM_046. Default: every .npz in --probability-dir.",
    )
    parser.add_argument("--patches-per-case", type=int, default=6)
    parser.add_argument(
        "--upsample-scale",
        type=int,
        default=2,
        help="Compare 1x against this scale. Memory grows as scale**3.",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-synthetic-check",
        action="store_true",
        help="Skip the ideal-tube mechanism check.",
    )
    parser.add_argument(
        "--no-gradient-probe",
        action="store_true",
        help="Skip the measured-gradient and feasibility sections (they need a backward pass).",
    )
    return parser.parse_args()


def _is_out_of_memory(error: BaseException) -> bool:
    """Both torch.OutOfMemoryError and torch.AcceleratorError subclass RuntimeError."""
    message = str(error).lower()
    return "out of memory" in message or "cudaerrormemoryallocation" in message


def _release_gpu(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------
# Skeleton variants. The plain path is the pinned trainer function. The checkpointed path must be
# numerically identical to it; ``_assert_checkpoint_equivalence`` enforces that at run time.
# ---------------------------------------------------------------------------------------------------
def _skeleton_step(x: torch.Tensor, skeleton: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One iteration of ``_soft_skeleton3d``, transcribed so it can be recomputed."""
    x = _soft_erode3d(x)
    opened = _soft_open3d(x)
    delta = F.relu(x - opened)
    return x, skeleton + F.relu(delta - skeleton * delta)


def _checkpointed_soft_skeleton3d(x: torch.Tensor, iterations: int) -> torch.Tensor:
    """``_soft_skeleton3d`` with each iteration recomputed in the backward pass.

    The loop is pure min/max pooling: cheap to recompute, expensive to store. This
    trades the per-iteration activations for one extra forward pass over the loop.
    """
    opened = _soft_open3d(x)
    skeleton = F.relu(x - opened)
    for _ in range(int(iterations)):
        x, skeleton = checkpoint(_skeleton_step, x, skeleton, use_reentrant=False)
    return skeleton


def _assert_checkpoint_equivalence(device: torch.device) -> None:
    generator = torch.Generator(device="cpu").manual_seed(0)
    probe = torch.rand((1, 1, 24, 28, 20), generator=generator).to(device)
    with torch.no_grad():
        reference = _soft_skeleton3d(probe, BASE_ITERATIONS)
        candidate = _checkpointed_soft_skeleton3d(probe, BASE_ITERATIONS)
    deviation = float((reference - candidate).abs().max())
    if deviation != 0.0:
        raise AssertionError(
            f"Checkpointed skeleton deviates from the pinned trainer implementation by {deviation:.3e}."
        )


def _upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return x
    return F.interpolate(x, scale_factor=scale, mode="trilinear", align_corners=False)


def _nearest_upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return x
    return F.interpolate(x, scale_factor=scale, mode="nearest")


# ---------------------------------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------------------------------
def _load_plan_patch_shape(
    probability_dir: Path,
    zooms: tuple[float, ...],
) -> tuple[tuple[int, int, int], float]:
    """Map the 3d_fullres patch size onto nibabel (x, y, z) axis order.

    nnU-Net's array order reverses nibabel's, then ``transpose_forward`` permutes it,
    so plan axis ``j`` corresponds to nibabel axis ``2 - transpose_forward[j]``.

    The plan spacing is the dataset median, so individual cases deviate from it. The
    soft skeleton is measured on the case's own grid; the returned deviation records
    how far that grid is from the one the trainer optimises on. The morphology is
    isotropic in index units, so a few per cent of spacing deviation shifts only the
    physical radius each iteration covers, not the operation itself.
    """
    plans = json.loads((probability_dir / "plans.json").read_text())
    transpose_forward = plans["transpose_forward"]
    configuration = plans["configurations"]["3d_fullres"]
    patch_size = configuration["patch_size"]
    plan_spacing = configuration["spacing"]

    patch_shape = [0, 0, 0]
    deviation = 0.0
    for plan_axis, (size, spacing) in enumerate(zip(patch_size, plan_spacing)):
        nibabel_axis = 2 - transpose_forward[plan_axis]
        patch_shape[nibabel_axis] = int(size)
        deviation = max(deviation, abs(spacing - zooms[nibabel_axis]) / spacing)
    if deviation > 0.15:
        raise ValueError(
            "This case is too far from the plan grid for the measured skeleton to stand in for the "
            f"trainer's: plan spacing {plan_spacing} against zooms {zooms} ({deviation:.1%} deviation)."
        )
    return tuple(patch_shape), deviation  # type: ignore[return-value]


def _load_case(
    case_id: str,
    probability_dir: Path,
    ground_truth_dir: Path,
) -> dict[str, object]:
    probability_path = probability_dir / f"{case_id}.npz"
    mask_path = probability_dir / f"{case_id}.nii.gz"
    ground_truth_path = ground_truth_dir / f"{case_id}_0000.nii.gz"
    for path in (probability_path, mask_path, ground_truth_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing probe input: {path}")

    mask_image = nib.load(mask_path)
    zooms = tuple(float(value) for value in mask_image.header.get_zooms()[:3])
    # nnU-Net writes probabilities in its own axis order, which reverses nibabel's.
    with np.load(probability_path) as archive:
        foreground = np.ascontiguousarray(archive["probabilities"][1].transpose(2, 1, 0), dtype=np.float32)
    saved_mask = np.asanyarray(mask_image.dataobj).astype(np.uint8)
    if foreground.shape != saved_mask.shape:
        raise ValueError(f"{case_id}: probability shape {foreground.shape} against mask {saved_mask.shape}.")
    # Self-check on the axis mapping: thresholding must reproduce the exported mask.
    if not np.array_equal((foreground > 0.5).astype(np.uint8), saved_mask):
        raise ValueError(
            f"{case_id}: probabilities thresholded at 0.5 do not reproduce the exported mask, so the "
            "assumed axis order is wrong."
        )

    ground_truth_image = nib.load(ground_truth_path)
    if ground_truth_image.shape != saved_mask.shape:
        raise ValueError(
            f"{case_id}: ground truth shape {ground_truth_image.shape} against prediction {saved_mask.shape}."
        )
    ground_truth = np.ascontiguousarray(np.asanyarray(ground_truth_image.dataobj) > 0)

    return {
        "case_id": case_id,
        "foreground": foreground,
        "ground_truth": ground_truth,
        "zooms": zooms,
    }


def _bounding_box(mask: np.ndarray, margin: int) -> tuple[slice, ...]:
    slices = []
    for axis in range(mask.ndim):
        other_axes = tuple(index for index in range(mask.ndim) if index != axis)
        present = np.flatnonzero(mask.any(axis=other_axes))
        low = max(0, int(present[0]) - margin)
        high = min(mask.shape[axis], int(present[-1]) + 1 + margin)
        slices.append(slice(low, high))
    return tuple(slices)


def _case_centreline_radius_maps(
    ground_truth: np.ndarray,
    zooms: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Calibre of the GT structure each voxel belongs to, in index units and mm.

    Computed once per case over the GT bounding box rather than per patch: it is both
    cheaper and free of the patch-face artefacts a per-patch skeletonisation carries.

    Each voxel takes the distance-transform value at its nearest GT centreline voxel.
    Bucketing a voxel by its OWN distance transform would instead reproduce the
    wall-distance shell that earlier ``distal r=1`` numbers were retired for: it
    mixes tracheal surface voxels into the thinnest bin.
    """
    radius_index = np.zeros(ground_truth.shape, dtype=np.float32)
    radius_mm = np.zeros(ground_truth.shape, dtype=np.float32)
    if not ground_truth.any():
        return radius_index, radius_mm

    box = _bounding_box(ground_truth, RADIUS_MARGIN_VOXELS)
    inside = ground_truth[box]
    centreline = _skeletonize(inside)
    if not centreline.any():
        return radius_index, radius_mm

    distance_index = ndimage.distance_transform_edt(inside)
    distance_mm = ndimage.distance_transform_edt(inside, sampling=zooms)
    nearest = tuple(
        ndimage.distance_transform_edt(~centreline, return_distances=False, return_indices=True)
    )
    radius_index[box] = np.where(inside, distance_index[nearest], 0.0)
    radius_mm[box] = np.where(inside, distance_mm[nearest], 0.0)
    return radius_index, radius_mm


def _sample_patch_origins(
    foreground: np.ndarray,
    patch_shape: tuple[int, int, int],
    count: int,
    generator: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Draw patch origins centred on teacher hard foreground.

    Every logged unlabelled patch in the reported run had ``hard_active_patches = 1.000``,
    so foreground-centred sampling reproduces the population the diagnostics describe.
    """
    candidates = np.flatnonzero(foreground.ravel() > 0.5)
    if candidates.size == 0:
        return []
    chosen = generator.choice(candidates, size=min(count, candidates.size), replace=False)
    centres = np.unravel_index(chosen, foreground.shape)

    origins = []
    for centre in zip(*centres):
        origin = []
        for axis, (position, size) in enumerate(zip(centre, patch_shape)):
            limit = foreground.shape[axis] - size
            if limit < 0:
                raise ValueError(f"Patch {patch_shape} does not fit in volume {foreground.shape}.")
            origin.append(int(np.clip(position - size // 2, 0, limit)))
        origins.append(tuple(origin))
    return origins  # type: ignore[return-value]


def _patch_slices(origin: tuple[int, int, int], patch_shape: tuple[int, int, int]) -> tuple[slice, ...]:
    return tuple(slice(start, start + size) for start, size in zip(origin, patch_shape))


# ---------------------------------------------------------------------------------------------------
# Per-patch measurement
# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def _measure_patch_at_scale(
    probability: torch.Tensor,
    radius_index_1x: torch.Tensor,
    scale: int,
) -> dict[str, float]:
    """Degeneracy census, bucketed skeleton share and amputation cost at one scale.

    The teacher skeleton is the most expensive tensor here, so it is computed once and
    shared between the census and every amputation cut.
    """
    iterations = BASE_ITERATIONS * scale
    teacher = _upsample(probability, scale)
    radius = _nearest_upsample(radius_index_1x, scale)
    teacher_skeleton = _soft_skeleton3d(teacher, iterations)

    hard = (teacher > 0.5).float()
    eroded = _soft_erode3d(hard)
    foreground_voxels = float(hard.sum())
    skeleton_mass = float(teacher_skeleton.sum())
    annihilated = float(((hard > 0) & (eroded <= 0)).sum())

    record: dict[str, float] = {
        "iterations": float(iterations),
        "foreground_voxels": foreground_voxels,
        "skeleton_mass": skeleton_mass,
        # Dimensionless, so it is comparable across scales despite the 8x voxel count.
        "skeleton_mass_share_of_foreground": skeleton_mass / max(foreground_voxels, 1.0),
        "annihilated_foreground_fraction": annihilated / max(foreground_voxels, 1.0),
    }

    for name, low, high in RADIUS_BUCKETS:
        mask = (radius >= low) & (radius < high) & (radius > 0)
        bucket_skeleton = float((teacher_skeleton * mask).sum())
        bucket_foreground = float((hard * mask).sum())
        record[f"skel_share__{name}"] = bucket_skeleton / max(skeleton_mass, 1e-12)
        record[f"fg_share__{name}"] = bucket_foreground / max(foreground_voxels, 1.0)
        record[f"skel_per_fg__{name}"] = bucket_skeleton / max(bucket_foreground, 1.0)
    del hard, eroded

    intact_loss, intact_tprec, intact_tsens = _cldice_from_skeletons(
        teacher,
        teacher,
        teacher_skeleton,
        teacher_skeleton,
        beta=BASE_BETA,
    )
    record.update(
        {
            "intact_loss": float(intact_loss),
            "intact_tprec": float(intact_tprec),
            "intact_tsens": float(intact_tsens),
        }
    )

    teacher_hard_voxels = max(float((teacher > 0.5).sum()), 1.0)
    for cut in AMPUTATION_RADII:
        keep = ((radius >= cut) | (radius <= 0)).float()
        student = teacher * keep
        student_skeleton = _soft_skeleton3d(student, iterations)
        loss, tprec, tsens = _cldice_from_skeletons(
            student,
            teacher,
            student_skeleton,
            teacher_skeleton,
            beta=BASE_BETA,
        )
        tag = f"cut{cut:g}"
        record[f"amputated_loss__{tag}"] = float(loss)
        record[f"amputated_tsens__{tag}"] = float(tsens)
        record[f"amputated_tprec__{tag}"] = float(tprec)
        record[f"amputation_loss_penalty__{tag}"] = float(loss) - float(intact_loss)
        record[f"amputated_voxel_fraction__{tag}"] = float(
            ((teacher > 0.5) & (keep < 0.5)).sum() / teacher_hard_voxels
        )
        del keep, student, student_skeleton
    return record


def _measure_gradient(
    probability: torch.Tensor,
    radius_index_1x: torch.Tensor,
    scale: int,
    checkpointed: bool,
) -> dict[str, float]:
    """Measured gradient on distal versus proximal student voxels, plus cost.

    The gradient is taken with respect to the 1x student tensor at both scales, so
    the two numbers are directly comparable: at 2x it flows back through the
    trilinear interpolation to the same voxels.
    """
    iterations = BASE_ITERATIONS * scale
    teacher = _upsample(probability, scale)
    with torch.no_grad():
        teacher_skeleton = _soft_skeleton3d(teacher, iterations)

    keep_1x = ((radius_index_1x >= GRADIENT_AMPUTATION_RADIUS) | (radius_index_1x <= 0)).float()
    skeleton_fn = _checkpointed_soft_skeleton3d if checkpointed else _soft_skeleton3d
    device_type = probability.device.type

    def _forward_backward() -> tuple[torch.Tensor, torch.Tensor]:
        student_1x = (probability * keep_1x).clone().requires_grad_(True)
        student = _upsample(student_1x, scale)
        student_skeleton = skeleton_fn(student, iterations)
        loss, _, _ = _cldice_from_skeletons(
            student,
            teacher,
            student_skeleton,
            teacher_skeleton,
            beta=BASE_BETA,
        )
        loss.backward()
        return loss.detach(), student_1x.grad.detach()

    # The first pass at a new tensor size pays a one-off CUDA allocator and kernel
    # warm-up that can exceed the steady-state cost by two orders of magnitude. Time
    # the second pass, or the reported cost is an artefact rather than the method's.
    warmup_started = time.perf_counter()
    _forward_backward()
    if device_type == "cuda":
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    if device_type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss_value, gradient_tensor = _forward_backward()
    if device_type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_bytes = float(torch.cuda.max_memory_allocated()) if device_type == "cuda" else float("nan")

    gradient = gradient_tensor.abs()
    teacher_foreground = probability > 0.5
    distal = teacher_foreground & (radius_index_1x > 0) & (radius_index_1x < GRADIENT_AMPUTATION_RADIUS)
    proximal = teacher_foreground & (radius_index_1x >= GRADIENT_AMPUTATION_RADIUS)

    def _mean(mask: torch.Tensor) -> float:
        count = float(mask.sum())
        if count == 0.0:
            return float("nan")
        return float((gradient * mask).sum() / count)

    distal_mean = _mean(distal)
    proximal_mean = _mean(proximal)
    total_gradient = float(gradient.sum())
    return {
        "loss": float(loss_value),
        "distal_mean_abs_gradient": distal_mean,
        "proximal_mean_abs_gradient": proximal_mean,
        "distal_to_proximal_gradient_ratio": distal_mean / proximal_mean if proximal_mean else float("nan"),
        "distal_share_of_total_gradient": float((gradient * distal).sum()) / max(total_gradient, 1e-30),
        "distal_voxels": float(distal.sum()),
        "proximal_voxels": float(proximal.sum()),
        "seconds": elapsed,
        "warmup_seconds": warmup_seconds,
        "peak_gpu_bytes": peak_bytes,
        "checkpointed": float(checkpointed),
    }


# ---------------------------------------------------------------------------------------------------
# Synthetic mechanism check
# ---------------------------------------------------------------------------------------------------
@torch.no_grad()
def _synthetic_tube_check(device: torch.device) -> list[dict[str, float]]:
    """Ideal tubes of known thickness, to confirm the degeneracy exists at all.

    Each tube is a square-section cylinder of the given thickness in voxels, run
    along the last axis of a 48-voxel box, with probability 1 inside and 0 outside.
    """
    rows = []
    for thickness in range(1, 7):
        volume = torch.zeros((1, 1, 48, 48, 48), device=device)
        low = 24 - thickness // 2
        high = low + thickness
        volume[..., low:high, low:high, 4:44] = 1.0
        for scale in SCALES:
            iterations = BASE_ITERATIONS * scale
            scaled = _upsample(volume, scale)
            hard = (scaled > 0.5).float()
            eroded = _soft_erode3d(hard)
            skeleton = _soft_skeleton3d(scaled, iterations)
            foreground = float(hard.sum())
            rows.append(
                {
                    "thickness_voxels": float(thickness),
                    "scale": float(scale),
                    "iterations": float(iterations),
                    "foreground_voxels": foreground,
                    "skeleton_mass": float(skeleton.sum()),
                    "skeleton_mass_share_of_foreground": float(skeleton.sum()) / max(foreground, 1.0),
                    "annihilated_foreground_fraction": float(((hard > 0) & (eroded <= 0)).sum())
                    / max(foreground, 1.0),
                }
            )
    return rows


# ---------------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------------
def _mean_over(records: list[dict[str, float]], key: str) -> float:
    values = [record[key] for record in records if key in record and np.isfinite(record[key])]
    return float(np.mean(values)) if values else float("nan")


def _git_blob(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "hash-object", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _print_scale_table(per_scale: dict[int, list[dict[str, float]]]) -> None:
    print("\n=== Soft-skeleton mass share by GT centreline radius (index units, teacher map) ===")
    print(f"{'bucket':>11} {'fg share':>9} {'skel 1x':>9} {'skel 2x':>9} {'2x/1x':>7}")
    for name, _, _ in RADIUS_BUCKETS:
        foreground_share = _mean_over(per_scale[1], f"fg_share__{name}")
        share_1x = _mean_over(per_scale[SCALES[0]], f"skel_share__{name}")
        share_2x = _mean_over(per_scale[SCALES[1]], f"skel_share__{name}")
        ratio = share_2x / share_1x if share_1x else float("nan")
        print(f"{name:>11} {foreground_share:>9.4f} {share_1x:>9.4f} {share_2x:>9.4f} {ratio:>7.3f}")

    print("\n=== Degeneracy census and skeleton mass ===")
    for scale in SCALES:
        annihilated = _mean_over(per_scale[scale], "annihilated_foreground_fraction")
        share = _mean_over(per_scale[scale], "skeleton_mass_share_of_foreground")
        print(
            f"  {scale}x, {BASE_ITERATIONS * scale} iterations: "
            f"cross-erosion annihilates {annihilated:.4f} of teacher foreground; "
            f"skeleton mass is {share:.4f} of foreground"
        )


def _print_amputation_table(per_scale: dict[int, list[dict[str, float]]]) -> None:
    print("\n=== Amputation counterfactual: cost of a student that lost the distal tree ===")
    print(
        f"{'cut (vox)':>10} {'fg lost':>8} {'dLoss 1x':>10} {'dLoss 2x':>10} {'2x/1x':>7} "
        f"{'tsens 1x':>9} {'tsens 2x':>9}"
    )
    for cut in AMPUTATION_RADII:
        tag = f"cut{cut:g}"
        lost = _mean_over(per_scale[1], f"amputated_voxel_fraction__{tag}")
        delta_1x = _mean_over(per_scale[1], f"amputation_loss_penalty__{tag}")
        delta_2x = _mean_over(per_scale[SCALES[1]], f"amputation_loss_penalty__{tag}")
        ratio = delta_2x / delta_1x if delta_1x else float("nan")
        tsens_1x = _mean_over(per_scale[1], f"amputated_tsens__{tag}")
        tsens_2x = _mean_over(per_scale[SCALES[1]], f"amputated_tsens__{tag}")
        print(
            f"{cut:>10g} {lost:>8.4f} {delta_1x:>10.4f} {delta_2x:>10.4f} {ratio:>7.3f} "
            f"{tsens_1x:>9.4f} {tsens_2x:>9.4f}"
        )


def _print_gradient_table(gradient_records: list[dict[str, object]]) -> None:
    usable = [record for record in gradient_records if record.get("status") == "ok"]
    if not usable:
        return
    print("\n=== Measured gradient on the surrogate student (w.r.t. the 1x tensor) ===")
    print(
        f"{'scale':>5} {'ckpt':>5} {'distal':>11} {'proximal':>11} {'d/p':>7} "
        f"{'distal share':>13} {'s':>7} {'warm s':>8} {'peak GiB':>9}"
    )
    for record in usable:
        peak = record["peak_gpu_bytes"]
        peak_text = f"{peak / 2**30:.2f}" if np.isfinite(peak) else "n/a"  # type: ignore[operator]
        print(
            f"{record['scale']:>5g} {'yes' if record['checkpointed'] else 'no':>5} "
            f"{record['distal_mean_abs_gradient']:>11.3e} {record['proximal_mean_abs_gradient']:>11.3e} "
            f"{record['distal_to_proximal_gradient_ratio']:>7.3f} "
            f"{record['distal_share_of_total_gradient']:>13.4f} "
            f"{record['seconds']:>7.2f} {record['warmup_seconds']:>8.2f} {peak_text:>9}"
        )
    failed = [record for record in gradient_records if record.get("status") != "ok"]
    for record in failed:
        print(
            f"  {record['scale']}x checkpointed={bool(record['checkpointed'])} on "
            f"{record['case_id']}: {record['status']}"
        )


def _print_synthetic_table(rows: list[dict[str, float]]) -> None:
    print("\n=== Synthetic ideal tubes: does the degeneracy exist in isolation ===")
    print(f"{'thickness':>10} {'scale':>6} {'skel/fg':>9} {'annihilated':>12}")
    for row in rows:
        print(
            f"{row['thickness_voxels']:>10g} {row['scale']:>6g} "
            f"{row['skeleton_mass_share_of_foreground']:>9.4f} "
            f"{row['annihilated_foreground_fraction']:>12.4f}"
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    completed_cases: list[str],
    per_scale: dict[int, list[dict[str, float]]],
    patch_rows: list[dict[str, object]],
    gradient_rows: list[dict[str, object]],
    synthetic_rows: list[dict[str, float]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "script": "measure_soft_skeleton_scale.py",
        "seed": args.seed,
        "device": str(args.device),
        "completed_cases": completed_cases,
        "patches_per_case": args.patches_per_case,
        "base_iterations": BASE_ITERATIONS,
        "beta": BASE_BETA,
        "scales": list(SCALES),
        "radius_buckets": [[name, low, high] for name, low, high in RADIUS_BUCKETS],
        "amputation_radii": list(AMPUTATION_RADII),
        "gradient_amputation_radius": GRADIENT_AMPUTATION_RADIUS,
        "trainer_blobs": {
            "nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py": _git_blob(
                ROOT / "nnunet_trainers" / "nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py"
            ),
            "nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring.py": _git_blob(
                ROOT
                / "nnunet_trainers"
                / "nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring.py"
            ),
        },
        "aggregate": {
            f"{scale}x": {
                key: _mean_over(per_scale[scale], key)
                for key in sorted({key for record in per_scale[scale] for key in record})
            }
            for scale in SCALES
        },
        "gradient_probe": gradient_rows,
        "synthetic_tubes": synthetic_rows,
    }
    path = output_dir / "skeleton_scale_probe.json"
    path.write_text(json.dumps(summary, indent=2))
    _write_csv(output_dir / "skeleton_scale_probe_per_patch.csv", patch_rows)
    if synthetic_rows:
        _write_csv(output_dir / "skeleton_scale_probe_synthetic.csv", synthetic_rows)
    return path


def main() -> None:
    args = _parse_args()
    global SCALES
    if args.upsample_scale < 2:
        raise SystemExit("--upsample-scale must be at least 2.")
    SCALES = (1, int(args.upsample_scale))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)

    _assert_checkpoint_equivalence(device)
    _release_gpu(device)

    case_ids = args.cases
    if not case_ids:
        case_ids = sorted(path.stem for path in args.probability_dir.glob("*.npz"))
    if not case_ids:
        raise FileNotFoundError(
            f"No probability maps in {args.probability_dir}. Export them with --save_probabilities first."
        )

    # The ideal-tube check is cheap and independent of the cases, so run it first: a
    # later failure still leaves the mechanism result on disk.
    synthetic_rows = [] if args.no_synthetic_check else _synthetic_tube_check(device)
    _release_gpu(device)

    per_scale: dict[int, list[dict[str, float]]] = {scale: [] for scale in SCALES}
    patch_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    completed_cases: list[str] = []

    for case_number, case_id in enumerate(case_ids):
        case = _load_case(case_id, args.probability_dir, args.ground_truth_dir)
        zooms = case["zooms"]  # type: ignore[assignment]
        patch_shape, spacing_deviation = _load_plan_patch_shape(args.probability_dir, zooms)
        foreground = case["foreground"]  # type: ignore[assignment]
        ground_truth = case["ground_truth"]  # type: ignore[assignment]
        origins = _sample_patch_origins(foreground, patch_shape, args.patches_per_case, generator)
        radius_index_volume, radius_mm_volume = _case_centreline_radius_maps(ground_truth, zooms)
        print(
            f"[{case_id}] patch {patch_shape} voxels {zooms} mm "
            f"({spacing_deviation:.1%} from plan spacing), {len(origins)} patches",
            flush=True,
        )

        for patch_index, origin in enumerate(origins):
            slices = _patch_slices(origin, patch_shape)
            probability = torch.from_numpy(np.ascontiguousarray(foreground[slices])).to(device)[None, None]
            radius_index = torch.from_numpy(
                np.ascontiguousarray(radius_index_volume[slices])
            ).to(device)[None, None]
            radius_mm_patch = radius_mm_volume[slices]

            identity = {
                "case_id": case_id,
                "patch_index": patch_index,
                "spacing_deviation_from_plan": spacing_deviation,
                "origin": "-".join(str(value) for value in origin),
                "gt_voxels_in_patch": float(ground_truth[slices].sum()),
                "gt_centreline_radius_mm_p50": float(
                    np.median(radius_mm_patch[radius_mm_patch > 0])
                    if (radius_mm_patch > 0).any()
                    else np.nan
                ),
            }

            for scale in SCALES:
                record = dict(identity)
                record["scale"] = scale
                record.update(_measure_patch_at_scale(probability, radius_index, scale))
                per_scale[scale].append(
                    {key: value for key, value in record.items() if isinstance(value, float)}
                )
                patch_rows.append(record)
                _release_gpu(device)

            if not args.no_gradient_probe and patch_index == 0:
                # The non-checkpointed 2x pass is expected to exhaust memory. Record
                # that once rather than fragmenting the allocator on every case.
                configurations = [(1, False), (1, True), (SCALES[1], True)]
                if case_number == 0:
                    configurations.append((SCALES[1], False))
                for scale, checkpointed in configurations:
                    try:
                        stats: dict[str, object] = dict(
                            _measure_gradient(probability, radius_index, scale, checkpointed)
                        )
                        stats["status"] = "ok"
                    except RuntimeError as error:
                        if not _is_out_of_memory(error):
                            raise
                        stats = {
                            "checkpointed": float(checkpointed),
                            "status": "out_of_memory",
                            "distal_mean_abs_gradient": float("nan"),
                            "proximal_mean_abs_gradient": float("nan"),
                            "distal_to_proximal_gradient_ratio": float("nan"),
                            "distal_share_of_total_gradient": float("nan"),
                            "seconds": float("nan"),
                            "warmup_seconds": float("nan"),
                            "peak_gpu_bytes": float("nan"),
                        }
                        print(
                            f"  gradient probe {scale}x checkpointed={checkpointed}: out of memory",
                            flush=True,
                        )
                    stats.update({"case_id": case_id, "scale": scale})
                    gradient_rows.append(stats)
                    _release_gpu(device)

            del probability, radius_index
            _release_gpu(device)

        completed_cases.append(case_id)
        del case, foreground, ground_truth, radius_index_volume, radius_mm_volume
        # Written after every case so a later failure keeps what already finished.
        _write_outputs(
            args.output_dir, args, completed_cases, per_scale, patch_rows, gradient_rows, synthetic_rows
        )

    _print_scale_table(per_scale)
    _print_amputation_table(per_scale)
    _print_gradient_table(gradient_rows)
    if synthetic_rows:
        _print_synthetic_table(synthetic_rows)
    path = _write_outputs(
        args.output_dir, args, completed_cases, per_scale, patch_rows, gradient_rows, synthetic_rows
    )
    print(f"\nWrote {path} and the per-patch CSV for {len(completed_cases)} cases.")


if __name__ == "__main__":
    main()
