"""Precompute distal crop-class maps to disk (airway hard-mining).

Writes a uint8 NIfTI per labelled ATM'22 case, aligned to its airway mask, with
``0 = background`` / ``1 = proximal airway`` / ``2 = distal skeleton`` (EDT radius
<= ``--radius``). Training loads these instead of skeletonising every epoch, which
removes the per-epoch ``skeletonize`` + EDT (the transient-allocation source behind
the OOM) and speeds every epoch. Run ONCE per (dataset, radius) before launching a
distal-sampling run; re-running skips files that already exist (use ``--overwrite``).

Per-case work is independent and CPU-bound (~26 s/case skeletonize+EDT on full ATM
volumes), so it runs in parallel by default across ``--jobs`` processes (auto = the
PBS-allocated cores, so the precompute step in train.pbs uses all 8 instead of one).

Behaviour-preserving: the supervised pipeline does not resample, and the CT-foreground
crop never cuts the airway, so skeletonising the native-resolution mask here reproduces
the legacy on-the-fly ``crop_classes`` exactly (see ``compute_distal_crop_classes``).

Example:
    python -u -m scripts.precompute_distal_classes --batch-root data/ATM22 --radius 2.0
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

# Import from the DL-free module (not monai_atm22) so spawned workers do NOT re-import
# the ~18 s monai/torch stack. ``resolve_project_path`` (which pulls torch via
# training.config) is imported lazily inside main(), parent-process only.
from lung_airway_segmentation.datasets.distal_classes import compute_distal_crop_classes
from lung_airway_segmentation.io.atm22_layout import (
    list_case_ids,
    resolve_case_paths,
    resolve_distal_classes_path,
)


def _process_case(
    case_id: str,
    batch_root: Path,
    radius: float,
    classes_root: Path | None,
    overwrite: bool,
) -> tuple[str, str, str | None]:
    """Compute + save one case's distal class map. Returns (case_id, status, info).

    Top-level (picklable) so it can run in a ``ProcessPoolExecutor`` worker.
    ``status`` is one of ``written`` / ``skipped`` / ``no_label``.
    """
    paths = resolve_case_paths(case_id, batch_root=batch_root)
    if paths["airway"] is None:
        return (str(case_id).zfill(3), "no_label", None)

    out_path = resolve_distal_classes_path(
        paths["case_id"], batch_root=batch_root, radius=radius, classes_root=classes_root
    )
    if out_path.is_file() and not overwrite:
        return (paths["case_id"], "skipped", str(out_path))

    mask_img = nib.load(str(paths["airway"]))
    mask = np.asanyarray(mask_img.dataobj)
    classes = compute_distal_crop_classes(mask, distal_radius_voxels=radius)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Fresh header from the mask's affine (no scl_slope/inter rescaling on reload);
    # same grid as the mask so the index-based CropForegroundd/SpatialPadd stay aligned.
    out_img = nib.Nifti1Image(classes.astype(np.uint8), affine=mask_img.affine)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(out_path))
    info = f"{out_path.name}  distal={int((classes == 2).sum())}  airway={int((classes >= 1).sum())}"
    return (paths["case_id"], "written", info)


def _default_jobs() -> int:
    """Default worker count.

    Each worker holds a full-volume float64 EDT (~1.5 GB on 512x512x700 ATM masks),
    so parallelism is RAM-bound, not just core-bound. Inside a PBS job we trust the
    allocation (train.pbs: 64 GB / 8 cores) and use ``$NCPUS``. Off PBS we fall back
    to a conservative min(4, cores) so a typical laptop does not thrash; bump it with
    ``--jobs`` if you have the RAM (budget ~2 GB/worker).
    """
    env = os.environ.get("NCPUS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(4, os.cpu_count() or 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--batch-root",
        required=True,
        help="ATM'22 root containing imagesTr/ + labelsTr/ (project-relative paths are resolved).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="EDT radius cutoff in voxels (<= this on the skeleton = distal). Default 2.0.",
    )
    parser.add_argument(
        "--classes-root",
        default=None,
        help="Output directory (default: <batch-root>/distalTr). Must be writable.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Specific case ids (e.g. 016 027); default = every labelled case under batch-root.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel worker processes (default: $NCPUS in a PBS job, else min(8, cores)). 1 = serial.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even when the output map already exists.",
    )
    args = parser.parse_args()

    if args.radius <= 0.0:
        parser.error("--radius must be positive.")

    # Lazy (parent-process only): training.config imports torch — keep it out of the
    # module top level so spawned workers stay light.
    from lung_airway_segmentation.training.config import resolve_project_path

    batch_root = resolve_project_path(args.batch_root)
    classes_root = resolve_project_path(args.classes_root) if args.classes_root else None
    case_ids = args.case_ids if args.case_ids else list_case_ids(batch_root)

    requested_jobs = args.jobs if (args.jobs and args.jobs > 0) else _default_jobs()
    jobs = max(1, min(requested_jobs, len(case_ids)))

    counts = {"written": 0, "skipped": 0, "no_label": 0}

    def tally(result: tuple[str, str, str | None]) -> None:
        case_id, status, info = result
        counts[status] += 1
        if status == "written":
            print(f"  {case_id}: {info}", flush=True)

    print(f"Precomputing {len(case_ids)} case(s) with {jobs} worker(s), radius={args.radius:g} ...", flush=True)
    if jobs == 1:
        for case_id in case_ids:
            tally(_process_case(case_id, batch_root, args.radius, classes_root, args.overwrite))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(_process_case, case_id, batch_root, args.radius, classes_root, args.overwrite)
                for case_id in case_ids
            ]
            for future in as_completed(futures):
                tally(future.result())

    resolved_root = classes_root if classes_root is not None else (batch_root / "distalTr")
    print(
        f"Done. written={counts['written']} skipped={counts['skipped']} "
        f"no_label={counts['no_label']} jobs={jobs} radius={args.radius:g} -> {resolved_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
