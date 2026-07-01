"""Precompute lung masks to disk for lung-cropped training (pipeline infrastructure).

Runs the ``lungmask`` pretrained lung-segmentation U-Net on each ATM'22 CT and saves a
binary lung mask NIfTI (``0`` background / ``1`` lung) aligned to the CT grid. Training
then crops CT + airway (+ distal maps) to the LUNG bounding box instead of the whole-body
CT foreground, concentrating capacity on the airway region — the biggest fixable gap vs
ATM'22 pipelines (PROJECT_STATE 2026-07-01). Run ONCE per dataset before a lung-cropped
run; re-running skips existing files (use ``--overwrite``).

WEIGHTS / OFFLINE NOTE: ``lungmask`` downloads pretrained weights on first use into the
torch-hub cache (``~/.cache/torch/hub/checkpoints`` unless ``TORCH_HOME`` is set). Compute
nodes usually have NO internet, so pre-fetch once on a node that DOES (the CX3 data-transfer
node dtn-c.cx3.hpc.ic.ac.uk) — either by running one case there, or with the one-liner:
    python -c "from lungmask import LMInferer; LMInferer()"
Point ``TORCH_HOME`` at shared storage if ``~/.cache`` is not visible from compute nodes.

Geometry: lungmask consumes a SimpleITK image and returns a label array on the SAME grid;
we copy the CT's geometry onto the saved mask, so it reloads aligned to the CT and airway
(both read from NIfTIs with matching affines).

Example:
    python -u -m scripts.precompute_lung_masks --batch-root data/ATM22
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from lung_airway_segmentation.io.atm22_layout import (
    list_case_ids,
    resolve_case_paths,
    resolve_lung_mask_path,
)


def build_inferer(force_cpu: bool):
    """Load the lungmask inferer once (weights are cached after the first download).

    Supports both the current ``LMInferer`` class and the legacy ``mask`` module; both
    expose ``.apply(sitk_image) -> np.ndarray`` (labels 0=bg, 1=right lung, 2=left lung).
    """
    try:
        from lungmask import LMInferer

        return LMInferer(force_cpu=force_cpu)
    except ImportError:
        # Legacy lungmask (< 0.2.14): module-level apply, no force_cpu knob.
        from lungmask import mask as legacy_mask

        return legacy_mask


def segment_lungs(inferer, ct_image: sitk.Image) -> np.ndarray:
    """Binary lung mask (uint8, z,y,x) for a CT SimpleITK image."""
    labels = inferer.apply(ct_image)  # 0=bg, 1/2 = lungs (or lobes); we only need lung>0
    return (np.asarray(labels) > 0).astype(np.uint8)


def process_case(
    case_id: str,
    batch_root: Path,
    lung_root: Path | None,
    inferer,
    overwrite: bool,
) -> tuple[str, str, str | None]:
    """Segment + save one case's lung mask. Returns (case_id, status, info)."""
    paths = resolve_case_paths(case_id, batch_root=batch_root)
    padded = paths["case_id"]
    out_path = resolve_lung_mask_path(padded, batch_root=batch_root, lung_root=lung_root)
    if out_path.is_file() and not overwrite:
        return (padded, "skipped", str(out_path))

    ct_image = sitk.ReadImage(str(paths["ct"]))
    lung = segment_lungs(inferer, ct_image)
    if int(lung.sum()) == 0:
        return (padded, "empty", None)  # lungmask found no lung — inspect this case

    lung_image = sitk.GetImageFromArray(lung)
    lung_image.CopyInformation(ct_image)  # identical geometry to the CT -> aligns with airway
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(lung_image, str(out_path))
    return (padded, "written", f"{out_path.name}  lung_voxels={int(lung.sum()):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-root", required=True,
                        help="ATM'22 root containing imagesTr/ (project-relative paths are resolved).")
    parser.add_argument("--lung-root", default=None,
                        help="Output directory (default: <batch-root>/lungTr). Must be writable.")
    parser.add_argument("--case-ids", nargs="*", default=None,
                        help="Specific case ids (e.g. 016 027); default = every case under batch-root.")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Run lungmask on CPU (default: GPU if available).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even when the output mask already exists.")
    args = parser.parse_args()

    # Lazy (keeps the module import light): training.config pulls torch.
    from lung_airway_segmentation.training.config import resolve_project_path

    batch_root = resolve_project_path(args.batch_root)
    lung_root = resolve_project_path(args.lung_root) if args.lung_root else None
    case_ids = args.case_ids if args.case_ids else list_case_ids(batch_root)

    print(f"Loading lungmask (force_cpu={args.force_cpu}) ...", flush=True)
    inferer = build_inferer(args.force_cpu)

    counts = {"written": 0, "skipped": 0, "empty": 0}
    print(f"Precomputing lung masks for {len(case_ids)} case(s) ...", flush=True)
    for case_id in case_ids:
        padded, status, info = process_case(case_id, batch_root, lung_root, inferer, args.overwrite)
        counts[status] = counts.get(status, 0) + 1
        if status in ("written", "empty"):
            print(f"  {padded}: {info if info else 'NO LUNG FOUND — inspect'}", flush=True)

    resolved_root = lung_root if lung_root is not None else (batch_root / "lungTr")
    print(
        f"Done. written={counts['written']} skipped={counts['skipped']} empty={counts.get('empty', 0)} "
        f"-> {resolved_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
