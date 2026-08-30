"""Scout coronal display windows for the soft-skeletonisation Methods figure.

The figure needs a window whose max projection reads as ONE trunk with side
branches coming off it, rather than a single tube or an unreadable thicket.
This script scores candidate windows on the ground-truth skeleton and renders a
contact sheet so the choice is made by looking, not by guessing coordinates.

Run from the repository root::

    .venv\\Scripts\\python.exe dissertation\\scripts\\scout_soft_skeleton_patch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASE = "ATM_044"
GT_PATH = ROOT / "data" / "ATM22" / "labelsTr" / f"{CASE}_0000.nii.gz"
CT_PATH = ROOT / "data" / "ATM22" / "imagesTr" / f"{CASE}_0000.nii.gz"
OUT = ROOT / "dissertation" / "build" / "figures" / "scout"

# Display window size in voxels, matching the existing figure's 60 x 60 in-plane
# footprint and its five-slice coronal slab.
WIN_X, WIN_Z, SLAB_Y = 64, 64, 7
STRIDE = 16


def branch_score(skeleton_slab: np.ndarray) -> tuple[int, int]:
    """Return (branch-point count, skeleton voxel count) for a projected slab."""
    projected = np.max(skeleton_slab, axis=1)
    if not projected.any():
        return 0, 0
    thin = skeletonize(projected > 0)
    neighbours = ndimage.convolve(
        thin.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
    )
    junctions = int(np.count_nonzero(thin & (neighbours >= 4)))
    return junctions, int(thin.sum())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt = np.asarray(nib.load(GT_PATH).dataobj) > 0
    print("volume", gt.shape, "airway voxels", int(gt.sum()))

    # 3-D skeleton once, then slabs are taken from it.
    skeleton = skeletonize(gt)
    print("skeleton voxels", int(skeleton.sum()))

    xs = range(0, gt.shape[0] - WIN_X, STRIDE)
    zs = range(0, gt.shape[2] - WIN_Z, STRIDE)
    ys = range(0, gt.shape[1] - SLAB_Y, STRIDE // 2)

    candidates = []
    for y in ys:
        for x in xs:
            for z in zs:
                slab = skeleton[x:x + WIN_X, y:y + SLAB_Y, z:z + WIN_Z]
                junctions, length = branch_score(slab)
                # Want a trunk plus branches: enough length to read as a tree,
                # at least two divisions, but not a dense thicket.
                if length < 55 or length > 240 or junctions < 2 or junctions > 9:
                    continue
                candidates.append((junctions, length, x, y, z))

    candidates.sort(key=lambda row: (-row[0], -row[1]))
    print("candidates", len(candidates))
    if not candidates:
        print("no candidate windows; relax the thresholds")
        return

    # Spread the contact sheet over distinct locations rather than 24 near-copies.
    chosen: list[tuple[int, int, int, int, int]] = []
    for cand in candidates:
        if all(
            abs(cand[2] - other[2]) > WIN_X // 2
            or abs(cand[3] - other[3]) > SLAB_Y * 3
            or abs(cand[4] - other[4]) > WIN_Z // 2
            for other in chosen
        ):
            chosen.append(cand)
        if len(chosen) == 24:
            break

    ct = np.asarray(nib.load(CT_PATH).dataobj, dtype=np.float32)
    fig, axes = plt.subplots(4, 6, figsize=(15, 10.5), layout="constrained")
    for ax, (junctions, length, x, y, z) in zip(axes.ravel(), chosen):
        ct_slab = ct[x:x + WIN_X, y + SLAB_Y // 2, z:z + WIN_Z].T
        gt_slab = np.max(gt[x:x + WIN_X, y:y + SLAB_Y, z:z + WIN_Z], axis=1).T
        ax.imshow(ct_slab, cmap="gray", vmin=-1000, vmax=-350, origin="lower")
        ax.contour(gt_slab.astype(float), levels=[0.5], colors=["#2ca25f"], linewidths=1.0)
        ax.set_title(f"x{x} y{y} z{z}  j={junctions} L={length}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")

    path = OUT / "soft_skeleton_window_scout.png"
    fig.savefig(path, dpi=110)
    print("wrote", path)
    for row in chosen:
        print("  junctions=%d length=%3d  x=%3d y=%3d z=%3d" % row)


if __name__ == "__main__":
    main()
