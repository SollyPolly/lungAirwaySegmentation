"""Distal crop-class computation — numpy/scipy/skimage only, no DL dependencies.

Kept deliberately light (no monai/torch) so the offline precompute
(``scripts/precompute_distal_classes.py``) can spawn many parallel workers without
each one importing the ~18 s monai/torch stack. ``monai_atm22`` re-exports this for
the training transform + tests.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def compute_distal_crop_classes(
        mask: np.ndarray,
        *,
        distal_radius_voxels: float = 2.0,
) -> np.ndarray:
    """3-class crop-guidance map from a binary airway mask (spatial dims only).

    * 0 = background
    * 1 = proximal / non-distal airway
    * 2 = distal airway = skeleton voxels whose EDT radius <= ``distal_radius_voxels``

    Class 2 is never left empty when any airway is present (falls back to the whole
    skeleton, then the mask) so ``RandCropByLabelClassesd`` does not warn / mis-weight.
    Shared by ``ComputeDistalCropClassesd`` and the offline precompute script so the
    on-disk maps are identical to the legacy on-the-fly transform. The pipeline does
    not resample, and the CT-foreground crop never cuts the airway, so skeletonising
    the native-resolution mask offline reproduces the cropped/padded class map exactly.
    """
    if distal_radius_voxels <= 0.0:
        raise ValueError("distal_radius_voxels must be positive.")
    binary = np.asarray(mask) > 0
    classes = np.zeros(binary.shape, dtype=np.uint8)
    if binary.any():
        classes[binary] = 1  # all airway starts as proximal
        radius = ndimage.distance_transform_edt(binary)
        skel = skeletonize(binary)
        distal = skel & (radius <= float(distal_radius_voxels))
        if not distal.any():
            distal = skel if skel.any() else binary
        classes[distal] = 2
    return classes
