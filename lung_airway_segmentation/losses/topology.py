"""Topology-aware and geometry-aware losses.

Soft centerline Dice (clDice, Shit et al. 2021) — a differentiable surrogate for
centerline/tree-length overlap. Volumetric Dice is dominated by thick proximal
airways, so it barely penalises hedging on thin distal branches; clDice weights a
one-voxel branch the same as the trachea by operating on the (soft) skeleton.

Soft skeletonisation uses iterated soft morphology (min/max pooling), so the whole
thing is differentiable and runs on the GPU. Sanity-check against the hard
`metrics.topology.cldice_score_from_masks`: 1 - soft_cldice_loss should track it.

Reference: https://github.com/jocpae/clDice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    """Reference 3D clDice erosion using three directional min-pools.

    A full ``3x3x3`` min-pool over-erodes diagonal neighbours and leaves a thick
    morphological remnant instead of the intended centreline. The reference clDice
    implementation erodes independently along each spatial axis and takes the
    minimum of those responses.
    """
    if image.ndim != 5:
        raise ValueError(f"soft_erode expects a 5D (B,C,D,H,W) tensor, got {tuple(image.shape)}.")

    depth = -F.max_pool3d(-image, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    height = -F.max_pool3d(-image, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    width = -F.max_pool3d(-image, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.minimum(torch.minimum(depth, height), width)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    """3D soft dilation: max-pool with unit stride to preserve shape."""
    return F.max_pool3d(image, kernel_size=3, stride=1, padding=1)


def soft_open(image: torch.Tensor) -> torch.Tensor:
    """Soft morphological opening (erode then dilate)."""
    return soft_dilate(soft_erode(image))


def soft_skeleton(image: torch.Tensor, iterations: int) -> torch.Tensor:
    """Differentiable soft skeleton of a [0, 1] mask, shape (B, C, D, H, W).

    ``iterations`` bounds the maximum structure radius that can be fully thinned;
    set it >= the largest expected airway radius in voxels (~10 at this resolution).
    """
    if image.ndim != 5:
        raise ValueError(f"soft_skeleton expects a 5D (B,C,D,H,W) tensor, got {tuple(image.shape)}.")
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(int(iterations)):
        image = soft_erode(image)
        opened = soft_open(image)
        delta = F.relu(image - opened)
        # add the newly exposed centerline voxels, without double-counting
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    iterations: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Return ``1 - soft clDice`` averaged over the batch.

    ``probabilities`` and ``targets`` are (B, 1, D, H, W) in [0, 1]
    (probabilities = sigmoid(logits); targets = binary mask).
    """
    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probabilities and targets must match: {tuple(probabilities.shape)} "
            f"!= {tuple(targets.shape)}"
        )

    skeleton_pred = soft_skeleton(probabilities, iterations)
    skeleton_true = soft_skeleton(targets, iterations)

    batch = probabilities.shape[0]
    skel_pred = skeleton_pred.reshape(batch, -1)
    skel_true = skeleton_true.reshape(batch, -1)
    pred = probabilities.reshape(batch, -1)
    true = targets.reshape(batch, -1)

    # topology precision: predicted centerline that lies inside the target
    t_prec = (torch.sum(skel_pred * true, dim=1) + smooth) / (torch.sum(skel_pred, dim=1) + smooth)
    # topology sensitivity: target centerline that lies inside the prediction
    t_sens = (torch.sum(skel_true * pred, dim=1) + smooth) / (torch.sum(skel_true, dim=1) + smooth)

    cl_dice = 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens)
    return cl_dice.mean()


class SoftClDiceLoss(nn.Module):
    """Module wrapper around :func:`soft_cldice_loss` (expects probabilities)."""

    def __init__(self, iterations: int = 10, smooth: float = 1.0):
        super().__init__()
        if int(iterations) < 1:
            raise ValueError("clDice iterations must be >= 1.")
        self.iterations = int(iterations)
        self.smooth = float(smooth)

    def forward(self, probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return soft_cldice_loss(probabilities, targets.float(), self.iterations, self.smooth)


# ===========================================================================
# Centerline Boundary Dice (cbDice) — geometry-aware topology loss.
#
# Reference: Pengcheng Shi, Jiesi Hu, Yanwu Yang, Zilve Gao, Wei Liu, Ting Ma,
#   "Centerline Boundary Dice Loss for Vascular Segmentation", MICCAI 2024.
#   Code: https://github.com/PengchengShi1220/cbDice  (loss/cbdice_loss.py).
#
# IDEA. clDice (above) scores the one-voxel skeleton overlap and is *radius-blind*:
# combined with Dice it under-weights thin branches and tolerates over-thick tubes
# ("diameter imbalance favouring larger vessels"). cbDice folds the Euclidean
# distance transform (= local radius) into the centerline precision/sensitivity, so
# the objective is matched to vessel *calibre* — it up-weights thin branches and
# penalises radius (wall-thickness) mismatch between prediction and GT. This is the
# loss-level lever for our fat-wall / radial-over-segmentation problem that the
# operating point, pos_weight and clDice-weight provably cannot move.
#
# Faithful binary (single-channel sigmoid) adaptation of the repo's SoftcbDiceLoss.
# As in the reference, the skeleton is taken on the *hard* mask under no_grad and the
# distance/radius maps are fixed weights (the EDT is non-differentiable); the gradient
# reaches the network only through the soft probabilities at the skeleton/mask. The
# structure below mirrors the repo's ``get_weights`` / ``combine_tensors`` 1:1 in
# VOXEL units with the literal one-voxel floors (``max(skel_radius, 1)``).
#
# The ONLY intentional deviations from the repo, each behaviour-preserving:
#   (1) sigmoid foreground prob instead of the softmax max-over-foreground-channels
#       (we are single-channel binary; numerically the same foreground prob map);
#   (2) the EDT runs per-sample via scipy in voxel units — identical in result to the
#       repo's ``monai.transforms.distance_transform_edt`` (which is also voxel-unit,
#       no physical spacing) — chosen only for unambiguous 3D handling;
#   (3) the morphological ``soft_skeleton`` above is used (the repo's default; its
#       optional Euler-characteristic skeletonizer is an nnU-Net dependency we skip);
#   (4) we return ``1 - cbDice`` (a proper [0, 1] loss, consistent with
#       soft_cldice_loss) rather than the repo's ``-cbDice`` — a constant offset with
#       an identical gradient.
#
# NOTE (2026-07-04): an earlier version of this port used a *physical-spacing* EDT
# (scipy ``sampling=spacing``) and a ``min(spacing)`` radius floor instead of the
# literal 1. That is NOT what the repo does (the repo is voxel-unit throughout) and it
# changed the radius normalisation / thin-branch importance. Reverted here to the
# faithful voxel-unit form; ``voxel_spacing`` is accepted-and-ignored for backward
# compatibility with existing callers/configs.
# ===========================================================================


def _edt_voxel_per_sample(binary_mask: torch.Tensor) -> torch.Tensor:
    """Per-sample Euclidean distance transform in VOXEL units.

    ``binary_mask`` is (B, D, H, W) in {0, 1}; returns the same shape/device. Matches
    the reference repo's ``monai.transforms.distance_transform_edt(mask)``, which is
    voxel-unit (no physical spacing). Computed as a fixed, non-differentiable radius
    weight, so it runs on CPU via scipy per sample (unambiguous 3D); ``sampling`` is
    left at its default (unit / voxel), identical in result to the repo's MONAI EDT.
    """
    import numpy as np
    from scipy import ndimage

    array = binary_mask.detach().cpu().numpy().astype(bool)
    out = np.zeros(array.shape, dtype=np.float32)
    for index in range(array.shape[0]):
        if array[index].any():
            out[index] = ndimage.distance_transform_edt(array[index])
    return torch.from_numpy(out).to(device=binary_mask.device, dtype=binary_mask.dtype)


def _cbdice_combine_tensors(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Repo ``combine_tensors``: take ``b*c``, but ``a*c`` where a is set and b is not.

    This is the radius-aware denominator: use the GT skeleton radius where it exists,
    else the predicted skeleton radius — so a calibre mismatch between the predicted
    and the true tube is exactly what the normalisation penalises.
    """
    a_c = a * c
    b_c = b * c
    combined = b_c.clone()
    only_a = (a != 0) & (b == 0)
    combined[only_a] = a_c[only_a]
    return combined


def _cbdice_get_weights(
    mask_input: torch.Tensor,
    skel_input: torch.Tensor,
    dim: int,
    prob_flag: bool,
):
    """Radius-normalised (distance, skeleton-radius, thin-branch-importance) maps.

    Faithful binary port of the repo ``get_weights`` in VOXEL units. When
    ``prob_flag`` the inputs are the SOFT prediction (mask prob / soft skeleton) and
    carry the gradient; otherwise they are the binary GT. All three maps are fixed
    weights computed under no_grad from the binarised mask/skeleton — exactly as in
    the repo, the radius normalisation uses ``max(skel_radius.max(), 1)`` and
    ``max(skel_radius.min(), 1)`` (the literal one-voxel floor). The gradient reaches
    the network only through the final multiply by the soft input.
    """
    if prob_flag:
        mask_soft = mask_input
        skel_soft = skel_input
        mask = (mask_soft > 0.5).float()
        skel = (skel_soft > 0.5).float()
    else:
        mask = mask_input
        skel = skel_input
        mask_soft = mask
        skel_soft = skel

    with torch.no_grad():
        distances = _edt_voxel_per_sample(mask) * mask  # distances[mask == 0] = 0
        skel_radius = distances * skel

        dist_map_norm = torch.zeros_like(distances)
        skel_radius_norm = torch.zeros_like(distances)
        importance = torch.zeros_like(distances)
        for index in range(distances.shape[0]):
            skel_i = skel_radius[index]
            # Repo: skel_radius_max = max(skel_i.max(), 1); skel_radius_min =
            # max(skel_i.min(), 1). skel_i is 0 off the skeleton, so its min over the
            # volume is 0 => radius_min is the constant one voxel. The literal 1 is the
            # one-voxel floor of the voxel-unit EDT; because skel_i >= 1 on the
            # skeleton, importance below is bounded in (0, 1] with no extra clamp.
            radius_max = torch.clamp(skel_i.max(), min=1.0)
            radius_min = torch.clamp(skel_i.min(), min=1.0)

            dist_map_norm[index] = torch.clamp(distances[index], max=radius_max) / radius_max
            skel_radius_norm[index] = skel_i / radius_max
            # subtraction-based inverse (linear): thin branches (small radius) get the
            # largest weight; squared in 3D. Non-skeleton voxels are zeroed next.
            linear = (radius_max - skel_i + radius_min) / radius_max
            importance[index] = linear if dim == 2 else linear ** 2
        importance = importance * skel  # 0 for non-skeleton voxels

    return dist_map_norm * mask_soft, skel_radius_norm * mask_soft, importance * skel_soft


def soft_cbdice_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    iterations: int = 10,
    smooth: float = 1.0,
    voxel_spacing=None,
) -> torch.Tensor:
    """Return ``1 - soft cbDice`` (centerline boundary Dice), averaged over the batch.

    ``probabilities`` and ``targets`` are (B, 1, D, H, W) in [0, 1]
    (probabilities = sigmoid(logits); targets = binary mask). Faithful binary port of
    the repo ``SoftcbDiceLoss`` (Shi et al., MICCAI 2024): VOXEL-unit EDT radius
    weights with the literal one-voxel floors. ``voxel_spacing`` is accepted for
    backward compatibility but IGNORED — the reference operates in voxel units, and
    passing physical spacing was a deviation (see the module note above).
    """
    del voxel_spacing  # accepted for backward compatibility; the repo is voxel-unit.

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probabilities and targets must match: {tuple(probabilities.shape)} "
            f"!= {tuple(targets.shape)}"
        )
    if probabilities.ndim != 5:
        raise ValueError(
            f"soft_cbdice_loss expects 5D (B,1,D,H,W) tensors, got {tuple(probabilities.shape)}."
        )

    dim = 3
    pred_prob = probabilities[:, 0]  # (B, D, H, W); carries the gradient

    # Skeletons on the HARD masks, detached: the EDT and skeleton are fixed weights;
    # the gradient reaches the network through the soft probabilities below.
    with torch.no_grad():
        true_mask = (targets[:, 0] > 0.5).float()
        pred_hard = (pred_prob > 0.5).float()
        skel_pred_hard = soft_skeleton(pred_hard.unsqueeze(1), iterations).squeeze(1)
        skel_true = soft_skeleton(true_mask.unsqueeze(1), iterations).squeeze(1)

    skel_pred_prob = skel_pred_hard * pred_prob  # gradient via pred_prob

    q_vl, q_slvl, q_sl = _cbdice_get_weights(true_mask, skel_true, dim, prob_flag=False)
    q_vp, q_spvp, q_sp = _cbdice_get_weights(pred_prob, skel_pred_prob, dim, prob_flag=True)

    w_tprec = (torch.sum(q_sp * q_vl) + smooth) / (
        torch.sum(_cbdice_combine_tensors(q_spvp, q_slvl, q_sp)) + smooth
    )
    w_tsens = (torch.sum(q_sl * q_vp) + smooth) / (
        torch.sum(_cbdice_combine_tensors(q_slvl, q_spvp, q_sl)) + smooth
    )
    # Repo returns ``-2·(prec·sens)/(prec+sens)``; we return ``1 - that`` so the term
    # is a proper [0, 1] loss consistent with soft_cldice_loss (constant offset,
    # identical gradient).
    return 1.0 - 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)


class SoftCbDiceLoss(nn.Module):
    """Module wrapper around :func:`soft_cbdice_loss` (expects probabilities)."""

    def __init__(self, iterations: int = 10, smooth: float = 1.0):
        super().__init__()
        if int(iterations) < 1:
            raise ValueError("cbDice iterations must be >= 1.")
        self.iterations = int(iterations)
        self.smooth = float(smooth)

    def forward(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        voxel_spacing=None,
    ) -> torch.Tensor:
        return soft_cbdice_loss(
            probabilities,
            targets.float(),
            self.iterations,
            self.smooth,
            voxel_spacing=voxel_spacing,
        )


# ===========================================================================
# *** EXPERIMENTAL / STRETCH — persistent-homology topology loss ***
#
# Wired into CombinedSegmentationLoss behind `topo_weight` (default 0.0 = OFF),
# ramped like clDice — but OFF by default and NOT on the dissertation critical
# path. clDice (above) is the primary, validated topology loss. This is a reference /
# starting point for "penalise wrong topology natively" — i.e. teach the network
# to produce the single connected tree that LCC currently enforces in post-
# processing. It is UNTESTED, SLOW (cubical persistence per volume per step), and
# must be validated on a few cases before any real run. See PROJECT_STATE.md.
#
# Backend: optional `torch-topological` (differentiable cubical PH + Wasserstein).
# `gudhi` can compute the diagrams too, but routing gradients to the critical
# voxels by hand is version-dependent and error-prone — prefer torch-topological.
#
# WHERE IT IS CLEANEST: FULL volumes, where the airway tree's target topology is a
# single connected component. On 96^3 patches the per-patch topology is noisy
# (branches are cut at the patch faces, creating artificial births/deaths) — which
# is one reason topology losses and larger/full-volume training pair well (Part 2).
# ===========================================================================


def persistent_homology_loss(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    dimensions: tuple = (0,),
) -> torch.Tensor:
    """Topology loss = distance between predicted and GT persistence diagrams. [EXPERIMENTAL]

    Hu et al. (2019): a segmentation has the right topology when its persistence
    diagram matches the ground truth's. Build the (differentiable) cubical
    persistence diagram of the predicted probability map and of the GT mask, then
    minimise the Wasserstein distance between them. This pushes predicted features
    (connected components in dim 0; loops in dim 1) toward the GT's — strengthening
    real branches and annihilating spurious disconnected blobs — WITHOUT assuming a
    fixed component count: a branch present in the GT is kept, one that is not is
    removed, by the matching.

    Why matching (not 'keep the k most-persistent components'): an airway tree is
    one component but its filtration shows many transient ones (branches appear,
    then merge). Real branches AND spurious blobs are both finite persistence pairs;
    only matching to the GT diagram distinguishes them. A naive top-k rule is wrong
    for trees — it keeps isolated blobs and kills connected branches.

    Args:
        probabilities: (B, 1, D, H, W) in [0, 1] (= sigmoid(logits)).
        targets: (B, 1, D, H, W) binary ground truth.
        dimensions: homology dimensions to penalise; (0,) = connected components only
            (cheapest, most relevant to the disconnected-blob problem). Add 1 to also
            penalise spurious loops (airway trees should have none).

    Returns:
        Scalar — mean Wasserstein diagram distance over the batch.

    Wiring (when ready): like clDice, behind its own weight and an epoch warm-up
    (the topology of an untrained network is noise):
        total += topo_weight * persistent_homology_loss(probabilities, targets)
    """
    try:
        from torch_topological.nn import CubicalComplex, WassersteinDistance
    except ImportError as exc:  # pragma: no cover - optional backend
        raise ImportError(
            "persistent_homology_loss requires the optional `torch-topological` backend "
            "(pip install torch-topological). EXPERIMENTAL / stretch — not on the critical path."
        ) from exc

    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probabilities and targets must match: {tuple(probabilities.shape)} "
            f"!= {tuple(targets.shape)}"
        )

    # VERIFY against your torch-topological version: the CubicalComplex / Distance
    # API and the per-dimension structure it returns have shifted across releases.
    cubical = CubicalComplex(dim=3)
    wasserstein = WassersteinDistance(q=2)

    batch_losses = []
    for predicted, target in zip(probabilities[:, 0], targets[:, 0].float()):  # (D, H, W)
        predicted_diagram = cubical(predicted)
        target_diagram = cubical(target)
        batch_losses.append(
            sum(wasserstein(predicted_diagram[d], target_diagram[d]) for d in dimensions)
        )

    return torch.stack(batch_losses).mean()
