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
# reaches the network only through the soft probabilities at the skeleton/mask.
# Behaviour-preserving differences from the repo: (1) sigmoid foreground prob instead
# of softmax max-channel; (2) the EDT is computed per-sample with scipy (unambiguous
# 3D, no MONAI/cucim shape dependency); (3) the morphological soft_skeleton above is
# used (the repo's default; its optional Euler-characteristic skeletonizer is an
# nnU-Net dependency we skip); (4) we return ``1 - cbDice`` (>=0, identical gradient)
# to match soft_cldice_loss rather than the repo's ``-cbDice``.
# ===========================================================================


def _normalise_voxel_spacing(
    voxel_spacing,
    *,
    batch_size: int,
    spatial_dims: int,
):
    """Return positive per-sample spacing with shape ``(B, spatial_dims)``.

    ``MetaTensor.pixdim`` is a tensor for one sample and a list of tensors after
    MONAI collation, so both forms are accepted alongside ordinary tuples/arrays.
    ``None`` intentionally means unit spacing for plain tensors and legacy callers.
    """
    import numpy as np

    if voxel_spacing is None:
        return np.ones((batch_size, spatial_dims), dtype=np.float64)

    if isinstance(voxel_spacing, (list, tuple)) and voxel_spacing and any(
        torch.is_tensor(value) for value in voxel_spacing
    ):
        spacing = np.stack(
            [torch.as_tensor(value).detach().cpu().numpy() for value in voxel_spacing]
        )
    elif torch.is_tensor(voxel_spacing):
        spacing = voxel_spacing.detach().cpu().numpy()
    else:
        spacing = np.asarray(voxel_spacing)

    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.ndim == 1:
        if spacing.shape[0] != spatial_dims:
            raise ValueError(
                f"voxel_spacing must contain {spatial_dims} values, got shape {spacing.shape}."
            )
        spacing = np.repeat(spacing[None, :], batch_size, axis=0)
    elif spacing.shape != (batch_size, spatial_dims):
        raise ValueError(
            "voxel_spacing must have shape "
            f"({spatial_dims},) or ({batch_size}, {spatial_dims}), got {spacing.shape}."
        )

    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("voxel_spacing values must be finite and positive.")
    return spacing


def _edt_per_sample(binary_mask: torch.Tensor, voxel_spacing=None) -> torch.Tensor:
    """Per-sample physical Euclidean distance transform.

    ``binary_mask`` is (B, D, H, W) in {0, 1}; returns the same shape/device. Used as
    a fixed (non-differentiable) radius weight, so it runs on CPU via scipy. Spacing
    may be one ``(D,H,W)`` tuple or one tuple per batch item; distances are then in
    physical units. Plain tensors default explicitly to unit spacing.
    """
    import numpy as np
    from scipy import ndimage

    array = binary_mask.detach().cpu().numpy().astype(bool)
    spacing = _normalise_voxel_spacing(
        voxel_spacing,
        batch_size=array.shape[0],
        spatial_dims=array.ndim - 1,
    )
    out = np.zeros(array.shape, dtype=np.float32)
    for index in range(array.shape[0]):
        if array[index].any():
            out[index] = ndimage.distance_transform_edt(
                array[index],
                sampling=tuple(float(value) for value in spacing[index]),
            )
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
    mask_soft: torch.Tensor,
    skel_soft: torch.Tensor,
    dim: int,
    voxel_spacing=None,
):
    """Radius-normalised (distance, skeleton-radius, thin-branch-importance) maps.

    Mirrors the repo ``get_weights``. ``mask_soft``/``skel_soft`` are soft for the
    prediction (carry gradient) and binary for the GT. The radius normalisation is a
    fixed weight (computed under no_grad from the hard mask); each returned map is
    multiplied by the soft input so the gradient flows through the probabilities only.
    """
    mask_hard = (mask_soft > 0.5).float()
    skel_hard = (skel_soft > 0.5).float()

    with torch.no_grad():
        spacing = _normalise_voxel_spacing(
            voxel_spacing,
            batch_size=mask_hard.shape[0],
            spatial_dims=mask_hard.ndim - 1,
        )
        distances = _edt_per_sample(mask_hard, voxel_spacing=spacing) * mask_hard
        skel_radius = distances * skel_hard

        dist_map_norm = torch.zeros_like(distances)
        skel_radius_norm = torch.zeros_like(distances)
        importance = torch.zeros_like(distances)
        eps = torch.finfo(distances.dtype).eps
        for index in range(distances.shape[0]):
            radius_i = skel_radius[index]
            positive_radius = radius_i[radius_i > 0]
            # The repo floors radius normalisation at the literal `1` (= one voxel,
            # because its EDT is in voxel units). With a physical EDT the faithful
            # floor is one voxel's smallest physical extent, min(spacing): the
            # minimum non-zero EDT distance is exactly min(spacing) (the nearest
            # background voxel one step along the finest axis), so every skeleton
            # radius is >= radius_floor. eps only guards a pathological zero spacing.
            radius_floor = torch.as_tensor(
                spacing[index].min(),
                device=distances.device,
                dtype=distances.dtype,
            )
            radius_floor = torch.clamp(radius_floor, min=eps)
            if positive_radius.numel() == 0:
                positive_distance = distances[index][distances[index] > 0]
                if positive_distance.numel() == 0:
                    continue
                radius_max = torch.clamp(positive_distance.max(), min=radius_floor)
            else:
                radius_max = torch.clamp(positive_radius.max(), min=radius_floor)
            # Reference: skel_radius_min = max(skel_radius.min(), 1). skel_radius is
            # zero off the skeleton, so skel_radius.min() == 0 over the volume and the
            # term collapses to the constant one-voxel floor — here radius_floor (and
            # radius_max >= radius_floor, so the minimum just returns the floor). With
            # radius_min == radius_floor <= every skeleton radius, the importance below
            # is bounded in (0, 1]; the first geomfix's literal `1.0` mm floor instead
            # exceeded sub-mm radii, pushing importance > 1 and the loss negative.
            radius_min = torch.minimum(radius_floor, radius_max)

            dist_map_norm[index] = torch.clamp(distances[index], max=radius_max) / radius_max
            skel_radius_norm[index] = radius_i / radius_max
            # subtraction-based inverse (linear): thin branches (small radius) get the
            # largest weight; squared in 3D. The public repo floors this at 1 voxel;
            # for physical EDTs the same-units floor is one voxel's smallest spacing.
            # Non-skeleton voxels are zeroed next.
            linear = (radius_max - radius_i + radius_min) / radius_max
            importance[index] = torch.clamp(linear if dim == 2 else linear ** 2, 0.0, 1.0)
        importance = importance * skel_hard

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
    (probabilities = sigmoid(logits); targets = binary mask). Radius-aware centerline
    Dice — see the reference/idea note above (Shi et al., MICCAI 2024). When
    ``voxel_spacing`` is omitted and ``targets`` is a MONAI ``MetaTensor``, spacing
    is read from ``targets.pixdim``; plain tensors use unit spacing.
    """
    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probabilities and targets must match: {tuple(probabilities.shape)} "
            f"!= {tuple(targets.shape)}"
        )
    if probabilities.ndim != 5:
        raise ValueError(
            f"soft_cbdice_loss expects 5D (B,1,D,H,W) tensors, got {tuple(probabilities.shape)}."
        )

    if voxel_spacing is None:
        voxel_spacing = getattr(targets, "pixdim", None)

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

    q_vl, q_slvl, q_sl = _cbdice_get_weights(
        true_mask, skel_true, dim, voxel_spacing=voxel_spacing
    )
    q_vp, q_spvp, q_sp = _cbdice_get_weights(
        pred_prob, skel_pred_prob, dim, voxel_spacing=voxel_spacing
    )

    w_tprec = (torch.sum(q_sp * q_vl) + smooth) / (
        torch.sum(_cbdice_combine_tensors(q_spvp, q_slvl, q_sp)) + smooth
    )
    w_tsens = (torch.sum(q_sl * q_vp) + smooth) / (
        torch.sum(_cbdice_combine_tensors(q_slvl, q_spvp, q_sl)) + smooth
    )
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
