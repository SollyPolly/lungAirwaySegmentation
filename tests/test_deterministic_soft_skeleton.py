from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nnunet_trainers.nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring import (
    _max3_along,
    _soft_erode3d,
    _soft_open3d,
    _soft_skeleton3d,
)


def _pooled_erode(x: torch.Tensor) -> torch.Tensor:
    """The pre-fix implementation, kept here as the reference to match."""
    d = -F.max_pool3d(-x, (3, 1, 1), 1, (1, 0, 0))
    h = -F.max_pool3d(-x, (1, 3, 1), 1, (0, 1, 0))
    w = -F.max_pool3d(-x, (1, 1, 3), 1, (0, 0, 1))
    return torch.minimum(torch.minimum(d, h), w)


@pytest.mark.parametrize("dim,kernel,padding", [
    (2, (3, 1, 1), (1, 0, 0)),
    (3, (1, 3, 1), (0, 1, 0)),
    (4, (1, 1, 3), (0, 0, 1)),
])
def test_axis_max_matches_max_pool3d(dim, kernel, padding):
    generator = torch.Generator().manual_seed(20260830)
    x = torch.rand((2, 1, 6, 7, 5), generator=generator)
    torch.testing.assert_close(
        _max3_along(x, dim), F.max_pool3d(x, kernel, 1, padding), rtol=0, atol=0
    )


def test_erode_and_open_match_the_pooling_implementation():
    generator = torch.Generator().manual_seed(4321)
    x = torch.rand((2, 1, 6, 7, 5), generator=generator)
    torch.testing.assert_close(_soft_erode3d(x), _pooled_erode(x), rtol=0, atol=0)
    torch.testing.assert_close(
        _soft_open3d(x), F.max_pool3d(_pooled_erode(x), 3, 1, 1), rtol=0, atol=0
    )


def test_skeleton_backward_runs_under_strict_determinism():
    """The regression: soft-clDice used to raise on backward at epoch 5."""
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        generator = torch.Generator().manual_seed(99)
        x = torch.rand((1, 1, 8, 8, 8), generator=generator).requires_grad_(True)
        _soft_skeleton3d(x, 10).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
    finally:
        torch.use_deterministic_algorithms(False)
