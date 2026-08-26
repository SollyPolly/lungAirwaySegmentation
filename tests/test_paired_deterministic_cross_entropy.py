from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nnunet_trainers.nnUNetTrainer_MT240_PairedReplicates_NoDeepSupervision_NoMirroring import (
    DeterministicCrossEntropyLoss,
)


def test_deterministic_cross_entropy_matches_stock_loss_and_gradient():
    generator = torch.Generator().manual_seed(817)
    logits = torch.randn((1, 2, 7, 9, 5), generator=generator)
    target = torch.randint(0, 2, (1, 1, 7, 9, 5), generator=generator)
    target[..., 0, 0, 0] = -1

    stock_logits = logits.clone().requires_grad_(True)
    fixed_logits = logits.clone().requires_grad_(True)
    stock_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(
        stock_logits, target[:, 0]
    )
    fixed_loss = DeterministicCrossEntropyLoss(ignore_index=-1)(fixed_logits, target)
    stock_loss.backward()
    fixed_loss.backward()

    torch.testing.assert_close(fixed_loss, stock_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(fixed_logits.grad, stock_logits.grad, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_deterministic_cross_entropy_runs_under_strict_cuda_determinism():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        generator = torch.Generator(device="cuda").manual_seed(991)
        logits = torch.randn(
            (1, 2, 16, 20, 12), device="cuda", generator=generator
        )
        target = torch.randint(
            0, 2, (1, 1, 16, 20, 12), device="cuda", generator=generator
        )

        gradients = []
        losses = []
        for _ in range(2):
            trial = logits.clone().requires_grad_(True)
            loss = DeterministicCrossEntropyLoss()(trial, target)
            loss.backward()
            losses.append(loss.detach().cpu())
            gradients.append(trial.grad.detach().cpu())

        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=False)


def test_runner_only_archives_the_known_protocol_v1_failure():
    runner = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_nnunet_mt240_paired.sh"
    ).read_text()
    assert '"protocol_version": "mt240_full_state_epoch_seeded_v1"' in runner
    assert "failed_protocol_v1_" in runner
    assert "Refusing fresh start in unrecognized non-empty result directory" in runner
    assert "find \"$RESULT_DIR\" -maxdepth 1 -type f -name 'checkpoint*.pth'" in runner
