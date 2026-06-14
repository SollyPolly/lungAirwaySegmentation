"""Reproducibility helpers: deterministic seeding and run-environment capture.

The dissertation's headline result is a *paired* comparison (supervised vs
+clDice) decided by small mean differences over a handful of cases. For that
difference to be trustworthy, run-to-run noise must be controlled and every run
must record exactly which code produced it.

This module provides:
- ``seed_everything`` — seed Python, NumPy, and torch (CPU + CUDA), and (by
  default) put cuDNN into deterministic mode. Call once, early, per run.
- ``seed_worker`` / ``make_seeded_generator`` — make DataLoader shuffling and any
  per-worker NumPy/random draws reproducible.
- ``collect_environment_metadata`` — git commit/branch/dirty flag plus key
  library versions, for stamping into ``run_metadata.json``.
"""

import os
import platform
import random
import subprocess

import numpy as np
import torch

from lung_airway_segmentation.settings import PROJECT_ROOT


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed all RNGs and configure cuDNN for a reproducible run.

    ``deterministic=True`` (the default for the dissertation's paired
    comparisons) forces cuDNN to pick deterministic convolution algorithms and
    disables the autotuner. This costs some throughput — notably on 3D convs —
    so set ``deterministic: false`` in the training config when a run only needs
    speed (e.g. a throwaway sanity check), not a defensible number.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs this to use a deterministic workspace (CUDA >= 10.2).
        # setdefault so an explicit user/HPC value is never overridden.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: some 3D upsample/pool backward kernels have no deterministic
        # implementation; warn rather than crash a 12h job over them.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch without warn_only
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def make_seeded_generator(seed: int) -> torch.Generator:
    """Return a torch.Generator seeded for reproducible DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: seed NumPy/random per worker.

    Each worker derives its seed from torch's per-worker initial seed, which is
    itself a function of the loader's seeded generator — so the whole pipeline is
    reproducible across runs without every worker drawing the same samples.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _git_info() -> dict:
    """Return the current git commit, branch, and working-tree dirty flag."""

    def _run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status = _run(["status", "--porcelain"])
    return {
        "commit": _run(["rev-parse", "HEAD"]),
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
    }


def _package_versions() -> dict:
    """Return installed versions of the libraries that affect numerical results."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in ("torch", "monai", "numpy", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def collect_environment_metadata() -> dict:
    """Capture code version + environment for stamping into run metadata.

    Never raises: a missing git binary or metadata entry yields ``None`` for that
    field rather than failing the run.
    """
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git": _git_info(),
        "cuda": {
            "available": cuda_available,
            "torch_cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        },
    }
