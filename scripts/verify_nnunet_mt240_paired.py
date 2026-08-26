#!/usr/bin/env python3
"""Verify that paired MT240 arms are identical immediately before treatment.

Run this after epoch 5 checkpoints exist for the requested arms. The command
recomputes tensor hashes from the student and EMA checkpoint payloads rather
than trusting filenames or the emitted metadata alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


PROTOCOL_VERSION = "mt240_full_state_epoch_seeded_v2"
REPLICATE_SEEDS = {index: 2026082100 + index for index in range(1, 6)}
ARM_TRAINERS = {
    "control": "nnUNetTrainer_MT240Paired_Control_NoDeepSupervision_NoMirroring",
    "softcldice_w010": "nnUNetTrainer_MT240Paired_SoftCLDiceW010_NoDeepSupervision_NoMirroring",
    "plainmse_w010": "nnUNetTrainer_MT240Paired_PlainMSEW010_NoDeepSupervision_NoMirroring",
}


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_path(results_root: Path, replicate: int, arm: str) -> Path:
    trainer = ARM_TRAINERS[arm]
    return (
        results_root
        / f"rep_{replicate}"
        / "Dataset126_ATM22MT240LungCrop"
        / f"{trainer}__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "checkpoint_pretreatment_epoch005.pth"
    )


def _read_arm(results_root: Path, replicate: int, arm: str) -> dict[str, object]:
    checkpoint_path = _checkpoint_path(results_root, replicate, arm)
    teacher_path = Path(str(checkpoint_path) + ".mt")
    metadata_path = Path(str(checkpoint_path) + ".paired.json")
    for path in (checkpoint_path, teacher_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing paired pre-treatment artifact: {path}")

    main = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "paired_protocol_version": PROTOCOL_VERSION,
        "paired_arm": arm,
        "paired_replicate": replicate,
        "paired_seed": REPLICATE_SEEDS[replicate],
        "current_epoch": 5,
    }
    for field, value in expected.items():
        if main.get(field) != value or teacher.get(field) != value:
            raise RuntimeError(
                f"{arm} replicate {replicate}: {field} is not consistently {value!r}"
            )
        if metadata.get(field) != value:
            raise RuntimeError(
                f"{arm} replicate {replicate}: metadata {field} is not {value!r}"
            )
    transaction = main.get("paired_transaction_id")
    if not transaction or transaction != teacher.get("paired_transaction_id"):
        raise RuntimeError(f"{arm} replicate {replicate}: student/teacher transaction mismatch")
    if transaction != metadata.get("paired_transaction_id"):
        raise RuntimeError(f"{arm} replicate {replicate}: checkpoint/metadata transaction mismatch")

    student_hash = state_dict_sha256(main["network_weights"])
    teacher_hash = state_dict_sha256(teacher["teacher_weights"])
    if metadata.get("student_network_sha256") != student_hash:
        raise RuntimeError(f"{arm} replicate {replicate}: student metadata hash is false")
    if metadata.get("teacher_network_sha256") != teacher_hash:
        raise RuntimeError(f"{arm} replicate {replicate}: teacher metadata hash is false")
    return {
        "checkpoint": str(checkpoint_path),
        "transaction_id": transaction,
        "initial_checkpoint_sha256": main.get("paired_initial_checkpoint_sha256"),
        "initial_network_sha256": main.get("paired_initial_network_sha256"),
        "student_network_sha256": student_hash,
        "teacher_network_sha256": teacher_hash,
    }


def verify_replicate(results_root: Path, replicate: int, arms: list[str]) -> dict[str, object]:
    if replicate not in REPLICATE_SEEDS:
        raise ValueError(f"Replicate must be 1-5, got {replicate}")
    if len(arms) < 2:
        raise ValueError("At least two arms are required to verify a pair")
    records = {arm: _read_arm(results_root, replicate, arm) for arm in arms}
    comparison_fields = (
        "initial_checkpoint_sha256",
        "initial_network_sha256",
        "student_network_sha256",
        "teacher_network_sha256",
    )
    for field in comparison_fields:
        values = {str(record[field]) for record in records.values()}
        if len(values) != 1:
            raise RuntimeError(
                f"Replicate {replicate} is not pre-treatment matched for {field}: {records}"
            )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "replicate": replicate,
        "seed": REPLICATE_SEEDS[replicate],
        "arms": records,
        "pretreatment_state_exact_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("data/nnunet/nnUNet_results_paired"),
    )
    parser.add_argument("--replicate", type=int, action="append", required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(ARM_TRAINERS),
        default=["control", "softcldice_w010"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "results_root": str(args.results_root.resolve()),
        "replicates": [
            verify_replicate(args.results_root, replicate, args.arms)
            for replicate in args.replicate
        ],
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
