#!/bin/bash
# Shared fail-closed runner for the three MT240 paired-replicate PBS arrays.

set -euo pipefail

PROJECT=/rds/general/user/dl525/home/projects/dissertation/LungAirwaySegmentation
cd "$PROJECT"

ARM="${PAIRED_ARM:?Set PAIRED_ARM to control, softcldice_w010, or plainmse_w010}"
REPLICATE="${PBS_ARRAY_INDEX:?Submit this runner through a paired PBS array}"
case "$REPLICATE" in
    1|2|3|4|5) ;;
    *) echo "Unsupported paired replicate '$REPLICATE'"; exit 1 ;;
esac
SEED=$((2026082100 + REPLICATE))

module load miniforge/3
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ctfm

export nnUNet_raw="$PROJECT/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT/data/nnunet/nnUNet_preprocessed"
CANONICAL_RESULTS="$PROJECT/data/nnunet/nnUNet_results"
export nnUNet_results="$PROJECT/data/nnunet/nnUNet_results_paired/rep_${REPLICATE}"
export TORCH_HOME="$HOME/.torchcache"
# Required by the trainer: multiprocessing completion order must not change
# which augmentation RNG draws reach a particular optimizer step.
export nnUNet_n_proc_DA=0
export MT_PAIRED_DETERMINISTIC=1
export MT_PAIRED_REPLICATE="$REPLICATE"
export MT_PAIRED_SEED="$SEED"
export MT_PAIRED_INIT_CHECKPOINT="$CANONICAL_RESULTS/Dataset123_ATM22L20LungCrop/nnUNetTrainer_NoDeepSupervision_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

DATASET=126
FOLD=0
MODULE=nnUNetTrainer_MT240_PairedReplicates_NoDeepSupervision_NoMirroring
case "$ARM" in
    control)
        TRAINER=nnUNetTrainer_MT240Paired_Control_NoDeepSupervision_NoMirroring
        ;;
    softcldice_w010)
        TRAINER=nnUNetTrainer_MT240Paired_SoftCLDiceW010_NoDeepSupervision_NoMirroring
        ;;
    plainmse_w010)
        TRAINER=nnUNetTrainer_MT240Paired_PlainMSEW010_NoDeepSupervision_NoMirroring
        ;;
    *) echo "Unknown PAIRED_ARM '$ARM'"; exit 1 ;;
esac

RESULT_DIR="$nnUNet_results/Dataset126_ATM22MT240LungCrop/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}"
MANIFEST_DIR="$nnUNet_results/paired_launch_manifests"
MANIFEST="$MANIFEST_DIR/${ARM}.json"
mkdir -p "$MANIFEST_DIR"

# Ten array members may become eligible together. Keep different
# arm/replicate pairs independent while refusing an accidental duplicate of
# the same pair. An advisory lock is released automatically if the job exits
# or its node fails, so it cannot leave a stale sentinel that blocks resume.
command -v flock >/dev/null 2>&1 || {
    echo "flock is required to protect paired result directories"
    exit 1
}
LOCK_DIR="$PROJECT/logs/nnunet_locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/mt240_paired_${ARM}_rep${REPLICATE}.lock"
flock -n 9 || {
    echo "Another job is already running paired arm=$ARM replicate=$REPLICATE"
    exit 1
}

NNUNET_PACKAGE=$(python -c 'from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).resolve().parent)')
TRAINER_DIR="$NNUNET_PACKAGE/training/nnUNetTrainer/variants/network_architecture"
test -d "$TRAINER_DIR" || { echo "Cannot find nnU-Net trainer directory: $TRAINER_DIR"; exit 1; }

verify_blob() {
    local source_file=$1
    local expected_blob=$2
    local actual_blob
    local head_blob
    test -f "$source_file" || { echo "Missing source: $source_file"; exit 1; }
    git ls-files --error-unmatch "$source_file" >/dev/null 2>&1 || {
        echo "Refusing paired launch: source is not tracked: $source_file"
        exit 1
    }
    actual_blob=$(git hash-object "$source_file")
    head_blob=$(git rev-parse "HEAD:$source_file")
    test "$actual_blob" = "$expected_blob" && test "$head_blob" = "$expected_blob" || {
        echo "Refusing paired launch: source changed: $source_file"
        echo "expected=$expected_blob working_tree=$actual_blob HEAD=$head_blob"
        exit 1
    }
}

verify_blob nnunet_trainers/nnUNetTrainer_NoDeepSupervision_NoMirroring.py \
    23af9756e44a5f63d7e136b2177276c8a9299ae0
verify_blob nnunet_trainers/nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py \
    7d840298c711ca4c6a5e60c1c47f5b63d9370d95
verify_blob nnunet_trainers/nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring.py \
    c80fe415da7823bc491333c4e70fb4ae94d5f8cc
verify_blob nnunet_trainers/nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring.py \
    34131e294f560a5b34d57298cf0ebe7e95eb1af1
verify_blob nnunet_trainers/nnUNetTrainer_MeanTeacher_VoxelMSE_NoDeepSupervision_NoMirroring.py \
    ba7071d6473e6eda130bfea4ff39cfda9428bbc4
verify_blob "nnunet_trainers/${MODULE}.py" 6b57d7f8c835fb4fec92e97a605cb48dc3c35984
verify_blob configs/nnunet/mt240_paired_replicates.json efade3659ce973bb4f1cee44488e634a93e0a9f6

ensure_same_trainer() {
    local source_file="$PROJECT/nnunet_trainers/$1"
    local target_file="$TRAINER_DIR/$1"
    test -f "$source_file" || { echo "Missing trainer: $source_file"; exit 1; }
    if test -e "$target_file" || test -L "$target_file"; then
        cmp -s "$source_file" "$target_file" || {
            echo "Refusing to replace different installed trainer: $target_file"
            exit 1
        }
    elif ! ln -s "$source_file" "$target_file" 2>/dev/null; then
        # Concurrent array members can all observe an absent link before the
        # first one creates it. Accept the winner only when it installed the
        # exact reviewed source; otherwise fail closed.
        if test -e "$target_file" || test -L "$target_file"; then
            cmp -s "$source_file" "$target_file" || {
                echo "Concurrent install produced a different trainer: $target_file"
                exit 1
            }
        else
            echo "Could not install trainer: $target_file"
            exit 1
        fi
    fi
}

for TRAINER_FILE in \
    nnUNetTrainer_NoDeepSupervision_NoMirroring.py \
    nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py \
    nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring.py \
    nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring.py \
    nnUNetTrainer_MeanTeacher_VoxelMSE_NoDeepSupervision_NoMirroring.py \
    "${MODULE}.py"
do
    ensure_same_trainer "$TRAINER_FILE"
done

test -s "$MT_PAIRED_INIT_CHECKPOINT" || {
    echo "Missing Dataset123 full-state checkpoint: $MT_PAIRED_INIT_CHECKPOINT"
    exit 1
}
ACTUAL_SEED_SHA=$(sha256sum "$MT_PAIRED_INIT_CHECKPOINT" | awk '{print $1}')
EXPECTED_SEED_SHA=2f7344a2cdab8d2fa4e43c600a8234f7c73585903df8068d92a25bb6c2e42c5e
test "$ACTUAL_SEED_SHA" = "$EXPECTED_SEED_SHA" || {
    echo "Dataset123 checkpoint SHA-256 mismatch: $ACTUAL_SEED_SHA"
    exit 1
}

export PAIRED_ARM TRAINER RESULT_DIR MANIFEST PROJECT MODULE ACTUAL_SEED_SHA

# Protocol v1 reached epoch 0 but failed before its first optimizer step because
# PyTorch's reduced CUDA NLL kernel rejects strict deterministic mode. nnU-Net
# nevertheless populated RESULT_DIR with logs/configuration files, which the
# normal fresh-start guard correctly treats as unsafe. Permit exactly that
# known manifest state to be archived (never deleted) before the v2 retry.
if test -d "$RESULT_DIR" && test -n "$(find "$RESULT_DIR" -mindepth 1 -print -quit)" && \
        ! test -f "$RESULT_DIR/checkpoint_latest.pth" && \
        ! test -f "$RESULT_DIR/checkpoint_final.pth"; then
    if find "$RESULT_DIR" -maxdepth 1 -type f -name 'checkpoint*.pth' -print -quit | grep -q .; then
        echo "Refusing to archive incomplete result directory containing a checkpoint: $RESULT_DIR"
        exit 1
    fi
    if python - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
if not manifest.is_file():
    raise SystemExit(1)
payload = json.loads(manifest.read_text())
expected = {
    "status": "launching",
    "protocol_version": "mt240_full_state_epoch_seeded_v1",
    "arm": os.environ["PAIRED_ARM"],
    "replicate": int(os.environ["MT_PAIRED_REPLICATE"]),
}
if any(payload.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
if Path(payload.get("result_dir", "")).resolve() != Path(os.environ["RESULT_DIR"]).resolve():
    raise SystemExit(1)
PY
    then
        FAILED_RESULT_DIR="${RESULT_DIR}.failed_protocol_v1_$(date -u +%Y%m%dT%H%M%SZ)"
        test ! -e "$FAILED_RESULT_DIR" || {
            echo "Refusing to overwrite archived failed result: $FAILED_RESULT_DIR"
            exit 1
        }
        mv -- "$RESULT_DIR" "$FAILED_RESULT_DIR"
        echo "Archived protocol-v1 pre-checkpoint failure to $FAILED_RESULT_DIR"
    else
        echo "Refusing fresh start in unrecognized non-empty result directory: $RESULT_DIR"
        exit 1
    fi
fi

python - <<'PY'
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_MT240_PairedReplicates_NoDeepSupervision_NoMirroring import (
    PAIRED_PROTOCOL_VERSION,
    REPLICATE_SEEDS,
)
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2

replicate = int(os.environ["MT_PAIRED_REPLICATE"])
seed = int(os.environ["MT_PAIRED_SEED"])
assert REPLICATE_SEEDS[replicate] == seed
found = recursive_find_python_class(
    os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer"),
    os.environ["TRAINER"],
    "nnunetv2.training.nnUNetTrainer",
)
assert found is not None, f"nnU-Net cannot discover {os.environ['TRAINER']}"
assert found.paired_arm == os.environ["PAIRED_ARM"]
assert found.paired_protocol_version == PAIRED_PROTOCOL_VERSION

root = Path(os.environ["nnUNet_preprocessed"])
seed_splits = json.loads(
    (root / "Dataset123_ATM22L20LungCrop" / "splits_final.json").read_text()
)
mt_root = root / "Dataset126_ATM22MT240LungCrop"
mt_splits = json.loads((mt_root / "splits_final.json").read_text())
dataset = json.loads((mt_root / "dataset.json").read_text())
assert len(seed_splits) == len(mt_splits) == 5
provenance = {
    str(key): str(value).lower()
    for key, value in dataset["semi_supervised"]["case_provenance"].items()
}
seed_fold = seed_splits[0]
mt_fold = mt_splits[0]
gt_train = {key for key in mt_fold["train"] if provenance[key] == "gt"}
u_train = {key for key in mt_fold["train"] if provenance[key] == "ignore"}
assert len(gt_train) == 16 and len(u_train) == 240
assert set(seed_fold["train"]) == gt_train
assert set(seed_fold["val"]) == set(mt_fold["val"])

manifest_path = Path(os.environ["MANIFEST"])
result_path = Path(os.environ["RESULT_DIR"])
if (result_path / "checkpoint_final.pth").is_file():
    launch_mode = "already_complete"
elif (result_path / "checkpoint_latest.pth").is_file():
    launch_mode = "resume"
else:
    launch_mode = "fresh"
exact_command = (
    f"nnUNetv2_train 126 3d_fullres 0 -tr {os.environ['TRAINER']} -p nnUNetPlans"
    + (" --c" if launch_mode == "resume" else "")
)
source_files = [
    "nnunet_trainers/nnUNetTrainer_NoDeepSupervision_NoMirroring.py",
    "nnunet_trainers/nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py",
    "nnunet_trainers/nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring.py",
    "nnunet_trainers/nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring.py",
    "nnunet_trainers/nnUNetTrainer_MeanTeacher_VoxelMSE_NoDeepSupervision_NoMirroring.py",
    f"nnunet_trainers/{os.environ['MODULE']}.py",
    "configs/nnunet/mt240_paired_replicates.json",
    "scripts/run_nnunet_mt240_paired.sh",
]
source_blobs = {
    path: subprocess.check_output(["git", "hash-object", path], text=True).strip()
    for path in source_files
}
attempt = {
    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "pbs_jobid": os.environ.get("PBS_JOBID"),
    "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
    "hostname": os.environ.get("HOSTNAME"),
}
previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
attempts = list(previous.get("attempts", []))
attempts.append(attempt)
payload = {
    "status": "launching",
    "protocol_version": PAIRED_PROTOCOL_VERSION,
    "arm": os.environ["PAIRED_ARM"],
    "trainer": os.environ["TRAINER"],
    "replicate": replicate,
    "seed": seed,
    "dataset": 126,
    "fold": 0,
    "initial_checkpoint": os.environ["MT_PAIRED_INIT_CHECKPOINT"],
    "initial_checkpoint_sha256": os.environ["ACTUAL_SEED_SHA"],
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "source_git_blobs": source_blobs,
    "attempts": attempts,
    "result_dir": os.environ["RESULT_DIR"],
    "launch_mode": launch_mode,
    "exact_command": exact_command,
    "tta": False,
}
temporary = manifest_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(manifest_path)
print(f"Paired preflight OK: {payload['arm']} replicate={replicate} seed={seed}")
PY

if test -f "$RESULT_DIR/checkpoint_final.pth"; then
    test -s "$RESULT_DIR/checkpoint_final.pth.mt" || {
        echo "Final student exists without its bound EMA sidecar"
        exit 1
    }
    test -s "$RESULT_DIR/checkpoint_final.pth.paired.json" || {
        echo "Final checkpoint lacks paired transaction metadata"
        exit 1
    }
    echo "Paired $ARM replicate $REPLICATE is already complete"
else
    if test -f "$RESULT_DIR/checkpoint_latest.pth"; then
        test -s "$RESULT_DIR/checkpoint_latest.pth.mt" || {
            echo "Cannot resume: checkpoint_latest.pth.mt is missing"
            exit 1
        }
        nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -tr "$TRAINER" -p nnUNetPlans --c
    else
        if test -d "$RESULT_DIR" && test -n "$(find "$RESULT_DIR" -mindepth 1 -print -quit)"; then
            echo "Refusing fresh start in non-empty incomplete result directory: $RESULT_DIR"
            exit 1
        fi
        # Deliberately no -pretrained_weights: the paired trainer performs a
        # strict complete load instead of nnU-Net's head-skipping transfer load.
        nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -tr "$TRAINER" -p nnUNetPlans
    fi
fi

test -s "$RESULT_DIR/checkpoint_final.pth"
test -s "$RESULT_DIR/checkpoint_final.pth.mt"
test -s "$RESULT_DIR/checkpoint_final.pth.paired.json"
export RESULT_DIR
python - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(os.environ["MANIFEST"])
payload = json.loads(manifest_path.read_text())
result = Path(os.environ["RESULT_DIR"])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

payload.update(
    status="complete",
    completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    final_checkpoint_sha256=sha256(result / "checkpoint_final.pth"),
    final_teacher_sidecar_sha256=sha256(result / "checkpoint_final.pth.mt"),
    pretreatment_checkpoint=str(result / "checkpoint_pretreatment_epoch005.pth"),
)
temporary = manifest_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(manifest_path)
PY

echo "Paired $ARM replicate $REPLICATE completed"
