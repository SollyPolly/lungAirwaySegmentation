#!/bin/bash
# Shared fail-closed runner for the reduced-label (@10, @5) Mean-Teacher arms.
#
# One runner, two arms, two rungs. PBS_ARRAY_INDEX is the nnU-Net dataset id, so
# the array bounds name the rungs directly: 130 = @10, 131 = @5. Every rung is a
# dataset of its own with a single fold 0, so nothing here indexes into a shared
# splits_final.json and Dataset123/Dataset126 are never touched. See
# scripts/prepare_lowlabel_rungs.py for why the rungs are not extra folds.
#
# Both arms read the SAME 240 unlabelled cases, so within a rung the only thing
# that varies between treatment and control is whether the consistency term has
# a gradient.
#
# Protocol note. These run the NON-deterministic warm-start protocol -- the one
# the sealed-test claim rests on -- not the paired-replicate protocol. Paired
# costs 3.2x for replicate-level statistics a two-rung label curve does not
# need. Each rung nevertheless gets its own control, run with the same code as
# its treatment, so the reported quantity (MT minus control, within a rung) is
# internally consistent even though the @20 rung predates the 2026-08-30
# deterministic soft-skeleton change.

set -euo pipefail

PROJECT=/rds/general/user/dl525/home/projects/dissertation/LungAirwaySegmentation
cd "$PROJECT"

ARM="${LOWLABEL_ARM:?Set LOWLABEL_ARM to softcldice_w010 or control}"
DATASET="${PBS_ARRAY_INDEX:?Submit this runner through a low-label PBS array}"
case "$DATASET" in
    130) RUNG=L10; MT_DATASET=Dataset130_ATM22MT10LungCrop; SEED_DATASET=Dataset128_ATM22L10LungCrop
         EXPECTED_GT_TRAIN=8; EXPECTED_VAL=2 ;;
    131) RUNG=L5;  MT_DATASET=Dataset131_ATM22MT5LungCrop;  SEED_DATASET=Dataset129_ATM22L5LungCrop
         EXPECTED_GT_TRAIN=4; EXPECTED_VAL=1 ;;
    *) echo "Unsupported low-label dataset '$DATASET'; this runner covers 130 (@10) and 131 (@5) only"; exit 1 ;;
esac

module load miniforge/3
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ctfm

export nnUNet_raw="$PROJECT/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PROJECT/data/nnunet/nnUNet_results"
export TORCH_HOME="$HOME/.torchcache"
export nnUNet_n_proc_DA=4

FOLD=0
EXPECTED_UNLABELLED=240
SEED_TRAINER=nnUNetTrainer_NoDeepSupervision_NoMirroring
MODULE=nnUNetTrainer_MeanTeacher_SoftCLDice_Diagnostics_NoDeepSupervision_NoMirroring

case "$ARM" in
    softcldice_w010)
        TRAINER=nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_SoftCLDiceDiagnostics_NoDeepSupervision_NoMirroring
        EXPECTED_OBJECTIVE=soft_probability_cldice
        EXPECTED_SOFT=True
        EXPECTED_WEIGHT=0.1
        ;;
    control)
        TRAINER=nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_ControlDiagnostics_NoDeepSupervision_NoMirroring
        EXPECTED_OBJECTIVE=supervised_control
        EXPECTED_SOFT=False
        EXPECTED_WEIGHT=0.0
        ;;
    *) echo "Unknown LOWLABEL_ARM '$ARM'"; exit 1 ;;
esac

SEED_CHECKPOINT="$nnUNet_results/${SEED_DATASET}/${SEED_TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/checkpoint_final.pth"
RESULT_DIR="$nnUNet_results/${MT_DATASET}/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}"

# Four members may become eligible at once. Keep different arm/rung pairs
# independent while refusing an accidental duplicate of the same pair. The
# advisory lock is released if the job exits or its node fails, so it cannot
# leave a stale sentinel that blocks a resume.
command -v flock >/dev/null 2>&1 || {
    echo "flock is required to protect low-label result directories"
    exit 1
}
LOCK_DIR="$PROJECT/logs/nnunet_locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/lowlabel_${ARM}_d${DATASET}.lock"
flock -n 9 || {
    echo "Another job is already running low-label arm=$ARM dataset=$DATASET"
    exit 1
}

NNUNET_PACKAGE=$(python -c 'from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).resolve().parent)')
TRAINER_DIR="$NNUNET_PACKAGE/training/nnUNetTrainer/variants/network_architecture"
test -d "$TRAINER_DIR" || {
    echo "Cannot find nnU-Net trainer directory: $TRAINER_DIR"
    exit 1
}

# Fail closed if any trainer in the executed chain has changed since this
# protocol was reviewed. Working tree AND HEAD must both match, so an
# uncommitted edit cannot reach a GPU.
verify_blob() {
    local source_file=$1
    local expected_blob=$2
    local actual_blob
    local head_blob
    test -f "$source_file" || { echo "Missing source: $source_file"; exit 1; }
    git ls-files --error-unmatch "$source_file" >/dev/null 2>&1 || {
        echo "Refusing low-label launch: source is not tracked: $source_file"
        exit 1
    }
    actual_blob=$(git hash-object "$source_file")
    head_blob=$(git rev-parse "HEAD:$source_file")
    test "$actual_blob" = "$expected_blob" && test "$head_blob" = "$expected_blob" || {
        echo "Refusing low-label launch: source changed: $source_file"
        echo "expected=$expected_blob working_tree=$actual_blob HEAD=$head_blob"
        exit 1
    }
}

# nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py is pinned at the
# post-2026-08-30 blob: the deterministic separable-max soft skeleton. The older
# non-paired PBS files still pin 1e401b0f and will refuse to launch until they
# are repinned. That is correct -- they were reviewed against different code.
verify_blob nnunet_trainers/nnUNetTrainer_NoDeepSupervision_NoMirroring.py \
    23af9756e44a5f63d7e136b2177276c8a9299ae0
verify_blob nnunet_trainers/nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py \
    7d840298c711ca4c6a5e60c1c47f5b63d9370d95
verify_blob nnunet_trainers/nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring.py \
    c80fe415da7823bc491333c4e70fb4ae94d5f8cc
verify_blob "nnunet_trainers/${MODULE}.py" \
    34131e294f560a5b34d57298cf0ebe7e95eb1af1

# Never replace a different installed trainer. Existing identical files and
# symlinks to this checkout are accepted; absent modules are linked once.
ensure_same_trainer() {
    local source_file="$PROJECT/nnunet_trainers/$1"
    local target_file="$TRAINER_DIR/$1"
    test -f "$source_file" || { echo "Missing repository trainer: $source_file"; exit 1; }
    if test -L "$target_file"; then
        test "$(readlink -f "$target_file")" = "$(readlink -f "$source_file")" || {
            echo "Refusing to replace different trainer symlink: $target_file"
            exit 1
        }
    elif test -e "$target_file"; then
        cmp -s "$source_file" "$target_file" || {
            echo "Refusing to replace different installed trainer: $target_file"
            exit 1
        }
    elif ! ln -s "$source_file" "$target_file" 2>/dev/null; then
        # Four independent jobs may race to create this same symlink.
        if test -L "$target_file"; then
            test "$(readlink -f "$target_file")" = "$(readlink -f "$source_file")" || {
                echo "Concurrent install produced a different trainer symlink: $target_file"
                exit 1
            }
        elif test -e "$target_file"; then
            cmp -s "$source_file" "$target_file" || {
                echo "Concurrent install produced a different trainer file: $target_file"
                exit 1
            }
        else
            echo "Could not install trainer module: $target_file"
            exit 1
        fi
    fi
}

for TRAINER_FILE in \
    nnUNetTrainer_NoDeepSupervision_NoMirroring.py \
    nnUNetTrainer_MeanTeacher_NoDeepSupervision_NoMirroring.py \
    nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring.py \
    "${MODULE}.py"
do
    ensure_same_trainer "$TRAINER_FILE"
done

TRAINER="$TRAINER" MODULE="$MODULE" EXPECTED_OBJECTIVE="$EXPECTED_OBJECTIVE" \
EXPECTED_SOFT="$EXPECTED_SOFT" EXPECTED_WEIGHT="$EXPECTED_WEIGHT" python - <<'PY'
import importlib
import os

module = importlib.import_module(
    "nnunetv2.training.nnUNetTrainer.variants.network_architecture." + os.environ["MODULE"]
)
base = importlib.import_module(
    "nnunetv2.training.nnUNetTrainer.variants.network_architecture."
    "nnUNetTrainer_MeanTeacher_TwoStream_NoDeepSupervision_NoMirroring"
)
K1Base = base.nnUNetTrainer_MeanTeacher_WarmStart_TwoStream_NoDeepSupervision_NoMirroring
Trainer = getattr(module, os.environ["TRAINER"])

assert issubclass(Trainer, K1Base)
assert Trainer.protocol_exposure == "K1"
assert Trainer.expected_labelled_per_step == 1
assert Trainer.expected_unlabelled_per_step == 1
assert Trainer.expected_local_batch_size == 2
assert Trainer.diagnostic_objective == os.environ["EXPECTED_OBJECTIVE"]
assert Trainer.enable_soft_probability_consistency is (os.environ["EXPECTED_SOFT"] == "True")
assert float(Trainer.configured_consistency_max) == float(os.environ["EXPECTED_WEIGHT"])
print(f"Trainer discovery/protocol preflight OK: {Trainer.__name__}")
PY

test -s "$SEED_CHECKPOINT" || {
    echo "Missing or empty seed checkpoint: $SEED_CHECKPOINT"
    echo "Run nnunet_lungcrop_lowlabel_seed_train.pbs for this rung first."
    exit 1
}

test -d "$nnUNet_preprocessed/${MT_DATASET}/nnUNetPlans_3d_fullres" || {
    echo "${MT_DATASET} is not built; run scripts/prepare_lowlabel_rungs.py --apply first."
    exit 1
}

MT_DATASET="$MT_DATASET" SEED_DATASET="$SEED_DATASET" RUNG="$RUNG" \
EXPECTED_GT_TRAIN="$EXPECTED_GT_TRAIN" EXPECTED_VAL="$EXPECTED_VAL" \
EXPECTED_UNLABELLED="$EXPECTED_UNLABELLED" python - <<'PY'
import json
import os
from pathlib import Path

rung = os.environ["RUNG"]
expected_gt = int(os.environ["EXPECTED_GT_TRAIN"])
expected_val = int(os.environ["EXPECTED_VAL"])
expected_unlabelled = int(os.environ["EXPECTED_UNLABELLED"])

root = Path(os.environ["nnUNet_preprocessed"])
mt_root = root / os.environ["MT_DATASET"]
seed_root = root / os.environ["SEED_DATASET"]

mt_splits = json.loads((mt_root / "splits_final.json").read_text())
seed_splits = json.loads((seed_root / "splits_final.json").read_text())
dataset = json.loads((mt_root / "dataset.json").read_text())
contract = dataset["semi_supervised"]
provenance = {str(k): str(v).lower() for k, v in contract["case_provenance"].items()}

# A rung is a single-fold dataset. More than one fold means something appended
# to it, which is exactly the shared-state failure this layout exists to avoid.
assert len(mt_splits) == 1, f"{mt_root.name} must hold exactly one fold, got {len(mt_splits)}"
assert len(seed_splits) == 1, f"{seed_root.name} must hold exactly one fold, got {len(seed_splits)}"

# The plans must point at THIS dataset. If they still name the source, the
# trainer would read the source's split and train on 16 labels claiming fewer.
plans = json.loads((mt_root / "nnUNetPlans.json").read_text())
assert plans["dataset_name"] == mt_root.name, (
    f"{mt_root.name}/nnUNetPlans.json names {plans['dataset_name']!r}; the trainer would "
    "read another dataset's splits_final.json"
)
seed_plans = json.loads((seed_root / "nnUNetPlans.json").read_text())
assert seed_plans["dataset_name"] == seed_root.name, seed_plans["dataset_name"]

mt, seed = mt_splits[0], seed_splits[0]
gt_train = {key for key in mt["train"] if provenance[key] == "gt"}
u_train = {key for key in mt["train"] if provenance[key] == "ignore"}

assert list(provenance.values()).count("gt") == 20
assert list(provenance.values()).count("ignore") == expected_unlabelled
assert len(gt_train) == expected_gt, (len(gt_train), expected_gt)
assert len(u_train) == expected_unlabelled, (len(u_train), expected_unlabelled)
assert len(mt["val"]) == expected_val, (len(mt["val"]), expected_val)
assert len(set(mt["train"])) == len(mt["train"])
assert len(set(mt["val"])) == len(mt["val"])
assert not set(mt["train"]) & set(mt["val"])
assert all(provenance[key] == "gt" for key in mt["val"]), "rung validates on an ignore case"

# The contract must be armed on this fold, so the trainer's own guard is live.
declared = contract.get("folds", {}).get("0")
assert declared is not None, "semi_supervised contract has no fold 0 entry; trainer guard inert"
assert set(declared["train"]) == set(mt["train"]) and set(declared["val"]) == set(mt["val"]), \
    "dataset.json contract disagrees with splits_final.json"

# The warm start is only meaningful if the seed saw exactly these labels and
# was scored on exactly these held-out cases.
assert set(seed["train"]) == gt_train, "seed split train differs from the rung's GT train"
assert set(seed["val"]) == set(mt["val"]), "seed split val differs from the rung's val"

print(
    f"{mt_root.name} ({rung}) contract OK: {len(gt_train)} GT train + {len(u_train)} "
    f"unlabelled / {len(mt['val'])} GT val, seed-matched, plans self-named"
)
print(f"  labelled train: {sorted(gt_train)}")
print(f"  val:            {sorted(mt['val'])}")
PY

if test -f "$RESULT_DIR/checkpoint_final.pth"; then
    test -s "$RESULT_DIR/checkpoint_final.pth.mt" || {
        echo "Final checkpoint exists but its Mean-Teacher sidecar is missing"
        exit 1
    }
    echo "Low-label $ARM $RUNG (dataset $DATASET) is already complete"
    exit 0
fi

if test -f "$RESULT_DIR/checkpoint_latest.pth"; then
    test -s "$RESULT_DIR/checkpoint_latest.pth.mt" || {
        echo "Cannot resume: checkpoint_latest.pth.mt is missing"
        exit 1
    }
    nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" \
        -tr "$TRAINER" \
        -p nnUNetPlans \
        --c
else
    if test -d "$RESULT_DIR" && test -n "$(find "$RESULT_DIR" -mindepth 1 -print -quit)"; then
        echo "Refusing a fresh start in non-empty incomplete result directory: $RESULT_DIR"
        exit 1
    fi
    nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" \
        -tr "$TRAINER" \
        -p nnUNetPlans \
        -pretrained_weights "$SEED_CHECKPOINT"
fi

test -s "$RESULT_DIR/checkpoint_final.pth.mt" || {
    echo "Training returned without a final Mean-Teacher sidecar"
    exit 1
}
echo "Low-label $ARM $RUNG (dataset $DATASET) completed"
