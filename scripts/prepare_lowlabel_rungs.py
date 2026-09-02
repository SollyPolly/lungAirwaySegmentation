"""Build the reduced-label rungs (@10 and @5) as their own nnU-Net datasets.

The label-efficiency curve currently has one semi-supervised rung: 20 labels
(16 train + 4 internal validation) against 240 unlabelled cases.  This script
adds two more, @10 and @5, as four new datasets:

===================================  ======  ==============================
dataset                              links   split
===================================  ======  ==============================
``Dataset128_ATM22L10LungCrop``      123     8 train / 2 val
``Dataset129_ATM22L5LungCrop``       123     4 train / 1 val
``Dataset130_ATM22MT10LungCrop``     126     8 GT + 240 unlabelled / 2 val
``Dataset131_ATM22MT5LungCrop``      126     4 GT + 240 unlabelled / 1 val
===================================  ======  ==============================

**Why new datasets rather than extra folds.**  nnU-Net's only mechanism for a
different train/val partition over the same arrays is an integer index into
``splits_final.json``, so a rung *can* be expressed as fold 5 and fold 6 of the
existing datasets.  It must not be.  ``splits_final.json`` is shared,
globally-read state, and six scripts assert that Dataset123 and Dataset126 hold
exactly five folds -- including ``scripts/run_nnunet_mt240_paired.sh``, which is
blob-pinned by three paired PBS files.  Appending folds would break a requeue or
resume of the paired wave, and relaxing the assertions would force that
blob cascade mid-wave.  A fold index also conflates "cross-validation member"
with "experimental arm": ``-f all`` and ensemble globbing would sweep the rungs
into the @20 CV.  Datasets 123 and 126 are therefore never touched.

**Why this is still cheap.**  The preprocessed arrays are byte-identical across
rungs -- same CTs, same lung crops, same plans, same labels.  Only *which cases
the split names* differs.  So the arrays are hardlinked rather than recomputed
or copied: no preprocessing, no measurable disk.  Hardlinks (not symlinks) so a
rung survives deletion of its source dataset.

**The one trap.**  ``nnUNetPlans.json`` carries ``dataset_name``, and the trainer
derives ``preprocessed_dataset_folder_base`` from it
(``nnUNetTrainer.py:130``), which is where it reads ``splits_final.json``
(``:615``) and where it writes results (``:132``).  Copy the plans verbatim and
the rung would silently read *the source dataset's* split and train on 16 labels
while claiming 8.  The plans are rewritten with the rung's own name -- the same
single field ``nnUNetv2_move_plans_between_datasets`` changes.  That command is
not used directly because it requires a ``nnUNet_raw`` folder for the target,
which a link-farm rung has no reason to own; with source and target plans
identifiers equal it makes no other change that matters here.

**Nesting.**  One shuffled ordering is drawn per pool and every rung takes a
prefix of it, so @5's cases are a subset of @10's, which are a subset of @20's,
for both the training and the validation halves.  A shuffle rather than a sort
prefix, because ATM identifiers carry acquisition order and a sort prefix would
confound label count with scanner batch.

**The unlabelled pool is held at all 240 for every rung.**  Demoted labelled
cases are dropped from the experiment rather than moved into the unlabelled
pool: moving them would change the unlabelled count too, giving the curve two
moving parts instead of one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from random import Random

SEED_SOURCE = "Dataset123_ATM22L20LungCrop"
MT_SOURCE = "Dataset126_ATM22MT240LungCrop"

BASE_FOLD = 0
EXPECTED_GT = 20
EXPECTED_UNLABELLED = 240
EXPECTED_BASE_TRAIN = 16
EXPECTED_BASE_VAL = 4
EXPECTED_SOURCE_FOLDS = 5

SELECTION_SEED = 20260902

# Files the rung owns outright. Everything else in the source dataset is
# mirrored: small metadata by copy, the preprocessed arrays by hardlink.
OWNED = ("splits_final.json", "nnUNetPlans.json", "dataset.json", "lowlabel_rung_manifest.json")


@dataclass(frozen=True)
class Rung:
    name: str
    n_train: int
    n_val: int
    seed_dataset: str
    mt_dataset: str


RUNGS = (
    Rung("L10", 8, 2, "Dataset128_ATM22L10LungCrop", "Dataset130_ATM22MT10LungCrop"),
    Rung("L5", 4, 1, "Dataset129_ATM22L5LungCrop", "Dataset131_ATM22MT5LungCrop"),
)


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _case_provenance(mt_dataset_json: dict) -> dict[str, str]:
    contract = mt_dataset_json.get("semi_supervised")
    if not isinstance(contract, dict):
        raise ValueError(f"{MT_SOURCE} dataset.json has no semi_supervised contract.")
    raw = contract.get("case_provenance")
    if not isinstance(raw, dict):
        raise ValueError(f"{MT_SOURCE} semi_supervised contract has no case_provenance.")
    provenance = {str(key): str(value).lower() for key, value in raw.items()}
    counts = {value: list(provenance.values()).count(value) for value in set(provenance.values())}
    if counts.get("gt") != EXPECTED_GT or counts.get("ignore") != EXPECTED_UNLABELLED:
        raise ValueError(
            f"{MT_SOURCE} provenance must be {EXPECTED_GT} gt / {EXPECTED_UNLABELLED} ignore, "
            f"got {counts}."
        )
    return provenance


def _validate_sources(seed_folds: list, mt_folds: list, provenance: dict[str, str]) -> None:
    """The rungs are carved out of fold 0, so fold 0 must be what we think it is."""
    for name, folds in ((SEED_SOURCE, seed_folds), (MT_SOURCE, mt_folds)):
        if not isinstance(folds, list) or len(folds) != EXPECTED_SOURCE_FOLDS:
            raise ValueError(
                f"{name} splits_final.json must hold exactly {EXPECTED_SOURCE_FOLDS} folds, "
                f"got {len(folds) if isinstance(folds, list) else type(folds).__name__}. "
                "This script must never see a source dataset that has been appended to."
            )

    seed_fold = seed_folds[BASE_FOLD]
    mt_fold = mt_folds[BASE_FOLD]

    unknown = (set(mt_fold["train"]) | set(mt_fold["val"])) - set(provenance)
    if unknown:
        raise ValueError(f"{MT_SOURCE} fold 0 references cases with no provenance: {sorted(unknown)}")

    gt_train = {key for key in mt_fold["train"] if provenance[key] == "gt"}
    unlabelled_train = {key for key in mt_fold["train"] if provenance[key] == "ignore"}
    if len(gt_train) != EXPECTED_BASE_TRAIN or len(unlabelled_train) != EXPECTED_UNLABELLED:
        raise ValueError(
            f"{MT_SOURCE} fold 0 must be {EXPECTED_BASE_TRAIN} GT + {EXPECTED_UNLABELLED} "
            f"unlabelled, got {len(gt_train)} + {len(unlabelled_train)}."
        )
    if len(mt_fold["val"]) != EXPECTED_BASE_VAL:
        raise ValueError(f"{MT_SOURCE} fold 0 must validate on {EXPECTED_BASE_VAL} cases.")
    if any(provenance[key] != "gt" for key in mt_fold["val"]):
        raise ValueError(f"{MT_SOURCE} fold 0 validates on a non-GT case.")

    # The MT arms warm-start from the seed model, so the two sources must agree
    # on which labels that model was allowed to see.
    if set(seed_fold["train"]) != gt_train:
        raise ValueError(f"{SEED_SOURCE} fold 0 train differs from {MT_SOURCE} fold 0 GT train.")
    if set(seed_fold["val"]) != set(mt_fold["val"]):
        raise ValueError(f"{SEED_SOURCE} fold 0 val differs from {MT_SOURCE} fold 0 val.")


def build_rung_splits(seed_folds: list, mt_folds: list, provenance: dict[str, str]):
    """Return, per rung, the seed split, the MT split, and a provenance record."""
    mt_fold = mt_folds[BASE_FOLD]
    gt_train = sorted(key for key in mt_fold["train"] if provenance[key] == "gt")
    unlabelled_train = sorted(key for key in mt_fold["train"] if provenance[key] == "ignore")
    gt_val = sorted(mt_fold["val"])

    # One RNG, drawn train-then-val, so both orderings are reproducible from the
    # single documented seed and every rung is a prefix of the one above it.
    rng = Random(SELECTION_SEED)
    train_order = list(gt_train)
    rng.shuffle(train_order)
    val_order = list(gt_val)
    rng.shuffle(val_order)

    plan = []
    for rung in RUNGS:
        if rung.n_train > len(train_order) or rung.n_val > len(val_order):
            raise ValueError(f"Rung {rung.name} asks for more labels than fold 0 holds.")
        train = sorted(train_order[: rung.n_train])
        val = sorted(val_order[: rung.n_val])
        plan.append(
            {
                "rung": rung,
                "seed_split": [{"train": train, "val": val}],
                "mt_split": [{"train": sorted(train + unlabelled_train), "val": val}],
                "record": {
                    "rung": rung.name,
                    "seed_dataset": rung.seed_dataset,
                    "mt_dataset": rung.mt_dataset,
                    "labelled_total": rung.n_train + rung.n_val,
                    "labelled_train": train,
                    "labelled_val": val,
                    "unlabelled_train": len(unlabelled_train),
                },
            }
        )

    # Nesting is the property the curve rests on; assert it rather than trust it.
    records = [item["record"] for item in plan]
    for finer, coarser in zip(records[1:], records[:-1]):
        if not set(finer["labelled_train"]) <= set(coarser["labelled_train"]):
            raise ValueError(f"{finer['rung']} train is not nested inside {coarser['rung']}.")
        if not set(finer["labelled_val"]) <= set(coarser["labelled_val"]):
            raise ValueError(f"{finer['rung']} val is not nested inside {coarser['rung']}.")
    for record in records:
        if set(record["labelled_train"]) & set(record["labelled_val"]):
            raise ValueError(f"{record['rung']} train and val overlap.")
        if not set(record["labelled_train"]) <= set(gt_train):
            raise ValueError(f"{record['rung']} train escapes fold 0's labelled pool.")
        if not set(record["labelled_val"]) <= set(gt_val):
            raise ValueError(f"{record['rung']} val escapes fold 0's validation pool.")

    return plan


def _link_file(source: Path, target: Path) -> str:
    """Hardlink, degrading to symlink then copy. Returns the mode actually used."""
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        pass
    try:
        target.symlink_to(source)
        return "symlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _mirror(source_dir: Path, target_dir: Path) -> dict[str, int]:
    """Mirror a preprocessed dataset: metadata copied, array directories linked.

    Top-level metadata is copied rather than linked so that editing a rung's
    manifest can never write through to the source dataset. The arrays are
    linked because they are large, identical, and never written to.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    modes: dict[str, int] = {}
    for entry in sorted(source_dir.iterdir()):
        if entry.name in OWNED:
            continue  # the rung writes its own
        if entry.is_file():
            shutil.copy2(entry, target_dir / entry.name)
            modes["copy"] = modes.get("copy", 0) + 1
        elif entry.is_dir():
            for source_file in sorted(entry.rglob("*")):
                if not source_file.is_file():
                    continue
                target_file = target_dir / source_file.relative_to(source_dir)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                mode = _link_file(source_file, target_file)
                modes[mode] = modes.get(mode, 0) + 1
    return modes


def _write_plans(source_dir: Path, target_dir: Path, dataset_name: str) -> None:
    """Copy the plans with dataset_name repointed at the rung.

    Without this the trainer resolves preprocessed_dataset_folder_base to the
    SOURCE dataset and silently trains on the source's split.
    """
    plans = _read_json(source_dir / "nnUNetPlans.json")
    if plans.get("dataset_name") not in (source_dir.name, dataset_name):
        raise ValueError(
            f"{source_dir}/nnUNetPlans.json names dataset {plans.get('dataset_name')!r}, "
            f"expected {source_dir.name!r}."
        )
    plans["dataset_name"] = dataset_name
    _write_json(target_dir / "nnUNetPlans.json", plans)


def _write_dataset_json(source_dir: Path, target_dir: Path, split: dict) -> None:
    """Copy dataset.json, arming the semi-supervised contract on the rung's fold.

    The trainer's _assert_contract_matches_split is a no-op for a fold the
    contract has no entry for. Recording the rung's own fold 0 turns that guard
    back on, so a later edit to splits_final.json cannot silently change GT
    exposure.
    """
    dataset_json = _read_json(source_dir / "dataset.json")
    contract = dataset_json.get("semi_supervised")
    if isinstance(contract, dict):
        contract["folds"] = {"0": split}
    _write_json(target_dir / "dataset.json", dataset_json)


def build_rung_dataset(
    source_dir: Path, target_dir: Path, split: list, record: dict, *, role: str
) -> dict[str, int]:
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(
            f"{target_dir} already exists and is not empty. Move an incomplete build aside "
            "rather than letting this script write into it."
        )
    modes = _mirror(source_dir, target_dir)
    _write_plans(source_dir, target_dir, target_dir.name)
    _write_dataset_json(source_dir, target_dir, split[0])
    _write_json(target_dir / "splits_final.json", split)
    _write_json(
        target_dir / "lowlabel_rung_manifest.json",
        {
            "role": role,
            "source_dataset": source_dir.name,
            "selection_seed": SELECTION_SEED,
            "unlabelled_pool_held_fixed": EXPECTED_UNLABELLED,
            "link_modes": modes,
            **record,
        },
    )
    return modes


def plan_rungs(root: Path):
    """Validate the sources and return the per-rung build plan."""
    seed_folds = _read_json(root / SEED_SOURCE / "splits_final.json")
    mt_folds = _read_json(root / MT_SOURCE / "splits_final.json")
    provenance = _case_provenance(_read_json(root / MT_SOURCE / "dataset.json"))
    _validate_sources(seed_folds, mt_folds, provenance)
    return build_rung_splits(seed_folds, mt_folds, provenance)


def build_rung_datasets_from_root(root: Path) -> dict[str, dict[str, int]]:
    """Build all four rung datasets under ``root``. Returns per-dataset link modes."""
    built: dict[str, dict[str, int]] = {}
    for item in plan_rungs(root):
        rung, record = item["rung"], item["record"]
        built[rung.seed_dataset] = build_rung_dataset(
            root / SEED_SOURCE,
            root / rung.seed_dataset,
            item["seed_split"],
            record,
            role="supervised_seed",
        )
        built[rung.mt_dataset] = build_rung_dataset(
            root / MT_SOURCE,
            root / rung.mt_dataset,
            item["mt_split"],
            record,
            role="mean_teacher",
        )
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nnunet-preprocessed", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Build the datasets. Without it the script only reports what it would do.",
    )
    args = parser.parse_args()

    root = args.nnunet_preprocessed
    for item in plan_rungs(root):
        rung, record = item["rung"], item["record"]
        print(
            f"{rung.name:>3}  {rung.n_train} train + {rung.n_val} val = "
            f"{record['labelled_total']} labels, {record['unlabelled_train']} unlabelled"
        )
        print(f"     {rung.seed_dataset}  (links {SEED_SOURCE})")
        print(f"     {rung.mt_dataset}  (links {MT_SOURCE})")
        print(f"     train: {record['labelled_train']}")
        print(f"     val:   {record['labelled_val']}")

    if not args.apply:
        print("Dry run only. Rerun with --apply to build.")
        return

    for name, modes in build_rung_datasets_from_root(root).items():
        print(f"{name}: built {modes}")

    print(f"{SEED_SOURCE} and {MT_SOURCE} were not modified.")


if __name__ == "__main__":
    main()
