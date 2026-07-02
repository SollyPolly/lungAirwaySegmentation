"""Export ATM'22 into the nnU-Net v2 raw-dataset layout (Track A: stock nnU-Net control).

Builds ``<out_dir>/imagesTr/`` (CT, channel ``0000``), ``labelsTr/`` (airway) and a v2
``dataset.json`` for a chosen set of ATM cases, so nnU-Net's own rule-based pipeline
(``plan_and_preprocess`` -> ``train`` 5-fold -> ``predict``) can run on the SAME data. This
separates the PIPELINE gap from the METHOD gap and pre-empts the "weak baseline" critique;
nnU-Net optimises Dice, so it is a CONTROL, not the topology-aware method.

Test hygiene lives in the CLI (``scripts/convert_atm_to_nnunet.py``), which computes the
sealed split (seed 15) and NEVER passes our test cases here.

nnU-Net v2 naming already matches ATM'22 — images ``ATM_XXX_0000.nii.gz`` (channel 0),
labels ``ATM_XXX.nii.gz`` — so this mostly links/copies + writes ``dataset.json``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from lung_airway_segmentation.io.atm22_layout import resolve_case_paths

# ATM'22 is a single-channel CT with a binary airway label; nnU-Net wants consecutive
# integer labels starting at 0 (the ATM masks are already 0/1).
_CHANNEL_NAMES = {"0": "CT"}
_LABELS = {"background": 0, "airway": 1}
_LINK_MODES = ("symlink", "hardlink", "copy")


def nnunet_dataset_json(num_training: int, *, channel_names=None, labels=None,
                        file_ending: str = ".nii.gz") -> dict:
    """The minimal valid nnU-Net v2 ``dataset.json`` as a dict."""
    return {
        "channel_names": dict(channel_names or _CHANNEL_NAMES),
        "labels": dict(labels or _LABELS),
        "numTraining": int(num_training),
        "file_ending": file_ending,
    }


def _place(src: Path, dst: Path, mode: str) -> None:
    """Symlink (default) / hardlink / copy ``src`` -> ``dst``, replacing any existing dst.

    Falls back to a copy when linking is unavailable (e.g. Windows without the symlink
    privilege, or a cross-device hardlink), so the export is portable local + on HPC.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        if mode == "hardlink":
            os.link(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def export_atm_to_nnunet(case_ids, *, batch_root, out_dir, mode: str = "symlink") -> dict:
    """Write the nnU-Net v2 raw dataset for ``case_ids``. Returns a summary dict.

    Each case must be labelled (has an airway mask) — a missing label raises, since a
    stock nnU-Net control needs the label. ``mode`` in {'symlink', 'hardlink', 'copy'}.
    """
    if mode not in _LINK_MODES:
        raise ValueError(f"mode must be one of {_LINK_MODES}, got {mode!r}.")
    batch_root = Path(batch_root)
    out_dir = Path(out_dir)
    images_dir = out_dir / "imagesTr"
    labels_dir = out_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    for case_id in case_ids:
        paths = resolve_case_paths(case_id, batch_root=batch_root)
        if paths["airway"] is None:
            raise ValueError(f"ATM'22 case {case_id} has no airway label — cannot export to nnU-Net.")
        padded = paths["case_id"]
        _place(paths["ct"], images_dir / f"ATM_{padded}_0000.nii.gz", mode)
        _place(paths["airway"], labels_dir / f"ATM_{padded}.nii.gz", mode)
        exported.append(padded)

    (out_dir / "dataset.json").write_text(json.dumps(nnunet_dataset_json(len(exported)), indent=2))
    return {"dataset_dir": str(out_dir), "num_training": len(exported), "cases": exported}
