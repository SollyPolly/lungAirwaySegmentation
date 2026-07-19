# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.3",
#     "anywidget==0.11.0",
#     "ipyniivue==2.4.4",
#     "nibabel",
#     "numpy",
#     "plotly",
#     "scikit-image",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # Distance-to-wall radius bins in voxels (label, lo, hi). Mirrors
    # scripts/analyse_distal.py::RADIUS_BINS so the viewer's confidence curve uses
    # the same distal definition as the headline numbers. Kept local (not imported)
    # because analyse_distal pulls in torch, which this viewer deliberately avoids.
    RADIUS_BINS = [
        ("r=1 (distal)", 0.5, 1.5),
        ("r=2", 1.5, 2.5),
        ("r=3", 2.5, 3.5),
        ("r=4-5", 3.5, 5.5),
        ("r>=6 (proximal)", 5.5, 1e9),
    ]
    return (RADIUS_BINS,)


@app.cell(hide_code=True)
def _():
    import hashlib
    import json
    import os
    import shutil
    import tempfile
    from pathlib import Path

    import marimo as mo
    import nibabel as nib
    import numpy as np
    import plotly.graph_objects as go
    from scipy import ndimage
    from skimage.morphology import skeletonize

    from ipyniivue import NiiVue, SliceType

    from lung_airway_segmentation.settings import (
        DEFAULT_HU_WINDOW,
        RAW_AEROPATH_ROOT,
        RAW_ATM22_ROOT,
    )
    from lung_airway_segmentation.io.case_layout import (
        list_case_ids,
        resolve_case_paths,
    )
    from lung_airway_segmentation.io.atm22_layout import (
        list_case_ids as list_atm22_case_ids,
        resolve_case_paths as resolve_atm22_case_paths,
    )
    from lung_airway_segmentation.io.nifti import load_canonical_image

    return (
        DEFAULT_HU_WINDOW,
        NiiVue,
        Path,
        RAW_AEROPATH_ROOT,
        RAW_ATM22_ROOT,
        SliceType,
        go,
        hashlib,
        json,
        list_atm22_case_ids,
        list_case_ids,
        load_canonical_image,
        mo,
        ndimage,
        nib,
        np,
        os,
        resolve_atm22_case_paths,
        resolve_case_paths,
        shutil,
        skeletonize,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Airway Segmentation Viewer

    GPU-rendered CT + airway masks via **NiiVue** ([`ipyniivue`](https://github.com/niivue/ipyniivue)).
    The **Saved Prediction Viewer** is the primary panel and defaults to the
    nnU-Net runs; the labelled-data inspector below it is for eyeballing raw
    ATM'22 / AeroPath ground truth.

    Each NiiVue canvas gives synced axial/coronal/sagittal slices plus a 3D
    volume render in one widget — scrub slices by dragging, rotate the render,
    and right-click for its own contrast/opacity menu. Mask toggles update the
    view without re-loading the CT.

    Full chest CTs are often 200–300 MB even when compressed. Choose one canvas
    below when you are ready to load it; keeping the canvases off initially
    makes the app shell render reliably in browsers and VS Code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    active_viewer = mo.ui.dropdown(
        {
            "Off (controls and analysis only)": "off",
            "Saved prediction": "prediction",
            "Labelled data": "labelled",
        },
        value="Off (controls and analysis only)",
        label="NIfTI canvas",
    )
    return (active_viewer,)


@app.cell(hide_code=True)
def _(NiiVue, mo):
    # ipyniivue 2.4.4 owns one module-level WebGL canvas. Keep one Python model
    # and one marimo wrapper alive for the whole session; replacing either when
    # the case changes can move the singleton canvas into a stale widget.
    nv_shared = NiiVue(height=1)
    nv_widget = mo.ui.anywidget(nv_shared)
    return nv_shared, nv_widget


@app.cell(hide_code=True)
def _(active_viewer, labelled_panel, mo, nv_widget, prediction_panel):
    # Whole UI, rendered at the top of the page. Placed here (not last) because
    # marimo lays out cell OUTPUTS in file order; the plumbing cells below emit no
    # output, so only this viewer + the title above it appear.
    _load_note = mo.md(
        "Select **Saved prediction** or **Labelled data** to load one canvas. "
        "Volumes are fetched as files rather than pushed through the widget socket. "
        "The first load can still take several seconds."
    )
    mo.vstack(
        [
            mo.hstack([active_viewer, _load_note], widths=[1.0, 2.5], align="center"),
            mo.vstack([nv_widget]),
            prediction_panel,
            mo.md("---"),
            labelled_panel,
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(Path, SliceType, hashlib, nib, np, os, shutil, tempfile):
    # NiiVue built-in colormaps. The CT is the gray base; each mask is a solid-hue
    # overlay whose zero background is transparent (cal_min sits above 0 so empty
    # voxels fall below the colormap floor and NiiVue renders them clear).
    SLICE_TYPES = {
        "Multiplanar": SliceType.MULTIPLANAR,
        "3D render": SliceType.RENDER,
        "Axial": SliceType.AXIAL,
        "Coronal": SliceType.CORONAL,
        "Sagittal": SliceType.SAGITTAL,
    }

    # ipyniivue serialises a `path` by reading the entire file into widget state.
    # A chest CT is commonly 200-300 MB, so stage selected files under marimo's
    # public directory and give NiiVue a URL instead. Hard links avoid copying the
    # data; the copy fallback covers filesystems that do not support hard links.
    _PUBLIC_CACHE = Path(__file__).resolve().parent / "public" / "niivue-cache"
    _PUBLIC_CACHE.mkdir(parents=True, exist_ok=True)

    # Derived overlays are session-specific so parallel browser kernels cannot
    # overwrite one another's missed-truth mask before it is staged.
    _VIEWER_CACHE = Path(tempfile.gettempdir()) / f"airway_viewer_cache_{os.getpid()}"
    _VIEWER_CACHE.mkdir(exist_ok=True)

    def stage_volume(path):
        source = Path(path).resolve(strict=True)
        stat = source.stat()
        fingerprint = hashlib.sha256(
            f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:20]
        suffix = "".join(source.suffixes) or ".nii"
        staged = _PUBLIC_CACHE / f"{fingerprint}{suffix}"

        if not staged.exists():
            temporary = staged.with_name(f".{staged.name}.{os.getpid()}.tmp")
            try:
                try:
                    os.link(source, temporary)
                except OSError:
                    shutil.copy2(source, temporary)
                os.replace(temporary, staged)
            finally:
                if temporary.exists():
                    temporary.unlink()

        return f"./public/niivue-cache/{staged.name}"

    def ct_volume(path, hu_window):
        """Base CT volume dict for NiiVue.load_volumes, windowed for contrast."""
        return {
            "url": stage_volume(path),
            "name": Path(path).name,
            "colormap": "gray",
            "cal_min": float(hu_window[0]),
            "cal_max": float(hu_window[1]),
            "opacity": 1.0,
        }

    def mask_volume(path, colormap, opacity, name):
        """Binary-mask overlay dict: cal_min=0.5 keeps the 0 background transparent.
        NiiVue Volumes expose no `visible` trait, so the controls cells show/hide a
        mask by driving its (reactive) `opacity` — 0.0 means hidden."""
        return {
            "url": stage_volume(path),
            "name": name,
            "colormap": colormap,
            "cal_min": 0.5,
            "cal_max": 1.0,
            "opacity": float(opacity),
        }

    def configure_niivue(nv, volumes, height, slice_type):
        """Load a case into the session's single persistent NiiVue model."""
        nv.load_volumes(volumes)
        nv.height = int(height)
        nv.set_slice_type(slice_type)
        return nv

    def write_binary_overlay(mask, affine, name):
        out_path = _VIEWER_CACHE / name
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), out_path)
        return out_path

    def missed_truth_overlay(true_path, pred_path, name):
        """GT AND NOT prediction, written from the native-orientation label/pred
        files so its affine matches the CT NiiVue loads by path. Returns None when
        the shapes disagree or nothing is missed."""
        true_img = nib.load(str(true_path))
        pred_img = nib.load(str(pred_path))
        if true_img.shape != pred_img.shape:
            return None
        missed = (np.asarray(true_img.dataobj) > 0) & ~(np.asarray(pred_img.dataobj) > 0)
        if not missed.any():
            return None
        return write_binary_overlay(missed, true_img.affine, name)

    return (
        SLICE_TYPES,
        configure_niivue,
        ct_volume,
        mask_volume,
        missed_truth_overlay,
    )


@app.cell(hide_code=True)
def _(
    DEFAULT_HU_WINDOW,
    Path,
    RAW_AEROPATH_ROOT,
    RAW_ATM22_ROOT,
    json,
    nib,
    resolve_atm22_case_paths,
    resolve_case_paths,
):
    prediction_run_root = Path(__file__).resolve().parent / "runs"
    project_root = Path(__file__).resolve().parent

    def list_prediction_run_names(run_root):
        if not run_root.exists():
            return []

        run_dirs = []
        for metadata_path in run_root.rglob("run_metadata.json"):
            run_dir = metadata_path.parent
            has_case_predictions = any(
                (case_dir / "prediction_metadata.json").is_file()
                for predictions_dir in run_dir.glob("predictions*")
                if predictions_dir.is_dir()
                for case_dir in predictions_dir.iterdir()
                if case_dir.is_dir()
            )
            if has_case_predictions:
                run_dirs.append(run_dir)

        # Newest first, then float the nnU-Net study buckets to the top: nnU-Net is
        # the model vehicle now, so its runs should lead the dropdown. Python's sort
        # is stable, so the mtime order is preserved within each group.
        run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        run_dirs.sort(
            key=lambda path: 0
            if path.relative_to(run_root).parts[0].startswith("nnunet")
            else 1
        )
        return [str(path.relative_to(run_root)) for path in run_dirs]

    def resolve_prediction_run_dir(run_root, run_name):
        return (run_root / run_name).resolve()

    def resolve_prediction_data_root(run_metadata, resolved_config):
        candidate_paths = []

        recorded_data_root = run_metadata.get("data_root")
        if recorded_data_root:
            candidate_paths.append(Path(recorded_data_root))

        data_config = resolved_config.get("data", {})
        configured_data_root = data_config.get(
            "raw_data_root",
            data_config.get("batch_root"),
        )
        if configured_data_root:
            configured_path = Path(configured_data_root)
            if configured_path.is_absolute():
                candidate_paths.append(configured_path)
            else:
                candidate_paths.append((project_root / configured_path).resolve())

        dataset_name = str(data_config.get("dataset_name", "aeropath")).lower()
        candidate_paths.append(
            RAW_ATM22_ROOT if dataset_name == "atm22" else RAW_AEROPATH_ROOT
        )

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path.resolve()

        return (
            RAW_ATM22_ROOT.resolve()
            if dataset_name == "atm22"
            else RAW_AEROPATH_ROOT.resolve()
        )

    def list_prediction_set_names(run_root, run_name):
        if run_name is None:
            return []

        run_dir = resolve_prediction_run_dir(run_root, run_name)
        prediction_sets = [
            path.name
            for path in run_dir.glob("predictions*")
            if path.is_dir()
            and any(
                case_dir.is_dir() and (case_dir / "prediction_metadata.json").is_file()
                for case_dir in path.iterdir()
            )
        ]
        return sorted(prediction_sets, key=lambda name: (name != "predictions_safe_crop", name))

    def list_prediction_case_ids(run_root, run_name, prediction_set_name):
        if run_name is None or prediction_set_name is None:
            return []

        predictions_dir = resolve_prediction_run_dir(run_root, run_name) / prediction_set_name
        case_ids = [
            case_dir.name
            for case_dir in predictions_dir.iterdir()
            if case_dir.is_dir() and (case_dir / "prediction_metadata.json").is_file()
        ]
        return sorted(case_ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))

    def list_prediction_mask_options(run_root, run_name, prediction_set_name, case_id):
        if run_name is None or prediction_set_name is None or case_id is None:
            return {}

        case_dir = resolve_prediction_run_dir(run_root, run_name) / prediction_set_name / str(case_id)
        mask_paths = [
            path
            for path in case_dir.glob("airway_pred*_full.nii.gz")
            if "lung_masked" not in path.name
        ]
        label_by_name = {
            "airway_pred_full.nii.gz": "Raw prediction",
            "airway_pred_lcc_full.nii.gz": "Largest component (6-connectivity)",
            "airway_pred_lcc18_full.nii.gz": "Largest component (18-connectivity)",
            "airway_pred_lcc26_full.nii.gz": "Largest component (26-connectivity)",
        }
        return {
            label_by_name.get(path.name, path.name): path.name
            for path in sorted(mask_paths, key=lambda path: (path.name != "airway_pred_lcc_full.nii.gz", path.name))
        }

    def list_distal_analysis_options(run_dir):
        if run_dir is None:
            return {}

        analysis_paths = []
        for pattern in ("distal_analysis*.json", "nnunet*_topology.json"):
            analysis_paths.extend(path for path in run_dir.glob(pattern) if path.is_file())

        unique_paths = sorted(
            set(analysis_paths),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
        return {path.name: path.name for path in unique_paths}

    def load_distal_analysis(run_dir, analysis_filename):
        if run_dir is None or analysis_filename is None:
            return None

        run_dir = run_dir.resolve()
        analysis_path = (run_dir / analysis_filename).resolve()
        if analysis_path.parent != run_dir:
            raise ValueError(f"Analysis file is outside the selected run: {analysis_filename}")
        if not analysis_path.is_file():
            raise FileNotFoundError(f"Missing distal analysis JSON: {analysis_path}")

        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis["_path"] = analysis_path
        return analysis

    def preferred_prediction_run_name(run_names):
        # nnU-Net is the model vehicle: lead with the exported Track-A ensemble,
        # then any other nnU-Net study, then whatever is newest.
        for run_name in run_names:
            if Path(run_name).parts[0] == "nnunet-track-a":
                return run_name
        for run_name in run_names:
            if Path(run_name).parts[0].startswith("nnunet"):
                return run_name
        return run_names[0] if run_names else None

    def preferred_prediction_case_id(case_ids):
        # Case IDs arrive numerically sorted; the first held-out case is a fine
        # default (the old AeroPath-specific "20" hard-code is gone).
        return case_ids[0] if case_ids else None

    def load_prediction_bundle(run_root, run_name, prediction_set_name, case_id, prediction_mask_filename):
        """Resolve the paths + metadata NiiVue and the analysis panels need. This
        deliberately loads NO image arrays — NiiVue reads the NIfTIs directly, and
        the GT-confidence diagnostic loads what it needs lazily — so switching case
        no longer pulls several hundred MB of volumes into Python."""
        run_dir = resolve_prediction_run_dir(run_root, run_name)
        prediction_case_dir = run_dir / prediction_set_name / str(case_id)
        prediction_metadata_path = prediction_case_dir / "prediction_metadata.json"
        prediction_mask_path = prediction_case_dir / prediction_mask_filename

        if not prediction_metadata_path.exists():
            raise FileNotFoundError(f"Missing prediction metadata: {prediction_metadata_path}")
        if not prediction_mask_path.exists():
            raise FileNotFoundError(f"Missing full prediction mask: {prediction_mask_path}")

        prediction_metadata = json.loads(prediction_metadata_path.read_text(encoding="utf-8"))
        run_metadata_path = run_dir / "run_metadata.json"
        history_path = run_dir / "history.json"
        resolved_config_path = run_dir / "resolved_config.json"

        run_metadata = (
            json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if run_metadata_path.exists()
            else {}
        )
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.exists()
            else {}
        )
        resolved_config = (
            json.loads(resolved_config_path.read_text(encoding="utf-8"))
            if resolved_config_path.exists()
            else {}
        )

        data_config = resolved_config.get("data", {})
        dataset_name = str(data_config.get("dataset_name", "aeropath")).lower()
        data_root = resolve_prediction_data_root(run_metadata, resolved_config)
        if dataset_name == "atm22":
            case_paths = resolve_atm22_case_paths(case_id, batch_root=data_root)
        elif dataset_name == "aeropath":
            case_paths = resolve_case_paths(case_id, data_root=data_root)
        else:
            raise ValueError(f"Unsupported prediction dataset: {dataset_name}")

        if case_paths["airway"] is None:
            raise ValueError(f"Case {case_id} does not have a reference airway mask.")

        # Header-only geometry read (no voxel data pulled into memory).
        ct_header = nib.load(str(case_paths["ct"])).header
        spacing = tuple(float(value) for value in ct_header.get_zooms()[:3])
        hu_window = tuple(
            float(value)
            for value in data_config.get("preprocessing", {}).get("hu_window", DEFAULT_HU_WINDOW)
        )

        probability_path = prediction_case_dir / "airway_prob_full.nii.gz"
        history_entries = history.get("history", []) if isinstance(history, dict) else []
        best_metrics = history.get("best", {}) if isinstance(history, dict) else {}

        return {
            "run_dir": run_dir,
            "run_name": run_name,
            "study_name": run_metadata.get("study_name")
            or resolved_config.get("training", {}).get("study_name"),
            "run_label": run_metadata.get("run_label")
            or resolved_config.get("training", {}).get("run_label"),
            "experiment_name": run_metadata.get("experiment_name")
            or resolved_config.get("training", {}).get("experiment_name"),
            "dataset_name": dataset_name,
            "prediction_set_name": prediction_set_name,
            "prediction_mask_filename": prediction_mask_filename,
            "case_id": str(case_id),
            "ct_path": case_paths["ct"],
            "lung_mask_path": case_paths["lung"],
            "true_mask_path": case_paths["airway"],
            "prediction_mask_path": prediction_mask_path,
            "probability_path": probability_path if probability_path.exists() else None,
            "prediction_metadata_path": prediction_metadata_path,
            "hu_window": hu_window,
            "spacing": spacing,
            "best_val_dice": best_metrics.get("val_dice"),
            "best_epoch": best_metrics.get("epoch"),
            "last_epoch_metrics": history_entries[-1] if history_entries else None,
            "checkpoint_epoch": prediction_metadata.get("checkpoint_epoch"),
            "threshold": prediction_metadata.get("threshold"),
        }

    return (
        list_distal_analysis_options,
        list_prediction_case_ids,
        list_prediction_mask_options,
        list_prediction_run_names,
        list_prediction_set_names,
        load_distal_analysis,
        load_prediction_bundle,
        prediction_run_root,
        preferred_prediction_case_id,
        preferred_prediction_run_name,
    )


@app.cell(hide_code=True)
def _(mo):
    prediction_refresh_button = mo.ui.run_button(label="Refresh prediction runs")
    return (prediction_refresh_button,)


@app.cell(hide_code=True)
def _(
    SLICE_TYPES,
    list_prediction_run_names,
    mo,
    prediction_refresh_button,
    prediction_run_root,
    preferred_prediction_run_name,
):
    # Filesystem changes are not reactive, so explicitly depend on this button.
    _ = prediction_refresh_button.value
    prediction_run_names = list_prediction_run_names(prediction_run_root)
    prediction_runs_available = len(prediction_run_names) > 0

    prediction_run_selector = (
        mo.ui.dropdown(
            prediction_run_names,
            value=preferred_prediction_run_name(prediction_run_names),
            label="Prediction run",
            searchable=True,
        )
        if prediction_runs_available
        else None
    )
    prediction_view_mode = mo.ui.dropdown(
        list(SLICE_TYPES.keys()),
        value="Multiplanar",
        label="View",
    )
    prediction_view_height_slider = mo.ui.slider(
        320, 900, value=600, step=20, label="Viewer height",
    )
    show_prediction_lung_mask = mo.ui.switch(value=False, label="Lung")
    show_true_prediction_mask = mo.ui.switch(value=True, label="Truth")
    show_predicted_mask = mo.ui.switch(value=True, label="Prediction")
    show_missed_truth_mask = mo.ui.switch(value=False, label="Missed truth")
    show_gt_confidence = mo.ui.switch(value=False, label="GT confidence curve")
    predicted_mask_opacity = mo.ui.slider(
        0.05, 1.0, value=0.70, step=0.05, label="Prediction opacity",
    )
    return (
        predicted_mask_opacity,
        prediction_run_selector,
        prediction_runs_available,
        prediction_view_height_slider,
        prediction_view_mode,
        show_gt_confidence,
        show_missed_truth_mask,
        show_predicted_mask,
        show_prediction_lung_mask,
        show_true_prediction_mask,
    )


@app.cell(hide_code=True)
def _(
    list_prediction_set_names,
    mo,
    prediction_run_root,
    prediction_run_selector,
    prediction_runs_available,
):
    if not prediction_runs_available or prediction_run_selector is None:
        prediction_set_selector = None
    else:
        prediction_set_names = list_prediction_set_names(
            prediction_run_root,
            prediction_run_selector.value,
        )
        prediction_set_selector = (
            mo.ui.dropdown(
                prediction_set_names,
                value=prediction_set_names[0],
                label="Prediction set",
                searchable=True,
            )
            if prediction_set_names
            else None
        )
    return (prediction_set_selector,)


@app.cell(hide_code=True)
def _(
    list_prediction_case_ids,
    mo,
    prediction_run_root,
    prediction_run_selector,
    prediction_runs_available,
    prediction_set_selector,
    preferred_prediction_case_id,
):
    if (
        not prediction_runs_available
        or prediction_run_selector is None
        or prediction_set_selector is None
    ):
        prediction_case_selector = None
    else:
        prediction_case_ids = list_prediction_case_ids(
            prediction_run_root,
            prediction_run_selector.value,
            prediction_set_selector.value,
        )
        prediction_case_selector = (
            mo.ui.dropdown(
                prediction_case_ids,
                value=preferred_prediction_case_id(prediction_case_ids),
                label="Predicted case",
                searchable=True,
            )
            if prediction_case_ids
            else None
        )
    return (prediction_case_selector,)


@app.cell(hide_code=True)
def _(
    list_prediction_mask_options,
    mo,
    prediction_case_selector,
    prediction_run_root,
    prediction_run_selector,
    prediction_set_selector,
):
    if (
        prediction_run_selector is None
        or prediction_set_selector is None
        or prediction_case_selector is None
    ):
        prediction_mask_selector = None
    else:
        prediction_mask_options = list_prediction_mask_options(
            prediction_run_root,
            prediction_run_selector.value,
            prediction_set_selector.value,
            prediction_case_selector.value,
        )
        prediction_mask_selector = (
            mo.ui.dropdown(
                prediction_mask_options,
                value=next(iter(prediction_mask_options)),
                label="Prediction mask",
            )
            if prediction_mask_options
            else None
        )
    return (prediction_mask_selector,)


@app.cell(hide_code=True)
def _(
    load_prediction_bundle,
    prediction_case_selector,
    prediction_mask_selector,
    prediction_run_root,
    prediction_run_selector,
    prediction_runs_available,
    prediction_set_selector,
):
    if not prediction_runs_available or prediction_run_selector is None:
        prediction_bundle = None
        prediction_bundle_error = None
    elif (
        prediction_set_selector is None
        or prediction_case_selector is None
        or prediction_mask_selector is None
    ):
        prediction_bundle = None
        prediction_bundle_error = "Selected run does not contain a usable saved prediction mask."
    else:
        try:
            prediction_bundle = load_prediction_bundle(
                prediction_run_root,
                prediction_run_selector.value,
                prediction_set_selector.value,
                prediction_case_selector.value,
                prediction_mask_selector.value,
            )
            prediction_bundle_error = None
        except Exception as error:
            prediction_bundle = None
            prediction_bundle_error = str(error)
    return prediction_bundle, prediction_bundle_error


@app.cell(hide_code=True)
def _(
    DEFAULT_HU_WINDOW,
    SLICE_TYPES,
    active_viewer,
    configure_niivue,
    ct_volume,
    labelled_paths,
    mask_volume,
    missed_truth_overlay,
    nv_shared,
    prediction_bundle,
):
    nv_prediction = None
    nv_labelled = None
    prediction_layer_index = {}
    prediction_layer_shown = {}
    labelled_layer_index = {}
    labelled_layer_shown = {}

    if active_viewer.value == "prediction" and prediction_bundle is not None:
        _hu = prediction_bundle["hu_window"]
        _volumes = [ct_volume(prediction_bundle["ct_path"], _hu)]

        def _add(name, path, colormap, opacity):
            _volumes.append(mask_volume(path, colormap, opacity, name))
            prediction_layer_index[name] = len(_volumes) - 1
            prediction_layer_shown[name] = opacity

        if prediction_bundle["lung_mask_path"]:
            _add("lung", prediction_bundle["lung_mask_path"], "blue", 0.15)
        _add("truth", prediction_bundle["true_mask_path"], "green", 0.40)
        _add("prediction", prediction_bundle["prediction_mask_path"], "red", 0.70)

        _missed_path = missed_truth_overlay(
            prediction_bundle["true_mask_path"],
            prediction_bundle["prediction_mask_path"],
            f"missed_{prediction_bundle['case_id']}.nii.gz",
        )
        if _missed_path is not None:
            _add("missed", _missed_path, "warm", 0.90)

        # All overlays are loaded up front; the controls cell below drives
        # visibility/opacity by mutating traits, so no toggle re-fetches the CT.
        nv_prediction = configure_niivue(
            nv_shared,
            _volumes,
            height=600,
            slice_type=SLICE_TYPES["Multiplanar"],
        )
    elif (
        active_viewer.value == "labelled"
        and labelled_paths is not None
        and labelled_paths.get("airway") is not None
    ):
        _volumes = [ct_volume(labelled_paths["ct"], DEFAULT_HU_WINDOW)]
        if labelled_paths.get("lung"):
            _volumes.append(mask_volume(labelled_paths["lung"], "blue", 0.15, "lung"))
            labelled_layer_index["lung"] = len(_volumes) - 1
            labelled_layer_shown["lung"] = 0.15
        _volumes.append(mask_volume(labelled_paths["airway"], "red", 0.70, "airway"))
        labelled_layer_index["airway"] = len(_volumes) - 1
        labelled_layer_shown["airway"] = 0.70
        nv_labelled = configure_niivue(
            nv_shared,
            _volumes,
            height=520,
            slice_type=SLICE_TYPES["Multiplanar"],
        )
    else:
        # Keep the one widget mounted, but make its empty canvas unobtrusive.
        nv_shared.load_volumes([])
        nv_shared.height = 1
    return (
        labelled_layer_index,
        labelled_layer_shown,
        nv_labelled,
        nv_prediction,
        prediction_layer_index,
        prediction_layer_shown,
    )


@app.cell(hide_code=True)
def _(
    SLICE_TYPES,
    nv_prediction,
    predicted_mask_opacity,
    prediction_layer_index,
    prediction_layer_shown,
    prediction_view_height_slider,
    prediction_view_mode,
    show_missed_truth_mask,
    show_predicted_mask,
    show_prediction_lung_mask,
    show_true_prediction_mask,
):
    # Reactive control of the existing widget — mutating each overlay's (reactive)
    # opacity updates the canvas in place, so no toggle re-sends the CT. A hidden
    # mask is opacity 0; the prediction's shown opacity tracks its slider. Runs on
    # any toggle/slider change, and right after a rebuild so a fresh widget adopts
    # the current switch states.
    if nv_prediction is not None:
        _toggles = {
            "lung": show_prediction_lung_mask.value,
            "truth": show_true_prediction_mask.value,
            "prediction": show_predicted_mask.value,
            "missed": show_missed_truth_mask.value,
        }
        _shown = dict(prediction_layer_shown)
        _shown["prediction"] = float(predicted_mask_opacity.value)
        for _name, _idx in prediction_layer_index.items():
            if _idx < len(nv_prediction.volumes):
                _on = _toggles.get(_name, True)
                nv_prediction.volumes[_idx].opacity = _shown.get(_name, 0.7) if _on else 0.0
        nv_prediction.height = int(prediction_view_height_slider.value)
        nv_prediction.set_slice_type(SLICE_TYPES[prediction_view_mode.value])
    return


@app.cell(hide_code=True)
def _(
    active_viewer,
    mo,
    nv_prediction,
    prediction_bundle,
    prediction_bundle_error,
):
    if active_viewer.value != "prediction":
        prediction_viewer = mo.md(
            "The saved-prediction canvas is not loaded. Select **Saved prediction** "
            "from the NIfTI canvas control above to load it."
        )
    elif prediction_bundle_error is not None:
        prediction_viewer = mo.md(f"Prediction viewer error: `{prediction_bundle_error}`")
    elif nv_prediction is None or prediction_bundle is None:
        prediction_viewer = mo.md(
            "Prediction viewer will appear once a saved run is available under `runs/`."
        )
    else:
        prediction_viewer = mo.md(
            "The active saved-prediction canvas is shown above the panel controls."
        )
    return (prediction_viewer,)


@app.cell(hide_code=True)
def _(
    list_distal_analysis_options,
    mo,
    prediction_bundle,
    prediction_bundle_error,
):
    if prediction_bundle is None or prediction_bundle_error is not None:
        distal_analysis_selector = None
    else:
        distal_analysis_options = list_distal_analysis_options(prediction_bundle["run_dir"])
        distal_analysis_selector = (
            mo.ui.dropdown(
                distal_analysis_options,
                value=next(iter(distal_analysis_options)),
                label="Analysis table",
                searchable=True,
            )
            if distal_analysis_options
            else None
        )
    return (distal_analysis_selector,)


@app.function
def format_distal_analysis_markdown(analysis):
    def fmt_metric(value, digits=3):
        if value is None:
            return "-"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number != number:
            return "-"
        return f"{number:.{digits}f}"

    def fmt_int(value):
        if value is None:
            return "-"
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)

    def fmt_percent(value):
        if value is None:
            return "-"
        try:
            return f"{100.0 * float(value):.1f}%"
        except (TypeError, ValueError):
            return str(value)

    if analysis is None:
        return "### Distal Analysis\nNo analysis JSON is selected."

    analysis_path = analysis.get("_path")
    table_mean = analysis.get("table_mean") or {}
    operating_point = analysis.get("operating_point") or {}
    report_cases = analysis.get("report_cases") or []
    postprocessing = analysis.get("postprocessing") or {}

    file_label = analysis_path.name if analysis_path is not None else "analysis JSON"
    threshold = operating_point.get("threshold")
    threshold_label = (
        f"`{float(threshold):.2f}`"
        if isinstance(threshold, (int, float))
        else f"`{threshold}`" if threshold is not None else "`unavailable`"
    )
    split_label = analysis.get("report_split", "unavailable")
    selected_on = operating_point.get("selected_on")
    selection_label = f", selected on `{selected_on}`" if selected_on else ""
    checkpoint = analysis.get("checkpoint")
    checkpoint_label = f", checkpoint `{checkpoint}`" if checkpoint else ""
    postproc = postprocessing.get("lcc")
    postproc_label = f", postprocess `{postproc}`" if postproc else ""

    lines = [
        "### Distal Analysis",
        "",
        (
            f"`{file_label}` - split `{split_label}`, n={len(report_cases)}, "
            f"threshold {threshold_label}{selection_label}{checkpoint_label}{postproc_label}"
        ),
    ]

    reason = operating_point.get("reason") or operating_point.get("note")
    if reason:
        lines.append(f"- **Operating point**: {reason}")

    if not table_mean:
        lines.append("")
        lines.append("No `table_mean` block was found in this analysis JSON.")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "| Mask | Dice | TD/TLD | BD | clDice | TPrec | Precision | LCC kept |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                "| Raw | "
                f"{fmt_metric(table_mean.get('dice_raw'))} | "
                f"{fmt_metric(table_mean.get('td_raw'))} | "
                "- | - | - | "
                f"{fmt_metric(table_mean.get('prec_raw'))} | "
                "- |"
            ),
            (
                "| +LCC | "
                f"{fmt_metric(table_mean.get('dice_lcc'))} | "
                f"{fmt_metric(table_mean.get('td_lcc'))} | "
                f"{fmt_metric(table_mean.get('bd_lcc'))} | "
                f"{fmt_metric(table_mean.get('cldice_lcc'))} | "
                f"{fmt_metric(table_mean.get('tprec_lcc'))} | "
                f"{fmt_metric(table_mean.get('prec_lcc'))} | "
                f"{fmt_metric(table_mean.get('lcc_retained_fraction'))} |"
            ),
        ]
    )

    if table_mean.get("cldice_atm") is not None or table_mean.get("td_atm") is not None:
        lines.extend(
            [
                "",
                "| ATM branch parser | clDice | TD/TLD |",
                "| --- | ---: | ---: |",
                (
                    "| ATM | "
                    f"{fmt_metric(table_mean.get('cldice_atm'))} | "
                    f"{fmt_metric(table_mean.get('td_atm'))} |"
                ),
            ]
        )

    bins = analysis.get("bins") or []
    if bins:
        lines.extend(
            [
                "",
                "| Radius bin | Voxels | % airway | Mean P | Median P | Recall @ op | Recall @ 0.5 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in bins:
            recall_at_op = row.get("recall_at_threshold", row.get("recall"))
            lines.append(
                "| "
                f"{row.get('bin', '-')} | "
                f"{fmt_int(row.get('voxels'))} | "
                f"{fmt_metric(row.get('pct_airway'), digits=1)} | "
                f"{fmt_metric(row.get('mean_prob'))} | "
                f"{fmt_metric(row.get('median_prob'))} | "
                f"{fmt_percent(recall_at_op)} | "
                f"{fmt_percent(row.get('recall_at_0.5'))} |"
            )

    return "\n".join(lines)


@app.cell(hide_code=True)
def _(
    distal_analysis_selector,
    load_distal_analysis,
    mo,
    prediction_bundle,
    prediction_bundle_error,
):
    if prediction_bundle is None or prediction_bundle_error is not None:
        distal_analysis_panel = mo.md("")
    elif distal_analysis_selector is None:
        distal_analysis_panel = mo.md(
            "### Distal Analysis\n"
            "No `distal_analysis*.json` or `nnunet*_topology.json` file was found in this run folder."
        )
    else:
        try:
            distal_analysis = load_distal_analysis(
                prediction_bundle["run_dir"],
                distal_analysis_selector.value,
            )
            distal_analysis_panel = mo.vstack(
                [
                    distal_analysis_selector,
                    mo.md(format_distal_analysis_markdown(distal_analysis)),
                ],
                gap=0.45,
            )
        except Exception as error:
            distal_analysis_panel = mo.vstack(
                [
                    distal_analysis_selector,
                    mo.md(f"### Distal Analysis\nCould not load analysis table: `{error}`"),
                ],
                gap=0.45,
            )
    return (distal_analysis_panel,)


@app.cell(hide_code=True)
def _(
    RADIUS_BINS,
    load_canonical_image,
    ndimage,
    np,
    prediction_bundle,
    show_gt_confidence,
    skeletonize,
):
    # Predicted probability restricted to the ground-truth airway, stratified by
    # branch calibre (distance-to-wall at the nearest centreline voxel). A
    # recall/confidence DIAGNOSTIC — it discards all false positives, so it is
    # never a performance number; pair it with the precision-side metrics.
    # nnU-Net runs export hard masks only, so this stays empty for them; it lights
    # up for the from-scratch runs that saved airway_prob_full.nii.gz.
    SHELL_VOXELS = 2.5  # background-shell thickness around the airway, in voxels.
    if prediction_bundle is None or not show_gt_confidence.value:
        gt_confidence = None
    elif prediction_bundle["probability_path"] is None:
        gt_confidence = {
            "error": "No airway_prob_full.nii.gz saved for this run (nnU-Net exports hard masks only)."
        }
    else:
        _prob = np.asarray(
            load_canonical_image(prediction_bundle["probability_path"]).dataobj,
            dtype=np.float32,
        )
        _gt = np.asarray(
            load_canonical_image(prediction_bundle["true_mask_path"]).dataobj
        ) > 0
        if _prob.shape != _gt.shape:
            gt_confidence = {"error": f"probability shape {_prob.shape} != GT shape {_gt.shape}"}
        elif not _gt.any():
            gt_confidence = {"error": "case has no ground-truth airway voxels"}
        else:
            # Work inside a padded bounding box of the airway: the shell distance
            # transform is memory-heavy on a full volume, and everything we need is
            # local to the tree. Pad > shell width.
            _where = np.where(_gt)
            _bbox = tuple(
                slice(max(0, int(ax.min()) - 4), min(int(size), int(ax.max()) + 5))
                for ax, size in zip(_where, _gt.shape)
            )
            _gt_c = _gt[_bbox]
            _prob_c = _prob[_bbox]

            # Bin BOTH airway and shell voxels by the calibre of their nearest
            # centreline voxel, so a whole branch is classified by its thickness —
            # a cleaner distal/generation proxy than per-voxel wall distance.
            _skeleton_c = skeletonize(_gt_c)
            _radius = ndimage.distance_transform_edt(_gt_c)  # distance to wall
            _, _skel_idx = ndimage.distance_transform_edt(~_skeleton_c, return_indices=True)
            _calibre = _radius[tuple(_skel_idx)]

            _prob_gt = _prob_c[_gt_c]
            _calibre_gt = _calibre[_gt_c]

            # Background shell: non-airway voxels within SHELL_VOXELS of the wall,
            # binned by the calibre of the branch they hug.
            _bg_dist = ndimage.distance_transform_edt(~_gt_c)
            _shell = (~_gt_c) & (_bg_dist > 0) & (_bg_dist <= SHELL_VOXELS)
            _prob_shell = _prob_c[_shell]
            _calibre_shell = _calibre[_shell]

            def _stratify(values, calibres):
                rows = []
                total = int(values.size)
                for label, lo, hi in RADIUS_BINS:
                    m = (calibres >= lo) & (calibres < hi)
                    count = int(m.sum())
                    if count == 0:
                        continue
                    p = values[m]
                    rows.append(
                        {
                            "bin": label,
                            "count": count,
                            "pct": float(100.0 * count / total) if total else 0.0,
                            "mean_prob": float(p.mean()),
                            "median_prob": float(np.median(p)),
                            "p10_prob": float(np.percentile(p, 10)),
                            "p90_prob": float(np.percentile(p, 90)),
                        }
                    )
                return rows

            gt_confidence = {
                "radius_curve": _stratify(_prob_gt, _calibre_gt),
                "shell_curve": _stratify(_prob_shell, _calibre_shell),
                "shell_voxels": SHELL_VOXELS,
                "mean_prob": float(_prob_gt.mean()) if _prob_gt.size else 0.0,
                "shell_mean_prob": float(_prob_shell.mean()) if _prob_shell.size else 0.0,
            }
    return (gt_confidence,)


@app.cell(hide_code=True)
def _(
    go,
    gt_confidence,
    mo,
    prediction_bundle,
    prediction_bundle_error,
    prediction_view_height_slider,
    show_gt_confidence,
):
    # Confidence-vs-calibre curve: mean predicted probability on the true airway
    # and on the adjacent background shell, stratified by branch calibre, with the
    # operating threshold drawn on so it is visible which bins are thresholded away.
    if (
        not show_gt_confidence.value
        or prediction_bundle is None
        or prediction_bundle_error is not None
    ):
        gt_confidence_curve_view = mo.md("")
    elif gt_confidence is not None and "error" in gt_confidence:
        gt_confidence_curve_view = mo.md(
            f"GT-confidence curve unavailable: `{gt_confidence['error']}`"
        )
    elif gt_confidence is None or not gt_confidence["radius_curve"]:
        gt_confidence_curve_view = mo.md("No ground-truth airway voxels to stratify.")
    else:
        curve_rows = gt_confidence["radius_curve"]
        shell_by_bin = {row["bin"]: row for row in gt_confidence.get("shell_curve", [])}
        curve_labels = [row["bin"] for row in curve_rows]
        curve_means = [row["mean_prob"] for row in curve_rows]
        curve_err_up = [max(0.0, row["p90_prob"] - row["mean_prob"]) for row in curve_rows]
        curve_err_down = [max(0.0, row["mean_prob"] - row["p10_prob"]) for row in curve_rows]
        curve_customdata = [
            (row["median_prob"], row["count"], row["pct"]) for row in curve_rows
        ]
        shell_means = [
            shell_by_bin[label]["mean_prob"] if label in shell_by_bin else None
            for label in curve_labels
        ]
        shell_customdata = [
            (
                shell_by_bin[label]["median_prob"] if label in shell_by_bin else float("nan"),
                shell_by_bin[label]["count"] if label in shell_by_bin else 0,
            )
            for label in curve_labels
        ]

        gt_confidence_curve_figure = go.Figure()
        gt_confidence_curve_figure.add_trace(
            go.Bar(
                x=curve_labels,
                y=curve_means,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=curve_err_up,
                    arrayminus=curve_err_down,
                    thickness=1,
                    width=4,
                ),
                marker_color="#277da1",
                name="airway (true)",
                customdata=curve_customdata,
                hovertemplate=(
                    "%{x}<br>airway mean P=%{y:.3f}<br>median P=%{customdata[0]:.3f}"
                    "<br>voxels=%{customdata[1]:,} (%{customdata[2]:.1f}% of airway)"
                    "<extra></extra>"
                ),
            )
        )
        gt_confidence_curve_figure.add_trace(
            go.Bar(
                x=curve_labels,
                y=shell_means,
                marker_color="#adb5bd",
                name=f"background shell (≤{gt_confidence.get('shell_voxels', 2.5):g} vox)",
                customdata=shell_customdata,
                hovertemplate=(
                    "%{x}<br>shell mean P=%{y:.3f}<br>median P=%{customdata[0]:.3f}"
                    "<br>voxels=%{customdata[1]:,}<extra></extra>"
                ),
            )
        )
        gt_confidence_threshold = prediction_bundle["threshold"]
        if gt_confidence_threshold is not None:
            gt_confidence_curve_figure.add_hline(
                y=float(gt_confidence_threshold),
                line=dict(color="#d81b60", dash="dash", width=1.5),
                annotation_text=f"op threshold {float(gt_confidence_threshold):.2f}",
                annotation_position="top left",
            )
        _gt_conf_run = (
            prediction_bundle.get("run_label")
            or prediction_bundle.get("experiment_name")
            or prediction_bundle.get("run_name")
        )
        _gt_conf_provenance = (
            f"run: {_gt_conf_run}"
            f"  ·  set: {prediction_bundle.get('prediction_set_name') or '—'}"
            f"  ·  case: {prediction_bundle.get('case_id')}"
        )
        gt_confidence_curve_figure.update_layout(
            title=dict(
                text=(
                    "Predicted confidence: true airway vs adjacent tissue, by distance to wall"
                    f"<br><span style='font-size:12px;color:#6c757d'>{_gt_conf_provenance}</span>"
                ),
                x=0.01,
                xanchor="left",
                y=0.97,
                yanchor="top",
            ),
            height=int(prediction_view_height_slider.value),
            margin=dict(l=10, r=10, t=84, b=0),
            yaxis=dict(title="P(airway)", range=[0, 1]),
            xaxis=dict(title="branch calibre (centreline radius, voxels)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
            barmode="group",
            bargap=0.3,
            bargroupgap=0.08,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        gt_confidence_separability = (
            gt_confidence["mean_prob"] - gt_confidence.get("shell_mean_prob", 0.0)
        )
        gt_confidence_curve_view = mo.vstack(
            [
                mo.md(
                    f"### GT Confidence — Radius Stratification "
                    f"(airway {gt_confidence['mean_prob']:.3f} vs shell "
                    f"{gt_confidence.get('shell_mean_prob', 0.0):.3f} → "
                    f"separability gap {gt_confidence_separability:+.3f})"
                ),
                mo.as_html(gt_confidence_curve_figure),
            ],
            gap=0.5,
        )
    return (gt_confidence_curve_view,)


@app.cell(hide_code=True)
def _(
    mo,
    predicted_mask_opacity,
    prediction_bundle,
    prediction_bundle_error,
    prediction_case_selector,
    prediction_mask_selector,
    prediction_refresh_button,
    prediction_run_selector,
    prediction_runs_available,
    prediction_set_selector,
    prediction_view_height_slider,
    prediction_view_mode,
    show_gt_confidence,
    show_missed_truth_mask,
    show_predicted_mask,
    show_prediction_lung_mask,
    show_true_prediction_mask,
):
    # Compact, full-width control strip that sits ABOVE the (full-width) viewer.
    # The run summary is returned separately so it can go beside the table below.
    prediction_summary = mo.md("")
    if not prediction_runs_available:
        prediction_controls = mo.vstack(
            [
                mo.md(
                    "## Saved Prediction Viewer\n\n"
                    "No run folders with saved predictions were found under `runs/`. "
                    "Export nnU-Net predictions with `scripts.export_nnunet_viewer_run` first."
                ),
                prediction_refresh_button,
            ],
            gap=0.75,
        )
    elif prediction_bundle_error is not None:
        prediction_controls = mo.vstack(
            [
                mo.md("## Saved Prediction Viewer"),
                mo.hstack(
                    [
                        prediction_run_selector,
                        prediction_set_selector,
                        prediction_case_selector,
                        prediction_refresh_button,
                    ],
                    gap=0.8,
                    wrap=True,
                    align="end",
                ),
                mo.md(f"Prediction bundle error: `{prediction_bundle_error}`"),
            ],
            gap=0.75,
        )
    else:
        selectors_row = mo.hstack(
            [
                prediction_run_selector,
                prediction_set_selector,
                prediction_case_selector,
                prediction_mask_selector,
                prediction_view_mode,
                prediction_view_height_slider,
                prediction_refresh_button,
            ],
            gap=0.8,
            wrap=True,
            align="end",
        )
        toggles_row = mo.hstack(
            [
                show_prediction_lung_mask,
                show_true_prediction_mask,
                show_predicted_mask,
                show_missed_truth_mask,
                show_gt_confidence,
                predicted_mask_opacity,
            ],
            gap=1.2,
            justify="start",
            wrap=True,
            align="center",
        )
        prediction_controls = mo.vstack(
            [mo.md("## Saved Prediction Viewer"), selectors_row, toggles_row],
            gap=0.6,
        )

        prediction_notes = [
            f"- **Study**: `{prediction_bundle['study_name']}`" if prediction_bundle["study_name"] else "- **Study**: legacy run",
            f"- **Variant**: `{prediction_bundle['run_label']}`" if prediction_bundle["run_label"] else "- **Variant**: legacy run",
            f"- **Experiment**: `{prediction_bundle['experiment_name']}`" if prediction_bundle["experiment_name"] else "- **Experiment**: unavailable",
            f"- **Run**: {prediction_bundle['run_name']}",
            f"- **Dataset**: `{prediction_bundle['dataset_name']}`",
            f"- **Prediction set**: `{prediction_bundle['prediction_set_name']}`",
            f"- **Prediction mask**: `{prediction_bundle['prediction_mask_filename']}`",
            f"- **Case**: `{prediction_bundle['case_id']}`",
            f"- **Best validation Dice**: `{float(prediction_bundle['best_val_dice']):.4f}`" if prediction_bundle["best_val_dice"] is not None else "- **Best validation Dice**: unavailable",
            f"- **Checkpoint epoch**: `{int(prediction_bundle['checkpoint_epoch'])}`" if prediction_bundle["checkpoint_epoch"] is not None else "- **Checkpoint epoch**: unavailable",
            f"- **Threshold**: `{float(prediction_bundle['threshold']):.2f}`" if prediction_bundle["threshold"] is not None else "- **Threshold**: unavailable",
        ]
        prediction_summary_blocks = [mo.md("### Run Summary"), mo.md("\n".join(prediction_notes))]

        last_epoch_metrics = prediction_bundle["last_epoch_metrics"]
        if last_epoch_metrics is not None:
            last_epoch_lines = []
            epoch_value = last_epoch_metrics.get("epoch")
            if epoch_value is not None:
                last_epoch_lines.append(f"- **Epoch**: `{int(epoch_value)}`")
            train_loss = last_epoch_metrics.get("train_loss")
            if train_loss is not None:
                last_epoch_lines.append(f"- **Train loss**: `{float(train_loss):.4f}`")
            train_dice = last_epoch_metrics.get("train_dice")
            if train_dice is not None:
                last_epoch_lines.append(f"- **Train Dice**: `{float(train_dice):.4f}`")
            val_dice = last_epoch_metrics.get("val_dice")
            if val_dice is not None:
                last_epoch_lines.append(f"- **Validation Dice**: `{float(val_dice):.4f}`")
            if last_epoch_lines:
                prediction_summary_blocks.extend(
                    [mo.md("### Last Epoch"), mo.md("\n".join(last_epoch_lines))]
                )

        prediction_summary = mo.vstack(prediction_summary_blocks, gap=0.4)
    return prediction_controls, prediction_summary


@app.cell(hide_code=True)
def _(
    distal_analysis_panel,
    gt_confidence_curve_view,
    mo,
    prediction_controls,
    prediction_summary,
    prediction_viewer,
):
    # Full-width layout: control strip, then the viewer spanning the page (so its
    # canvas has real width), then the analysis table beside the run summary, then
    # the GT-confidence curve (empty unless that toggle is on).
    prediction_panel = mo.vstack(
        [
            prediction_controls,
            prediction_viewer,
            mo.hstack(
                [distal_analysis_panel, prediction_summary],
                widths=[1.6, 1.0],
                gap=1.5,
                align="start",
                wrap=True,
            ),
            gt_confidence_curve_view,
        ],
        gap=0.9,
    )
    return (prediction_panel,)


@app.cell(hide_code=True)
def _(SLICE_TYPES, mo):
    dataset_selector = mo.ui.dropdown(
        {"ATM'22": "atm22", "AeroPath": "aeropath"},
        value="ATM'22",
        label="Dataset",
    )
    labelled_view_mode = mo.ui.dropdown(
        list(SLICE_TYPES.keys()), value="Multiplanar", label="View"
    )
    labelled_height_slider = mo.ui.slider(
        320, 900, value=520, step=20, label="Viewer height"
    )
    show_labelled_lung = mo.ui.switch(value=False, label="Lung")
    show_labelled_airway = mo.ui.switch(value=True, label="Airway")
    return (
        dataset_selector,
        labelled_height_slider,
        labelled_view_mode,
        show_labelled_airway,
        show_labelled_lung,
    )


@app.cell(hide_code=True)
def _(
    RAW_AEROPATH_ROOT,
    RAW_ATM22_ROOT,
    dataset_selector,
    list_atm22_case_ids,
    list_case_ids,
    mo,
):
    if dataset_selector.value == "atm22":
        labelled_root = RAW_ATM22_ROOT
        labelled_case_ids = list_atm22_case_ids(labelled_root)
    else:
        labelled_root = RAW_AEROPATH_ROOT
        labelled_case_ids = list_case_ids(labelled_root)

    labelled_case_selector = mo.ui.dropdown(
        labelled_case_ids,
        value=labelled_case_ids[0],
        label="Case",
        searchable=True,
    )
    return labelled_case_selector, labelled_root


@app.cell(hide_code=True)
def _(
    dataset_selector,
    labelled_case_selector,
    labelled_root,
    resolve_atm22_case_paths,
    resolve_case_paths,
):
    if dataset_selector.value == "atm22":
        labelled_paths = resolve_atm22_case_paths(
            labelled_case_selector.value, batch_root=labelled_root
        )
    else:
        labelled_paths = resolve_case_paths(
            labelled_case_selector.value, data_root=labelled_root
        )
    return (labelled_paths,)


@app.cell(hide_code=True)
def _(
    SLICE_TYPES,
    labelled_height_slider,
    labelled_layer_index,
    labelled_layer_shown,
    labelled_view_mode,
    nv_labelled,
    show_labelled_airway,
    show_labelled_lung,
):
    if nv_labelled is not None:
        _toggles = {"lung": show_labelled_lung.value, "airway": show_labelled_airway.value}
        for _name, _idx in labelled_layer_index.items():
            if _idx < len(nv_labelled.volumes):
                _on = _toggles.get(_name, True)
                nv_labelled.volumes[_idx].opacity = (
                    labelled_layer_shown.get(_name, 0.7) if _on else 0.0
                )
        nv_labelled.height = int(labelled_height_slider.value)
        nv_labelled.set_slice_type(SLICE_TYPES[labelled_view_mode.value])
    return


@app.cell(hide_code=True)
def _(
    active_viewer,
    dataset_selector,
    labelled_case_selector,
    labelled_height_slider,
    labelled_paths,
    labelled_view_mode,
    mo,
    nib,
    nv_labelled,
    show_labelled_airway,
    show_labelled_lung,
):
    _dataset_label = "ATM'22" if dataset_selector.value == "atm22" else "AeroPath"
    if active_viewer.value != "labelled":
        labelled_body = mo.md(
            "The labelled-data canvas is not loaded. Select **Labelled data** "
            "from the NIfTI canvas control above to load it."
        )
    elif nv_labelled is None:
        labelled_body = mo.md(
            f"Case `{labelled_case_selector.value}` has no airway label to display."
        )
    else:
        _zooms = nib.load(str(labelled_paths["ct"])).header.get_zooms()[:3]
        _spacing = " x ".join(f"{float(v):.3f}" for v in _zooms)
        _note_line = (
            f"**{_dataset_label}** · case `{labelled_case_selector.value}` · "
            f"spacing `{_spacing} mm` · lung mask "
            f"`{'available' if labelled_paths.get('lung') else 'none'}`"
        )
        labelled_body = mo.vstack(
            [
                mo.hstack(
                    [mo.md(_note_line), show_labelled_lung, show_labelled_airway],
                    gap=1.2,
                    justify="start",
                    align="center",
                    wrap=True,
                ),
                mo.md("The active labelled-data canvas is shown above."),
            ],
            gap=0.6,
        )

    labelled_panel = mo.vstack(
        [
            mo.md("## Labelled-Data Inspector"),
            mo.hstack(
                [dataset_selector, labelled_case_selector, labelled_view_mode, labelled_height_slider],
                widths=[1.0, 1.2, 1.0, 1.4],
                gap=0.8,
                wrap=True,
                align="end",
            ),
            labelled_body,
        ],
        gap=0.8,
    )
    return (labelled_panel,)


if __name__ == "__main__":
    app.run()
