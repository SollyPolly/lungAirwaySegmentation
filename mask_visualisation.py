# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.1",
#     "nibabel",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-image",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    REPORT_CAMERA = dict(
        eye=dict(x=1.45, y=1.55, z=0.95),
        center=dict(x=0.0, y=0.0, z=0.0),
        up=dict(x=0.0, y=0.0, z=1.0),
    )
    return (REPORT_CAMERA,)


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import nibabel as nib
    import numpy as np
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes

    from lung_airway_segmentation.settings import (
        DEFAULT_CROP_MARGIN,
        DEFAULT_HU_WINDOW,
        RAW_AEROPATH_ROOT,
    )
    from lung_airway_segmentation.preprocessing.geometry import crop_volume
    from lung_airway_segmentation.preprocessing.intensity import clip_ct_to_hu_window
    from lung_airway_segmentation.preprocessing.pipeline import preprocess_case
    from lung_airway_segmentation.io.case_layout import list_case_ids, resolve_case_paths
    from lung_airway_segmentation.io.nifti import load_canonical_image

    return (
        DEFAULT_CROP_MARGIN,
        DEFAULT_HU_WINDOW,
        Path,
        RAW_AEROPATH_ROOT,
        clip_ct_to_hu_window,
        crop_volume,
        go,
        json,
        list_case_ids,
        load_canonical_image,
        marching_cubes,
        mo,
        nib,
        np,
        preprocess_case,
        resolve_case_paths,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # AeroPath Raw vs Cropped Viewer

    This notebook uses the shared `preprocess_case(...)` function from
    `lung_airway_segmentation.preprocessing.pipeline`. The preprocessing shown here keeps the original
    voxel spacing and performs:

    - alignment checks
    - lung-box cropping
    - CT clipping and normalization for training output
    - binary airway and lung masks

    The 3D views show raw and cropped masks. The slice viewers below show
    CT images with mask overlays for direct inspection.
    """)
    return


@app.cell
def _(viewer_panel):
    viewer_panel
    return


@app.cell
def _(DEFAULT_CROP_MARGIN, RAW_AEROPATH_ROOT, list_case_ids, mo):
    data_root = RAW_AEROPATH_ROOT
    case_ids = list_case_ids(data_root)

    case_selector = mo.ui.dropdown(
        case_ids,
        value=case_ids[0],
        label="Case",
        searchable=True,
    )
    crop_margin_slider = mo.ui.slider(
        0,
        20,
        value=DEFAULT_CROP_MARGIN,
        step=1,
        label="Crop margin (voxels)",
    )
    mesh_height_slider = mo.ui.slider(
        320,
        720,
        value=460,
        step=20,
        label="3D viewer height",
    )
    slice_height_slider = mo.ui.slider(
        320,
        720,
        value=420,
        step=20,
        label="Slice viewer height",
    )
    show_lung_mask = mo.ui.switch(value=True, label="Show lung mask")
    show_airway_mask = mo.ui.switch(value=True, label="Show airway mask")
    return (
        case_selector,
        crop_margin_slider,
        data_root,
        mesh_height_slider,
        show_airway_mask,
        show_lung_mask,
        slice_height_slider,
    )


@app.cell
def _(
    case_selector,
    crop_margin_slider,
    data_root,
    load_canonical_image,
    np,
    preprocess_case,
    resolve_case_paths,
):
    case_id = case_selector.value
    crop_margin_value = int(crop_margin_slider.value)
    paths = resolve_case_paths(case_id, data_root=data_root)

    ct_img = load_canonical_image(paths["ct"])
    lung_img = load_canonical_image(paths["lung"])
    airway_img = load_canonical_image(paths["airway"])

    raw_ct = np.asarray(ct_img.dataobj, dtype=np.float32)
    raw_lung_mask = np.asarray(lung_img.dataobj) > 0
    raw_airway_mask = np.asarray(airway_img.dataobj) > 0
    original_spacing = tuple(float(value) for value in ct_img.header.get_zooms()[:3])

    processed_case = preprocess_case(
        case_id,
        data_root=data_root,
        include_lung_mask=True,
        crop_margin=crop_margin_value,
    )
    return (
        case_id,
        crop_margin_value,
        original_spacing,
        processed_case,
        raw_airway_mask,
        raw_ct,
        raw_lung_mask,
    )


@app.cell
def _(crop_volume, processed_case, raw_ct):
    cropped_ct_hu = crop_volume(raw_ct, processed_case.crop_box).astype("float32", copy=False)
    return (cropped_ct_hu,)


@app.cell
def _(
    DEFAULT_HU_WINDOW,
    clip_ct_to_hu_window,
    cropped_ct_hu,
    processed_case,
    raw_ct,
):
    hu_window = DEFAULT_HU_WINDOW
    raw_ct_display = clip_ct_to_hu_window(raw_ct, hu_window)
    cropped_ct_display = clip_ct_to_hu_window(cropped_ct_hu, hu_window)
    normalized_ct = processed_case.ct
    return cropped_ct_display, hu_window, normalized_ct, raw_ct_display


@app.cell
def _(mo):
    plane_selector = mo.ui.dropdown(
        {
            "Axial (z)": 0,
            "Coronal (y)": 1,
            "Sagittal (x)": 2,
        },
        value="Axial (z)",
        label="Slice plane",
    )
    return (plane_selector,)


@app.cell
def _(
    default_mask_index,
    mo,
    plane_selector,
    raw_airway_mask,
    raw_ct,
    raw_lung_mask,
):
    axis = int(plane_selector.value)
    shared_slice_slider = mo.ui.slider(
        0,
        int(raw_ct.shape[axis]) - 1,
        value=default_mask_index(raw_lung_mask | raw_airway_mask, axis),
        step=1,
        label="Slice",
    )
    return axis, shared_slice_slider


@app.cell
def _(marching_cubes, np):
    def build_mask_mesh(mask_volume, voxel_spacing, stride):
        volume = np.asarray(mask_volume[::stride, ::stride, ::stride]) > 0
        if not volume.any():
            return None

        spacing = tuple(float(value) * stride for value in voxel_spacing)
        mesh_vertices, mesh_faces, _, _ = marching_cubes(
            volume.astype(np.uint8),
            level=0.5,
            spacing=spacing,
        )
        return {
            "vertices": mesh_vertices,
            "faces": mesh_faces,
        }

    def extract_plane(volume, axis, index):
        if axis == 0:
            plane = volume[index, :, :]
        elif axis == 1:
            plane = volume[:, index, :]
        else:
            plane = volume[:, :, index]
        return np.rot90(plane)

    def overlay_mask(mask, axis, index):
        plane = extract_plane(mask.astype(np.uint8, copy=False), axis, index)
        return np.where(plane > 0, 1.0, np.nan)

    def default_mask_index(mask_volume, axis):
        positive_projection = np.any(mask_volume > 0, axis=tuple(dim for dim in range(mask_volume.ndim) if dim != axis))
        positive_indices = np.flatnonzero(positive_projection)
        if positive_indices.size == 0:
            return int(mask_volume.shape[axis]) // 2
        return int(positive_indices[positive_indices.size // 2])

    return build_mask_mesh, default_mask_index, extract_plane, overlay_mask


@app.cell
def _(build_mask_mesh, processed_case):
    cropped_lung_mesh = build_mask_mesh(processed_case.lung_mask, processed_case.spacing, stride=6)
    cropped_airway_mesh = build_mask_mesh(processed_case.airway_mask, processed_case.spacing, stride=4)
    return cropped_airway_mesh, cropped_lung_mesh


@app.cell
def _(
    case_id,
    case_selector,
    crop_margin_slider,
    crop_margin_value,
    hu_window,
    mesh_height_slider,
    mo,
    original_spacing,
    processed_case,
    show_airway_mask,
    show_lung_mask,
    slice_height_slider,
):
    notes = [
        f"Original spacing: `{original_spacing[0]:.3f} x {original_spacing[1]:.3f} x {original_spacing[2]:.3f} mm`",
        f"Crop margin: `{crop_margin_value}` voxels",
        f"Crop box: `{processed_case.crop_box}`",
        f"Processed shape: `{processed_case.metadata['processed_shape']}`",
        f"Display HU window: `{hu_window}`",
        "Mesh detail is fixed to a single high-detail setting.",
    ]

    controls = mo.vstack(
        [
            case_selector,
            crop_margin_slider,
            mesh_height_slider,
            slice_height_slider,
            mo.hstack(
                [show_lung_mask, show_airway_mask],
                justify="start",
                gap=1.0,
            ),
            mo.md(f"### Case {case_id}"),
            mo.md("\n".join(f"- {note}" for note in notes)),
        ],
        gap=1.0,
    )
    return (controls,)


@app.cell
def _(
    REPORT_CAMERA,
    cropped_airway_mesh,
    cropped_lung_mesh,
    go,
    mesh_height_slider,
    mo,
    show_airway_mask,
    show_lung_mask,
):
    cropped_mesh_figure = go.Figure()

    if show_lung_mask.value and cropped_lung_mesh is not None:
        cropped_lung_vertices = cropped_lung_mesh["vertices"]
        cropped_lung_faces = cropped_lung_mesh["faces"]
        cropped_mesh_figure.add_trace(
            go.Mesh3d(
                x=cropped_lung_vertices[:, 0],
                y=cropped_lung_vertices[:, 1],
                z=cropped_lung_vertices[:, 2],
                i=cropped_lung_faces[:, 0],
                j=cropped_lung_faces[:, 1],
                k=cropped_lung_faces[:, 2],
                color="#2a9d8f",
                opacity=0.18,
                name="Cropped lung mask",
                hovertemplate="Cropped lung mask<extra></extra>",
                flatshading=True,
            )
        )

    if show_airway_mask.value and cropped_airway_mesh is not None:
        cropped_airway_vertices = cropped_airway_mesh["vertices"]
        cropped_airway_faces = cropped_airway_mesh["faces"]
        cropped_mesh_figure.add_trace(
            go.Mesh3d(
                x=cropped_airway_vertices[:, 0],
                y=cropped_airway_vertices[:, 1],
                z=cropped_airway_vertices[:, 2],
                i=cropped_airway_faces[:, 0],
                j=cropped_airway_faces[:, 1],
                k=cropped_airway_faces[:, 2],
                color="#d81b60",
                opacity=0.92,
                name="Cropped airway mask",
                hovertemplate="Cropped airway mask<extra></extra>",
                flatshading=True,
            )
        )

    if not cropped_mesh_figure.data:
        cropped_3d_view = mo.md("Enable at least one mask to display the cropped 3D view.")
    else:
        cropped_mesh_figure.update_layout(
            height=int(mesh_height_slider.value),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=0.02,
                bgcolor="rgba(255,255,255,0.7)",
            ),
            scene=dict(
                aspectmode="data",
                dragmode="turntable",
                xaxis_title="x (mm)",
                yaxis_title="y (mm)",
                zaxis_title="z (mm)",
                bgcolor="white",
                camera=REPORT_CAMERA,
            ),
        )
        cropped_3d_view = mo.vstack(
            [
                mo.md("### Cropped 3D Viewer"),
                mo.as_html(cropped_mesh_figure),
            ],
            gap=0.5,
        )
    return (cropped_3d_view,)


@app.cell
def _(
    axis,
    extract_plane,
    go,
    overlay_mask,
    raw_airway_mask,
    raw_ct_display,
    raw_lung_mask,
    shared_slice_slider,
    show_airway_mask,
    show_lung_mask,
    slice_height_slider,
):
    raw_shared_slice_index = int(shared_slice_slider.value)
    raw_slice = extract_plane(raw_ct_display, axis, raw_shared_slice_index)
    raw_slice_figure = go.Figure()
    raw_slice_figure.add_trace(
        go.Heatmap(
            z=raw_slice,
            colorscale="Gray",
            showscale=False,
            hovertemplate="CT %{z:.1f}<extra></extra>",
        )
    )
    if show_lung_mask.value:
        raw_slice_figure.add_trace(
            go.Heatmap(
                z=overlay_mask(raw_lung_mask, axis, raw_shared_slice_index),
                colorscale=[[0.0, "rgba(42,157,143,0.0)"], [1.0, "rgba(42,157,143,0.30)"]],
                showscale=False,
                hoverinfo="skip",
            )
        )
    if show_airway_mask.value:
        raw_slice_figure.add_trace(
            go.Heatmap(
                z=overlay_mask(raw_airway_mask, axis, raw_shared_slice_index),
                colorscale=[[0.0, "rgba(216,27,96,0.0)"], [1.0, "rgba(216,27,96,0.78)"]],
                showscale=False,
                hoverinfo="skip",
            )
        )
    raw_slice_figure.update_layout(
        title=f"Raw CT slice {raw_shared_slice_index}",
        height=int(slice_height_slider.value),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed", scaleanchor="x", scaleratio=1),
    )
    return (raw_slice_figure,)


@app.cell
def _(
    axis,
    cropped_ct_display,
    extract_plane,
    go,
    overlay_mask,
    processed_case,
    shared_slice_slider,
    show_airway_mask,
    show_lung_mask,
    slice_height_slider,
):
    cropped_slice_figure = go.Figure()
    cropped_source_slice_index = int(shared_slice_slider.value)
    crop_bounds = processed_case.crop_box[axis]
    crop_start = int(crop_bounds[0])
    crop_stop = int(crop_bounds[1])

    if crop_start <= cropped_source_slice_index < crop_stop:
        cropped_slice_index = cropped_source_slice_index - crop_start
        cropped_slice = extract_plane(cropped_ct_display, axis, cropped_slice_index)
        cropped_slice_figure.add_trace(
            go.Heatmap(
                z=cropped_slice,
                colorscale="Gray",
                showscale=False,
                hovertemplate="CT %{z:.1f}<extra></extra>",
            )
        )
        if show_lung_mask.value and processed_case.lung_mask is not None:
            cropped_slice_figure.add_trace(
                go.Heatmap(
                    z=overlay_mask(processed_case.lung_mask, axis, cropped_slice_index),
                    colorscale=[[0.0, "rgba(42,157,143,0.0)"], [1.0, "rgba(42,157,143,0.30)"]],
                    showscale=False,
                    hoverinfo="skip",
                )
            )
        if show_airway_mask.value:
            cropped_slice_figure.add_trace(
                go.Heatmap(
                    z=overlay_mask(processed_case.airway_mask, axis, cropped_slice_index),
                    colorscale=[[0.0, "rgba(216,27,96,0.0)"], [1.0, "rgba(216,27,96,0.78)"]],
                    showscale=False,
                    hoverinfo="skip",
                )
            )
        cropped_title = (
            f"Cropped CT slice {cropped_slice_index} "
            f"(raw slice {cropped_source_slice_index})"
        )
    else:
        cropped_title = (
            f"Cropped CT unavailable for raw slice {cropped_source_slice_index} "
            f"(crop range {crop_start} to {crop_stop - 1})"
        )
        cropped_slice_figure.add_annotation(
            text=(
                "Selected raw slice is outside the cropped volume.<br>"
                f"Crop range on this axis: {crop_start} to {crop_stop - 1}"
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=15, color="#333333"),
        )
        cropped_slice_figure.update_xaxes(visible=False)
        cropped_slice_figure.update_yaxes(visible=False)

    cropped_slice_figure.update_layout(
        title=cropped_title,
        height=int(slice_height_slider.value),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed", scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return (cropped_slice_figure,)


@app.cell
def _(controls, cropped_3d_view, mo):
    top_panel = mo.hstack(
        [
            controls,
            cropped_3d_view,
        ],
        widths=[0.95, 2.2],
        gap=1.25,
        align="start",
        wrap=True,
    )
    return (top_panel,)


@app.cell
def _(
    cropped_slice_figure,
    mo,
    plane_selector,
    raw_slice_figure,
    shared_slice_slider,
):
    slice_panel = mo.vstack(
        [
            mo.md("## Slice Viewer"),
            mo.hstack(
                [plane_selector, shared_slice_slider],
                widths=[0.85, 2.15],
                gap=1.0,
                wrap=True,
                align="center",
            ),
            mo.hstack(
                [mo.as_html(raw_slice_figure), mo.as_html(cropped_slice_figure)],
                widths="equal",
                gap=1.0,
                wrap=True,
            ),
        ],
        gap=0.75,
    )
    return (slice_panel,)


@app.cell
def _(
    DEFAULT_HU_WINDOW,
    Path,
    clip_ct_to_hu_window,
    json,
    load_canonical_image,
    nib,
    np,
    resolve_case_paths,
):
    prediction_run_root = Path(__file__).resolve().parent / "runs"

    def list_prediction_run_names(run_root):
        if not run_root.exists():
            return []

        run_dirs = []
        for metadata_path in run_root.rglob("run_metadata.json"):
            run_dir = metadata_path.parent
            predictions_dir = run_dir / "predictions"
            if not predictions_dir.is_dir():
                continue
            has_case_predictions = any(
                (case_dir / "prediction_metadata.json").is_file()
                for case_dir in predictions_dir.iterdir()
                if case_dir.is_dir()
            )
            if has_case_predictions:
                run_dirs.append(run_dir)

        run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return [str(path.relative_to(run_root)) for path in run_dirs]

    def resolve_prediction_run_dir(run_root, run_name):
        return (run_root / run_name).resolve()

    def list_prediction_case_ids(run_root, run_name):
        if run_name is None:
            return []

        predictions_dir = resolve_prediction_run_dir(run_root, run_name) / "predictions"
        if not predictions_dir.exists():
            return []

        case_ids = [
            case_dir.name
            for case_dir in predictions_dir.iterdir()
            if case_dir.is_dir() and (case_dir / "prediction_metadata.json").is_file()
        ]
        return sorted(case_ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))

    def load_prediction_bundle(run_root, run_name, case_id):
        run_dir = resolve_prediction_run_dir(run_root, run_name)
        prediction_case_dir = run_dir / "predictions" / str(case_id)
        prediction_metadata_path = prediction_case_dir / "prediction_metadata.json"
        prediction_mask_path = prediction_case_dir / "airway_pred_full.nii.gz"

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

        data_root = Path(run_metadata["data_root"]).resolve() if run_metadata.get("data_root") else None
        case_paths = resolve_case_paths(case_id, data_root=data_root) if data_root is not None else resolve_case_paths(case_id)

        ct_image = load_canonical_image(case_paths["ct"])
        ct = np.asarray(ct_image.dataobj, dtype=np.float32)
        hu_window = tuple(
            float(value)
            for value in resolved_config.get("data", {}).get("preprocessing", {}).get("hu_window", DEFAULT_HU_WINDOW)
        )
        ct_display = clip_ct_to_hu_window(ct, hu_window)
        spacing = tuple(float(value) for value in ct_image.header.get_zooms()[:3])

        if case_paths["airway"] is None:
            raise ValueError(f"Case {case_id} does not have a reference airway mask.")
        true_mask = np.asarray(load_canonical_image(case_paths["airway"]).dataobj) > 0
        lung_mask = (
            np.asarray(load_canonical_image(case_paths["lung"]).dataobj) > 0
            if case_paths["lung"] is not None
            else None
        )
        prediction_mask = np.asarray(nib.load(str(prediction_mask_path)).dataobj) > 0

        if ct.shape != true_mask.shape or ct.shape != prediction_mask.shape:
            raise ValueError(
                "Prediction bundle shape mismatch: "
                f"ct={ct.shape}, true_mask={true_mask.shape}, prediction_mask={prediction_mask.shape}"
            )
        if lung_mask is not None and ct.shape != lung_mask.shape:
            raise ValueError(
                "Prediction bundle lung mask shape mismatch: "
                f"ct={ct.shape}, lung_mask={lung_mask.shape}"
            )

        probability_path = prediction_case_dir / "airway_prob_full.nii.gz"
        probability_volume = (
            np.asarray(nib.load(str(probability_path)).dataobj, dtype=np.float32)
            if probability_path.exists()
            else None
        )

        history_entries = history.get("history", []) if isinstance(history, dict) else []
        best_metrics = history.get("best", {}) if isinstance(history, dict) else {}

        return {
            "run_dir": run_dir,
            "run_name": run_name,
            "case_id": str(case_id),
            "ct_path": case_paths["ct"],
            "lung_mask_path": case_paths["lung"],
            "true_mask_path": case_paths["airway"],
            "prediction_mask_path": prediction_mask_path,
            "probability_path": probability_path if probability_path.exists() else None,
            "prediction_metadata_path": prediction_metadata_path,
            "ct": ct,
            "ct_display": ct_display,
            "hu_window": hu_window,
            "lung_mask": lung_mask,
            "true_mask": true_mask,
            "prediction_mask": prediction_mask,
            "probability_volume": probability_volume,
            "spacing": spacing,
            "best_val_dice": best_metrics.get("val_dice"),
            "best_epoch": best_metrics.get("epoch"),
            "last_epoch_metrics": history_entries[-1] if history_entries else None,
            "checkpoint_epoch": prediction_metadata.get("checkpoint_epoch"),
            "threshold": prediction_metadata.get("threshold"),
        }

    return (
        list_prediction_case_ids,
        list_prediction_run_names,
        load_prediction_bundle,
        prediction_run_root,
    )


@app.cell
def _(list_prediction_run_names, mo, prediction_run_root):
    prediction_run_names = list_prediction_run_names(prediction_run_root)
    prediction_runs_available = len(prediction_run_names) > 0

    prediction_run_selector = (
        mo.ui.dropdown(
            prediction_run_names,
            value=prediction_run_names[0],
            label="Prediction run",
            searchable=True,
        )
        if prediction_runs_available
        else None
    )
    prediction_plane_selector = mo.ui.dropdown(
        {
            "Axial (z)": 0,
            "Coronal (y)": 1,
            "Sagittal (x)": 2,
        },
        value="Axial (z)",
        label="Prediction plane",
    )
    prediction_view_height_slider = mo.ui.slider(
        320,
        720,
        value=420,
        step=20,
        label="Prediction viewer height",
    )
    show_prediction_lung_mask = mo.ui.switch(value=False, label="Lung")
    show_true_prediction_mask = mo.ui.switch(value=True, label="Truth")
    show_predicted_mask = mo.ui.switch(value=True, label="Prediction")
    predicted_mask_opacity = mo.ui.slider(
        0.05,
        1.0,
        value=0.70,
        step=0.05,
        label="Predicted mask opacity",
    )
    return (
        predicted_mask_opacity,
        prediction_plane_selector,
        prediction_run_selector,
        prediction_runs_available,
        prediction_view_height_slider,
        show_predicted_mask,
        show_prediction_lung_mask,
        show_true_prediction_mask,
    )


@app.cell
def _(
    list_prediction_case_ids,
    mo,
    prediction_run_root,
    prediction_run_selector,
    prediction_runs_available,
):
    if not prediction_runs_available or prediction_run_selector is None:
        prediction_case_selector = None
    else:
        prediction_case_ids = list_prediction_case_ids(
            prediction_run_root,
            prediction_run_selector.value,
        )
        prediction_case_selector = (
            mo.ui.dropdown(
                prediction_case_ids,
                value=prediction_case_ids[0],
                label="Predicted case",
                searchable=True,
            )
            if prediction_case_ids
            else None
        )
    return (prediction_case_selector,)


@app.cell
def _(
    load_prediction_bundle,
    prediction_case_selector,
    prediction_run_root,
    prediction_run_selector,
    prediction_runs_available,
):
    if not prediction_runs_available or prediction_run_selector is None:
        prediction_bundle = None
        prediction_bundle_error = None
    elif prediction_case_selector is None:
        prediction_bundle = None
        prediction_bundle_error = "Selected run does not contain any saved case predictions."
    else:
        try:
            prediction_bundle = load_prediction_bundle(
                prediction_run_root,
                prediction_run_selector.value,
                prediction_case_selector.value,
            )
            prediction_bundle_error = None
        except Exception as error:
            prediction_bundle = None
            prediction_bundle_error = str(error)
    return prediction_bundle, prediction_bundle_error


@app.cell
def _(default_mask_index, mo, prediction_bundle, prediction_plane_selector):
    prediction_axis = int(prediction_plane_selector.value)
    if prediction_bundle is None:
        prediction_slice_slider = None
    else:
        prediction_union_mask = prediction_bundle["true_mask"] | prediction_bundle["prediction_mask"]
        if prediction_bundle["lung_mask"] is not None:
            prediction_union_mask = prediction_union_mask | prediction_bundle["lung_mask"]
        prediction_slice_slider = mo.ui.slider(
            0,
            int(prediction_bundle["ct"].shape[prediction_axis]) - 1,
            value=default_mask_index(prediction_union_mask, prediction_axis),
            step=1,
            label="Prediction slice",
        )
    return prediction_axis, prediction_slice_slider


@app.cell
def _(build_mask_mesh, prediction_bundle):
    if prediction_bundle is None:
        prediction_lung_mesh = None
        prediction_true_mesh = None
        prediction_mask_mesh = None
    else:
        prediction_lung_mesh = (
            build_mask_mesh(
                prediction_bundle["lung_mask"],
                prediction_bundle["spacing"],
                stride=6,
            )
            if prediction_bundle["lung_mask"] is not None
            else None
        )
        prediction_true_mesh = build_mask_mesh(
            prediction_bundle["true_mask"],
            prediction_bundle["spacing"],
            stride=5,
        )
        prediction_mask_mesh = build_mask_mesh(
            prediction_bundle["prediction_mask"],
            prediction_bundle["spacing"],
            stride=5,
        )
    return prediction_lung_mesh, prediction_mask_mesh, prediction_true_mesh


@app.cell
def _(
    REPORT_CAMERA,
    go,
    mo,
    predicted_mask_opacity,
    prediction_bundle,
    prediction_bundle_error,
    prediction_lung_mesh,
    prediction_mask_mesh,
    prediction_true_mesh,
    prediction_view_height_slider,
    show_predicted_mask,
    show_prediction_lung_mask,
    show_true_prediction_mask,
):
    if prediction_bundle_error is not None:
        prediction_3d_view = mo.md(f"Prediction 3D viewer error: `{prediction_bundle_error}`")
    elif prediction_bundle is None:
        prediction_3d_view = mo.md("Prediction 3D viewer will appear here once a saved run is available.")
    else:
        prediction_mesh_figure = go.Figure()

        if show_prediction_lung_mask.value and prediction_lung_mesh is not None:
            prediction_lung_vertices = prediction_lung_mesh["vertices"]
            prediction_lung_faces = prediction_lung_mesh["faces"]
            prediction_mesh_figure.add_trace(
                go.Mesh3d(
                    x=prediction_lung_vertices[:, 0],
                    y=prediction_lung_vertices[:, 1],
                    z=prediction_lung_vertices[:, 2],
                    i=prediction_lung_faces[:, 0],
                    j=prediction_lung_faces[:, 1],
                    k=prediction_lung_faces[:, 2],
                    color="#2a9d8f",
                    opacity=0.14,
                    name="Lung mask",
                    hovertemplate="Lung mask<extra></extra>",
                    flatshading=True,
                )
            )

        if show_true_prediction_mask.value and prediction_true_mesh is not None:
            prediction_true_vertices = prediction_true_mesh["vertices"]
            prediction_true_faces = prediction_true_mesh["faces"]
            prediction_mesh_figure.add_trace(
                go.Mesh3d(
                    x=prediction_true_vertices[:, 0],
                    y=prediction_true_vertices[:, 1],
                    z=prediction_true_vertices[:, 2],
                    i=prediction_true_faces[:, 0],
                    j=prediction_true_faces[:, 1],
                    k=prediction_true_faces[:, 2],
                    color="#277da1",
                    opacity=0.38,
                    name="True airway mask",
                    hovertemplate="True airway mask<extra></extra>",
                    flatshading=True,
                )
            )

        if show_predicted_mask.value and prediction_mask_mesh is not None:
            prediction_mesh_vertices = prediction_mask_mesh["vertices"]
            prediction_mesh_faces = prediction_mask_mesh["faces"]
            prediction_mesh_figure.add_trace(
                go.Mesh3d(
                    x=prediction_mesh_vertices[:, 0],
                    y=prediction_mesh_vertices[:, 1],
                    z=prediction_mesh_vertices[:, 2],
                    i=prediction_mesh_faces[:, 0],
                    j=prediction_mesh_faces[:, 1],
                    k=prediction_mesh_faces[:, 2],
                    color="#d81b60",
                    opacity=float(predicted_mask_opacity.value),
                    name="Predicted airway mask",
                    hovertemplate="Predicted airway mask<extra></extra>",
                    flatshading=True,
                )
            )

        if not prediction_mesh_figure.data:
            prediction_3d_view = mo.md("Enable at least one mask to display the saved prediction 3D view.")
        else:
            prediction_mesh_figure.update_layout(
                height=int(prediction_view_height_slider.value),
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=0.02,
                    bgcolor="rgba(255,255,255,0.7)",
                ),
                scene=dict(
                    aspectmode="data",
                    dragmode="turntable",
                    xaxis_title="x (mm)",
                    yaxis_title="y (mm)",
                    zaxis_title="z (mm)",
                    bgcolor="white",
                    camera=REPORT_CAMERA,
                ),
            )
            prediction_3d_view = mo.vstack(
                [
                    mo.md(f"### Saved Prediction 3D Viewer | Case {prediction_bundle['case_id']}"),
                    mo.as_html(prediction_mesh_figure),
                ],
                gap=0.5,
            )
    return (prediction_3d_view,)


@app.cell
def _(
    mo,
    predicted_mask_opacity,
    prediction_bundle,
    prediction_bundle_error,
    prediction_case_selector,
    prediction_plane_selector,
    prediction_run_selector,
    prediction_runs_available,
    prediction_slice_slider,
    prediction_view_height_slider,
    show_predicted_mask,
    show_prediction_lung_mask,
    show_true_prediction_mask,
):
    if not prediction_runs_available:
        prediction_controls = mo.md(
            "## Saved Prediction Viewer\n\n"
            "No run folders with saved predictions were found under `runs/`. "
            "Run `scripts.predict_case` first."
        )
    elif prediction_bundle_error is not None:
        prediction_controls = mo.vstack(
            [
                mo.md("## Saved Prediction Viewer"),
                mo.hstack(
                    [prediction_run_selector, prediction_case_selector],
                    widths=[1.6, 1.0],
                    gap=0.8,
                    wrap=True,
                    align="end",
                ),
                mo.md(f"Prediction bundle error: `{prediction_bundle_error}`"),
            ],
            gap=0.75,
        )
    else:
        prediction_primary_controls = mo.hstack(
            [prediction_run_selector, prediction_case_selector],
            widths=[1.6, 1.0],
            gap=0.8,
            wrap=True,
            align="end",
        )
        prediction_secondary_controls = mo.hstack(
            [prediction_plane_selector, prediction_slice_slider],
            widths=[1.0, 1.6],
            gap=0.8,
            wrap=True,
            align="end",
        )
        prediction_slider_controls = mo.hstack(
            [prediction_view_height_slider, predicted_mask_opacity],
            widths="equal",
            gap=0.8,
            wrap=True,
            align="end",
        )
        prediction_mask_controls = mo.hstack(
            [show_prediction_lung_mask, show_true_prediction_mask, show_predicted_mask],
            widths="equal",
            justify="start",
            gap=0.8,
            wrap=True,
            align="center",
        )

        prediction_notes = [
            f"- **Run**: {prediction_bundle['run_name']}",
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
                    [
                        mo.md("### Last Epoch"),
                        mo.md("\n".join(last_epoch_lines)),
                    ]
                )

        prediction_controls = mo.vstack(
            [
                mo.md("## Saved Prediction Viewer"),
                prediction_primary_controls,
                prediction_secondary_controls,
                prediction_slider_controls,
                mo.md("### Masks"),
                prediction_mask_controls,
                *prediction_summary_blocks,
            ],
            gap=0.8,
        )
    return (prediction_controls,)


@app.cell
def _(
    extract_plane,
    go,
    mo,
    overlay_mask,
    predicted_mask_opacity,
    prediction_axis,
    prediction_bundle,
    prediction_bundle_error,
    prediction_slice_slider,
    prediction_view_height_slider,
    show_predicted_mask,
    show_prediction_lung_mask,
    show_true_prediction_mask,
):
    if prediction_bundle_error is not None:
        prediction_slice_view = mo.md(f"Prediction slice viewer error: `{prediction_bundle_error}`")
    elif prediction_bundle is None or prediction_slice_slider is None:
        prediction_slice_view = mo.md("Prediction slice viewer will appear here once a saved run is available.")
    else:
        prediction_slice_index = int(prediction_slice_slider.value)
        prediction_ct_slice = extract_plane(
            prediction_bundle["ct_display"],
            prediction_axis,
            prediction_slice_index,
        )

        prediction_slice_figure = go.Figure()
        prediction_slice_figure.add_trace(
            go.Heatmap(
                z=prediction_ct_slice,
                colorscale="Gray",
                showscale=False,
                hovertemplate="CT %{z:.1f}<extra></extra>",
            )
        )

        if show_prediction_lung_mask.value and prediction_bundle["lung_mask"] is not None:
            prediction_slice_figure.add_trace(
                go.Heatmap(
                    z=overlay_mask(
                        prediction_bundle["lung_mask"],
                        prediction_axis,
                        prediction_slice_index,
                    ),
                    colorscale=[[0.0, "rgba(42,157,143,0.0)"], [1.0, "rgba(42,157,143,0.26)"]],
                    showscale=False,
                    hoverinfo="skip",
                )
            )

        if show_true_prediction_mask.value:
            prediction_slice_figure.add_trace(
                go.Heatmap(
                    z=overlay_mask(
                        prediction_bundle["true_mask"],
                        prediction_axis,
                        prediction_slice_index,
                    ),
                    colorscale=[[0.0, "rgba(39,125,161,0.0)"], [1.0, "rgba(39,125,161,0.55)"]],
                    showscale=False,
                    hoverinfo="skip",
                )
            )

        if show_predicted_mask.value:
            predicted_alpha = float(predicted_mask_opacity.value)
            prediction_slice_figure.add_trace(
                go.Heatmap(
                    z=overlay_mask(
                        prediction_bundle["prediction_mask"],
                        prediction_axis,
                        prediction_slice_index,
                    ),
                    colorscale=[
                        [0.0, "rgba(216,27,96,0.0)"],
                        [1.0, f"rgba(216,27,96,{predicted_alpha:.2f})"],
                    ],
                    showscale=False,
                    hoverinfo="skip",
                )
            )

        prediction_slice_figure.update_layout(
            title=(
                f"Saved prediction slice {prediction_slice_index} "
                f"| case {prediction_bundle['case_id']}"
            ),
            height=int(prediction_view_height_slider.value),
            margin=dict(l=0, r=0, t=45, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, autorange="reversed", scaleanchor="x", scaleratio=1),
        )
        prediction_slice_view = mo.vstack(
            [
                mo.md("### Prediction Slice Viewer"),
                mo.as_html(prediction_slice_figure),
            ],
            gap=0.5,
        )
    return (prediction_slice_view,)


@app.cell
def _(mo, prediction_3d_view, prediction_controls):
    prediction_top_panel = mo.hstack(
        [
            prediction_controls,
            prediction_3d_view,
        ],
        widths=[1.2, 1.8],
        gap=1.25,
        align="start",
        wrap=True,
    )
    return (prediction_top_panel,)


@app.cell
def _(mo, prediction_slice_view, prediction_top_panel):
    prediction_panel = mo.vstack(
        [
            prediction_top_panel,
            prediction_slice_view,
        ],
        gap=0.75,
    )
    return (prediction_panel,)


@app.cell
def _(mo, normalized_ct, prediction_panel, slice_panel, top_panel):
    note = mo.md(
        f"""
        ## Notes

        - `preprocess_case(...)` returns a normalized cropped CT for training with shape `{normalized_ct.shape}`.
        - The slice viewers display clipped CT values in the HU window used for preprocessing, not the normalized `[0, 1]` array.
        - The cropped views remain at the original scan spacing.
        """
    )
    viewer_panel = mo.vstack([top_panel, slice_panel, prediction_panel, note], gap=1.0)
    return (viewer_panel,)


if __name__ == "__main__":
    app.run()
