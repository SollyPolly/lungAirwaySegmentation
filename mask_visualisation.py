# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.3",
#     "nibabel>=5.0",
#     "numpy>=2.0",
#     "plotly>=6.0",
#     "scikit-image>=0.25",
# ]
# ///

"""Interactive airway prediction viewer.

Run with:

    marimo run mask_visualisation.py

The browser view uses compact Plotly meshes and reads only the selected CT
slice.  Full NIfTI volumes are never sent to the browser.  The app discovers
both exported ``runs/`` predictions and native ``data/nnunet/predict_out``
outputs, and can open the selected CT/GT/prediction directly in ITK-SNAP.
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from lung_airway_segmentation.visualization.prediction_viewer import (
        build_mask_mesh,
        combine_cropped_masks,
        default_slice_index,
        discover_prediction_sources,
        extract_ct_plane,
        extract_mask_plane,
        find_itksnap_executable,
        launch_itksnap,
        list_prediction_cases,
        load_prediction_bundle,
    )

    PROJECT_ROOT = Path(__file__).resolve().parent
    REPORT_CAMERA = {
        "eye": {"x": 1.45, "y": 1.55, "z": 0.95},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    }
    return (
        PROJECT_ROOT,
        REPORT_CAMERA,
        Path,
        build_mask_mesh,
        combine_cropped_masks,
        default_slice_index,
        discover_prediction_sources,
        extract_ct_plane,
        extract_mask_plane,
        find_itksnap_executable,
        go,
        launch_itksnap,
        list_prediction_cases,
        load_prediction_bundle,
        mo,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    _introduction = (
        "# Airway Prediction Viewer\n\n"
        "A local, dependency-light viewer for saved airway segmentations. It "
        "discovers both **viewer exports under `runs/`** and **native nnU-Net "
        "outputs under `data/nnunet/predict_out/`**. The in-browser view is "
        "Plotly-based; use the ITK-SNAP button for full-resolution clinical "
        "navigation with CT, ground truth, and prediction loaded together."
    )
    mo.md(_introduction)
    return


@app.cell(hide_code=True)
def _(mo):
    refresh_sources = mo.ui.run_button(
        label="Refresh predictions",
        tooltip="Rescan runs/ and native nnU-Net output folders.",
    )
    return (refresh_sources,)


@app.cell(hide_code=True)
def _(PROJECT_ROOT, discover_prediction_sources, mo, refresh_sources):
    _ = refresh_sources.value
    prediction_sources = discover_prediction_sources(PROJECT_ROOT)
    prediction_sources_by_key = {source.key: source for source in prediction_sources}
    _source_options = {source.label: source.key for source in prediction_sources}
    source_selector = (
        mo.ui.dropdown(
            _source_options,
            value=next(iter(_source_options)),
            label="Prediction collection",
            searchable=True,
            full_width=True,
        )
        if _source_options
        else None
    )
    return prediction_sources, prediction_sources_by_key, source_selector


@app.cell(hide_code=True)
def _(list_prediction_cases, mo, prediction_sources_by_key, source_selector):
    selected_source = (
        prediction_sources_by_key.get(source_selector.value)
        if source_selector is not None
        else None
    )
    prediction_cases = (
        list_prediction_cases(selected_source) if selected_source is not None else []
    )
    prediction_cases_by_id = {case.case_id: case for case in prediction_cases}
    _case_options = {
        f"{case.case_id} · {len(case.masks)} mask{'s' if len(case.masks) != 1 else ''}": case.case_id
        for case in prediction_cases
    }
    case_selector = (
        mo.ui.dropdown(
            _case_options,
            value=next(iter(_case_options)),
            label="Case",
            searchable=True,
        )
        if _case_options
        else None
    )
    return case_selector, prediction_cases, prediction_cases_by_id, selected_source


@app.cell(hide_code=True)
def _(case_selector, mo, prediction_cases_by_id):
    selected_case = (
        prediction_cases_by_id.get(case_selector.value)
        if case_selector is not None
        else None
    )
    _mask_options = {}
    if selected_case is not None:
        for _mask in selected_case.masks:
            _label = _mask.label
            if _label in _mask_options:
                _label = f"{_label} · {_mask.path.name}"
            _mask_options[_label] = str(_mask.path)
    mask_selector = (
        mo.ui.dropdown(
            _mask_options,
            value=next(iter(_mask_options)),
            label="Mask variant",
            searchable=True,
        )
        if _mask_options
        else None
    )
    return mask_selector, selected_case


@app.cell(hide_code=True)
def _(Path, mask_selector):
    selected_prediction_path = (
        Path(mask_selector.value) if mask_selector is not None else None
    )
    return (selected_prediction_path,)


@app.cell(hide_code=True)
def _(mo):
    comparison_mode = mo.ui.dropdown(
        {
            "Prediction + ground truth": "overlay",
            "Agreement / errors": "errors",
            "Prediction only": "prediction",
            "Ground truth only": "truth",
        },
        value="Prediction + ground truth",
        label="3D layers",
    )
    mesh_detail = mo.ui.dropdown(
        {
            "Full detail": 1,
            "Faster (stride 2)": 2,
            "Preview (stride 3)": 3,
        },
        value="Full detail",
        label="Mesh detail",
    )
    prediction_opacity = mo.ui.slider(
        0.1,
        1.0,
        value=0.70,
        step=0.05,
        label="Prediction opacity",
        show_value=True,
    )
    truth_opacity = mo.ui.slider(
        0.1,
        1.0,
        value=0.42,
        step=0.05,
        label="GT opacity",
        show_value=True,
    )
    mesh_height = mo.ui.slider(
        420,
        900,
        value=650,
        step=20,
        label="3D height",
        show_value=True,
    )
    slice_height = mo.ui.slider(
        360,
        800,
        value=620,
        step=20,
        label="Slice height",
        show_value=True,
    )
    plane_selector = mo.ui.dropdown(
        {"Axial": 2, "Coronal": 1, "Sagittal": 0},
        value="Axial",
        label="Slice plane",
    )
    hu_window_selector = mo.ui.dropdown(
        {
            "Lung (-1000 to 400 HU)": (-1000.0, 400.0),
            "Airway (-1024 to 600 HU)": (-1024.0, 600.0),
            "Mediastinal (-160 to 240 HU)": (-160.0, 240.0),
        },
        value="Lung (-1000 to 400 HU)",
        label="CT window",
    )
    return (
        comparison_mode,
        hu_window_selector,
        mesh_detail,
        mesh_height,
        plane_selector,
        prediction_opacity,
        slice_height,
        truth_opacity,
    )


@app.cell(hide_code=True)
def _(
    PROJECT_ROOT,
    load_prediction_bundle,
    selected_case,
    selected_prediction_path,
    selected_source,
):
    if (
        selected_source is None
        or selected_case is None
        or selected_prediction_path is None
    ):
        prediction_bundle = None
        bundle_error = None
    else:
        try:
            prediction_bundle = load_prediction_bundle(
                selected_source,
                selected_case.case_id,
                selected_prediction_path,
                PROJECT_ROOT,
            )
            bundle_error = None
        except Exception as _error:
            prediction_bundle = None
            bundle_error = str(_error)
    return bundle_error, prediction_bundle


@app.cell(hide_code=True)
def _(build_mask_mesh, combine_cropped_masks, comparison_mode, mesh_detail, prediction_bundle):
    scene_meshes = []
    if prediction_bundle is not None:
        _stride = int(mesh_detail.value)
        _prediction = prediction_bundle.prediction
        _truth = prediction_bundle.ground_truth
        _mode = comparison_mode.value

        if _mode == "errors" and _truth is not None:
            _true_positive = combine_cropped_masks(_prediction, _truth, "and")
            _false_positive = combine_cropped_masks(
                _prediction, _truth, "first_only"
            )
            _missed = combine_cropped_masks(_truth, _prediction, "first_only")
            scene_meshes = [
                (
                    "True positive",
                    build_mask_mesh(
                        _true_positive,
                        prediction_bundle.affine,
                        preferred_stride=_stride,
                    ),
                    "#35d07f",
                    "error",
                ),
                (
                    "False positive",
                    build_mask_mesh(
                        _false_positive,
                        prediction_bundle.affine,
                        preferred_stride=_stride,
                    ),
                    "#ff4d5e",
                    "error",
                ),
                (
                    "Missed ground truth",
                    build_mask_mesh(
                        _missed,
                        prediction_bundle.affine,
                        preferred_stride=_stride,
                    ),
                    "#ffb020",
                    "error",
                ),
            ]
        else:
            if _mode in {"overlay", "truth"} and _truth is not None:
                scene_meshes.append(
                    (
                        "Ground truth",
                        build_mask_mesh(
                            _truth,
                            prediction_bundle.affine,
                            preferred_stride=_stride,
                        ),
                        "#19c3d8",
                        "truth",
                    )
                )
            if _mode in {"overlay", "prediction"} or _truth is None:
                scene_meshes.append(
                    (
                        "Prediction",
                        build_mask_mesh(
                            _prediction,
                            prediction_bundle.affine,
                            preferred_stride=_stride,
                        ),
                        "#ff3b9d",
                        "prediction",
                    )
                )
    return (scene_meshes,)


@app.cell(hide_code=True)
def _(
    REPORT_CAMERA,
    bundle_error,
    comparison_mode,
    go,
    mesh_height,
    mo,
    prediction_bundle,
    prediction_opacity,
    scene_meshes,
    truth_opacity,
):
    if bundle_error is not None:
        mesh_view = mo.callout(
            mo.md(f"**Could not load this prediction**\n\n`{bundle_error}`"),
            kind="danger",
        )
    elif prediction_bundle is None:
        mesh_view = mo.callout(
            "Select a prediction collection and case to build the 3D view.",
            kind="info",
        )
    else:
        _figure = go.Figure()
        _visible_meshes = 0
        for _name, _mesh, _color, _opacity_kind in scene_meshes:
            if _mesh is None:
                continue
            _visible_meshes += 1
            if _opacity_kind == "prediction":
                _opacity = float(prediction_opacity.value)
            elif _opacity_kind == "truth":
                _opacity = float(truth_opacity.value)
            else:
                _opacity = 0.94
            _vertices = _mesh.vertices
            _faces = _mesh.faces
            _figure.add_trace(
                go.Mesh3d(
                    x=_vertices[:, 0],
                    y=_vertices[:, 1],
                    z=_vertices[:, 2],
                    i=_faces[:, 0],
                    j=_faces[:, 1],
                    k=_faces[:, 2],
                    name=_name,
                    color=_color,
                    opacity=_opacity,
                    flatshading=True,
                    hovertemplate=f"{_name}<extra></extra>",
                    lighting={
                        "ambient": 0.45,
                        "diffuse": 0.72,
                        "specular": 0.18,
                        "roughness": 0.75,
                    },
                )
            )
        if _visible_meshes == 0:
            mesh_view = mo.callout(
                "The selected layer is empty, so no surface could be generated.",
                kind="warn",
            )
        else:
            _figure.update_layout(
                template="plotly_dark",
                height=int(mesh_height.value),
                margin={"l": 0, "r": 0, "t": 44, "b": 0},
                title={
                    "text": (
                        f"Case {prediction_bundle.case_id} · "
                        f"{comparison_mode.value.replace('_', ' ').title()}"
                    ),
                    "x": 0.02,
                },
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.01,
                    "xanchor": "right",
                    "x": 1.0,
                },
                scene={
                    "aspectmode": "data",
                    "dragmode": "turntable",
                    "camera": REPORT_CAMERA,
                    "xaxis": {"title": "world x (mm)", "showbackground": False},
                    "yaxis": {"title": "world y (mm)", "showbackground": False},
                    "zaxis": {"title": "world z (mm)", "showbackground": False},
                },
                uirevision=(
                    f"{prediction_bundle.source.key}:{prediction_bundle.case_id}:"
                    f"{comparison_mode.value}"
                ),
            )
            mesh_view = mo.as_html(_figure)
    return (mesh_view,)


@app.cell(hide_code=True)
def _(default_slice_index, mo, plane_selector, prediction_bundle):
    slice_axis = int(plane_selector.value)
    if prediction_bundle is None:
        slice_slider = mo.ui.slider(
            0,
            1,
            value=0,
            disabled=True,
            label="Slice",
            show_value=True,
        )
    else:
        _default_index = default_slice_index(
            [prediction_bundle.prediction, prediction_bundle.ground_truth],
            slice_axis,
        )
        slice_slider = mo.ui.slider(
            0,
            prediction_bundle.shape[slice_axis] - 1,
            value=_default_index,
            step=1,
            debounce=True,
            label="Slice index",
            show_value=True,
            full_width=True,
        )
    return slice_axis, slice_slider


@app.cell(hide_code=True)
def _(
    comparison_mode,
    extract_ct_plane,
    extract_mask_plane,
    go,
    hu_window_selector,
    mo,
    np,
    plane_selector,
    prediction_bundle,
    slice_axis,
    slice_height,
    slice_slider,
):
    if prediction_bundle is None:
        slice_view = mo.callout(
            "A CT slice will appear after a prediction is loaded.", kind="info"
        )
    else:
        try:
            _index = int(slice_slider.value)
            _ct = extract_ct_plane(
                prediction_bundle.ct_image, slice_axis, _index
            )
            _prediction = extract_mask_plane(
                prediction_bundle.prediction, slice_axis, _index
            )
            _truth = extract_mask_plane(
                prediction_bundle.ground_truth, slice_axis, _index
            )
            _low, _high = (float(value) for value in hu_window_selector.value)
            _figure = go.Figure()
            _figure.add_trace(
                go.Heatmap(
                    z=_ct,
                    colorscale="Gray",
                    zmin=_low,
                    zmax=_high,
                    showscale=True,
                    colorbar={"title": "HU", "thickness": 14},
                    hovertemplate="HU %{z:.0f}<extra></extra>",
                    name="CT",
                )
            )

            _mode = comparison_mode.value
            _overlay = np.full(_prediction.shape, np.nan, dtype=np.float32)
            if _mode == "errors" and _truth is not None:
                _overlay[_prediction & _truth] = 1
                _overlay[_prediction & ~_truth] = 2
                _overlay[_truth & ~_prediction] = 3
                _colors = ("#35d07f", "#ff4d5e", "#ffb020")
                _labels = ("True positive", "False positive", "Missed GT")
            elif _mode == "truth" and _truth is not None:
                _overlay[_truth] = 1
                _colors = ("#19c3d8",)
                _labels = ("Ground truth",)
            elif _mode == "prediction" or _truth is None:
                _overlay[_prediction] = 1
                _colors = ("#ff3b9d",)
                _labels = ("Prediction",)
            else:
                _overlay[_truth & ~_prediction] = 1
                _overlay[_prediction & ~_truth] = 2
                _overlay[_prediction & _truth] = 3
                _colors = ("#19c3d8", "#ff3b9d", "#d7ff6b")
                _labels = ("GT only", "Prediction only", "Overlap")

            _count = len(_colors)
            if _count == 1:
                _colorscale = [[0.0, _colors[0]], [1.0, _colors[0]]]
            else:
                _colorscale = []
                for _color_index, _color in enumerate(_colors):
                    _start = _color_index / _count
                    _stop = (_color_index + 1) / _count
                    _colorscale.extend(
                        [[_start, _color], [max(_start, _stop - 1e-6), _color]]
                    )
            _figure.add_trace(
                go.Heatmap(
                    z=_overlay,
                    zmin=1,
                    zmax=max(1, _count),
                    colorscale=_colorscale,
                    showscale=False,
                    opacity=0.72,
                    hoverinfo="skip",
                    name="Segmentation",
                )
            )
            for _label, _color in zip(_labels, _colors):
                _figure.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker={"size": 11, "symbol": "square", "color": _color},
                        name=_label,
                        hoverinfo="skip",
                    )
                )
            _figure.update_layout(
                template="plotly_dark",
                height=int(slice_height.value),
                margin={"l": 0, "r": 30, "t": 52, "b": 0},
                title={
                    "text": (
                        f"{plane_selector.selected_key} slice {_index} · "
                        f"window {_low:.0f} to {_high:.0f} HU"
                    ),
                    "x": 0.02,
                },
                xaxis={"visible": False, "scaleanchor": "y"},
                yaxis={"visible": False, "autorange": "reversed"},
                legend={"orientation": "h", "y": 1.04, "x": 0.48},
                uirevision=(
                    f"slice:{prediction_bundle.source.key}:"
                    f"{prediction_bundle.case_id}:{slice_axis}"
                ),
            )
            slice_view = mo.as_html(_figure)
        except Exception as _error:
            slice_view = mo.callout(
                mo.md(f"**Could not read this CT slice**\n\n`{_error}`"),
                kind="danger",
            )
    return (slice_view,)


@app.cell(hide_code=True)
def _(bundle_error, mo, prediction_bundle):
    if bundle_error is not None:
        summary_panel = mo.callout(bundle_error, kind="danger")
    elif prediction_bundle is None:
        summary_panel = mo.md("")
    else:
        _metrics = prediction_bundle.metrics

        def _score(name):
            _value = _metrics.get(name)
            return "—" if _value is None else f"{float(_value):.3f}"

        _stats = mo.hstack(
            [
                mo.stat(_score("dice"), label="Dice", bordered=True),
                mo.stat(_score("iou"), label="IoU", bordered=True),
                mo.stat(_score("precision"), label="Precision", bordered=True),
                mo.stat(_score("recall"), label="Recall", bordered=True),
                mo.stat(
                    f"{int(_metrics['predicted_voxels']):,}",
                    label="Predicted voxels",
                    bordered=True,
                ),
            ],
            widths="equal",
            gap=0.7,
            wrap=True,
        )
        _metadata = prediction_bundle.metadata
        _run_lines = [
            f"- **Source**: {_metadata.get('source_type', 'Unknown')}",
            f"- **Dataset**: `{_metadata.get('dataset_name', 'unknown')}`",
            f"- **Case**: `{prediction_bundle.case_id}`",
            (
                f"- **Spacing**: `{' × '.join(f'{value:.3f}' for value in prediction_bundle.spacing)} mm`"
            ),
            f"- **Shape**: `{prediction_bundle.shape}`",
        ]
        for _key, _label in (
            ("study_name", "Study"),
            ("run_label", "Run label"),
            ("experiment_name", "Experiment"),
            ("threshold", "Threshold"),
            ("checkpoint_epoch", "Checkpoint epoch"),
            ("best_val_dice", "Recorded best validation Dice"),
        ):
            if _metadata.get(_key) is not None:
                _run_lines.append(f"- **{_label}**: `{_metadata[_key]}`")

        _warning_view = (
            mo.callout(
                mo.md("\n".join(f"- {warning}" for warning in prediction_bundle.warnings)),
                kind="warn",
            )
            if prediction_bundle.warnings
            else mo.md("")
        )
        _file_lines = [
            f"- CT: `{prediction_bundle.ct_path}`",
            f"- Ground truth: `{prediction_bundle.ground_truth_path or 'not found'}`",
            f"- Prediction: `{prediction_bundle.prediction_path}`",
        ]
        summary_panel = mo.vstack(
            [
                mo.md("### Selected case"),
                _stats,
                mo.hstack(
                    [mo.md("\n".join(_run_lines)), _warning_view],
                    widths=[1.5, 1.0],
                    gap=1.2,
                    wrap=True,
                    align="start",
                ),
                mo.accordion({"Resolved files": mo.md("\n".join(_file_lines))}),
            ],
            gap=0.8,
        )
    return (summary_panel,)


@app.cell(hide_code=True)
def _(find_itksnap_executable, mo, prediction_bundle):
    itksnap_executable = find_itksnap_executable()
    open_itksnap = mo.ui.run_button(
        label="Open this case in ITK-SNAP",
        kind="success",
        disabled=prediction_bundle is None or itksnap_executable is None,
        tooltip=(
            "Open CT, ground truth, and prediction as separate layers."
            if itksnap_executable is not None
            else "ITK-SNAP was not found. Set ITKSNAP_PATH to its executable."
        ),
    )
    return itksnap_executable, open_itksnap


@app.cell(hide_code=True)
def _(
    itksnap_executable,
    launch_itksnap,
    mo,
    open_itksnap,
    prediction_bundle,
):
    if itksnap_executable is None:
        itksnap_status = mo.callout(
            mo.md(
                "ITK-SNAP was not found. Install it or set `ITKSNAP_PATH` to the "
                "ITK-SNAP executable, then restart the app."
            ),
            kind="warn",
        )
    elif prediction_bundle is None:
        itksnap_status = mo.callout(
            f"ITK-SNAP is ready at `{itksnap_executable}`. Select a usable case.",
            kind="neutral",
        )
    elif open_itksnap.value:
        try:
            _process = launch_itksnap(
                itksnap_executable,
                prediction_bundle.ct_path,
                prediction_bundle.prediction_path,
                prediction_bundle.ground_truth_path,
            )
            itksnap_status = mo.callout(
                (
                    f"Opened case `{prediction_bundle.case_id}` in ITK-SNAP "
                    f"(process `{_process.pid}`). GT and prediction are separate "
                    "segmentation layers."
                ),
                kind="success",
            )
        except Exception as _error:
            itksnap_status = mo.callout(
                mo.md(f"**ITK-SNAP did not open**\n\n`{_error}`"), kind="danger"
            )
    else:
        itksnap_status = mo.callout(
            (
                f"Ready: `{itksnap_executable}`. The button opens the selected CT "
                "with ground truth first and prediction second in the segmentation list."
            ),
            kind="info",
        )
    return (itksnap_status,)


@app.cell(hide_code=True)
def _(
    case_selector,
    comparison_mode,
    hu_window_selector,
    itksnap_status,
    mask_selector,
    mesh_detail,
    mesh_height,
    mesh_view,
    mo,
    open_itksnap,
    plane_selector,
    prediction_sources,
    prediction_opacity,
    refresh_sources,
    slice_height,
    slice_slider,
    slice_view,
    source_selector,
    summary_panel,
    truth_opacity,
):
    if source_selector is None:
        main_panel = mo.vstack(
            [
                mo.callout(
                    mo.md(
                        "No prediction masks were found. Expected either "
                        "`runs/**/predictions*/<case>/*.nii.gz` or "
                        "`data/nnunet/predict_out/<collection>/*.nii.gz`."
                    ),
                    kind="warn",
                ),
                refresh_sources,
            ],
            gap=1.0,
        )
    else:
        _primary_controls = [
            source_selector,
            case_selector,
            mask_selector,
            refresh_sources,
        ]
        _primary_controls = [item for item in _primary_controls if item is not None]
        _viewer_controls = [
            comparison_mode,
            mesh_detail,
            prediction_opacity,
            truth_opacity,
            mesh_height,
        ]
        _slice_controls = [
            plane_selector,
            slice_slider,
            hu_window_selector,
            slice_height,
        ]
        _tabs = mo.ui.tabs(
            {
                "3D surface": mo.vstack(
                    [
                        mo.hstack(
                            _viewer_controls,
                            gap=0.8,
                            wrap=True,
                            align="end",
                        ),
                        mesh_view,
                    ],
                    gap=0.7,
                ),
                "CT slice": mo.vstack(
                    [
                        mo.hstack(
                            _slice_controls,
                            gap=0.8,
                            wrap=True,
                            align="end",
                        ),
                        slice_view,
                    ],
                    gap=0.7,
                ),
            },
            value="3D surface",
        )
        main_panel = mo.vstack(
            [
                mo.hstack(
                    _primary_controls,
                    widths=[2.5, 0.7, 1.2, 0.7],
                    gap=0.8,
                    wrap=True,
                    align="end",
                ),
                mo.md(
                    f"Found **{len(prediction_sources)}** prediction collections. "
                    "Click legend entries in the 3D view to hide individual surfaces."
                ),
                _tabs,
                summary_panel,
                mo.md("### Full-resolution ITK-SNAP"),
                mo.hstack(
                    [open_itksnap, itksnap_status],
                    widths=[0.8, 2.2],
                    gap=1.0,
                    align="center",
                    wrap=True,
                ),
            ],
            gap=1.0,
        )
    main_panel
    return


if __name__ == "__main__":
    app.run()
