# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.3",
#     "nibabel",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scikit-image",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.23.9"
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
    # Probability colourscale for GT-confidence views (low=cool, high=warm).
    CONFIDENCE_COLORSCALE = "Turbo"
    return CONFIDENCE_COLORSCALE, RADIUS_BINS


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import nibabel as nib
    import numpy as np
    import plotly.graph_objects as go
    from scipy import ndimage
    from skimage.measure import marching_cubes
    from skimage.morphology import skeletonize

    from lung_airway_segmentation.settings import (
        DEFAULT_CROP_MARGIN,
        DEFAULT_HU_WINDOW,
        RAW_AEROPATH_ROOT,
        RAW_ATM22_ROOT,
    )
    from lung_airway_segmentation.schemas import PreprocessedCase
    from lung_airway_segmentation.preprocessing.geometry import crop_volume
    from lung_airway_segmentation.preprocessing.intensity import clip_ct_to_hu_window
    from lung_airway_segmentation.preprocessing.pipeline import preprocess_case
    from lung_airway_segmentation.io.case_layout import list_case_ids, resolve_case_paths
    from lung_airway_segmentation.io.atm22_layout import (
        list_case_ids as list_atm22_case_ids,
        resolve_case_paths as resolve_atm22_case_paths,
    )
    from lung_airway_segmentation.io.nifti import load_canonical_image, verify_alignment

    return (
        DEFAULT_CROP_MARGIN,
        DEFAULT_HU_WINDOW,
        Path,
        PreprocessedCase,
        RAW_AEROPATH_ROOT,
        RAW_ATM22_ROOT,
        clip_ct_to_hu_window,
        crop_volume,
        go,
        json,
        list_atm22_case_ids,
        list_case_ids,
        load_canonical_image,
        marching_cubes,
        mo,
        ndimage,
        nib,
        np,
        preprocess_case,
        resolve_atm22_case_paths,
        resolve_case_paths,
        skeletonize,
        verify_alignment,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Labelled Airway Dataset Viewer

    Select AeroPath or ATM'22 to inspect labelled CT volumes and airway masks.

    AeroPath uses the shared lung-cropped preprocessing path. ATM'22 has no lung
    masks in this layout, so its processed view retains the complete canonical
    CT volume. The prediction viewer below supports saved AeroPath and ATM'22 runs.
    """)
    return


@app.cell
def _(viewer_panel):
    viewer_panel
    return


@app.cell
def _(mo):
    dataset_selector = mo.ui.dropdown(
        {
            "AeroPath": "aeropath",
            "ATM'22": "atm22",
        },
        value="AeroPath",
        label="Dataset",
    )
    return (dataset_selector,)


@app.cell
def _(
    DEFAULT_CROP_MARGIN,
    RAW_AEROPATH_ROOT,
    RAW_ATM22_ROOT,
    dataset_selector,
    list_atm22_case_ids,
    list_case_ids,
    mo,
):
    if dataset_selector.value == "atm22":
        data_root = RAW_ATM22_ROOT
        case_ids = list_atm22_case_ids(data_root)
    else:
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
    show_lung_mask = mo.ui.switch(
        value=dataset_selector.value == "aeropath",
        label="Show lung mask",
    )
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
    DEFAULT_HU_WINDOW,
    PreprocessedCase,
    case_selector,
    crop_margin_slider,
    data_root,
    dataset_selector,
    load_canonical_image,
    np,
    preprocess_case,
    resolve_atm22_case_paths,
    resolve_case_paths,
    verify_alignment,
):
    case_id = case_selector.value
    crop_margin_value = int(crop_margin_slider.value)
    dataset_name = "ATM'22" if dataset_selector.value == "atm22" else "AeroPath"

    if dataset_selector.value == "atm22":
        paths = resolve_atm22_case_paths(case_id, batch_root=data_root)
        if paths["airway"] is None:
            raise ValueError(f"ATM'22 case {case_id} does not have an airway label.")

        ct_img = load_canonical_image(paths["ct"])
        airway_img = load_canonical_image(paths["airway"])
        verify_alignment(
            ct_img,
            airway_img,
            reference_name="ATM'22 CT",
            other_name="ATM'22 airway mask",
        )
        raw_ct = np.asarray(ct_img.dataobj, dtype=np.float32)
        raw_airway_mask = np.asarray(airway_img.dataobj) > 0
        raw_lung_mask = None
        original_spacing = tuple(float(value) for value in ct_img.header.get_zooms()[:3])
        full_crop_box = tuple((0, int(size)) for size in raw_ct.shape)
        processed_case = PreprocessedCase(
            case_id=str(case_id),
            ct=raw_ct,
            airway_mask=raw_airway_mask,
            lung_mask=None,
            spacing=original_spacing,
            affine=np.asarray(ct_img.affine, dtype=np.float64),
            crop_box=full_crop_box,
            metadata={
                "supervision": "labeled",
                "case_dir": paths["case_dir"],
                "ct_path": paths["ct"],
                "lung_mask_path": None,
                "airway_mask_path": paths["airway"],
                "original_shape": tuple(int(size) for size in raw_ct.shape),
                "processed_shape": tuple(int(size) for size in raw_ct.shape),
                "spacing": original_spacing,
                "original_affine": np.asarray(ct_img.affine, dtype=np.float64),
                "cropped_affine": np.asarray(ct_img.affine, dtype=np.float64),
                "crop_margin": (0, 0, 0),
                "hu_window": DEFAULT_HU_WINDOW,
                "crop_source": "full_volume",
            },
        )
    else:
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
        dataset_name,
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
def _(DEFAULT_HU_WINDOW, cropped_ct_hu, processed_case, raw_ct):
    hu_window = DEFAULT_HU_WINDOW
    raw_ct_display = raw_ct
    cropped_ct_display = cropped_ct_hu
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
        value=default_mask_index(
            raw_airway_mask
            if raw_lung_mask is None
            else raw_lung_mask | raw_airway_mask,
            axis,
        ),
        step=1,
        label="Slice",
    )
    return axis, shared_slice_slider


@app.cell
def _(marching_cubes, np):
    def build_mask_mesh(
        mask_volume,
        voxel_spacing,
        preferred_stride=1,
        max_sampled_foreground_voxels=400_000,
    ):
        """Build the highest-detail mesh that stays within a practical size budget."""
        mask = np.asarray(mask_volume) > 0
        stride = max(int(preferred_stride), 1)
        while (
            stride < 8
            and int(mask[::stride, ::stride, ::stride].sum()) > max_sampled_foreground_voxels
        ):
            stride += 1

        sampled_mask = mask[::stride, ::stride, ::stride]
        foreground_coordinates = np.argwhere(sampled_mask)
        if foreground_coordinates.size == 0:
            return None

        spacing = tuple(float(value) * stride for value in voxel_spacing)
        lower = np.maximum(foreground_coordinates.min(axis=0) - 1, 0)
        upper = np.minimum(
            foreground_coordinates.max(axis=0) + 2,
            sampled_mask.shape,
        )
        slices = tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))
        volume = sampled_mask[slices]
        mesh_vertices, mesh_faces, _, _ = marching_cubes(
            volume.astype(np.uint8),
            level=0.5,
            spacing=spacing,
        )
        mesh_vertices += lower * np.asarray(spacing, dtype=np.float32)
        return {
            "vertices": mesh_vertices,
            "faces": mesh_faces,
            "stride": stride,
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
def _(build_mask_mesh, processed_case, show_airway_mask, show_lung_mask):
    cropped_lung_mesh = (
        build_mask_mesh(processed_case.lung_mask, processed_case.spacing)
        if show_lung_mask.value and processed_case.lung_mask is not None
        else None
    )
    cropped_airway_mesh = (
        build_mask_mesh(processed_case.airway_mask, processed_case.spacing)
        if show_airway_mask.value
        else None
    )
    return cropped_airway_mesh, cropped_lung_mesh


@app.cell
def _(
    case_id,
    case_selector,
    crop_margin_slider,
    crop_margin_value,
    dataset_name,
    dataset_selector,
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
        f"Dataset: `{dataset_name}`",
        f"Original spacing: `{original_spacing[0]:.3f} x {original_spacing[1]:.3f} x {original_spacing[2]:.3f} mm`",
        (
            f"Crop margin: `{crop_margin_value}` voxels"
            if dataset_name == "AeroPath"
            else "Crop mode: `full volume` (ATM'22 has no lung mask)"
        ),
        f"Processed box: `{processed_case.crop_box}`",
        f"Processed shape: `{processed_case.metadata['processed_shape']}`",
        f"Display HU window: `{hu_window}`",
        "Mesh detail automatically uses the lowest safe stride for each mask.",
    ]

    controls = mo.vstack(
        [
            dataset_selector,
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
                mo.md("### Processed 3D Viewer"),
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
    hu_window,
    np,
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
    raw_slice = np.clip(
        extract_plane(raw_ct_display, axis, raw_shared_slice_index),
        hu_window[0],
        hu_window[1],
    )
    raw_slice_figure = go.Figure()
    raw_slice_figure.add_trace(
        go.Heatmap(
            z=raw_slice,
            colorscale="Gray",
            showscale=False,
            hovertemplate="CT %{z:.1f}<extra></extra>",
        )
    )
    if show_lung_mask.value and raw_lung_mask is not None:
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
    hu_window,
    np,
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
        cropped_slice = np.clip(
            extract_plane(cropped_ct_display, axis, cropped_slice_index),
            hu_window[0],
            hu_window[1],
        )
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
            f"Processed CT slice {cropped_slice_index} "
            f"(raw slice {cropped_source_slice_index})"
        )
    else:
        cropped_title = (
            f"Processed CT unavailable for raw slice {cropped_source_slice_index} "
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
    RAW_AEROPATH_ROOT,
    RAW_ATM22_ROOT,
    clip_ct_to_hu_window,
    json,
    load_canonical_image,
    nib,
    np,
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

        run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
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

    def preferred_prediction_run_name(run_names):
        preferred_suffix = (
            Path("aeropath-supervised")
            / "2026-04-17__23-21-26__baseline-p96-th05__baseline_unet"
        )
        preferred_parts = preferred_suffix.parts
        for run_name in run_names:
            run_parts = Path(run_name).parts
            if run_parts[-len(preferred_parts):] == preferred_parts:
                return run_name
        return run_names[0] if run_names else None

    def preferred_prediction_case_id(case_ids):
        return "20" if "20" in case_ids else case_ids[0] if case_ids else None

    def load_prediction_bundle(run_root, run_name, prediction_set_name, case_id, prediction_mask_filename):
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
            "ct": ct,
            "ct_display": ct_display,
            "hu_window": hu_window,
            "lung_mask": lung_mask,
            "true_mask": true_mask,
            "prediction_mask": prediction_mask,
            # Probability maps are large and unused by this viewer. Keep only
            # the path so opening ATM'22 predictions does not allocate ~600 MB.
            "probability_volume": None,
            "spacing": spacing,
            "best_val_dice": best_metrics.get("val_dice"),
            "best_epoch": best_metrics.get("epoch"),
            "last_epoch_metrics": history_entries[-1] if history_entries else None,
            "checkpoint_epoch": prediction_metadata.get("checkpoint_epoch"),
            "threshold": prediction_metadata.get("threshold"),
        }

    return (
        list_prediction_case_ids,
        list_prediction_mask_options,
        list_prediction_run_names,
        list_prediction_set_names,
        load_prediction_bundle,
        prediction_run_root,
        preferred_prediction_case_id,
        preferred_prediction_run_name,
    )


@app.cell
def _(mo):
    prediction_refresh_button = mo.ui.run_button(label="Refresh prediction runs")
    return (prediction_refresh_button,)


@app.cell
def _(
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
    show_gt_confidence = mo.ui.switch(value=False, label="GT confidence (prob×truth)")
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
        show_gt_confidence,
        show_predicted_mask,
        show_prediction_lung_mask,
        show_true_prediction_mask,
    )


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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
def _(
    build_mask_mesh,
    prediction_bundle,
    show_predicted_mask,
    show_prediction_lung_mask,
    show_true_prediction_mask,
):
    if prediction_bundle is None:
        prediction_lung_mesh = None
        prediction_true_mesh = None
        prediction_mask_mesh = None
    else:
        prediction_lung_mesh = (
            build_mask_mesh(
                prediction_bundle["lung_mask"],
                prediction_bundle["spacing"],
            )
            if show_prediction_lung_mask.value and prediction_bundle["lung_mask"] is not None
            else None
        )
        prediction_true_mesh = (
            build_mask_mesh(
                prediction_bundle["true_mask"],
                prediction_bundle["spacing"],
            )
            if show_true_prediction_mask.value
            else None
        )
        prediction_mask_mesh = (
            build_mask_mesh(
                prediction_bundle["prediction_mask"],
                prediction_bundle["spacing"],
            )
            if show_predicted_mask.value
            else None
        )
    return prediction_lung_mesh, prediction_mask_mesh, prediction_true_mesh


@app.cell
def _(
    RADIUS_BINS,
    ndimage,
    nib,
    np,
    prediction_bundle,
    show_gt_confidence,
    skeletonize,
):
    # GT-confidence diagnostic: the predicted probability restricted to the
    # ground-truth airway. Lazily loaded — the probability volume is large
    # (~600 MB on ATM'22), so we only touch it when the toggle is on. Computes
    # everything the views need: the full prob volume (slice overlay), the
    # GT-skeleton points coloured by prob (3D), the radius-stratified mean prob
    # (the "confidence falls off toward the periphery" curve), and a background
    # SHELL just outside the airway, binned by the radius of the nearest airway
    # voxel — the separability read (distal airway prob vs adjacent-tissue prob).
    #
    # NOTE: this masks the prediction to GT, discarding all false positives. It
    # is a recall/confidence DIAGNOSTIC, never a performance number — pair it
    # with the precision-side metrics (clDice / topology precision).
    SHELL_VOXELS = 2.5  # background-shell thickness around the airway, in voxels.
    if (
        prediction_bundle is None
        or not show_gt_confidence.value
    ):
        gt_confidence = None
    elif prediction_bundle["probability_path"] is None:
        gt_confidence = {
            "error": "No airway_prob_full.nii.gz saved for this case — re-run predict_atm.py."
        }
    else:
        _prob = np.asarray(
            nib.load(str(prediction_bundle["probability_path"])).dataobj,
            dtype=np.float32,
        )
        _gt = prediction_bundle["true_mask"]
        if _prob.shape != _gt.shape:
            gt_confidence = {
                "error": f"probability shape {_prob.shape} != GT shape {_gt.shape}"
            }
        elif not _gt.any():
            gt_confidence = {"error": "case has no ground-truth airway voxels"}
        else:
            _spacing = np.asarray(prediction_bundle["spacing"], dtype=np.float32)

            # Work inside a padded bounding box of the airway: the shell distance
            # transform with return_indices is memory-heavy on a full volume, and
            # everything we need is local to the tree. Pad > shell width.
            _where = np.where(_gt)
            _bbox = tuple(
                slice(
                    max(0, int(ax.min()) - 4),
                    min(int(size), int(ax.max()) + 5),
                )
                for ax, size in zip(_where, _gt.shape)
            )
            _offset = np.array([sl.start for sl in _bbox], dtype=np.float32)
            _gt_c = _gt[_bbox]
            _prob_c = _prob[_bbox]

            # Skeleton + calibre. distance-to-wall at a centreline voxel is the
            # local branch calibre (radius of the maximal inscribed sphere). We
            # bin BOTH airway and shell voxels by the calibre of their NEAREST
            # centreline voxel, so a whole branch is classified by its thickness
            # — a cleaner distal/generation proxy than per-voxel wall distance,
            # which lumps every branch's surface into r=1 and hides the distal
            # signal. (This is why these bins can differ from analyse_distal's
            # raw distance-to-wall RADIUS_BINS.)
            _skeleton_c = skeletonize(_gt_c)
            _radius = ndimage.distance_transform_edt(_gt_c)  # distance to wall
            # Calibre of the nearest centreline voxel, for every voxel in the box.
            _, _skel_idx = ndimage.distance_transform_edt(~_skeleton_c, return_indices=True)
            _calibre = _radius[tuple(_skel_idx)]

            # 3D: skeleton points (cropped index + offset → full-volume index → mm)
            # coloured by the predicted probability at each centreline voxel.
            _skeleton_points_mm = (
                np.argwhere(_skeleton_c).astype(np.float32) + _offset
            ) * _spacing
            _skeleton_prob = _prob_c[_skeleton_c]

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
                "prob": _prob,
                "gt": _gt,
                "skeleton_points_mm": _skeleton_points_mm,
                "skeleton_prob": _skeleton_prob,
                "radius_curve": _stratify(_prob_gt, _calibre_gt),
                "shell_curve": _stratify(_prob_shell, _calibre_shell),
                "shell_voxels": SHELL_VOXELS,
                "mean_prob": float(_prob_gt.mean()) if _prob_gt.size else 0.0,
                "shell_mean_prob": float(_prob_shell.mean()) if _prob_shell.size else 0.0,
            }
    return (gt_confidence,)


@app.cell
def _(
    CONFIDENCE_COLORSCALE,
    REPORT_CAMERA,
    go,
    gt_confidence,
    mo,
    predicted_mask_opacity,
    prediction_bundle,
    prediction_bundle_error,
    prediction_lung_mesh,
    prediction_mask_mesh,
    prediction_true_mesh,
    prediction_view_height_slider,
    show_gt_confidence,
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

        if (
            show_gt_confidence.value
            and gt_confidence is not None
            and "error" not in gt_confidence
        ):
            gt_confidence_points = gt_confidence["skeleton_points_mm"]
            prediction_mesh_figure.add_trace(
                go.Scatter3d(
                    x=gt_confidence_points[:, 0],
                    y=gt_confidence_points[:, 1],
                    z=gt_confidence_points[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=gt_confidence["skeleton_prob"],
                        colorscale=CONFIDENCE_COLORSCALE,
                        cmin=0.0,
                        cmax=1.0,
                        colorbar=dict(title="P(airway)", thickness=12, len=0.6, x=0.98),
                        showscale=True,
                    ),
                    name="GT centreline confidence",
                    hovertemplate="P(airway)=%{marker.color:.3f}<extra></extra>",
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
    prediction_mask_selector,
    prediction_plane_selector,
    prediction_refresh_button,
    prediction_run_selector,
    prediction_runs_available,
    prediction_set_selector,
    prediction_slice_slider,
    prediction_view_height_slider,
    show_gt_confidence,
    show_predicted_mask,
    show_prediction_lung_mask,
    show_true_prediction_mask,
):
    if not prediction_runs_available:
        prediction_controls = mo.vstack(
            [
                mo.md(
                    "## Saved Prediction Viewer\n\n"
                    "No run folders with saved predictions were found under `runs/`. "
                    "Run `scripts.predict_case` first."
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
                    widths=[1.6, 1.0, 0.8, 0.8],
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
            [
                prediction_run_selector,
                prediction_set_selector,
                prediction_case_selector,
                prediction_refresh_button,
            ],
            widths=[1.6, 1.0, 0.8, 0.8],
            gap=0.8,
            wrap=True,
            align="end",
        )
        prediction_secondary_controls = mo.hstack(
            [prediction_mask_selector, prediction_plane_selector, prediction_slice_slider],
            widths=[1.4, 1.0, 1.6],
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
            [
                show_prediction_lung_mask,
                show_true_prediction_mask,
                show_predicted_mask,
                show_gt_confidence,
            ],
            widths="equal",
            justify="start",
            gap=0.8,
            wrap=True,
            align="center",
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
    CONFIDENCE_COLORSCALE,
    extract_plane,
    go,
    gt_confidence,
    mo,
    np,
    overlay_mask,
    predicted_mask_opacity,
    prediction_axis,
    prediction_bundle,
    prediction_bundle_error,
    prediction_slice_slider,
    prediction_view_height_slider,
    show_gt_confidence,
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

        if (
            show_gt_confidence.value
            and gt_confidence is not None
            and "error" not in gt_confidence
        ):
            gt_confidence_prob_plane = extract_plane(
                gt_confidence["prob"], prediction_axis, prediction_slice_index
            )
            gt_confidence_truth_plane = extract_plane(
                gt_confidence["gt"], prediction_axis, prediction_slice_index
            )
            prediction_slice_figure.add_trace(
                go.Heatmap(
                    z=np.where(gt_confidence_truth_plane, gt_confidence_prob_plane, np.nan),
                    colorscale=CONFIDENCE_COLORSCALE,
                    zmin=0.0,
                    zmax=1.0,
                    colorbar=dict(title="P(airway)", thickness=12, len=0.85),
                    hovertemplate="P(airway)=%{z:.3f}<extra></extra>",
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
    # and on the adjacent background shell, stratified by branch calibre. The
    # operating threshold is drawn on so it is immediately visible which calibre
    # bins sit below it (thresholded away) and whether airway separates from shell.
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
        # Shell series aligned to the airway bins (None where a bin is absent).
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
def _(
    gt_confidence_curve_view,
    mo,
    prediction_slice_view,
    prediction_top_panel,
):
    prediction_panel = mo.vstack(
        [
            prediction_top_panel,
            prediction_slice_view,
            gt_confidence_curve_view,
        ],
        gap=0.75,
    )
    return (prediction_panel,)


@app.cell
def _(
    dataset_name,
    mo,
    normalized_ct,
    prediction_panel,
    slice_panel,
    top_panel,
):
    processed_note = (
        f"`preprocess_case(...)` returns a normalized cropped CT for training with shape `{normalized_ct.shape}`."
        if dataset_name == "AeroPath"
        else f"ATM'22 is shown as a canonical full-volume CT with shape `{normalized_ct.shape}` because no lung mask is available."
    )
    note = mo.md(
        f"""
        ## Notes

        - {processed_note}
        - The slice viewers display clipped CT values in the HU window used for preprocessing, not the normalized `[0, 1]` array.
        - The processed views remain at the original scan spacing.
        """
    )
    viewer_panel = mo.vstack([top_panel, slice_panel, prediction_panel, note], gap=1.0)
    return (viewer_panel,)


if __name__ == "__main__":
    app.run()
