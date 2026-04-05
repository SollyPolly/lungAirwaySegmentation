# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "nibabel",
#     "numpy",
#     "plotly",
#     "scipy",
#     "scikit-image",
# ]
# ///

import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import nibabel as nib
    import numpy as np
    import plotly.graph_objects as go
    from scipy.ndimage import zoom
    from skimage.measure import marching_cubes

    return Path, go, marching_cubes, mo, nib, np, zoom


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # AeroPath 3D Mask Viewer

    Select a case and toggle the lung and airway masks. The CT scan is
    loaded in the background for geometry and metadata, while the masks are
    shown as interactive 3D surfaces in the notebook.
    """)
    return


@app.cell
def _(viewer_panel):
    viewer_panel
    return


@app.cell
def _(Path, mo):
    project_root = Path(__file__).resolve().parent
    data_root = project_root / "data" / "Aeropath"

    mo.stop(
        not data_root.exists(),
        mo.md(f"Missing data directory: `{data_root}`"),
    )

    case_ids = sorted(
        [
            path.name
            for path in data_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ],
        key=int,
    )

    mo.stop(
        not case_ids,
        mo.md(f"No AeroPath case folders were found in `{data_root}`"),
    )
    return case_ids, data_root


@app.cell
def _(case_ids, mo):
    case_selector = mo.ui.dropdown(
        case_ids,
        value=case_ids[0],
        label="Case",
        searchable=True,
    )
    return (case_selector,)


@app.cell
def _(case_selector, data_root):
    case_id = case_selector.value
    case_dir = data_root / case_id

    ct_path = case_dir / f"{case_id}_CT_HR.nii.gz"
    lung_path = case_dir / f"{case_id}_CT_HR_label_lungs.nii.gz"
    airway_path = case_dir / f"{case_id}_CT_HR_label_airways.nii.gz"

    has_lung_mask = lung_path.exists()
    has_airway_mask = airway_path.exists()
    return (
        airway_path,
        case_id,
        ct_path,
        has_airway_mask,
        has_lung_mask,
        lung_path,
    )


@app.cell
def _(
    airway_path,
    ct_path,
    has_airway_mask,
    has_lung_mask,
    lung_path,
    mo,
    nib,
):
    mo.stop(
        not ct_path.exists(),
        mo.md(f"Missing CT volume: `{ct_path}`"),
    )

    ct_img = nib.as_closest_canonical(nib.load(str(ct_path)))
    lung_img = (
        nib.as_closest_canonical(nib.load(str(lung_path)))
        if has_lung_mask
        else None
    )
    airway_img = (
        nib.as_closest_canonical(nib.load(str(airway_path)))
        if has_airway_mask
        else None
    )

    ct_shape = ct_img.shape
    lung_mask_shape = lung_img.shape if lung_img is not None else None
    airway_mask_shape = airway_img.shape if airway_img is not None else None

    voxel_spacing = tuple(float(value) for value in ct_img.header.get_zooms()[:3])
    return (
        airway_img,
        airway_mask_shape,
        ct_shape,
        lung_img,
        lung_mask_shape,
        voxel_spacing,
    )


@app.cell
def _(has_airway_mask, has_lung_mask, mo):
    show_lung_mask = mo.ui.switch(
        value=has_lung_mask,
        label="Show lung mask",
        disabled=not has_lung_mask,
    )
    show_airway_mask = mo.ui.switch(
        value=has_airway_mask,
        label="Show airway mask",
        disabled=not has_airway_mask,
    )
    return show_airway_mask, show_lung_mask


@app.cell
def _(mo):
    mesh_quality = mo.ui.dropdown(
        {
            "Fast": "fast",
            "Medium (Recommended)": "medium",
            "Detailed": "detailed",
            "Maximum": "maximum",
        },
        value="Medium (Recommended)",
        label="3D quality",
    )
    return (mesh_quality,)


@app.cell
def _():
    mesh_quality_presets = {
        "fast": {
            "raw_lung_stride": 7,
            "raw_airway_stride": 5,
            "pre_lung_stride": 5,
            "pre_airway_stride": 4,
            "estimated_payload_mb": 4.0,
        },
        "medium": {
            "raw_lung_stride": 6,
            "raw_airway_stride": 4,
            "pre_lung_stride": 5,
            "pre_airway_stride": 3,
            "estimated_payload_mb": 5.2,
        },
        "detailed": {
            "raw_lung_stride": 5,
            "raw_airway_stride": 3,
            "pre_lung_stride": 4,
            "pre_airway_stride": 2,
            "estimated_payload_mb": 8.7,
        },
        "maximum": {
            "raw_lung_stride": 5,
            "raw_airway_stride": 3,
            "pre_lung_stride": 3,
            "pre_airway_stride": 2,
            "estimated_payload_mb": 10.9,
        },
    }
    return (mesh_quality_presets,)


@app.cell
def _(mesh_quality, mesh_quality_presets):
    mesh_config = mesh_quality_presets[mesh_quality.value]
    return (mesh_config,)


@app.cell
def _(marching_cubes, np):
    def build_mask_mesh(mask_img, stride):
        if mask_img is None:
            return None

        volume = np.asarray(mask_img.dataobj[::stride, ::stride, ::stride]) > 0
        if not volume.any():
            return None

        spacing = tuple(
            float(value) * stride for value in mask_img.header.get_zooms()[:3]
        )
        vertices, faces, _, _ = marching_cubes(
            volume.astype(np.uint8),
            level=0.5,
            spacing=spacing,
        )

        return {
            "vertices": vertices,
            "faces": faces,
            "spacing": spacing,
            "stride": stride,
        }

    def build_mask_mesh_from_array(mask_volume, voxel_spacing, stride):
        if mask_volume is None:
            return None

        volume = np.asarray(mask_volume[::stride, ::stride, ::stride]) > 0
        if not volume.any():
            return None

        spacing = tuple(float(value) * stride for value in voxel_spacing)
        vertices, faces, _, _ = marching_cubes(
            volume.astype(np.uint8),
            level=0.5,
            spacing=spacing,
        )

        return {
            "vertices": vertices,
            "faces": faces,
            "spacing": spacing,
            "stride": stride,
        }

    return build_mask_mesh, build_mask_mesh_from_array


@app.cell
def _(airway_img, build_mask_mesh, lung_img, mesh_config):
    lung_mesh = build_mask_mesh(lung_img, stride=mesh_config["raw_lung_stride"])
    airway_mesh = build_mask_mesh(airway_img, stride=mesh_config["raw_airway_stride"])
    return airway_mesh, lung_mesh


@app.cell
def _(
    airway_mask_shape,
    case_id,
    case_selector,
    ct_shape,
    has_airway_mask,
    has_lung_mask,
    lung_mask_shape,
    mesh_config,
    mesh_quality,
    mesh_quality_presets,
    mo,
    show_airway_mask,
    show_lung_mask,
    voxel_spacing,
):
    notes = [
        f"CT shape: `{ct_shape}`",
        f"Lung mask shape `{lung_mask_shape}`",
        f"Airway mask shape `{airway_mask_shape}`",
        (
            f"Voxel spacing: `{voxel_spacing[0]:.3f} x "
            f"{voxel_spacing[1]:.3f} x {voxel_spacing[2]:.3f} mm`"
        ),
        f"3D quality: `{mesh_quality.value}`",
        (
            "Estimated size for this preset: "
            f"`~{mesh_config['estimated_payload_mb']:.1f} MB`"
        ),
        (
            "Raw strides: "
            f"lung `{mesh_config['raw_lung_stride']}`, "
            f"airway `{mesh_config['raw_airway_stride']}`"
        ),
        (
            "Preprocessed strides: "
            f"lung `{mesh_config['pre_lung_stride']}`, "
            f"airway `{mesh_config['pre_airway_stride']}`"
        ),
        (
            "Available presets: "
            + ", ".join(f"`{name}`" for name in mesh_quality_presets)
        ),
    ]
    if not has_lung_mask:
        notes.append("No lung mask was found for this case.")
    if not has_airway_mask:
        notes.append("No airway mask was found for this case.")

    controls = mo.vstack(
        [
            case_selector,
            mesh_quality,
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
def _(airway_mesh, go, lung_mesh, mo, show_airway_mask, show_lung_mask):
    _figure = go.Figure()

    if show_lung_mask.value and lung_mesh is not None:
        _raw_lung_vertices = lung_mesh["vertices"]
        _raw_lung_faces = lung_mesh["faces"]
        _figure.add_trace(
            go.Mesh3d(
                x=_raw_lung_vertices[:, 0],
                y=_raw_lung_vertices[:, 1],
                z=_raw_lung_vertices[:, 2],
                i=_raw_lung_faces[:, 0],
                j=_raw_lung_faces[:, 1],
                k=_raw_lung_faces[:, 2],
                color="#59b5ff",
                opacity=0.14,
                name="Lung mask",
                hovertemplate="Lung mask<extra></extra>",
                flatshading=True,
            )
        )

    if show_airway_mask.value and airway_mesh is not None:
        _raw_airway_vertices = airway_mesh["vertices"]
        _raw_airway_faces = airway_mesh["faces"]
        _figure.add_trace(
            go.Mesh3d(
                x=_raw_airway_vertices[:, 0],
                y=_raw_airway_vertices[:, 1],
                z=_raw_airway_vertices[:, 2],
                i=_raw_airway_faces[:, 0],
                j=_raw_airway_faces[:, 1],
                k=_raw_airway_faces[:, 2],
                color="#ff7a00",
                opacity=0.85,
                name="Airway mask",
                hovertemplate="Airway mask<extra></extra>",
                flatshading=True,
            )
        )

    if not _figure.data:
        raw_view = mo.md("Enable at least one mask to display the raw 3D view.")
    else:
        _figure.update_layout(
            height=520,
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
            ),
        )

        raw_view = mo.vstack(
            [
                mo.md("### Raw 3D Viewer"),
                mo.as_html(_figure),
            ],
            gap=0.5,
        )
    return (raw_view,)


@app.cell
def _(controls, mo, preprocessed_3d_view, raw_view):
    comparison_panel = mo.hstack(
        [raw_view, preprocessed_3d_view],
        widths="equal",
        gap=1.0,
        align="start",
        wrap=True,
    )
    viewer_panel = mo.hstack(
        [controls, comparison_panel],
        widths=[0.9, 2.8],
        gap=1.25,
        align="start",
        wrap=True,
    )
    return (viewer_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Preprocessing

    The cells below define a reusable preprocessing function for one AeroPath
    case. It crops with the lung-mask bounding box, rescales to a target voxel
    spacing, normalizes the CT intensities, and returns arrays that are ready
    to feed into a training pipeline.
    """)
    return


@app.cell
def _():
    default_target_spacing = (1.0, 1.0, 1.0)
    default_intensity_window = (-1024.0, 600.0)
    default_bbox_margin = 5
    return (
        default_bbox_margin,
        default_intensity_window,
        default_target_spacing,
    )


@app.cell
def _(np, zoom):
    def bbox_from_mask(mask, margin=0):
        foreground = np.argwhere(mask > 0)
        if foreground.size == 0:
            raise ValueError("The lung mask is empty, so a crop box cannot be computed.")

        starts = foreground.min(axis=0)
        stops = foreground.max(axis=0) + 1

        starts = np.maximum(starts - margin, 0)
        stops = np.minimum(stops + margin, mask.shape)

        return tuple(
            slice(int(start), int(stop)) for start, stop in zip(starts, stops)
        )

    def crop_to_bbox(volume, bbox):
        return volume[bbox]

    def normalize_ct(volume, intensity_window):
        lower, upper = intensity_window
        clipped = np.clip(volume, lower, upper)
        normalized = (clipped - lower) / (upper - lower)
        return normalized.astype(np.float32)

    def resample_volume(volume, original_spacing, target_spacing, order):
        zoom_factors = tuple(
            float(source) / float(target)
            for source, target in zip(original_spacing, target_spacing)
        )
        return zoom(
            volume,
            zoom=zoom_factors,
            order=order,
            mode="nearest",
            prefilter=order > 0,
        )

    return bbox_from_mask, crop_to_bbox, normalize_ct, resample_volume


@app.cell
def _(
    bbox_from_mask,
    crop_to_bbox,
    data_root,
    default_bbox_margin,
    default_intensity_window,
    default_target_spacing,
    nib,
    normalize_ct,
    np,
    resample_volume,
):
    def preprocess_case(
        case_id,
        *,
        include_lung_mask=False,
        target_spacing=default_target_spacing,
        intensity_window=default_intensity_window,
        bbox_margin=default_bbox_margin,
    ):
        case_id = str(case_id)
        case_dir = data_root / case_id

        ct_path = case_dir / f"{case_id}_CT_HR.nii.gz"
        lung_path = case_dir / f"{case_id}_CT_HR_label_lungs.nii.gz"
        airway_path = case_dir / f"{case_id}_CT_HR_label_airways.nii.gz"

        if not ct_path.exists():
            raise FileNotFoundError(f"CT scan not found: {ct_path}")
        if not lung_path.exists():
            raise FileNotFoundError(
                f"Lung mask not found, but it is required for cropping: {lung_path}"
            )
        if not airway_path.exists():
            raise FileNotFoundError(f"Airway mask not found: {airway_path}")

        ct_img = nib.as_closest_canonical(nib.load(str(ct_path)))
        lung_img = nib.as_closest_canonical(nib.load(str(lung_path)))
        airway_img = nib.as_closest_canonical(nib.load(str(airway_path)))

        original_spacing = tuple(
            float(value) for value in ct_img.header.get_zooms()[:3]
        )

        ct = np.asarray(ct_img.dataobj, dtype=np.float32)
        lung_mask = np.asarray(lung_img.dataobj) > 0
        airway_mask = np.asarray(airway_img.dataobj) > 0

        bbox = bbox_from_mask(lung_mask, margin=bbox_margin)
        bbox_bounds = tuple((sl.start, sl.stop) for sl in bbox)

        cropped_ct = crop_to_bbox(ct, bbox).astype(np.float32, copy=False)
        cropped_airway_mask = crop_to_bbox(airway_mask, bbox).astype(np.uint8)
        cropped_lung_mask = crop_to_bbox(lung_mask, bbox).astype(np.uint8)

        resampled_ct = resample_volume(
            cropped_ct,
            original_spacing=original_spacing,
            target_spacing=target_spacing,
            order=1,
        ).astype(np.float32)
        normalized_ct = normalize_ct(resampled_ct, intensity_window)

        resampled_airway_mask = resample_volume(
            cropped_airway_mask,
            original_spacing=original_spacing,
            target_spacing=target_spacing,
            order=0,
        ).astype(np.uint8)

        resampled_lung_mask = None
        if include_lung_mask:
            resampled_lung_mask = resample_volume(
                cropped_lung_mask,
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                order=0,
            ).astype(np.uint8)

        training_sample = {
            "image": normalized_ct[None, ...],
            "airway_mask": resampled_airway_mask[None, ...],
        }
        if include_lung_mask:
            training_sample["lung_mask"] = resampled_lung_mask[None, ...]

        return {
            "case_id": case_id,
            "ct_path": ct_path,
            "airway_mask_path": airway_path,
            "lung_mask_path": lung_path,
            "bbox_slices": bbox,
            "bbox_bounds": bbox_bounds,
            "original_spacing": original_spacing,
            "target_spacing": tuple(float(value) for value in target_spacing),
            "intensity_window": tuple(float(value) for value in intensity_window),
            "cropped_ct": cropped_ct,
            "cropped_airway_mask": cropped_airway_mask,
            "cropped_lung_mask": cropped_lung_mask if include_lung_mask else None,
            "training_ct": normalized_ct,
            "training_airway_mask": resampled_airway_mask,
            "training_lung_mask": resampled_lung_mask,
            "training_sample": training_sample,
        }

    return (preprocess_case,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Call the preprocessing function manually when you need it, for example:

    ```python
    sample = preprocess_case("1", include_lung_mask=True)
    sample["training_sample"]["image"].shape
    ```
    """)
    return


@app.cell
def _(case_id, preprocess_case):
    preprocessed_case = preprocess_case(case_id, include_lung_mask=True)
    return (preprocessed_case,)


@app.cell
def _(build_mask_mesh_from_array, mesh_config, preprocessed_case):
    preprocessed_lung_mesh = build_mask_mesh_from_array(
        preprocessed_case["training_lung_mask"],
        preprocessed_case["target_spacing"],
        stride=mesh_config["pre_lung_stride"],
    )
    preprocessed_airway_mesh = build_mask_mesh_from_array(
        preprocessed_case["training_airway_mask"],
        preprocessed_case["target_spacing"],
        stride=mesh_config["pre_airway_stride"],
    )
    return preprocessed_airway_mesh, preprocessed_lung_mesh


@app.cell
def _(
    go,
    mo,
    preprocessed_airway_mesh,
    preprocessed_case,
    preprocessed_lung_mesh,
    show_airway_mask,
    show_lung_mask,
):
    _pre_figure = go.Figure()

    if show_lung_mask.value and preprocessed_lung_mesh is not None:
        _pre_lung_vertices = preprocessed_lung_mesh["vertices"]
        _pre_lung_faces = preprocessed_lung_mesh["faces"]
        _pre_figure.add_trace(
            go.Mesh3d(
                x=_pre_lung_vertices[:, 0],
                y=_pre_lung_vertices[:, 1],
                z=_pre_lung_vertices[:, 2],
                i=_pre_lung_faces[:, 0],
                j=_pre_lung_faces[:, 1],
                k=_pre_lung_faces[:, 2],
                color="#59b5ff",
                opacity=0.14,
                name="Preprocessed lung mask",
                hovertemplate="Preprocessed lung mask<extra></extra>",
                flatshading=True,
            )
        )

    if show_airway_mask.value and preprocessed_airway_mesh is not None:
        _pre_airway_vertices = preprocessed_airway_mesh["vertices"]
        _pre_airway_faces = preprocessed_airway_mesh["faces"]
        _pre_figure.add_trace(
            go.Mesh3d(
                x=_pre_airway_vertices[:, 0],
                y=_pre_airway_vertices[:, 1],
                z=_pre_airway_vertices[:, 2],
                i=_pre_airway_faces[:, 0],
                j=_pre_airway_faces[:, 1],
                k=_pre_airway_faces[:, 2],
                color="#ff7a00",
                opacity=0.85,
                name="Preprocessed airway mask",
                hovertemplate="Preprocessed airway mask<extra></extra>",
                flatshading=True,
            )
        )

    if not _pre_figure.data:
        preprocessed_3d_view = mo.md(
            "Enable at least one mask to display the preprocessed 3D view."
        )
    else:
        _pre_figure.update_layout(
            height=520,
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
            ),
        )

        preprocessed_3d_view = mo.vstack(
            [
                mo.md("### Preprocessed 3D Viewer"),
                mo.as_html(_pre_figure),
                mo.md(
                    "\n".join(
                        [
                            f"- Cropped CT: `{preprocessed_case['cropped_ct'].shape}`",
                            f"- Training CT: `{preprocessed_case['training_ct'].shape}`",
                            f"- Bounding box: `{preprocessed_case['bbox_bounds']}`",
                            (
                                "This 3D view shows the cropped and resampled masks. "
                                "The normalized CT itself is not surface-rendered."
                            ),
                        ]
                    )
                ),
            ],
            gap=0.5,
        )
    return (preprocessed_3d_view,)


if __name__ == "__main__":
    app.run()
