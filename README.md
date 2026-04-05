# Lung Airway Segmentation Notebook

This repo currently includes a marimo notebook for viewing AeroPath CT cases,
their masks, and a simple preprocessing pipeline.

The main notebook is:

- `basic_pipeline.py`

## What The Notebook Does

The marimo notebook provides:

- a case selector for AeroPath cases
- raw 3D lung and airway mask viewers
- preprocessed 3D lung and airway mask viewers
- a simple preprocessing function, `preprocess_case(...)`, that:
  - crops using the lung-mask bounding box
  - crops the airway mask
  - optionally keeps the cropped lung mask
  - resamples to a target spacing
  - normalizes the CT for training-style input

## Expected Data Layout

The notebook expects the AeroPath data under:

- `data/Aeropath/<case_id>/`

Each case folder should contain:

- `<case_id>_CT_HR.nii.gz`
- `<case_id>_CT_HR_label_lungs.nii.gz`
- `<case_id>_CT_HR_label_airways.nii.gz`

Example:

```text
data/
  Aeropath/
    1/
      1_CT_HR.nii.gz
      1_CT_HR_label_lungs.nii.gz
      1_CT_HR_label_airways.nii.gz
```

## Setup

Use Python `3.10+`.

From the repo root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the notebook dependencies:

```powershell
python -m pip install --upgrade pip
pip install marimo nibabel numpy plotly scipy scikit-image
```

## Open The Notebook

From the repo root:

```powershell
.\.venv\Scripts\python -m marimo edit basic_pipeline.py
```

marimo will print a local URL, usually something like:

```text
http://127.0.0.1:2718
```

Open that URL in your browser.

## Stop The Notebook

In the terminal where marimo is running:

```text
Ctrl+C
```

## Output Size Limit

This repo includes:

- `pyproject.toml`

with:

```toml
[tool.marimo.runtime]
output_max_bytes = 25_000_000
```

That raises marimo's output cap for the 3D viewers. If you change this file,
restart marimo for the new limit to take effect.

## Recommended Viewer Settings

The notebook includes a `3D quality` dropdown:

- `fast`
- `medium`
- `detailed`
- `maximum`

Recommended default:

- `medium`

Use `maximum` only when you need more geometric detail, since it can be much
slower and heavier in the browser.

## Using The Preprocessing Function

Inside the notebook, you can call:

```python
sample = preprocess_case("1", include_lung_mask=True)
```

The returned dictionary includes:

- `cropped_ct`
- `cropped_airway_mask`
- `cropped_lung_mask`
- `training_ct`
- `training_airway_mask`
- `training_lung_mask`
- `training_sample`

Example:

```python
sample["training_sample"]["image"].shape
```

## Troubleshooting

If the notebook opens but you still see old errors:

1. Stop marimo with `Ctrl+C`.
2. Start it again from the repo root.
3. Refresh the browser tab.

If the 3D viewer is too slow:

- switch the `3D quality` dropdown to `fast` or `medium`

If marimo says the output is too large:

- restart marimo from the repo root so it picks up `pyproject.toml`
- lower the viewer quality preset
