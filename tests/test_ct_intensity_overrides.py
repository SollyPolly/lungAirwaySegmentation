from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from lung_airway_segmentation.io.nnunet_lungcrop import write_lung_roi_ct
from scripts.build_lungcrop_expanded_meanteacher_nnunet import _preflight_added_inputs
from scripts.precompute_lung_masks import prepare_ct_for_lungmask, process_case


SCALE = 0.0625
OFFSET = -1024.0


def _save(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_lung_roi_intensity_decode_is_opt_in_and_keeps_outside_zero(tmp_path: Path):
    shape = (5, 5, 5)
    stored = np.full(shape, 16_400, dtype=np.uint16)  # 1 HU after decoding
    stored[0, 0, 0] = np.iinfo(np.uint16).max
    lung = np.zeros(shape, dtype=np.uint8)
    lung[1:4, 1:4, 1:4] = 1
    ct_path = tmp_path / "ct.nii.gz"
    lung_path = tmp_path / "lung.nii.gz"
    unchanged_path = tmp_path / "unchanged.nii.gz"
    decoded_path = tmp_path / "decoded.nii.gz"
    _save(ct_path, stored)
    _save(lung_path, lung)

    unchanged_record = write_lung_roi_ct(
        ct_path,
        lung_path,
        unchanged_path,
        margin_voxels=0,
        superior_margin_voxels=0,
    )
    decoded_record = write_lung_roi_ct(
        ct_path,
        lung_path,
        decoded_path,
        margin_voxels=0,
        superior_margin_voxels=0,
        intensity_scale=SCALE,
        intensity_offset=OFFSET,
    )

    unchanged_image = nib.load(str(unchanged_path))
    decoded_image = nib.load(str(decoded_path))
    unchanged = np.asarray(unchanged_image.dataobj)
    decoded = np.asarray(decoded_image.dataobj)
    assert unchanged_image.get_data_dtype() == np.dtype(np.uint16)
    assert decoded_image.get_data_dtype() == np.dtype(np.float32)
    assert unchanged[2, 2, 2] == 16_400
    assert decoded[2, 2, 2] == pytest.approx(1.0)
    assert unchanged[0, 0, 0] == 0
    assert decoded[0, 0, 0] == pytest.approx(0.0)
    assert unchanged_record["intensity_transform"] is None
    assert decoded_record["intensity_transform"] == {
        "scale": SCALE,
        "offset": OFFSET,
        "source_dtype": "uint16",
        "output_dtype": "float32",
        "source_range": [16_400.0, 65_535.0],
        "decoded_range": [1.0, 3_071.9375],
    }


def test_lungmask_decode_uses_original_geometry_and_strict_manifest():
    stored = np.full((3, 4, 5), 16_400, dtype=np.uint16)
    stored[0, 0, 0] = np.iinfo(np.uint16).max
    image = sitk.GetImageFromArray(stored)
    image.SetSpacing((0.7, 0.8, 1.5))
    image.SetOrigin((3.0, -2.0, 9.0))
    transform = {
        "group": "uint16_scaled_hu",
        "scale": SCALE,
        "offset": OFFSET,
    }

    decoded, record = prepare_ct_for_lungmask(
        image,
        intensity_transform=transform,
        strict_manifest=True,
    )
    decoded_array = sitk.GetArrayFromImage(decoded)
    assert decoded.GetPixelID() == sitk.sitkFloat32
    assert decoded.GetSpacing() == image.GetSpacing()
    assert decoded.GetOrigin() == image.GetOrigin()
    assert decoded_array[1, 1, 1] == pytest.approx(1.0)
    assert decoded_array[0, 0, 0] == pytest.approx(3_071.9375)
    assert record["source_range"] == [16_400.0, 65_535.0]
    assert record["decoded_range"] == [1.0, 3_071.9375]

    with pytest.raises(ValueError, match="undeclared uint16"):
        prepare_ct_for_lungmask(image, strict_manifest=True)

    signed_image = sitk.GetImageFromArray(stored.astype(np.int16))
    with pytest.raises(ValueError, match="not uint16"):
        prepare_ct_for_lungmask(
            signed_image,
            intensity_transform=transform,
            strict_manifest=True,
        )


def test_declared_override_regenerates_an_existing_mask(tmp_path: Path):
    batch_root = tmp_path / "ATM22"
    ct_path = batch_root / "imagesTr" / "ATM_242_0000.nii.gz"
    mask_path = batch_root / "lungTr" / "ATM_242_lung.nii.gz"
    ct_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    stored = np.full((3, 4, 5), 16_400, dtype=np.uint16)
    stored[0, 0, 0] = np.iinfo(np.uint16).max
    ct_image = sitk.GetImageFromArray(stored)
    sitk.WriteImage(ct_image, str(ct_path))
    stale_mask = sitk.GetImageFromArray(np.zeros_like(stored, dtype=np.uint8))
    stale_mask.CopyInformation(ct_image)
    sitk.WriteImage(stale_mask, str(mask_path))

    class FullLungInferer:
        decoded_pixel_id = None

        def apply(self, image):
            self.decoded_pixel_id = image.GetPixelID()
            return np.ones(sitk.GetArrayFromImage(image).shape, dtype=np.uint8)

    inferer = FullLungInferer()
    transform = {
        "group": "uint16_scaled_hu",
        "scale": SCALE,
        "offset": OFFSET,
    }
    _, status, _ = process_case(
        "242",
        batch_root,
        None,
        inferer,
        overwrite=False,
        intensity_transform=transform,
        strict_intensity_manifest=True,
    )
    assert status == "written"
    assert inferer.decoded_pixel_id == sitk.sitkFloat32
    assert np.all(sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) == 1)


def test_builder_preflight_rejects_empty_or_misaligned_mask(tmp_path: Path):
    batch_root = tmp_path / "ATM22"
    ct_path = batch_root / "imagesTr" / "ATM_242_0000.nii.gz"
    mask_path = batch_root / "lungTr" / "ATM_242_lung.nii.gz"
    ct_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    _save(ct_path, np.full((4, 4, 4), 16_400, dtype=np.uint16))
    transform = {
        "242": {
            "group": "uint16_scaled_hu",
            "scale": SCALE,
            "offset": OFFSET,
        }
    }

    _save(mask_path, np.zeros((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="empty"):
        _preflight_added_inputs(
            batch_root,
            {"242"},
            None,
            transform,
            margin_voxels=0,
            superior_margin_voxels=0,
        )

    _save(mask_path, np.ones((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Grid mismatch"):
        _preflight_added_inputs(
            batch_root,
            {"242"},
            None,
            transform,
            margin_voxels=0,
            superior_margin_voxels=0,
        )
