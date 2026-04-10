"""Dataset wrappers for AeroPath cases.

This module exposes the dataset-facing interface for labelled AeroPath data. It
uses the current AeroPath preprocessing pipeline to turn case IDs into
consistent sample dictionaries containing image arrays, masks, and spatial
metadata. It is the bridge between case-level preprocessing and downstream
training code.
"""
import numpy as np

from torch.utils.data import Dataset
from lung_airway_segmentation.preprocessing.pipeline import preprocess_case
from lung_airway_segmentation.io.case_layout import list_case_ids
from lung_airway_segmentation.settings import(
    DEFAULT_HU_WINDOW,
    RAW_AEROPATH_ROOT,
    DEFAULT_CROP_MARGIN
    
)

class AeroPathDataset(Dataset):
    """Dataset that loads labelled AeroPath cases on demand."""

    def __init__(
        self,
        case_ids = None,
        *,
        data_root=RAW_AEROPATH_ROOT,
        include_lung_mask=False,
        hu_window=DEFAULT_HU_WINDOW,
        crop_margin=DEFAULT_CROP_MARGIN,
        transform=None     
    ):
        if case_ids is None:
            self.case_ids = list_case_ids(data_root)
        else:
            self.case_ids = [str(case_id) for case_id in case_ids]
        
        self.data_root = data_root
        self.include_lung_mask = include_lung_mask
        self.hu_window = hu_window
        self.crop_margin = crop_margin
        self.transform = transform

    def __len__(self):
        """Return the number of cases available through the dataset."""
        return len(self.case_ids)

    def __getitem__(self, index):
        """Load one preprocessed case and package it as a sample dictionary."""
        case_id = self.case_ids[index]
    

        # This dataset wrapper is specific to AeroPath because it delegates to
        # the current AeroPath preprocessing pipeline and case layout rules.
        case = preprocess_case(
            case_id,
            data_root=self.data_root,
            include_lung_mask=self.include_lung_mask,
            hu_window=self.hu_window,
            crop_margin=self.crop_margin
        )

        image = pad_to_divisible(case.ct, divisor=32)
        airway_mask = pad_to_divisible(case.airway_mask, divisor=32)

        sample = {
            "case_id": case.case_id,
            "image": image,
            "airway_mask": airway_mask,
            #"lung_mask": case.lung_mask,
            #"spacing": case.spacing,
            #"affine": case.affine,
            #"crop_box": case.crop_box,
            #"metadata": case.metadata
        }

        if case.lung_mask is not None:
            sample["lung_mask"] = pad_to_divisible(case.lung_mask, divisor=16)


        if self.transform is not None:
            sample = self.transform(sample)

        #print("padded image shape:", image.shape)

        return sample

def pad_to_divisible(volume, divisor=32):
    z, y, x = volume.shape
    pad_z = (divisor - z % divisor) % divisor
    pad_y = (divisor - y % divisor) % divisor
    pad_x = (divisor - x % divisor) % divisor

    return np.pad(
        volume,
        ((0, pad_z), (0, pad_y), (0, pad_x)),
        mode="constant",
    )