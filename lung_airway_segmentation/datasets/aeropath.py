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

from lung_airway_segmentation.datasets.patches import (
    normalize_patch_size,
    extract_patch,
    sample_foreground_patch_start,
    sample_random_patch_start,
)

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
    

class AeroPathPatchDataset(Dataset):
    def __init__(
            self,
            case_ids=None,
            *,
            data_root=RAW_AEROPATH_ROOT,
            include_lung_mask=False,
            hu_window=DEFAULT_HU_WINDOW,
            crop_margin=DEFAULT_CROP_MARGIN,
            patch_size=(96,96,96),
            patches_per_case=4,
            foreground_probability=0.7,
            seed=15,
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
        self.patch_size = normalize_patch_size(patch_size)
        self.patches_per_case = patches_per_case
        self.foreground_probability = foreground_probability
        self.seed = seed
        self.transform = transform

    def __len__(self):
        """Return the number of patch samples exposed per epoch."""
        return len(self.case_ids) * self.patches_per_case

    def __getitem__(self, index):
        """Load one preprocessed case and package it as a sample dictionary."""
        case_index = index % len(self.case_ids)
        case_id = self.case_ids[case_index]
    
    
        # This dataset wrapper is specific to AeroPath because it delegates to
        # the current AeroPath preprocessing pipeline and case layout rules.
        case = preprocess_case(
            case_id,
            data_root=self.data_root,
            include_lung_mask=self.include_lung_mask,
            hu_window=self.hu_window,
            crop_margin=self.crop_margin
        )

        rng = np.random.default_rng(self.seed + index)

        use_foreground_patch = rng.random() < self.foreground_probability
        
        if use_foreground_patch:
            start = sample_foreground_patch_start(
                case.airway_mask,
                self.patch_size,
                rng
            )
        elif case.lung_mask is not None:
            start = sample_foreground_patch_start(
                case.lung_mask,
                self.patch_size,
                rng
            )
        else:
            volume_shape = (
                int(case.ct.shape[0]),
                int(case.ct.shape[1]),
                int(case.ct.shape[2]),                               
            )
            start = sample_random_patch_start(
                volume_shape,
                self.patch_size,
                rng
            )

        image_patch = extract_patch(case.ct, start, self.patch_size)
        airway_patch = extract_patch(case.airway_mask, start, self.patch_size)

        sample = {
            "case_id": case.case_id,
            "image": image_patch,
            "airway_mask": airway_patch,
            #"lung_mask": case.lung_mask,
            #"spacing": case.spacing,
            #"affine": case.affine,
            #"crop_box": case.crop_box,
            #"metadata": case.metadata
        }

        if case.lung_mask is not None:
            sample["lung_mask"] = extract_patch(case.lung_mask, start, self.patch_size)

        if self.transform is not None:
            sample["image"] = sample["image"][None, ...]
            sample["airway_mask"] = sample["airway_mask"][None, ...]
            if "lung_mask" in sample:
                sample["lung_mask"] = sample["lung_mask"][None, ...]
            sample = self.transform(sample)

        return sample        

def pad_to_divisible(volume, divisor=32):
    """Only used when model looks at an entire ct volume"""
    z, y, x = volume.shape
    pad_z = (divisor - z % divisor) % divisor
    pad_y = (divisor - y % divisor) % divisor
    pad_x = (divisor - x % divisor) % divisor

    return np.pad(
        volume,
        ((0, pad_z), (0, pad_y), (0, pad_x)),
        mode="constant",
    )