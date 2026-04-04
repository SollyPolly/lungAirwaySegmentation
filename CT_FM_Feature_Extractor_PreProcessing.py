from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from lighter_zoo import SegResEncoder
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms.compose import Compose
from monai.transforms.croppad.array import CropForeground
from monai.transforms.intensity.array import ScaleIntensityRange
from monai.transforms.io.array import LoadImage
from monai.transforms.spatial.array import Orientation
from monai.transforms.utility.array import EnsureType


DEFAULT_MODEL_DIR = Path("CT_FM_Feature_Extractor")
DEFAULT_INPUT_PATH = Path("data") / "AeroPath" / "1" / "1_CT_HR.nii.gz"
DEFAULT_OUTPUT_DIR = Path("features")

PREPROCESS = Compose(
    [
        LoadImage(ensure_channel_first=True),
        EnsureType(data_type="tensor", dtype=torch.float32),
        Orientation(axcodes="SPL", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        ScaleIntensityRange(
            a_min=-1024,
            a_max=2048,
            b_min=0,
            b_max=1,
            clip=True,
        ),
        CropForeground(allow_smaller=True),
    ]
)


class LastFeatureMap(torch.nn.Module):
    def __init__(self, model: SegResEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x)[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CT-FM features from a CT volume and save them for reuse.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input CT volume path, for example a .nii or .nii.gz file.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to the local Hugging Face model folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where .pt and .npy feature files will be written.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        help=(
            "Optional sliding-window ROI size. "
            "Use this when full-volume inference does not fit in GPU memory."
        ),
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
        help="Sliding-window batch size.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Sliding-window overlap fraction.",
    )
    return parser.parse_args()


def feature_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: Path, device: torch.device) -> SegResEncoder:
    model = SegResEncoder.from_pretrained(str(model_dir))
    model = model.to(device)
    model.eval()
    return model


def preprocess_input(input_path: Path) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    input_tensor = cast(torch.Tensor, PREPROCESS(str(input_path)))
    t1 = time.perf_counter()
    return input_tensor, (t1 - t0)


def run_full_volume(
    model: SegResEncoder,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batched = input_tensor.unsqueeze(0).to(device)
    return cast(torch.Tensor, model(batched)[-1])


def run_sliding_window(
    model: SegResEncoder,
    input_tensor: torch.Tensor,
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
) -> torch.Tensor:
    predictor = LastFeatureMap(model)
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="constant",
        sw_device=device,
        device=device,
        progress=True,
    )
    batched = input_tensor.unsqueeze(0).to(device)
    return cast(torch.Tensor, inferer(batched, predictor))


def extract_features(
    model: SegResEncoder,
    input_tensor: torch.Tensor,
    device: torch.device,
    roi_size: tuple[int, int, int] | None,
    sw_batch_size: int,
    overlap: float,
) -> tuple[torch.Tensor, tuple[int, ...], float]:
    t0 = time.perf_counter()
    with torch.no_grad():
        if roi_size is None:
            feature_maps = run_full_volume(model, input_tensor, device)
        else:
            feature_maps = run_sliding_window(
                model=model,
                input_tensor=input_tensor,
                device=device,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )
        features = torch.nn.functional.adaptive_avg_pool3d(feature_maps, 1).flatten()
    t1 = time.perf_counter()
    return features.cpu(), tuple(feature_maps.shape), (t1 - t0)


def save_features(
    features: torch.Tensor,
    input_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    base = feature_name(input_path)
    pt_path = output_dir / f"{base}_features.pt"
    npy_path = output_dir / f"{base}_features.npy"

    torch.save(features, pt_path)
    np.save(npy_path, features.numpy())

    return pt_path, npy_path


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input scan not found: {args.input}")

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Local model folder not found: {args.model_dir}")

    device = resolve_device(args.device)
    roi_size = tuple(args.roi_size) if args.roi_size is not None else None

    print(f"Python device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Input scan: {args.input}")
    print(f"Model folder: {args.model_dir}")
    if roi_size is None:
        print("Inference mode: full volume")
    else:
        print(f"Inference mode: sliding window, roi_size={roi_size}, overlap={args.overlap}")

    t0 = time.perf_counter()
    model = load_model(args.model_dir, device)
    t1 = time.perf_counter()

    input_tensor, preprocess_time = preprocess_input(args.input)
    print(f"Preprocessed tensor shape: {tuple(input_tensor.shape)}")

    try:
        features, feature_map_shape, inference_time = extract_features(
            model=model,
            input_tensor=input_tensor,
            device=device,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
        )
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA ran out of memory. Retry with a larger-memory GPU or pass "
            "--roi-size D H W for sliding-window inference, for example "
            "--roi-size 96 96 96."
        ) from exc

    pt_path, npy_path = save_features(features, args.input, args.output_dir)
    t2 = time.perf_counter()

    print("Feature extraction completed")
    print(f"Feature map shape: {feature_map_shape}")
    print(f"Feature vector shape: {tuple(features.shape)}")
    print(f"Model load time: {t1 - t0:.2f}s")
    print(f"Preprocess time: {preprocess_time:.2f}s")
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Total time: {t2 - t0:.2f}s")
    print(f"Saved PyTorch features to: {pt_path}")
    print(f"Saved NumPy features to: {npy_path}")


if __name__ == "__main__":
    main()
