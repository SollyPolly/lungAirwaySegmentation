from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from monai.inferers.utils import sliding_window_inference
from monai.losses.dice import DiceLoss
from monai.networks.nets.unet import UNet
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import RandCropByPosNegLabeld
from monai.transforms.io.array import LoadImage
from monai.transforms.utility.dictionary import EnsureTyped
from monai.utils.misc import set_determinism


DEFAULT_PROCESSED_ROOT = Path(__file__).resolve().parent / "data" / "processed"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs" / "basic_unet"
RoiSize3D = tuple[int, int, int]
TensorDict = dict[str, Tensor]


@dataclass(frozen=True)
class ProcessedCasePaths:
    case_id: str
    image: Path
    label: Path


@dataclass(frozen=True)
class LoadedCase:
    case_id: str
    image_path: Path
    label_path: Path
    affine: np.ndarray
    image: Tensor
    label: Tensor


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.dice = DiceLoss(sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        dice_term = self.dice(logits, target)
        bce_term = self.bce(logits, target)
        return self.dice_weight * dice_term + self.bce_weight * bce_term


class SingleCasePatchDataset(Dataset[TensorDict]):
    def __init__(
        self,
        image: Tensor,
        label: Tensor,
        roi_size: RoiSize3D,
        samples_per_epoch: int,
        pos: float = 1.0,
        neg: float = 1.0,
    ) -> None:
        self.image = image
        self.label = label
        self.samples_per_epoch = int(samples_per_epoch)
        self.transform = Compose(
            [
                EnsureTyped(
                    keys=("image", "label"),
                    dtype=torch.float32,
                    track_meta=False,
                ),
                RandCropByPosNegLabeld(
                    keys=("image", "label"),
                    label_key="label",
                    spatial_size=roi_size,
                    pos=pos,
                    neg=neg,
                    num_samples=1,
                    allow_smaller=True,
                ),
            ]
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> TensorDict:
        del index
        sample: TensorDict = {"image": self.image, "label": self.label}
        transformed = self.transform(sample)

        if isinstance(transformed, list):
            if len(transformed) != 1 or not isinstance(transformed[0], dict):
                raise TypeError("Expected a single transformed sample from MONAI crop transform.")
            transformed_dict: Mapping[str, object] = transformed[0]
        elif isinstance(transformed, dict):
            transformed_dict = transformed
        else:
            raise TypeError(f"Unexpected transform output type: {type(transformed)!r}")

        image = transformed_dict.get("image")
        label = transformed_dict.get("label")
        if not isinstance(image, torch.Tensor) or not isinstance(label, torch.Tensor):
            raise TypeError("MONAI crop transform must return torch.Tensor values for image and label.")

        return {"image": image, "label": label}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a minimal MONAI 3D U-Net on processed airway cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-case", default="1", help="Processed case id to train on.")
    parser.add_argument("--val-case", default="2", help="Processed case id to validate on.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root directory containing saved processed cases.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where checkpoints and metrics will be written.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=32,
        help="Number of random 3D training patches sampled from the train case per epoch.",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(96, 96, 96),
        help="Training and sliding-window inference patch size.",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=2,
        help="Sliding-window batch size used during full-volume inference.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save train and validation prediction masks from the best checkpoint.",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Also save sigmoid probability volumes when using --save-predictions.",
    )
    parser.add_argument(
        "--train-val-same-case",
        action="store_true",
        help="Override --val-case and evaluate on the training case as a pure overfit check.",
    )
    return parser


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_roi_size(values: object) -> RoiSize3D:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"ROI size must be a sequence of 3 integers, got {values!r}.")
    return (int(values[0]), int(values[1]), int(values[2]))


def resolve_processed_case_paths(case_id: str, processed_root: Path) -> ProcessedCasePaths:
    case_dir = processed_root / str(case_id)
    image_path = case_dir / f"{case_id}_ct_processed.nii.gz"
    label_path = case_dir / f"{case_id}_airway_mask_processed.nii.gz"

    missing_paths = [path for path in (image_path, label_path) if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Missing processed case files. Run preprocessing.py with --save-processed first.\n"
            f"{missing_text}"
        )

    return ProcessedCasePaths(case_id=str(case_id), image=image_path, label=label_path)


def load_case(paths: ProcessedCasePaths) -> LoadedCase:
    loader = LoadImage(image_only=True, ensure_channel_first=True)
    reference_image = nib.load(str(paths.image))
    affine = np.asarray(reference_image.affine, dtype=np.float64)
    image_array = np.asarray(loader(str(paths.image)), dtype=np.float32)
    label_array = np.asarray(loader(str(paths.label)), dtype=np.float32)
    binary_label_array = (label_array > 0.5).astype(np.float32, copy=False)

    if tuple(image_array.shape) != tuple(binary_label_array.shape):
        raise ValueError(
            f"Shape mismatch for case {paths.case_id}: image {tuple(image_array.shape)} vs label {tuple(binary_label_array.shape)}"
        )

    image = torch.from_numpy(np.ascontiguousarray(image_array))
    label = torch.from_numpy(np.ascontiguousarray(binary_label_array))

    return LoadedCase(
        case_id=paths.case_id,
        image_path=paths.image,
        label_path=paths.label,
        affine=affine,
        image=image,
        label=label,
    )


def build_train_loader(
    loaded_case: LoadedCase,
    roi_size: RoiSize3D,
    batch_size: int,
    samples_per_epoch: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[TensorDict]:
    dataset = SingleCasePatchDataset(
        image=loaded_case.image,
        label=loaded_case.label,
        roi_size=roi_size,
        samples_per_epoch=samples_per_epoch,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def build_model(device: torch.device) -> nn.Module:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
        norm="INSTANCE",
    )
    return model.to(device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[TensorDict],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    batch_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().item())
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("Training loader yielded no batches.")

    return running_loss / batch_count


def infer_full_volume(
    model: nn.Module,
    image: Tensor,
    roi_size: RoiSize3D,
    sw_batch_size: int,
    device: torch.device,
) -> Tensor:
    image_batch = image.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=image_batch,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=0.25,
            progress=False,
        )
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Expected tensor output from sliding_window_inference, got {type(logits)!r}.")
    return logits


def compute_binary_dice(prediction: Tensor, target: Tensor) -> float:
    prediction = prediction.float()
    target = target.float()
    intersection = 2.0 * torch.sum(prediction * target)
    denominator = torch.sum(prediction) + torch.sum(target)
    if float(denominator.item()) == 0.0:
        return 1.0
    score = (intersection + 1e-5) / (denominator + 1e-5)
    return float(score.item())


def compute_case_dice(
    model: nn.Module,
    loaded_case: LoadedCase,
    roi_size: RoiSize3D,
    sw_batch_size: int,
    device: torch.device,
) -> float:
    logits = infer_full_volume(
        model=model,
        image=loaded_case.image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        device=device,
    )
    label = loaded_case.label.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    prediction = (torch.sigmoid(logits) > 0.5).float()
    return compute_binary_dice(prediction, label)


def save_nifti_volume(
    volume: np.ndarray,
    affine: np.ndarray,
    output_path: Path,
    dtype: np.dtype[Any],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = nib.Nifti1Header()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(np.asarray(volume, dtype=dtype), affine, header=header)
    nib.save(image, str(output_path))
    return output_path


def export_case_prediction(
    model: nn.Module,
    loaded_case: LoadedCase,
    roi_size: RoiSize3D,
    sw_batch_size: int,
    device: torch.device,
    output_mask_path: Path,
    output_probability_path: Path | None = None,
) -> dict[str, Path]:
    logits = infer_full_volume(
        model=model,
        image=loaded_case.image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        device=device,
    )
    probabilities = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    prediction_mask = (probabilities > 0.5).astype(np.uint8)

    saved_paths = {
        "mask": save_nifti_volume(
            prediction_mask,
            loaded_case.affine,
            output_mask_path,
            np.uint8,
        )
    }

    if output_probability_path is not None:
        saved_paths["probability"] = save_nifti_volume(
            probabilities,
            loaded_case.affine,
            output_probability_path,
            np.float32,
        )

    return saved_paths


def make_run_dir(output_root: Path, train_case: str, val_case: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"train_{train_case}_val_{val_case}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    processed_root = args.processed_root.resolve()
    output_root = args.output_root.resolve()
    train_case_id = str(args.train_case)
    val_case_id = train_case_id if args.train_val_same_case else str(args.val_case)
    roi_size = parse_roi_size(args.roi_size)
    device = resolve_device(args.device)

    if any(value <= 0 for value in roi_size):
        raise ValueError(f"ROI size must be positive, got {roi_size}.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.samples_per_epoch <= 0:
        raise ValueError("--samples-per-epoch must be positive.")
    if args.sw_batch_size <= 0:
        raise ValueError("--sw-batch-size must be positive.")

    set_determinism(seed=args.seed)

    train_paths = resolve_processed_case_paths(train_case_id, processed_root=processed_root)
    val_paths = resolve_processed_case_paths(val_case_id, processed_root=processed_root)
    train_case = load_case(train_paths)
    val_case = load_case(val_paths)

    train_loader = build_train_loader(
        loaded_case=train_case,
        roi_size=roi_size,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        num_workers=args.num_workers,
        device=device,
    )
    model = build_model(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = make_run_dir(output_root=output_root, train_case=train_case_id, val_case=val_case_id)
    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    history_path = run_dir / "metrics.json"

    config: dict[str, object] = {
        "train_case": train_case_id,
        "val_case": val_case_id,
        "processed_root": str(processed_root),
        "train_image_path": str(train_case.image_path),
        "train_label_path": str(train_case.label_path),
        "val_image_path": str(val_case.image_path),
        "val_label_path": str(val_case.label_path),
        "device": str(device),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "samples_per_epoch": int(args.samples_per_epoch),
        "roi_size": [int(value) for value in roi_size],
        "sw_batch_size": int(args.sw_batch_size),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "train_image_shape": [int(value) for value in train_case.image.shape],
        "val_image_shape": [int(value) for value in val_case.image.shape],
    }
    save_json(run_dir / "config.json", config)

    history: list[dict[str, float | int]] = []
    best_val_dice = float("-inf")
    saved_prediction_paths: dict[str, str] = {}

    print(f"Using device: {device}")
    if device.type == "cuda":
        device_index = 0 if device.index is None else int(device.index)
        print(f"GPU: {torch.cuda.get_device_name(device_index)}")
    print(f"Training case: {train_case.case_id}  shape={tuple(train_case.image.shape)}")
    print(f"Validation case: {val_case.case_id}  shape={tuple(val_case.image.shape)}")
    print(f"Run directory: {run_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        train_dice = compute_case_dice(
            model=model,
            loaded_case=train_case,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            device=device,
        )
        val_dice = compute_case_dice(
            model=model,
            loaded_case=val_case,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            device=device,
        )

        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_dice": val_dice,
        }
        history.append(epoch_metrics)
        save_json(
            history_path,
            {
                "config": config,
                "history": history,
                "best_val_dice": best_val_dice,
                "prediction_paths": saved_prediction_paths,
            },
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"loss={train_loss:.4f} "
            f"train_dice={train_dice:.4f} "
            f"val_dice={val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_case": train_case_id,
                    "val_case": val_case_id,
                    "val_dice": val_dice,
                    "config": config,
                },
                best_model_path,
            )

        save_json(
            history_path,
            {
                "config": config,
                "history": history,
                "best_val_dice": best_val_dice,
                "prediction_paths": saved_prediction_paths,
            },
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": args.epochs,
            "train_case": train_case_id,
            "val_case": val_case_id,
            "best_val_dice": best_val_dice,
            "config": config,
        },
        last_model_path,
    )

    if args.save_predictions:
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])

        train_prediction_paths = export_case_prediction(
            model=model,
            loaded_case=train_case,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            device=device,
            output_mask_path=run_dir / f"train_case_{train_case.case_id}_prediction_mask.nii.gz",
            output_probability_path=(
                run_dir / f"train_case_{train_case.case_id}_prediction_probability.nii.gz"
                if args.save_probabilities
                else None
            ),
        )
        val_prediction_paths = export_case_prediction(
            model=model,
            loaded_case=val_case,
            roi_size=roi_size,
            sw_batch_size=args.sw_batch_size,
            device=device,
            output_mask_path=run_dir / f"val_case_{val_case.case_id}_prediction_mask.nii.gz",
            output_probability_path=(
                run_dir / f"val_case_{val_case.case_id}_prediction_probability.nii.gz"
                if args.save_probabilities
                else None
            ),
        )

        saved_prediction_paths = {
            "train_mask": str(train_prediction_paths["mask"]),
            "val_mask": str(val_prediction_paths["mask"]),
        }
        if "probability" in train_prediction_paths:
            saved_prediction_paths["train_probability"] = str(train_prediction_paths["probability"])
        if "probability" in val_prediction_paths:
            saved_prediction_paths["val_probability"] = str(val_prediction_paths["probability"])

        save_json(
            history_path,
            {
                "config": config,
                "history": history,
                "best_val_dice": best_val_dice,
                "prediction_paths": saved_prediction_paths,
            },
        )

    print(f"Best validation dice: {best_val_dice:.4f}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Last checkpoint: {last_model_path}")
    print(f"Metrics: {history_path}")
    if saved_prediction_paths:
        print(f"Train prediction mask: {saved_prediction_paths['train_mask']}")
        print(f"Validation prediction mask: {saved_prediction_paths['val_mask']}")
        if "train_probability" in saved_prediction_paths:
            print(f"Train probability map: {saved_prediction_paths['train_probability']}")
        if "val_probability" in saved_prediction_paths:
            print(f"Validation probability map: {saved_prediction_paths['val_probability']}")


if __name__ == "__main__":
    main()
