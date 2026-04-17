"""Full-volume inference helpers for 3D airway segmentation."""

import torch
from monai.inferers import sliding_window_inference


def predict_logits_for_volume(
    model,
    image,
    *,
    device,
    roi_size,
    sw_batch_size=1,
    overlap=0.5,
    use_amp=False,
):
    """Run sliding-window inference on one cropped 3D volume."""
    if image.ndim == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 4:
        image = image.unsqueeze(0)
    elif image.ndim != 5:
        raise ValueError(f"Expected image volume to be 3D, 4D, or 5D, got shape {tuple(image.shape)}")

    image = image.to(device).float()
    model.eval()

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            logits = sliding_window_inference(
                inputs=image,
                roi_size=tuple(int(value) for value in roi_size),
                sw_batch_size=int(sw_batch_size),
                predictor=model,
                overlap=float(overlap),
            )

    return logits
