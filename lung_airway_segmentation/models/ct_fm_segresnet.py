"""CT-FM initialized SegResNet definition for airway segmentation."""

from pathlib import Path

import torch
from lighter_zoo import SegResEncoder, SegResNet

from lung_airway_segmentation.settings import PROJECT_ROOT


DEFAULT_CT_FM_ENCODER_REPO = "project-lighter/ct_fm_feature_extractor"
DEFAULT_CT_FM_SEGRESNET_REPO = "project-lighter/ct_fm_segresnet"


def resolve_pretrained_path(path_value: str | Path) -> Path:
    """Resolve a pretrained checkpoint path relative to the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_ct_fm_encoder_state_dict(
    *,
    source: str,
    variant: str,
    path_value: str | Path | None = None,
    repo_id: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load CT-FM encoder weights from either a local bundle or the Hub."""
    source = source.lower()
    variant = variant.lower()

    if source == "local":
        if variant != "encoder":
            raise ValueError("Local CT-FM loading currently supports only variant='encoder'.")
        if path_value is None:
            raise ValueError("Local CT-FM loading requires pretrained.path.")

        resolved_path = resolve_pretrained_path(path_value)
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"Local CT-FM checkpoint directory does not exist: {resolved_path}")

        encoder = SegResEncoder.from_pretrained(str(resolved_path))
        return {f"encoder.{key}": value for key, value in encoder.state_dict().items()}

    if source == "hub":
        if variant == "encoder":
            encoder = SegResEncoder.from_pretrained(repo_id or DEFAULT_CT_FM_ENCODER_REPO)
            return {f"encoder.{key}": value for key, value in encoder.state_dict().items()}

        if variant == "segresnet":
            model = SegResNet.from_pretrained(repo_id or DEFAULT_CT_FM_SEGRESNET_REPO)
            return {
                key: value
                for key, value in model.state_dict().items()
                if key.startswith("encoder.")
            }

        raise ValueError(f"Unsupported CT-FM pretrained variant: {variant}")

    raise ValueError(f"Unsupported CT-FM pretrained source: {source}")


def build_ct_fm_segresnet(
    *,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    init_filters: int = 32,
    blocks_down: tuple[int, ...] = (1, 2, 2, 4, 4),
    dsdepth: int = 1,
    act: str = "relu",
    norm: str = "batch",
    pretrained_enabled: bool = True,
    pretrained_source: str = "local",
    pretrained_variant: str = "encoder",
    pretrained_path: str | Path | None = None,
    pretrained_repo_id: str | None = None,
    freeze_encoder: bool = False,
):
    """Construct a SegResNet initialized from CT-FM encoder weights."""
    model = SegResNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=blocks_down,
        dsdepth=dsdepth,
        act=act.lower(),
        norm=norm.lower(),
    )

    if pretrained_enabled:
        pretrained_state = load_ct_fm_encoder_state_dict(
            source=pretrained_source,
            variant=pretrained_variant,
            path_value=pretrained_path,
            repo_id=pretrained_repo_id,
        )
        load_result = model.load_state_dict(pretrained_state, strict=False)

        missing_encoder_keys = [
            key for key in load_result.missing_keys if key.startswith("encoder.")
        ]
        if missing_encoder_keys:
            raise ValueError(
                "CT-FM encoder weights did not fully load into SegResNet. "
                f"Missing encoder keys: {missing_encoder_keys[:5]}"
            )
        if load_result.unexpected_keys:
            raise ValueError(
                "Unexpected CT-FM checkpoint keys encountered while loading "
                f"SegResNet: {load_result.unexpected_keys[:5]}"
            )

    if freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    return model
