"""ParNet model utilities: input types, model loading, and prediction."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from parnet.models import RBPNet


class ParnetInputDict(dict):
    """A dict restricted to a single ``"sequences"`` key, enforcing ParNet's input contract."""

    def __init__(self, sequences: Any) -> None:
        super().__init__({"sequences": sequences})

    def __setitem__(self, key: str, value: Any) -> None:
        if key != "sequences":
            raise KeyError("Only 'sequences' key is allowed")
        super().__setitem__(key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if any(k != "sequences" for k in dict(*args, **kwargs).keys()):
            raise KeyError("Only 'sequences' key is allowed")
        super().update(*args, **kwargs)

    def __delitem__(self, key: str) -> None:
        raise KeyError("'sequences' key cannot be deleted")


class ParnetModelName(str, Enum):
    """Known ParNet model identifiers."""

    PARNET_NEW_11M_5_0 = "parnet_new.11m-5.0"
    PARNET_21M_NONE = "parnet.21m-none"
    PARNET_21M_0_0 = "parnet.21m-0.0"
    PARNET_21M_5_0 = "parnet.21m-5.0"
    PARNET_7M_0_0 = "parnet.7m-0.0"
    PARNET_7M_2_5 = "parnet.7m-2.5"
    PARNET_7M_10_0 = "parnet.7m-10.0"
    PARNET_7M_20_0 = "parnet.7m-20.0"
    PARNET_7M_80_0 = "parnet.7m-80.0"


@dataclass(frozen=True)
class _ParnetModelPostLoadConfig:
    use_maximum_target_control_logprob: bool
    reset_projection: bool
    control_nograd: bool | None  # None = do not set


_PARNET_MODEL_CONFIGS: dict[ParnetModelName, _ParnetModelPostLoadConfig] = {
    ParnetModelName.PARNET_NEW_11M_5_0: _ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=True,
        reset_projection=False,
        control_nograd=None,
    ),
    ParnetModelName.PARNET_21M_NONE: _ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=False,
        reset_projection=True,
        control_nograd=False,
    ),
    ParnetModelName.PARNET_21M_0_0: _ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=False,
        reset_projection=True,
        control_nograd=False,
    ),
    ParnetModelName.PARNET_21M_5_0: _ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=False,
        reset_projection=True,
        control_nograd=False,
    ),
    ParnetModelName.PARNET_7M_0_0: _ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=True,
        reset_projection=True,
        control_nograd=None,
    ),
    ParnetModelName.PARNET_7M_2_5: _ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=True,
        reset_projection=True,
        control_nograd=None,
    ),
    ParnetModelName.PARNET_7M_10_0: _ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=True,
        reset_projection=True,
        control_nograd=None,
    ),
    ParnetModelName.PARNET_7M_20_0: _ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=True,
        reset_projection=True,
        control_nograd=None,
    ),
    ParnetModelName.PARNET_7M_80_0: _ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=True,
        reset_projection=True,
        control_nograd=None,
    ),
}


def load_parnet_model(
    parnet_model_name: ParnetModelName | str,
    filepath: os.PathLike,
    dtype: torch.dtype,
    device: torch.device,
) -> RBPNet:
    """Load a ParNet model from a checkpoint and apply model-specific post-load configuration."""
    model_name = ParnetModelName(parnet_model_name)
    cfg = _PARNET_MODEL_CONFIGS[model_name]
    model: RBPNet = torch.load(filepath, map_location=device).to(dtype)
    if cfg.reset_projection:
        model.projection = lambda x: x
    model.head.use_maximum_target_control_logprob = cfg.use_maximum_target_control_logprob
    if cfg.control_nograd is not None:
        model.head.control_nograd = cfg.control_nograd
    return model


def parnet_predict(
    model: RBPNet,
    model_metadata: dict[str, Any],
    input_data: ParnetInputDict,
) -> dict[str, torch.Tensor]:
    """Run a ParNet model, routing input format based on model version."""
    if model_metadata["version"] == "0.5.0":
        return model(input_data["sequences"])
    return model(input_data)
