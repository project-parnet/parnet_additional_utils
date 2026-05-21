"""ParNet model utilities: input types, model loading, and prediction."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

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
    """Known ParNet model identifiers.

    NOTE: hard-coded for the main models worked with in the Marsico Lab as of May 2026.
    TODO: update to official models upon official release.
    """
    # Model trained end 2025.
    PARNET_NEW_11M_5_0 = "parnet_new.11m-5.0"
    #
    # Model trained end 2024/early 2025 ; main models. NOTE: 21m-5.0 is fully trained.
    PARNET_21M_NONE = "parnet.21m-none"
    PARNET_21M_0_0 = "parnet.21m-0.0"
    PARNET_21M_5_0 = "parnet.21m-5.0"
    #
    # Models trained from the 21M-0.0, introducing the penalty.
    PARNET_21M_fth_2_5 = "parnet.21m-ft-head-2.5"
    PARNET_21M_fth_5_0 = "parnet.21m-ft-head-5.0"
    PARNET_21M_fth_10_0 = "parnet.21m-ft-head-10.0"
    PARNET_21M_fth_20_0 = "parnet.21m-ft-head-20.0"
    PARNET_21M_fth_80_0 = "parnet.21m-ft-head-80.0"
    PARNET_21M_ftf_2_5 = "parnet.21m-ft-full-2.5"
    PARNET_21M_ftf_5_0 = "parnet.21m-ft-full-5.0"
    PARNET_21M_ftf_10_0 = "parnet.21m-ft-full-10.0"
    PARNET_21M_ftf_20_0 = "parnet.21m-ft-full-20.0"
    PARNET_21M_ftf_80_0 = "parnet.21m-ft-full-80.0"
    # Models fine-tuned from 21M-0.0 (head-only), max-logprob variant.
    PARNET_21M_fth_max_2_5 = "parnet.21m-ft-head-max-2.5"
    PARNET_21M_fth_max_5_0 = "parnet.21m-ft-head-max-5.0"
    PARNET_21M_fth_max_10_0 = "parnet.21m-ft-head-max-10.0"
    PARNET_21M_fth_max_20_0 = "parnet.21m-ft-head-max-20.0"
    PARNET_21M_fth_max_80_0 = "parnet.21m-ft-head-max-80.0"
    #
    # Models trained early 2025, fully trained, with the penalty.
    PARNET_7M_0_0 = "parnet.7m-0.0"
    PARNET_7M_2_5 = "parnet.7m-2.5"
    PARNET_7M_5_0 = "parnet.7m-5.0"
    PARNET_7M_10_0 = "parnet.7m-10.0"
    PARNET_7M_20_0 = "parnet.7m-20.0"
    PARNET_7M_80_0 = "parnet.7m-80.0"


@dataclass(frozen=True)
class ParnetModelPostLoadConfig:
    """Post-load attribute patching applied to a model after ``torch.load``.

    Attributes:
        use_maximum_target_control_logprob: Sets
            ``model.head.use_maximum_target_control_logprob`` after loading.
        bypass_projection: If True, replaces ``model.projection``
            (``nn.LazyConv1d`` embedding-dim reduction at the backbone end)
            with an identity lambda. Required for v0.1.0 / v0.3.0 models.
        control_nograd: If not None, sets ``model.head.control_nograd`` after
            loading.
        parnet_version: Lab-internal architecture version label (e.g.
            ``"v0.3.0"``). Reflects architecture changes only — not training
            hyperparameters or official parnet releases. ``None`` for models
            outside the known scheme.

    To register a config for a custom checkpoint, add an entry to
    ``PARNET_MODEL_CONFIGS`` at runtime::

        from parnet_additional_utils import PARNET_MODEL_CONFIGS, ParnetModelPostLoadConfig
        PARNET_MODEL_CONFIGS["my-model"] = ParnetModelPostLoadConfig(
            use_maximum_target_control_logprob=True,
            bypass_projection=True,
            control_nograd=None,
        )

    Or pass the config directly to ``load_parnet_model`` via ``post_load_config=``.
    """

    use_maximum_target_control_logprob: bool
    bypass_projection: bool
    control_nograd: bool | None  # None = do not set
    parnet_version: str | None = None  # lab-internal arch label; None = unregistered model


_FAMILY_CONFIGS: dict[str, ParnetModelPostLoadConfig] = {
    "parnet_new.11m": ParnetModelPostLoadConfig(
        # Max-logprob trick (AdditiveMix.forward: max stabilisation path).
        # 11M checkpoints may predate use_maximum_target_control_logprob attr → must patch.
        use_maximum_target_control_logprob=True,
        # v0.5.0 backbone has no nn.LazyConv1d projection layer → no bypass needed.
        bypass_projection=False,
        # control_nograd not present in v0.5.0 AdditiveMix variant → do not set.
        control_nograd=None,
        parnet_version="v0.5.0",
    ),
    "parnet.21m-base": ParnetModelPostLoadConfig(
        # Standard logsumexp mixing (AdditiveMix.forward: logsumexp path, default=False).
        use_maximum_target_control_logprob=False,
        # v0.1.0 backbone ends with nn.LazyConv1d projection → replace with identity.
        bypass_projection=True,
        # Explicitly disable control gradient; checkpoints may predate the default.
        control_nograd=False,
        parnet_version="v0.1.0",
    ),
    "parnet.7m": ParnetModelPostLoadConfig(
        # Max-logprob trick. 7M checkpoints were trained with True and may predate the attr.
        use_maximum_target_control_logprob=True,
        # v0.3.0 backbone ends with nn.LazyConv1d projection → replace with identity.
        bypass_projection=True,
        # control_nograd not explicitly set for 7M; rely on AdditiveMix default (False).
        control_nograd=None,
        parnet_version="v0.3.0",
    ),
    "parnet.21m-ft": ParnetModelPostLoadConfig(
        # Standard logsumexp; inherits from 21M-0.0 base (False) and AdditiveMix default.
        use_maximum_target_control_logprob=False,
        # Backbone is 21M (v0.1.0) → nn.LazyConv1d projection must be bypassed.
        bypass_projection=True,
        # Fresh AdditiveMix head on fine-tuning → do not override control_nograd.
        control_nograd=None,
        parnet_version="v0.1.2",
    ),
    "parnet.21m-fth-max": ParnetModelPostLoadConfig(
        # Max-logprob trick; explicitly set True on the head before fine-tuning.
        # Must be restored on load (attr not serialised in state_dict).
        use_maximum_target_control_logprob=True,
        # Backbone is 21M (v0.1.0) → nn.LazyConv1d projection must be bypassed.
        bypass_projection=True,
        # Fresh AdditiveMix head on fine-tuning → do not override control_nograd.
        control_nograd=None,
        parnet_version="v0.1.2",
    ),
}

_MODEL_FAMILY: dict[str, str] = {
    # parnet_new.11m (v0.5.0)
    "parnet_new.11m-5.0":          "parnet_new.11m",
    # parnet.21m base (v0.1.0)
    "parnet.21m-none":              "parnet.21m-base",
    "parnet.21m-0.0":               "parnet.21m-base",
    "parnet.21m-5.0":               "parnet.21m-base",
    # parnet.21m fine-tuned, head-only (v0.1.2, logsumexp)
    "parnet.21m-ft-head-2.5":       "parnet.21m-ft",
    "parnet.21m-ft-head-5.0":       "parnet.21m-ft",
    "parnet.21m-ft-head-10.0":      "parnet.21m-ft",
    "parnet.21m-ft-head-20.0":      "parnet.21m-ft",
    "parnet.21m-ft-head-80.0":      "parnet.21m-ft",
    # parnet.21m fine-tuned, full (v0.1.2, logsumexp)
    "parnet.21m-ft-full-2.5":       "parnet.21m-ft",
    "parnet.21m-ft-full-5.0":       "parnet.21m-ft",
    "parnet.21m-ft-full-10.0":      "parnet.21m-ft",
    "parnet.21m-ft-full-20.0":      "parnet.21m-ft",
    "parnet.21m-ft-full-80.0":      "parnet.21m-ft",
    # parnet.21m fine-tuned, head-only, max-logprob (v0.1.2, max-trick)
    "parnet.21m-ft-head-max-2.5":   "parnet.21m-fth-max",
    "parnet.21m-ft-head-max-5.0":   "parnet.21m-fth-max",
    "parnet.21m-ft-head-max-10.0":  "parnet.21m-fth-max",
    "parnet.21m-ft-head-max-20.0":  "parnet.21m-fth-max",
    "parnet.21m-ft-head-max-80.0":  "parnet.21m-fth-max",
    # parnet.7m (v0.3.0)
    "parnet.7m-0.0":                "parnet.7m",
    "parnet.7m-2.5":                "parnet.7m",
    "parnet.7m-5.0":                "parnet.7m",
    "parnet.7m-10.0":               "parnet.7m",
    "parnet.7m-20.0":               "parnet.7m",
    "parnet.7m-80.0":               "parnet.7m",
}

PARNET_MODEL_CONFIGS: dict[str, ParnetModelPostLoadConfig] = {
    model: _FAMILY_CONFIGS[family]
    for model, family in _MODEL_FAMILY.items()
}
"""Post-load configs for all known ParNet checkpoints, keyed by model name string.

Built from ``_FAMILY_CONFIGS`` and ``_MODEL_FAMILY``. To add a new model from an
existing family, add one entry to ``_MODEL_FAMILY``. To register a custom checkpoint
without modifying the library, extend at runtime::

    from parnet_additional_utils import PARNET_MODEL_CONFIGS, ParnetModelPostLoadConfig
    PARNET_MODEL_CONFIGS["my-model"] = ParnetModelPostLoadConfig(
        use_maximum_target_control_logprob=True,
        bypass_projection=True,
        control_nograd=None,
    )
"""


def _model_name_str(parnet_model_name: "ParnetModelName | str") -> str:
    """Return the plain string value of a model name (handles both enum and str inputs)."""
    return (
        parnet_model_name.value
        if isinstance(parnet_model_name, ParnetModelName)
        else str(parnet_model_name)
    )


def load_parnet_model(
    parnet_model_name: "ParnetModelName | str",
    filepath: os.PathLike,
    dtype: torch.dtype,
    device: torch.device,
    *,
    post_load_config: ParnetModelPostLoadConfig | None = None,
) -> RBPNet:
    """Load a ParNet model from a checkpoint and apply post-load configuration.

    Args:
        parnet_model_name: Model identifier. For known models use a
            ``ParnetModelName`` enum member; any string is accepted when
            ``post_load_config`` is provided.
        filepath: Path to the ``.pt`` checkpoint.
        dtype: Target dtype (e.g. ``torch.float32``).
        device: Target device.
        post_load_config: Explicit post-load configuration. If ``None``
            (default), the config is looked up from ``PARNET_MODEL_CONFIGS``
            by name. Pass this argument to load a checkpoint whose name is
            not yet registered in that dict.
    """
    if post_load_config is not None:
        cfg = post_load_config
    else:
        name_str = _model_name_str(parnet_model_name)
        if name_str not in PARNET_MODEL_CONFIGS:
            raise KeyError(
                f"No post-load config registered for {name_str!r}. "
                "Pass post_load_config= explicitly, or add an entry to PARNET_MODEL_CONFIGS."
            )
        cfg = PARNET_MODEL_CONFIGS[name_str]

    model: RBPNet = torch.load(filepath, map_location=device).to(dtype)

    if cfg.bypass_projection:
        model.projection = lambda x: x

    model.head.use_maximum_target_control_logprob = cfg.use_maximum_target_control_logprob

    if cfg.control_nograd is not None:
        model.head.control_nograd = cfg.control_nograd

    return model


def _get_head_num_tasks(model: RBPNet) -> int:
    """Return the output task count from a model head, tolerating known attribute renames."""
    head_target = model.head.head_target

    for attr in ("pointwise", "pointwise_conv"):
        layer = getattr(head_target, attr, None)
        if layer is not None:
            return cast(int, layer.out_channels)

    raise AttributeError(
        f"Cannot determine num_tasks from {type(head_target).__name__}: "
        "expected .pointwise or .pointwise_conv"
    )


def _resolve_parnet_layer_cls(class_name: str) -> type:
    """Resolve a parnet.layers class by name, with fallback for known renames.

    LinearProjection was renamed to LinearProjectionHead after commit 2e14278.
    Either name maps to whichever form the installed parnet version exposes.
    """
    import parnet.layers as _layers

    if hasattr(_layers, class_name):
        return cast(type, getattr(_layers, class_name))

    _aliases: dict[str, str] = {
        "LinearProjection": "LinearProjectionHead",
        "LinearProjectionHead": "LinearProjection",
    }

    alias = _aliases.get(class_name)

    if alias is not None and hasattr(_layers, alias):
        return cast(type, getattr(_layers, alias))

    raise ValueError(
        f"Cannot resolve {class_name!r} in parnet.layers. "
        "Add an entry to _aliases in parnet_additional_utils if the class was renamed."
    )


def save_parnet_model_as_statedict(
    model: RBPNet,
    filepath: os.PathLike,
    parnet_model_name: "ParnetModelName | str",
) -> None:
    """Save a fine-tuned ParNet model as a version-agnostic state_dict bundle.

    Unlike ``torch.save(model, ...)``, the bundle does not embed Python class
    references, so it survives renaming of internal parnet classes across library
    versions (e.g. ``LinearProjection`` → ``LinearProjectionHead``).

    The bundle is a dict with two keys:

    - ``state_dict`` — the full model parameters
    - ``model_config`` — architecture metadata needed to reconstruct the model

    Args:
        model: The fine-tuned RBPNet to save.
        filepath: Destination path (e.g. ``"results/model.statedict.pt"``).
        parnet_model_name: The base pretrained model identifier (e.g.
            ``ParnetModelName.PARNET_7M_0_0``). Stored in the bundle and
            used to reconstruct the backbone on load.
    """
    head = model.head
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "parnet_model_name": _model_name_str(parnet_model_name),
                "head_num_tasks": _get_head_num_tasks(model),
                "head_layer": type(head.head_target).__name__,
                "mix_coeff_layer": type(head.mix_coeff).__name__,
                "penalty_layer": type(head.penalty).__name__ if head.penalty is not None else None,
                "penalty_factor": head.penalty.factor if head.penalty is not None else None,
            },
        },
        filepath,
    )


def load_parnet_model_from_statedict(
    filepath: os.PathLike,
    pretrained_model_path: os.PathLike,
    device: torch.device,
    dtype: torch.dtype,
    *,
    post_load_config: ParnetModelPostLoadConfig | None = None,
) -> RBPNet:
    """Load a fine-tuned ParNet model from a state_dict bundle.

    Reconstructs the model architecture from ``model_config``, then loads the
    saved parameters. Class names in the bundle are resolved dynamically with
    fallbacks for known renames, so the bundle remains loadable across parnet
    library versions.

    Args:
        filepath: Path to a bundle saved by ``save_parnet_model_as_statedict``.
        pretrained_model_path: Path to the original pretrained checkpoint
            (.pt). Required to reconstruct the backbone; only the backbone
            weights matter here since the fine-tuned weights are loaded from
            the bundle afterwards.
        device: Target device.
        dtype: Target dtype.
        post_load_config: Explicit post-load configuration. If ``None``
            (default), looked up from ``PARNET_MODEL_CONFIGS`` using the name
            stored in the bundle. Pass this argument if the base model name is
            not in that dict.
    """
    from parnet.layers import AdditiveMix

    bundle = torch.load(filepath, map_location=device)
    config = bundle["model_config"]
    base_name = config["parnet_model_name"]

    # Rebuild backbone (applies bypass_projection, use_maximum_target_control_logprob, etc.)
    model = load_parnet_model(
        base_name,
        pretrained_model_path,
        dtype=dtype,
        device=device,
        post_load_config=post_load_config,
    )

    # Reconstruct head using current class implementations (class names are advisory)
    head_cls = _resolve_parnet_layer_cls(config["head_layer"])
    mix_cls = _resolve_parnet_layer_cls(config["mix_coeff_layer"])
    penalty_cls = (
        _resolve_parnet_layer_cls(config["penalty_layer"])
        if config["penalty_layer"] is not None
        else None
    )
    model.head = AdditiveMix(
        num_tasks=config["head_num_tasks"],
        head_layer=head_cls,
        mix_coeff_layer=mix_cls,
        penalty_layer=penalty_cls,
    )
    if model.head.penalty is not None and config["penalty_factor"] is not None:
        model.head.penalty.factor = config["penalty_factor"]

    # Re-apply post-load config attributes to the new head
    cfg = post_load_config if post_load_config is not None else PARNET_MODEL_CONFIGS[base_name]
    model.head.use_maximum_target_control_logprob = cfg.use_maximum_target_control_logprob
    if cfg.control_nograd is not None:
        model.head.control_nograd = cfg.control_nograd

    model = model.to(device).to(dtype)
    # load_state_dict materialises any LazyModule parameters from the saved tensor shapes
    model.load_state_dict(bundle["state_dict"])
    return model


def parnet_predict(
    model: RBPNet,
    model_metadata: dict[str, Any],
    input_data: ParnetInputDict,
) -> dict[str, torch.Tensor]:
    """Run a ParNet model, routing input format based on model version."""
    if model_metadata["version"] == "0.5.0":
        return cast(dict[str, torch.Tensor], model(input_data["sequences"]))

    return cast(dict[str, torch.Tensor], model(input_data))
