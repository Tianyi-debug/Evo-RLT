from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from torch import Tensor, nn


SUPPORTED_VLA_TYPES = {"pi05", "smolvla"}
log = logging.getLogger(__name__)


def _resolve_distributed_cuda_device(device: str | None) -> str | None:
    """Map a generic cuda device to this process's local rank under DDP."""
    if device != "cuda":
        return device
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return device
    try:
        rank_idx = int(local_rank)
    except ValueError:
        return device
    if rank_idx < 0:
        return device
    return f"cuda:{rank_idx}"


def normalize_vla_type(vla_type: str | None) -> str:
    """Normalize CLI/config aliases into the LeRobot policy type used here."""
    if vla_type is None or vla_type == "":
        return "auto"
    value = vla_type.lower().replace("-", "").replace("_", "")
    aliases = {
        "auto": "auto",
        "pi05": "pi05",
        "pi0.5": "pi05",
        "pi05base": "pi05",
        "smolvla": "smolvla",
    }
    if value not in aliases:
        raise ValueError(
            f"unsupported vla_type={vla_type!r}; expected one of auto, pi05, smolvla"
        )
    return aliases[value]


def infer_vla_type(pretrained_path: str, requested: str | None = "auto") -> str:
    requested_type = normalize_vla_type(requested)
    if requested_type != "auto":
        return requested_type

    config_path = Path(pretrained_path) / "config.json"
    if config_path.exists():
        raw = json.loads(config_path.read_text())
        raw_type = raw.get("type") or raw.get("policy_type")
        if raw_type is not None:
            inferred = normalize_vla_type(str(raw_type))
            if inferred in SUPPORTED_VLA_TYPES:
                return inferred

    name_hint = str(pretrained_path).lower()
    if "smolvla" in name_hint or "smol-vla" in name_hint or "smol_vla" in name_hint:
        return "smolvla"
    return "pi05"


def _load_config_from_dir(config_cls: type, pretrained_path: str):
    """Load LeRobot dataclass configs while stripping the polymorphic type field."""
    import draccus

    config_path = Path(pretrained_path) / "config.json"
    if not config_path.exists():
        return config_cls()

    raw = json.loads(config_path.read_text())
    raw.pop("type", None)
    raw.pop("policy_type", None)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(raw, tmp)
        tmp_path = tmp.name
    try:
        with draccus.config_type("json"):
            return draccus.parse(config_cls, tmp_path, args=[])
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _tie_pi05_embed_tokens(policy: nn.Module) -> None:
    """Restore tied language embeddings when older pi0.5 checkpoints omit them."""
    try:
        lm = policy.model.paligemma_with_expert.paligemma
        embed = lm.model.language_model.embed_tokens
    except AttributeError:
        return
    if embed is not None and lm.lm_head.weight.data_ptr() != embed.weight.data_ptr():
        embed.weight = lm.lm_head.weight


def _pi05_device_direct_load_enabled() -> bool:
    value = os.environ.get("EVO_RLT_PI05_DEVICE_DIRECT_LOAD", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _pi05_checkpoint_is_device_direct_compatible(model_file: Path) -> bool:
    """Return true when a pi0.5 checkpoint can skip LeRobot's CPU remap path."""
    if not model_file.exists():
        return False

    try:
        from safetensors import safe_open
    except Exception:
        return False

    try:
        with safe_open(model_file, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        log.debug("Could not inspect pi0.5 checkpoint keys for %s: %s", model_file, exc)
        return False

    if not keys:
        return False
    return all(key.startswith("model.") for key in keys)


def _load_pi05_policy_device_direct(
    policy_cls: type[nn.Module],
    cfg: Any,
    pretrained_path: str,
    *,
    strict: bool = False,
) -> nn.Module | None:
    """Load local current-format pi0.5 weights directly to the target device.

    LeRobot's PI05Policy.from_pretrained override loads the full safetensors file
    into a CPU state dict before remapping keys. Current Evo-RLT/pi0.5 checkpoints
    already use the expected ``model.*`` keys, so safetensors can stream them to
    the configured device and avoid a large CPU RAM peak. Older checkpoints keep
    using the upstream path via the caller's fallback.
    """
    if not _pi05_device_direct_load_enabled():
        return None

    model_file = Path(pretrained_path) / "model.safetensors"
    if not Path(pretrained_path).is_dir() or not _pi05_checkpoint_is_device_direct_compatible(model_file):
        return None

    try:
        from safetensors.torch import load_model
    except Exception:
        return None

    policy = None
    try:
        policy = policy_cls(cfg)
        missing_keys, unexpected_keys = load_model(
            policy,
            str(model_file),
            strict=strict,
            device=getattr(cfg, "device", "cpu"),
        )
        if missing_keys:
            log.info("pi0.5 device-direct load missing %d key(s)", len(missing_keys))
        if unexpected_keys:
            log.info("pi0.5 device-direct load ignored %d unexpected key(s)", len(unexpected_keys))
        _tie_pi05_embed_tokens(policy)
        policy.eval()
        log.info("Loaded pi0.5 checkpoint with safetensors device-direct path: %s", model_file)
        return policy
    except Exception as exc:
        log.warning(
            "pi0.5 device-direct load failed for %s; falling back to LeRobot loader: %s",
            model_file,
            exc,
        )
        if policy is not None:
            del policy
        return None


def load_vla_policy(
    pretrained_path: str,
    *,
    vla_type: str | None = "auto",
    revision: str | None = None,
    dtype: str | None = None,
    device: str | None = None,
) -> nn.Module:
    """Load a supported LeRobot VLA policy without importing every backend up front."""
    resolved_type = infer_vla_type(pretrained_path, vla_type)
    resolved_device = _resolve_distributed_cuda_device(device)
    if resolved_device != device:
        log.info("Resolved distributed VLA device %s -> %s", device, resolved_device)
    if resolved_type == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        cfg = _load_config_from_dir(PI05Config, pretrained_path)
        if dtype is not None:
            cfg.dtype = dtype
        if resolved_device is not None:
            cfg.device = resolved_device
        policy = None if revision is not None else _load_pi05_policy_device_direct(
            PI05Policy,
            cfg,
            pretrained_path,
            strict=False,
        )
        if policy is None:
            policy = PI05Policy.from_pretrained(
                pretrained_path,
                config=cfg,
                revision=revision,
                strict=False,
            )
        _tie_pi05_embed_tokens(policy)
        return policy

    if resolved_type == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        cfg = _load_config_from_dir(SmolVLAConfig, pretrained_path)
        if resolved_device is not None and hasattr(cfg, "device"):
            cfg.device = resolved_device
        if dtype is not None and hasattr(cfg, "dtype"):
            cfg.dtype = dtype
        return SmolVLAPolicy.from_pretrained(
            pretrained_path,
            config=cfg,
            revision=revision,
            strict=False,
        )

    raise ValueError(f"unsupported vla_type={resolved_type!r}")


def get_vla_prefix_target(policy: Any) -> Any:
    """Return the module whose forward emits prefix hidden states."""
    model = getattr(policy, "model", None)
    if model is None:
        raise AttributeError("VLA policy has no .model; cannot capture prefix hidden states")
    if hasattr(model, "paligemma_with_expert"):
        return model.paligemma_with_expert
    if hasattr(model, "vlm_with_expert"):
        return model.vlm_with_expert
    raise AttributeError(
        "unsupported VLA structure; expected .model.paligemma_with_expert "
        "or .model.vlm_with_expert"
    )


def extract_prefix_hidden(result: Any) -> Tensor | None:
    """Extract prefix output from PI05/SmolVLA expert forward return values."""
    if isinstance(result, dict):
        for key in ("prefix_hidden_states", "prefix_hidden", "prefix_output"):
            value = result.get(key)
            if value is not None:
                return value
        return None

    outputs = result[0] if isinstance(result, (tuple, list)) and len(result) > 0 else result
    if isinstance(outputs, dict):
        for key in ("prefix_hidden_states", "prefix_hidden", "prefix_output"):
            value = outputs.get(key)
            if value is not None:
                return value
        return None
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        prefix = outputs[0]
        return prefix if prefix is not None else None
    return None


def infer_vla_token_dim(policy: Any) -> int | None:
    """Infer the hidden size of prefix tokens for supported VLA policies."""
    paths = (
        ("model", "paligemma_with_expert", "paligemma", "config", "text_config", "hidden_size"),
        ("model", "vlm_with_expert", "config", "text_config", "hidden_size"),
        ("model", "vlm_with_expert", "vlm", "config", "text_config", "hidden_size"),
    )
    for path in paths:
        value = policy
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        if isinstance(value, int) and value > 0:
            return value
    if isinstance(policy, nn.Linear):
        return int(policy.out_features)
    return None


def _coerce_hw(value: Any) -> tuple[int, int] | None:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def _nested_attr(obj: Any, path: tuple[str, ...]) -> Any:
    value = obj
    for attr in path:
        value = getattr(value, attr, None)
        if value is None:
            return None
    return value


def _vision_config(policy: Any) -> Any:
    for path in (
        ("model", "paligemma_with_expert", "paligemma", "config", "vision_config"),
        ("model", "vlm_with_expert", "config", "vision_config"),
        ("model", "vlm_with_expert", "vlm", "config", "vision_config"),
    ):
        value = _nested_attr(policy, path)
        if value is not None:
            return value
    return None


def _num_cameras(policy: Any, camera_keys: list[str] | None = None) -> int:
    if camera_keys:
        return len(camera_keys)
    cfg = getattr(policy, "config", None)
    image_features = getattr(cfg, "image_features", None)
    if image_features:
        return len(image_features)
    return 1


def infer_num_image_tokens(
    policy: Any,
    *,
    image_resolution: tuple[int, int] | None = None,
    camera_keys: list[str] | None = None,
    num_per_camera: int = 0,
    required: bool = False,
) -> int:
    """Infer the leading image-token count for image-only prefix slicing."""
    if num_per_camera > 0:
        return int(num_per_camera) * _num_cameras(policy, camera_keys)

    vision_cfg = _vision_config(policy)
    if vision_cfg is not None:
        patch_hw = _coerce_hw(getattr(vision_cfg, "patch_size", None))
        model = getattr(policy, "model", None)
        is_pi05 = hasattr(model, "paligemma_with_expert")
        image_hw = image_resolution if is_pi05 else None
        image_hw = image_hw or _coerce_hw(getattr(vision_cfg, "image_size", None))
        if image_hw is None:
            resize = getattr(getattr(policy, "config", None), "resize_imgs_with_padding", None)
            image_hw = _coerce_hw(resize)
        if patch_hw is not None and image_hw is not None:
            per_camera = (int(image_hw[0]) // int(patch_hw[0])) * (
                int(image_hw[1]) // int(patch_hw[1])
            )
            if getattr(getattr(policy, "config", None), "add_image_special_tokens", False):
                per_camera += 3
            return per_camera * _num_cameras(policy, camera_keys)

    if required:
        raise ValueError(
            "could not infer VLA image-token count; pass policy.num_per_camera "
            "or disable policy.image_only/active_camera_indices"
        )
    return 0


def configure_vla_rtc(policy: Any, rtc_config: Any) -> None:
    cfg = getattr(policy, "config", None)
    if cfg is not None:
        cfg.rtc_config = rtc_config
    init_rtc = getattr(policy, "init_rtc_processor", None)
    if callable(init_rtc):
        init_rtc()
