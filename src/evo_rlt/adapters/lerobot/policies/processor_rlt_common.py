from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def _read_tokenizer_path(vla_pretrained_path: str) -> str:
    processor_path = Path(vla_pretrained_path) / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    processor_config = json.loads(processor_path.read_text())
    for step in processor_config["steps"]:
        if step.get("registry_name") == "tokenizer_processor":
            return step["config"]["tokenizer_name"]
    raise ValueError(f"tokenizer_processor not found in {processor_path}")


def _try_read_tokenizer_path(vla_pretrained_path: str) -> str | None:
    try:
        return _read_tokenizer_path(vla_pretrained_path)
    except ValueError:
        return None


def load_sft_vla_processors(
    vla_pretrained_path: str,
    tokenizer_path: str | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load the SFT VLA pre/post-processor pair verbatim from disk.

    The SFT JSON is the first source of truth for tokenizer identity. Callers
    may still pass an explicit local tokenizer snapshot to avoid network access.
    """
    if not vla_pretrained_path:
        raise ValueError(
            "vla_pretrained_path must be set; the SFT VLA ckpt is the single source of "
            "QUANTILES stats — deploy parity depends on it."
        )

    tokenizer_path = tokenizer_path or _try_read_tokenizer_path(vla_pretrained_path)
    overrides: dict[str, Any] = {}
    if tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        overrides = {
            "tokenizer_processor": {
                "tokenizer_name": tokenizer_path,
                "tokenizer": tokenizer,
            }
        }
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=vla_pretrained_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides=overrides,
    )
    post = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=vla_pretrained_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return pre, post


def load_sft_pi05_processors(
    vla_pretrained_path: str,
    tokenizer_path: str | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Backward-compatible alias for older pi0.5-specific imports."""
    return load_sft_vla_processors(vla_pretrained_path, tokenizer_path)
