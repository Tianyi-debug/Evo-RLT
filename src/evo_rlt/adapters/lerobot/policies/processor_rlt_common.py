from __future__ import annotations

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


def load_sft_pi05_processors(
    vla_pretrained_path: str,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load the SFT pi05 pre/post-processor pair verbatim from disk.

    The SFT JSON hard-codes ``tokenizer_name`` to an in-house path
    (``/llm_jzm/...``) that only exists on the original training cluster; we
    override to the public HF id so the loaded pipeline (and the JSON re-saved
    by the host policy's ckpt dir) is portable across deploy machines.
    """
    if not vla_pretrained_path:
        raise ValueError(
            "vla_pretrained_path must be set; the SFT pi05 ckpt is the single source of "
            "QUANTILES stats — deploy parity depends on it."
        )

    tokenizer_name = "google/paligemma-3b-pt-224"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    overrides = {
        "tokenizer_processor": {
            "tokenizer_name": tokenizer_name,
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
