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
    tokenizer_path: str = "google/paligemma-3b-pt-224",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load the SFT pi05 pre/post-processor pair verbatim from disk.

    The SFT JSON can hard-code ``tokenizer_name`` to an in-house path
    (``/llm_jzm/...``) that only exists on the original training cluster.  We
    override it so deployment can point at an explicit local tokenizer snapshot.
    """
    if not vla_pretrained_path:
        raise ValueError(
            "vla_pretrained_path must be set; the SFT pi05 ckpt is the single source of "
            "QUANTILES stats — deploy parity depends on it."
        )

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
