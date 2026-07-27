from __future__ import annotations

import json

import torch

from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy


def test_new_ac_config_uses_safe_v2_semantics():
    config = ChunkACPolicyConfig()

    assert config.ac_semantics_version == 2
    assert config.state_normalization == "rl_token_layer_norm"
    assert config.actor_action_residual is True
    assert config.actor_delta_scale == 0.1
    assert config.actor_ref_dropout_p == 0.0


def test_unversioned_local_checkpoint_is_pinned_to_v1(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"type": "rlt_ac"}),
        encoding="utf-8",
    )
    config = ChunkACPolicyConfig(pretrained_path=str(tmp_path))
    policy = object.__new__(ChunkACPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config

    policy._apply_checkpoint_semantics_compat()

    assert config.ac_semantics_version == 1
    assert config.state_normalization == "none"
    assert config.actor_action_residual is False


def test_versioned_v2_checkpoint_keeps_v2_semantics(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"type": "rlt_ac", "ac_semantics_version": 2}),
        encoding="utf-8",
    )
    config = ChunkACPolicyConfig(pretrained_path=str(tmp_path))
    policy = object.__new__(ChunkACPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config

    policy._apply_checkpoint_semantics_compat()

    assert config.ac_semantics_version == 2
    assert config.state_normalization == "rl_token_layer_norm"
    assert config.actor_action_residual is True
