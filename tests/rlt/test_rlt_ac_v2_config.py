from __future__ import annotations

import json

import pytest
import torch
from lerobot.configs.policies import PreTrainedConfig

from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy


def test_new_ac_config_uses_safe_v2_semantics():
    config = ChunkACPolicyConfig()

    assert config.ac_semantics_version == 2
    assert config.state_normalization == "rl_token_layer_norm"
    assert config.actor_action_residual is True
    assert config.actor_delta_scale == 0.1
    assert config.actor_ref_dropout_p == 0.0
    assert config.actor_bc_weight_mode == "fixed"
    assert config.actor_bc_uncertainty_kappa == 0.0
    assert config.actor_bc_uncertainty_threshold_mode == "fixed"
    assert config.actor_bc_uncertainty_ema_decay == pytest.approx(0.95)
    assert config.actor_bc_uncertainty_reset_ema_on_load is False
    assert config.critic_bootstrap_mode == "none"
    assert config.critic_bootstrap_keep_prob == pytest.approx(0.8)
    assert config.diagnostics_jsonl_path is None


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


def test_dynamic_bc_config_validates_schedule():
    config = ChunkACPolicyConfig(
        actor_bc_weight_mode="disagreement",
        actor_bc_uncertainty_tau_low=0.1,
        actor_bc_uncertainty_tau_high=0.2,
        actor_bc_uncertainty_kappa=3.0,
    )

    assert config.actor_bc_weight_mode == "disagreement"
    assert config.actor_bc_uncertainty_kappa == 3.0

    with pytest.raises(ValueError, match="greater"):
        ChunkACPolicyConfig(
            actor_bc_weight_mode="disagreement",
            actor_bc_uncertainty_tau_low=0.2,
            actor_bc_uncertainty_tau_high=0.2,
        )

    with pytest.raises(ValueError, match="non-negative"):
        ChunkACPolicyConfig(actor_bc_uncertainty_kappa=-1.0)

    with pytest.raises(ValueError, match="threshold_mode"):
        ChunkACPolicyConfig(actor_bc_uncertainty_threshold_mode="rolling")

    with pytest.raises(ValueError, match="ema_decay"):
        ChunkACPolicyConfig(actor_bc_uncertainty_ema_decay=1.0)

    with pytest.raises(ValueError, match="reset_ema_on_load requires"):
        ChunkACPolicyConfig(actor_bc_uncertainty_reset_ema_on_load=True)

    with pytest.raises(ValueError, match="reset_ema_on_load requires"):
        ChunkACPolicyConfig(
            actor_bc_weight_mode="disagreement",
            actor_bc_uncertainty_tau_low=0.1,
            actor_bc_uncertainty_tau_high=0.2,
            actor_bc_uncertainty_reset_ema_on_load=True,
        )

    with pytest.raises(ValueError, match="bootstrap_mode"):
        ChunkACPolicyConfig(critic_bootstrap_mode="random_each_epoch")

    with pytest.raises(ValueError, match="keep_prob"):
        ChunkACPolicyConfig(critic_bootstrap_keep_prob=0.0)

    with pytest.raises(ValueError, match="non-negative"):
        ChunkACPolicyConfig(critic_bootstrap_seed=-1)


def test_dynamic_bc_config_round_trips_through_pretrained_config(tmp_path):
    config = ChunkACPolicyConfig(
        actor_bc_weight_mode="disagreement",
        actor_bc_uncertainty_tau_low=0.015,
        actor_bc_uncertainty_tau_high=0.019,
        actor_bc_uncertainty_kappa=3.0,
        actor_bc_uncertainty_threshold_mode="ema_quantile",
        actor_bc_uncertainty_ema_decay=0.9,
        actor_bc_uncertainty_reset_ema_on_load=True,
        critic_bootstrap_mode="fixed_bernoulli",
        critic_bootstrap_keep_prob=0.75,
        critic_bootstrap_seed=77,
        diagnostics_jsonl_path="/tmp/evo-rlt-diagnostics.jsonl",
    )
    config.save_pretrained(tmp_path)

    ChunkACPolicyConfig.ensure_registered()
    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert loaded.actor_bc_weight_mode == "disagreement"
    assert loaded.actor_bc_uncertainty_tau_low == pytest.approx(0.015)
    assert loaded.actor_bc_uncertainty_tau_high == pytest.approx(0.019)
    assert loaded.actor_bc_uncertainty_kappa == pytest.approx(3.0)
    assert loaded.actor_bc_uncertainty_threshold_mode == "ema_quantile"
    assert loaded.actor_bc_uncertainty_ema_decay == pytest.approx(0.9)
    assert loaded.actor_bc_uncertainty_reset_ema_on_load is True
    assert loaded.critic_bootstrap_mode == "fixed_bernoulli"
    assert loaded.critic_bootstrap_keep_prob == pytest.approx(0.75)
    assert loaded.critic_bootstrap_seed == 77
    assert loaded.diagnostics_jsonl_path == "/tmp/evo-rlt-diagnostics.jsonl"
