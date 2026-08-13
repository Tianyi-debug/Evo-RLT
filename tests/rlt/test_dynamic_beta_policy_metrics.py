from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic


def _head_only_policy(
    *,
    ema: bool = False,
    diagnostics_jsonl_path: str | None = None,
    bootstrap: bool = False,
) -> ChunkACPolicy:
    policy = object.__new__(ChunkACPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        gamma=0.99,
        chunk_length=2,
        target_q_clip=100.0,
        tau=0.005,
        actor_update_interval=1,
        beta=0.3,
        actor_bc_weight_mode="disagreement",
        actor_bc_uncertainty_tau_low=0.0,
        actor_bc_uncertainty_tau_high=0.2,
        actor_bc_uncertainty_kappa=3.0,
        actor_bc_uncertainty_threshold_mode="ema_quantile" if ema else "fixed",
        actor_bc_uncertainty_ema_decay=0.5,
        actor_bc_uncertainty_min_gap=1e-6,
        actor_bc_uncertainty_reset_ema_on_load=False,
        critic_bootstrap_mode="fixed_bernoulli" if bootstrap else "none",
        critic_bootstrap_keep_prob=0.8,
        critic_bootstrap_seed=1000,
        diagnostics_jsonl_path=diagnostics_jsonl_path,
    )
    policy.actor = ChunkActor(
        state_dim=6,
        chunk_dim=4,
        hidden_dim=8,
        num_layers=2,
        proprio_dim=2,
        state_normalization="rl_token_layer_norm",
        action_residual=True,
    )
    policy.critic = TwinCritic(
        state_dim=6,
        chunk_dim=4,
        hidden_dim=8,
        num_layers=2,
        proprio_dim=2,
        state_normalization="rl_token_layer_norm",
    )
    policy.target_critic = copy.deepcopy(policy.critic)
    for parameter in policy.target_critic.parameters():
        parameter.requires_grad_(False)
    policy.register_buffer("_critic_step", torch.zeros((), dtype=torch.long))
    policy.register_buffer("_human_bc_step", torch.zeros((), dtype=torch.long))
    policy.register_buffer("_actor_bc_tau_low_ema", torch.tensor(0.0))
    policy.register_buffer("_actor_bc_tau_high_ema", torch.tensor(0.2))
    policy._diagnostics_jsonl_initialized = False
    return policy


def _batch() -> dict[str, torch.Tensor]:
    batch_size = 8
    return {
        "state_vec": torch.randn(batch_size, 6),
        "exec_chunk": torch.randn(batch_size, 2, 2),
        "ref_chunk": torch.randn(batch_size, 2, 2),
        "reward_seq": torch.zeros(batch_size, 2),
        "next_state_vec": torch.randn(batch_size, 6),
        "next_ref_chunk": torch.randn(batch_size, 2, 2),
        "done": torch.zeros(batch_size),
        "actual_steps": torch.full((batch_size,), 2),
        "source": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        "cache_index": torch.arange(batch_size),
    }


def test_dynamic_policy_forward_emits_finite_metrics():
    policy = _head_only_policy()

    loss, info = policy.forward(_batch())

    assert torch.isfinite(loss)
    assert info is not None
    for key in (
        "loss_actor_q",
        "loss_actor_bc_raw",
        "actor_disagreement_p95",
        "actor_beta_mean",
        "critic_target_mean",
        "critic_td_abs_mean",
        "source_3_beta_mean",
    ):
        assert isinstance(info[key], float)
        assert math.isfinite(info[key])
    assert "loss" not in info
    assert isinstance(info["loss_total_step"], float)
    assert info["actor_update"] is True


def test_actor_only_backward_does_not_accumulate_critic_gradients():
    policy = _head_only_policy()
    tx = policy._coerce_batch(_batch())

    actor_loss, _ = policy._actor_loss_without_critic_grads(tx)
    actor_loss.backward()

    assert any(parameter.grad is not None for parameter in policy.actor.parameters())
    assert all(parameter.grad is None for parameter in policy.critic.parameters())


def test_human_bc_forward_is_actor_only_and_rejects_non_human_batches():
    policy = _head_only_policy()
    policy.config.training_stage = "human_bc"
    for parameter in policy.critic.parameters():
        parameter.requires_grad_(False)
    batch = _batch()
    batch["source"] = torch.full((8,), 3)
    batch["proposal_chunk"] = batch.pop("ref_chunk")
    batch["bc_target_chunk"] = batch["proposal_chunk"] + 0.1
    batch["next_proposal_chunk"] = batch.pop("next_ref_chunk")

    loss, info = policy.forward(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert info["human_bc_stage"] is True
    assert info["human_bc_step"] == 1
    assert policy._critic_step.item() == 0
    assert info["human_sample_frac"] == pytest.approx(1.0)
    assert info["source_3_frac"] == pytest.approx(1.0)
    assert any(parameter.grad is not None for parameter in policy.actor.parameters())
    assert all(parameter.grad is None for parameter in policy.critic.parameters())
    optim_params = policy.get_optim_params()
    assert len(optim_params) == 1
    assert list(optim_params[0]["params"]) == list(policy.actor.parameters())

    batch["source"][0] = 1
    with pytest.raises(ValueError, match="accepts only source=3"):
        policy.forward(batch)


def test_ema_thresholds_are_used_before_batch_quantiles_update_them():
    policy = _head_only_policy(ema=True)
    tx = policy._coerce_batch(_batch())

    _, info = policy._actor_loss_without_critic_grads(tx)

    assert info["actor_tau_low_used"].item() == pytest.approx(0.0)
    assert info["actor_tau_high_used"].item() == pytest.approx(0.2)
    expected_low = 0.5 * info["actor_disagreement_p50"].item()
    expected_high = 0.5 * 0.2 + 0.5 * info["actor_disagreement_p95"].item()
    expected_high = max(expected_high, expected_low + 1e-6)
    assert policy._actor_bc_tau_low_ema.item() == pytest.approx(expected_low)
    assert policy._actor_bc_tau_high_ema.item() == pytest.approx(expected_high)
    assert info["actor_tau_low_updated"].item() == pytest.approx(expected_low)
    assert info["actor_tau_high_updated"].item() == pytest.approx(expected_high)


def test_ema_threshold_buffers_round_trip_with_state_dict():
    policy = _head_only_policy(ema=True)
    policy._actor_bc_tau_low_ema.fill_(0.123)
    policy._actor_bc_tau_high_ema.fill_(0.456)
    state = copy.deepcopy(policy.state_dict())

    restored = _head_only_policy(ema=True)
    restored.load_state_dict(state)

    assert restored._actor_bc_tau_low_ema.item() == pytest.approx(0.123)
    assert restored._actor_bc_tau_high_ema.item() == pytest.approx(0.456)


def test_ema_thresholds_can_be_reset_once_after_checkpoint_load(caplog):
    policy = _head_only_policy(ema=True)
    policy._actor_bc_tau_low_ema.fill_(0.0)
    policy._actor_bc_tau_high_ema.fill_(1.0)
    policy.config.actor_bc_uncertainty_tau_low = 0.0025
    policy.config.actor_bc_uncertainty_tau_high = 0.005
    policy.config.actor_bc_uncertainty_reset_ema_on_load = True

    with caplog.at_level("INFO"):
        applied = policy._reset_actor_uncertainty_ema_from_config()

    assert applied is True
    assert policy._actor_bc_tau_low_ema.item() == pytest.approx(0.0025)
    assert policy._actor_bc_tau_high_ema.item() == pytest.approx(0.005)
    assert policy.config.actor_bc_uncertainty_reset_ema_on_load is False
    assert "[0, 1] -> [0.0025, 0.005]" in caplog.text

    policy._actor_bc_tau_low_ema.fill_(0.003)
    policy._actor_bc_tau_high_ema.fill_(0.006)
    assert policy._reset_actor_uncertainty_ema_from_config() is False
    assert policy._actor_bc_tau_low_ema.item() == pytest.approx(0.003)
    assert policy._actor_bc_tau_high_ema.item() == pytest.approx(0.006)


def test_from_pretrained_applies_ema_reset_after_parent_load(monkeypatch):
    policy = _head_only_policy(ema=True)
    policy._actor_bc_tau_low_ema.fill_(0.0)
    policy._actor_bc_tau_high_ema.fill_(1.0)
    policy.config.actor_bc_uncertainty_tau_low = 0.0025
    policy.config.actor_bc_uncertainty_tau_high = 0.005
    policy.config.actor_bc_uncertainty_reset_ema_on_load = True

    def fake_parent_load(cls, pretrained_name_or_path, *, config, **kwargs):
        assert cls is ChunkACPolicy
        assert pretrained_name_or_path == "/tmp/fixed-checkpoint"
        assert policy._actor_bc_tau_low_ema.item() == pytest.approx(0.0)
        assert policy._actor_bc_tau_high_ema.item() == pytest.approx(1.0)
        return policy

    monkeypatch.setattr(
        PreTrainedPolicy,
        "from_pretrained",
        classmethod(fake_parent_load),
    )

    loaded = ChunkACPolicy.from_pretrained(
        "/tmp/fixed-checkpoint",
        config=policy.config,
    )

    assert loaded is policy
    assert loaded._actor_bc_tau_low_ema.item() == pytest.approx(0.0025)
    assert loaded._actor_bc_tau_high_ema.item() == pytest.approx(0.005)
    assert loaded.config.actor_bc_uncertainty_reset_ema_on_load is False


def test_ema_reset_state_dict_round_trip_preserves_learned_thresholds():
    policy = _head_only_policy(ema=True)
    policy.config.actor_bc_uncertainty_tau_low = 0.0025
    policy.config.actor_bc_uncertainty_tau_high = 0.005
    policy.config.actor_bc_uncertainty_reset_ema_on_load = True
    policy._actor_bc_tau_low_ema.fill_(0.0)
    policy._actor_bc_tau_high_ema.fill_(1.0)
    policy._reset_actor_uncertainty_ema_from_config()
    policy._actor_bc_tau_low_ema.fill_(0.003)
    policy._actor_bc_tau_high_ema.fill_(0.006)
    state = copy.deepcopy(policy.state_dict())

    restored = _head_only_policy(ema=True)
    restored.load_state_dict(state)
    restored.config.actor_bc_uncertainty_reset_ema_on_load = False
    assert restored._reset_actor_uncertainty_ema_from_config() is False
    assert restored._actor_bc_tau_low_ema.item() == pytest.approx(0.003)
    assert restored._actor_bc_tau_high_ema.item() == pytest.approx(0.006)


def test_jsonl_records_every_critic_step_and_actor_frequency(tmp_path):
    path = tmp_path / "diagnostics" / "train.jsonl"
    policy = _head_only_policy(
        ema=True,
        diagnostics_jsonl_path=str(path),
        bootstrap=True,
    )
    policy.config.actor_update_interval = 2

    _, first_info = policy.forward(_batch())
    _, second_info = policy.forward(_batch())
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert first_info["actor_update"] is False
    assert second_info["actor_update"] is True
    assert [row["critic_step"] for row in rows] == [1, 2]
    assert [row["actor_update"] for row in rows] == [False, True]
    assert "loss_actor" not in rows[0]
    assert rows[0]["source_0_frac"] == pytest.approx(0.25)
    assert rows[0]["source_3_frac"] == pytest.approx(0.25)
    assert "actor_beta_mean" in rows[1]
    assert "actor_tau_low_used" in rows[1]
    assert "critic_bootstrap_overlap_frac" in rows[0]
    assert all(
        not isinstance(value, (list, dict))
        for row in rows
        for value in row.values()
    )
