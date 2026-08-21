from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from safetensors.torch import save_file as save_safetensors_file

from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic
from evo_rlt.core.corrective_risk import (
    CorrectiveTakeoverRiskMLP,
    save_corrective_risk_checkpoint,
)


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
        action_dim=2,
        human_bc_target_mode="raw",
        human_bc_weight=1.0,
        teacher_distillation_weight=1.0,
        actor_human_weight=1.0,
        actor_teacher_weight=1.0,
        actor_q_weight_max=0.0,
        actor_q_trust_mode="fixed",
        corrective_risk_checkpoint="",
        corrective_risk_horizon_chunks=3,
        rl_token_dim=4,
        proprio_dim=2,
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
    policy.register_buffer("_teacher_bc_step", torch.zeros((), dtype=torch.long))
    policy.register_buffer("_actor_refine_step", torch.zeros((), dtype=torch.long))
    policy.register_buffer(
        "_actor_refine_batch_fingerprint", torch.zeros((), dtype=torch.long)
    )
    policy.register_buffer("_actor_bc_tau_low_ema", torch.tensor(0.0))
    policy.register_buffer("_actor_bc_tau_high_ema", torch.tensor(0.2))
    policy._diagnostics_jsonl_initialized = False
    object.__setattr__(policy, "_teacher_actor", None)
    object.__setattr__(policy, "_corrective_risk_head", None)
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


def test_policy_coerce_batch_preserves_semantic_v2_bootstrap_fields():
    policy = _head_only_policy()
    batch = _batch()
    batch["bootstrap_mask"] = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
    batch["cache_semantics_version"] = torch.full((8,), 2)

    tx = policy._coerce_batch(batch)

    assert torch.equal(tx["bootstrap_mask"], batch["bootstrap_mask"])
    assert torch.equal(
        tx["cache_semantics_version"], batch["cache_semantics_version"]
    )


def test_policy_forwards_zero_q_weight_to_actor_loss():
    policy = _head_only_policy()
    policy.config.actor_q_weight = 0.0

    _, info = policy.forward(_batch())

    assert info["actor_q_weight"] == pytest.approx(0.0)
    assert info["loss_actor_q"] == pytest.approx(0.0)
    assert info["loss_actor_q_weighted"] == pytest.approx(0.0)
    assert math.isfinite(info["loss_actor_q_raw"])


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


def test_teacher_bc_uses_disjoint_semantic_masks_and_frozen_actor(tmp_path):
    policy = _head_only_policy()
    teacher_state = {
        f"actor.{key}": value.detach().clone()
        for key, value in policy.actor.state_dict().items()
    }
    teacher_dir = tmp_path / "teacher" / "pretrained_model"
    teacher_dir.mkdir(parents=True)
    save_safetensors_file(teacher_state, teacher_dir / "model.safetensors")

    with torch.no_grad():
        for parameter in policy.actor.parameters():
            parameter.add_(0.05)
    policy.config.training_stage = "teacher_bc"
    policy.config.actor_q_weight = 0.0
    policy.config.actor_teacher_pretrained_path = str(teacher_dir)
    policy.config.teacher_distillation_weight = 1.0
    policy.config.human_bc_weight = 1.0
    for parameter in policy.critic.parameters():
        parameter.requires_grad_(False)

    batch = _batch()
    batch["source"] = torch.tensor([0, 0, 1, 2, 2, 2, 2, 3])
    batch["actor_bc_mask"] = torch.tensor([1, 1, 1, 1, 0, 0, 0, 1])
    batch["intervention_reason"] = torch.tensor([0, 0, 0, 0, 0, 1, 2, 1])
    batch["proposal_chunk"] = batch.pop("ref_chunk")
    batch["next_proposal_chunk"] = batch.pop("next_ref_chunk")
    batch["bc_target_chunk"] = batch["proposal_chunk"].clone()
    batch["bc_target_chunk"][-1] += 0.1

    tx = policy._coerce_batch(batch)
    masks = policy._teacher_supervision_masks(tx)
    assert masks["teacher"].tolist() == [True, True, True, True, False, False, True, False]
    assert masks["human"].tolist() == [False, False, False, False, False, False, False, True]
    assert masks["ignored"].tolist() == [False, False, False, False, True, True, False, False]
    assert not bool((masks["teacher"] & masks["human"]).any())

    loss, info = policy.forward(batch)
    loss.backward()

    teacher = policy._teacher_actor
    assert teacher is not None
    assert "_teacher_actor" not in dict(policy.named_modules())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(parameter.grad is not None for parameter in policy.actor.parameters())
    assert all(parameter.grad is None for parameter in policy.critic.parameters())
    assert info["teacher_bc_stage"] is True
    assert info["teacher_bc_step"] == 1
    assert info["teacher_sample_frac"] == pytest.approx(5 / 8)
    assert info["human_sample_frac"] == pytest.approx(1 / 8)
    assert info["teacher_ignored_sample_frac"] == pytest.approx(2 / 8)
    assert info["teacher_corrective_prefix_sample_frac"] == pytest.approx(1 / 8)
    assert info["teacher_proactive_prefix_sample_frac"] == pytest.approx(1 / 8)
    assert info["loss_actor_teacher_raw"] > 0
    assert info["loss_actor_human_bc_raw"] > 0

    optim_params = policy.get_optim_params()
    assert len(optim_params) == 1
    assert list(optim_params[0]["params"]) == list(policy.actor.parameters())


def test_teacher_bc_rejects_cache_without_typed_masks():
    policy = _head_only_policy()
    policy.config.training_stage = "teacher_bc"
    policy.config.actor_teacher_pretrained_path = "/tmp/unused"
    policy.config.actor_q_weight = 0.0

    with pytest.raises(KeyError, match="actor_bc_mask"):
        policy.forward(_batch())


def _configure_actor_refine(policy, tmp_path, *, q_weight: float = 0.0):
    teacher_state = {
        f"actor.{key}": value.detach().clone()
        for key, value in policy.actor.state_dict().items()
    }
    teacher_dir = tmp_path / "teacher" / "pretrained_model"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    save_safetensors_file(teacher_state, teacher_dir / "model.safetensors")
    policy.config.training_stage = "actor_refine"
    policy.config.actor_bc_weight_mode = "fixed"
    policy.config.actor_teacher_pretrained_path = str(teacher_dir)
    policy.config.actor_q_weight_max = q_weight
    for parameter in policy.critic.parameters():
        parameter.requires_grad_(False)
    batch = _batch()
    batch["source"] = torch.tensor([0, 0, 1, 2, 2, 2, 2, 3])
    batch["actor_bc_mask"] = torch.tensor([1, 1, 1, 1, 0, 0, 0, 1])
    batch["actor_q_mask"] = torch.tensor([1, 1, 1, 1, 0, 0, 1, 0])
    batch["intervention_reason"] = torch.tensor([0, 0, 0, 0, 0, 1, 2, 1])
    batch["proposal_chunk"] = batch.pop("ref_chunk")
    batch["next_proposal_chunk"] = batch.pop("next_ref_chunk")
    batch["bc_target_chunk"] = batch["proposal_chunk"].clone()
    batch["bc_target_chunk"][-1] += 0.1
    return batch


def test_actor_refine_q0_equals_teacher_human_and_fixed_q_math(tmp_path):
    policy = _head_only_policy()
    batch = _configure_actor_refine(policy, tmp_path, q_weight=0.0)
    tx = policy._coerce_batch(batch)

    q0_loss, q0_info = policy._forward_actor_refine(tx)
    teacher_loss, _ = policy._forward_teacher_bc(tx)
    assert q0_loss.item() == pytest.approx(teacher_loss.item(), abs=1e-7)
    assert q0_info["loss_actor_q_weighted"] == pytest.approx(0.0)
    assert q0_info["actor_q_trust_mean"] == pytest.approx(1.0)

    policy.config.actor_q_weight_max = 0.25
    fixed_loss, fixed_info = policy._forward_actor_refine(tx)
    with torch.no_grad():
        mu, _ = policy.actor(tx["state_vec"], tx["proposal_chunk_flat"], training=False)
        q = policy.critic.min_q(tx["state_vec"], mu).reshape(-1)
        mask = tx["actor_q_mask"].reshape(-1) > 0.5
        expected_q = -0.25 * q[mask].mean()
    assert fixed_info["loss_actor_q_weighted"] == pytest.approx(expected_q.item())
    assert fixed_loss.item() == pytest.approx(q0_loss.item() + expected_q.item())


def test_actor_refine_preserves_actor_q_gradient_and_freezes_other_heads(tmp_path):
    policy = _head_only_policy()
    batch = _configure_actor_refine(policy, tmp_path, q_weight=0.25)
    loss, _ = policy.forward(batch)
    loss.backward()

    assert any(parameter.grad is not None for parameter in policy.actor.parameters())
    assert all(parameter.grad is None for parameter in policy.critic.parameters())
    assert all(parameter.grad is None for parameter in policy._teacher_actor.parameters())
    assert list(policy.get_optim_params()[0]["params"]) == list(policy.actor.parameters())


def test_corrective_risk_trust_is_per_sample_and_detached(tmp_path):
    # Keep this autograd assertion independent from RNG consumed by earlier
    # training/audit tests in the same pytest process.
    torch.manual_seed(1234)
    policy = _head_only_policy()
    batch = _configure_actor_refine(policy, tmp_path, q_weight=0.25)
    risk = CorrectiveTakeoverRiskMLP(state_dim=6, action_dim=4, hidden_dims=(8, 4))
    with torch.no_grad():
        for parameter in risk.parameters():
            parameter.fill_(0.1)
    risk_path = tmp_path / "risk.pt"
    save_corrective_risk_checkpoint(
        risk_path,
        risk,
        {"primary_future_k": 3, "normalization": "none"},
    )
    policy.config.actor_q_trust_mode = "corrective_risk"
    policy.config.corrective_risk_checkpoint = str(risk_path)

    loss, info = policy.forward(batch)
    loss.backward()
    loaded_risk = policy._corrective_risk_head
    assert loaded_risk is not None
    assert info["actor_q_trust_p10"] != pytest.approx(info["actor_q_trust_p90"])
    assert all(parameter.grad is None for parameter in loaded_risk.parameters())
    assert all(parameter.grad is None for parameter in policy.critic.parameters())
    assert any(parameter.grad is not None for parameter in policy.actor.parameters())

    toy_action = torch.tensor([[0.2, -0.4]], requires_grad=True)
    toy_risk = torch.nn.Linear(2, 1, bias=False)
    toy_risk.weight.data.copy_(torch.tensor([[1.5, -0.5]]))
    direct_rho = toy_risk(toy_action).sigmoid()
    assert torch.autograd.grad(direct_rho.sum(), toy_action, retain_graph=True)[0].abs().sum() > 0
    with torch.no_grad():
        detached_trust = 1.0 - toy_risk(toy_action.detach()).sigmoid()
    objective = (detached_trust * toy_action.square().sum(dim=-1)).sum()
    detached_gradient = torch.autograd.grad(objective, toy_action)[0]
    assert torch.allclose(detached_gradient, 2.0 * detached_trust * toy_action)
    assert all(parameter.grad is None for parameter in toy_risk.parameters())


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


def test_projected_human_mode_changes_only_human_target_loss(tmp_path):
    policy = _head_only_policy()
    teacher_state = {
        f"actor.{key}": value.detach().clone()
        for key, value in policy.actor.state_dict().items()
    }
    teacher_dir = tmp_path / "teacher" / "pretrained_model"
    teacher_dir.mkdir(parents=True)
    save_safetensors_file(teacher_state, teacher_dir / "model.safetensors")
    policy.config.training_stage = "teacher_bc"
    policy.config.actor_q_weight = 0.0
    policy.config.actor_teacher_pretrained_path = str(teacher_dir)
    policy.config.teacher_distillation_weight = 1.0
    policy.config.human_bc_weight = 1.0

    batch = _batch()
    batch["source"] = torch.tensor([0, 0, 1, 2, 2, 2, 2, 3])
    batch["actor_bc_mask"] = torch.tensor([1, 1, 1, 1, 0, 0, 0, 1])
    batch["intervention_reason"] = torch.tensor([0, 0, 0, 0, 0, 1, 2, 1])
    batch["proposal_chunk"] = torch.zeros_like(batch.pop("ref_chunk"))
    batch["next_proposal_chunk"] = batch.pop("next_ref_chunk")
    batch["bc_target_chunk"] = batch["proposal_chunk"].clone()
    batch["bc_target_chunk"][-1] = 1.0

    policy.config.human_bc_target_mode = "raw"
    raw_loss, raw_info = policy.forward(batch)
    policy.config.human_bc_target_mode = "residual_feasible"
    projected_loss, projected_info = policy.forward(batch)

    assert projected_loss < raw_loss
    assert projected_info["loss_actor_human_bc_raw"] < raw_info[
        "loss_actor_human_bc_raw"
    ]
    assert projected_info["loss_actor_teacher_raw"] == pytest.approx(
        raw_info["loss_actor_teacher_raw"]
    )
    assert projected_info["human_projection_fraction"] == pytest.approx(1.0)
    assert projected_info["human_raw_target_rmse"] > projected_info[
        "human_feasible_target_rmse"
    ]


def test_legacy_raw_human_mode_keeps_unprojected_loss():
    policy = _head_only_policy()
    policy.config.training_stage = "human_bc"
    policy.config.human_bc_target_mode = "raw"
    batch = _batch()
    batch["source"] = torch.full((8,), 3)
    batch["proposal_chunk"] = torch.zeros_like(batch.pop("ref_chunk"))
    batch["next_proposal_chunk"] = batch.pop("next_ref_chunk")
    batch["bc_target_chunk"] = torch.full_like(batch["proposal_chunk"], 0.5)

    loss, info = policy.forward(batch)

    # Four flattened action values per sample, each with squared error 0.25.
    assert loss.item() == pytest.approx(1.0)
    assert info["human_projection_fraction"] == pytest.approx(1.0)
