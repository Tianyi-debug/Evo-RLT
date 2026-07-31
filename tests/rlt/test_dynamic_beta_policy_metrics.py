from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic


def _head_only_policy() -> ChunkACPolicy:
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
        assert info[key].shape == ()
        assert torch.isfinite(info[key])


def test_actor_only_backward_does_not_accumulate_critic_gradients():
    policy = _head_only_policy()
    tx = policy._coerce_batch(_batch())

    actor_loss, _ = policy._actor_loss_without_critic_grads(tx)
    actor_loss.backward()

    assert any(parameter.grad is not None for parameter in policy.actor.parameters())
    assert all(parameter.grad is None for parameter in policy.critic.parameters())
