from __future__ import annotations

import copy

import torch
import pytest

from evo_rlt.core.losses import critic_loss, actor_loss
from evo_rlt.core.utils import soft_update, flatten_chunk, compute_discount_vector
from evo_rlt.core.interfaces import Observation, ChunkTransition
from evo_rlt.core.replay_buffer import ReplayBuffer
from tests.rlt.helpers import (
    make_test_algorithm, make_batch, make_fake_transition,
    TOKEN_DIM, ACTION_DIM, PROPRIO_DIM, C, STATE_DIM, CHUNK_DIM,
)


@pytest.fixture
def algorithm():
    algo, _ = make_test_algorithm()
    return algo


@pytest.fixture
def batch():
    return make_batch()


def test_single_critic_step_loss_not_nan(algorithm, batch):
    """Single critic gradient step produces finite loss."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=1e-3)
    loss = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, gamma=0.99, C=C)
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify params changed
    loss2 = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, gamma=0.99, C=C)
    assert not torch.isnan(loss2)


def test_single_actor_step_loss_not_nan(algorithm, batch):
    """Single actor gradient step produces finite loss."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=1e-3)
    loss = actor_loss(algorithm.policy.actor, algorithm.critic, batch, beta=1.0)
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_frozen_params_stay_frozen(algorithm, batch):
    """After freezing VLA and encoder, their params should not get gradients."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    # Run both losses
    c_loss = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, gamma=0.99, C=C)
    c_loss.backward()

    # VLA params: no grad
    for p in algorithm.policy.vla.parameters():
        assert p.grad is None

    # RL token encoder params: no grad
    assert algorithm.policy.rl_token.rl_token_embed.grad is None
    for p in algorithm.policy.rl_token.encoder.parameters():
        assert p.grad is None


def test_full_forward_backward_pipeline(algorithm):
    """End-to-end: obs -> state -> action -> critic -> loss -> backward."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    obs = Observation(
        images={"base": torch.randn(4, 3, 64, 64)},
        proprio=torch.randn(4, PROPRIO_DIM),
    )

    with torch.no_grad():
        state = algorithm.policy.get_rl_state(obs)
        ref = algorithm.policy.get_reference_chunk(obs)
        ref_flat = flatten_chunk(ref)

    action, mu = algorithm.policy.actor.sample(state, ref_flat, training=True)
    q = algorithm.critic.min_q(state, action)
    loss = -q.mean()
    loss.backward()

    # Actor should have gradients
    for p in algorithm.policy.actor.parameters():
        assert p.grad is not None

    # Critic should have gradients (from Q computation)
    for p in algorithm.critic.parameters():
        assert p.grad is not None


def test_soft_update_changes_target(algorithm):
    """Verify soft update actually modifies target critic."""
    orig_params = [p.data.clone() for p in algorithm.target_critic.parameters()]

    # Perturb online critic
    for p in algorithm.critic.parameters():
        p.data += torch.randn_like(p.data) * 0.1

    soft_update(algorithm.target_critic, algorithm.critic, tau=0.1)

    changed = False
    for orig, p in zip(orig_params, algorithm.target_critic.parameters()):
        if not torch.allclose(orig, p.data):
            changed = True
            break
    assert changed


def test_compute_discount_vector():
    v = compute_discount_vector(0.99, 5)
    assert v.shape == (5,)
    assert torch.allclose(v[0], torch.tensor(1.0))
    assert torch.allclose(v[1], torch.tensor(0.99))
    assert torch.allclose(v[4], torch.tensor(0.99 ** 4))


def test_flatten_chunk():
    chunk = torch.randn(3, 10, 14)
    flat = flatten_chunk(chunk)
    assert flat.shape == (3, 140)


def test_replay_buffer_integration():
    """Test that buffer -> sample -> loss pipeline works end-to-end."""
    buf = ReplayBuffer(capacity=100)
    for _ in range(20):
        buf.add(make_fake_transition())

    from evo_rlt.core.actor import ChunkActor
    from evo_rlt.core.critic import TwinCritic

    actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1)
    critic = TwinCritic(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1)
    target_critic = copy.deepcopy(critic)

    batch = buf.sample(8)
    c_loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
    a_loss = actor_loss(actor, critic, batch, beta=1.0)

    assert not torch.isnan(c_loss)
    assert not torch.isnan(a_loss)


def test_lerobot_ac_actor_loss_does_not_add_critic_grads(batch):
    from types import SimpleNamespace

    from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    from evo_rlt.core.actor import ChunkActor
    from evo_rlt.core.critic import TwinCritic

    torch.manual_seed(0)
    policy = object.__new__(ChunkACPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        gamma=0.99,
        chunk_length=C,
        tau=0.005,
        actor_update_interval=2,
        beta=1.0,
        target_q_clip=100.0,
    )
    policy.actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=2)
    policy.critic = TwinCritic(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=2)
    policy.target_critic = copy.deepcopy(policy.critic)
    policy.register_buffer("_critic_step", torch.ones((), dtype=torch.long))

    lerobot_batch = dict(batch)
    lerobot_batch["exec_chunk"] = lerobot_batch.pop("exec_chunk_flat").view(-1, C, ACTION_DIM)
    lerobot_batch["ref_chunk"] = lerobot_batch.pop("ref_chunk_flat").view(-1, C, ACTION_DIM)
    lerobot_batch["next_ref_chunk"] = lerobot_batch.pop("next_ref_flat").view(-1, C, ACTION_DIM)

    critic_only = copy.deepcopy(policy.critic)
    target_only = copy.deepcopy(policy.target_critic)
    actor_only = copy.deepcopy(policy.actor)
    c_loss = critic_loss(
        critic_only, target_only, actor_only, batch,
        gamma=0.99, C=C, target_q_clip=100.0,
    )
    c_loss.backward()
    expected = {name: p.grad.clone() for name, p in critic_only.named_parameters()}

    loss, info = policy.forward(lerobot_batch)
    assert "loss_actor" in info
    loss.backward()

    for name, p in policy.critic.named_parameters():
        assert torch.allclose(p.grad, expected[name], atol=1e-6, rtol=1e-5)


def test_lerobot_ac_critic_only_forward_never_updates_actor(batch):
    from types import SimpleNamespace

    from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    from evo_rlt.core.actor import ChunkActor
    from evo_rlt.core.critic import TwinCritic

    policy = object.__new__(ChunkACPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        training_stage="critic_only",
        gamma=0.99,
        chunk_length=C,
        tau=0.005,
        actor_update_interval=2,
        beta=1.0,
        target_q_clip=100.0,
    )
    policy.actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=2)
    policy.critic = TwinCritic(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=2)
    policy.target_critic = copy.deepcopy(policy.critic)
    policy.register_buffer("_critic_step", torch.zeros((), dtype=torch.long))

    lerobot_batch = dict(batch)
    lerobot_batch["exec_chunk"] = lerobot_batch.pop("exec_chunk_flat").view(-1, C, ACTION_DIM)
    lerobot_batch["ref_chunk"] = lerobot_batch.pop("ref_chunk_flat").view(-1, C, ACTION_DIM)
    lerobot_batch["next_ref_chunk"] = lerobot_batch.pop("next_ref_flat").view(
        -1, C, ACTION_DIM
    )

    loss, info = policy.forward(lerobot_batch)
    loss.backward()

    assert info["critic_only_stage"] is True
    assert info["actor_update"] is False
    assert info["critic_step"] == 1
    assert all(parameter.grad is None for parameter in policy.actor.parameters())
    assert any(parameter.grad is not None for parameter in policy.critic.parameters())


def test_lerobot_ac_optimizer_groups_use_independent_learning_rates():
    from types import SimpleNamespace

    from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    from evo_rlt.core.actor import ChunkActor
    from evo_rlt.core.critic import TwinCritic

    policy = object.__new__(ChunkACPolicy)
    torch.nn.Module.__init__(policy)
    policy.actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=2)
    policy.critic = TwinCritic(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=2)
    policy.config = SimpleNamespace(
        training_stage="mixed_ac",
        actor_lr=1e-5,
        critic_lr=1e-4,
    )

    groups = policy.get_optim_params()
    assert len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(1e-5)
    assert groups[1]["lr"] == pytest.approx(1e-4)
    assert {id(parameter) for parameter in groups[0]["params"]} == {
        id(parameter) for parameter in policy.actor.parameters()
    }
    assert {id(parameter) for parameter in groups[1]["params"]} == {
        id(parameter) for parameter in policy.critic.parameters()
    }

    policy.config.training_stage = "critic_only"
    critic_only_groups = policy.get_optim_params()
    assert len(critic_only_groups) == 1
    assert critic_only_groups[0]["lr"] == pytest.approx(1e-4)
    assert {id(parameter) for parameter in critic_only_groups[0]["params"]} == {
        id(parameter) for parameter in policy.critic.parameters()
    }
    policy.config.training_stage = "human_bc"
    human_bc_groups = policy.get_optim_params()
    assert len(human_bc_groups) == 1
    assert human_bc_groups[0]["lr"] == pytest.approx(1e-5)
    assert {id(parameter) for parameter in human_bc_groups[0]["params"]} == {
        id(parameter) for parameter in policy.actor.parameters()
    }
