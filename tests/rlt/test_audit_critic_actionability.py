from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from evo_rlt.cli.audit_critic_actionability import (
    _actor_actions,
    _bootstrap_statistic,
    _collate,
    _rir_combined,
    _state_digest,
    _virtual_q_update,
)
from evo_rlt.cli.audit_actor_q_mechanism import _canonical_policy_config
from evo_rlt.cli.audit_matched_actor_refinement import _pair_summary
from evo_rlt.core.actor import ChunkActor


class _LinearTwinCritic(nn.Module):
    def __init__(self, sign: float):
        super().__init__()
        self.register_buffer("sign", torch.tensor(sign))

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        value = self.sign * actions.sum(dim=-1, keepdim=True)
        return value, value


class _ZeroTwinCritic(nn.Module):
    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        # Keep a zero-valued autograd path to the actor action.
        value = actions.sum(dim=-1, keepdim=True) * 0.0
        return value, value


def _actor() -> ChunkActor:
    torch.manual_seed(4)
    return ChunkActor(
        state_dim=4,
        chunk_dim=6,
        hidden_dim=12,
        num_layers=1,
        fixed_std=0.02,
        ref_dropout_p=0.0,
        proprio_dim=0,
        state_normalization="none",
        action_residual=True,
        delta_scale_per_action_dim=[0.2, 0.3],
    ).eval()


def _inputs():
    states = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4], [-0.2, 0.1, -0.1, 0.4]],
        dtype=torch.float32,
    )
    proposals = torch.tensor(
        [[0.1, -0.1, 0.0, 0.2, -0.2, 0.3], [0.0, 0.1, -0.2, 0.2, 0.1, -0.1]],
        dtype=torch.float32,
    )
    return states, proposals


def _update(actor, critic):
    states, proposals = _inputs()
    return _virtual_q_update(
        actor=actor,
        critic=critic,
        score_mode="min",
        states=states,
        proposals=proposals,
        lambda_q=0.25,
        actor_lr=5e-3,
    )[0]


def test_virtual_directions_share_theta0_and_do_not_pollute_models():
    actor = _actor()
    critic_1 = _LinearTwinCritic(1.0)
    critic_2 = _LinearTwinCritic(1.0)
    actor_before = _state_digest(actor)
    critic_1_before, critic_2_before = _state_digest(critic_1), _state_digest(critic_2)

    updated_1 = _update(actor, critic_1)
    updated_2 = _update(actor, critic_2)

    assert _state_digest(actor) == actor_before
    assert _state_digest(critic_1) == critic_1_before
    assert _state_digest(critic_2) == critic_2_before
    # Identical critics produce identical clones, verifying that each direction
    # starts from the same theta0 and fresh optimizer state.
    assert _state_digest(updated_1) == _state_digest(updated_2)


def test_controlled_self_gain_and_aligned_cross_rir_are_positive():
    actor = _actor()
    critic_1, critic_2 = _LinearTwinCritic(1.0), _LinearTwinCritic(0.5)
    states, proposals = _inputs()
    base = _actor_actions(actor, states, proposals)
    action_1 = _actor_actions(_update(actor, critic_1), states, proposals)
    action_2 = _actor_actions(_update(actor, critic_2), states, proposals)
    self_gain = critic_1(states, action_1)[0] - critic_1(states, base)[0]
    cross_12 = critic_2(states, action_1)[0] - critic_2(states, base)[0]
    cross_21 = critic_1(states, action_2)[0] - critic_1(states, base)[0]
    records = [
        {
            "cross_gain_1_to_2": float(cross_12[i].detach()),
            "cross_gain_2_to_1": float(cross_21[i].detach()),
        }
        for i in range(len(states))
    ]
    assert torch.all(self_gain > 0)
    assert _rir_combined(records) == 1.0


def test_opposed_critics_have_low_cross_rir():
    actor = _actor()
    critic_1, critic_2 = _LinearTwinCritic(1.0), _LinearTwinCritic(-1.0)
    states, proposals = _inputs()
    base = _actor_actions(actor, states, proposals)
    action_1 = _actor_actions(_update(actor, critic_1), states, proposals)
    action_2 = _actor_actions(_update(actor, critic_2), states, proposals)
    records = [
        {
            "cross_gain_1_to_2": float((
                critic_2(states[i : i + 1], action_1[i : i + 1])[0]
                - critic_2(states[i : i + 1], base[i : i + 1])[0]
            ).detach()),
            "cross_gain_2_to_1": float((
                critic_1(states[i : i + 1], action_2[i : i + 1])[0]
                - critic_1(states[i : i + 1], base[i : i + 1])[0]
            ).detach()),
        }
        for i in range(len(states))
    ]
    assert _rir_combined(records) == 0.0


def test_zero_gradient_critic_is_safe_and_leaves_actor_unchanged():
    actor = _actor()
    updated = _update(actor, _ZeroTwinCritic())
    assert _state_digest(updated) == _state_digest(actor)


def test_episode_bootstrap_is_reproducible():
    records = [
        {"episode_uid": "a", "value": 1.0},
        {"episode_uid": "a", "value": 3.0},
        {"episode_uid": "b", "value": 7.0},
    ]
    statistic = lambda sample: sum(row["value"] for row in sample) / len(sample)
    first = _bootstrap_statistic(records, statistic, seed=17, replicates=100)
    second = _bootstrap_statistic(records, statistic, seed=17, replicates=100)
    assert first == second
    assert first["replicates"] == 100


def test_residual_bounds_and_chunk_shape_survive_virtual_update():
    actor = _actor()
    states, proposals = _inputs()
    updated = _update(actor, _LinearTwinCritic(1.0))
    action = _actor_actions(updated, states, proposals)
    lower, upper = actor.residual_reachable_interval(proposals)
    assert action.shape == (2, 3 * 2)
    assert torch.all(action >= lower - 1e-7)
    assert torch.all(action <= upper + 1e-7)


def test_collate_preserves_current_critic_and_actor_q_masks():
    template = {
        "state_vec": torch.zeros(4),
        "exec_chunk": torch.zeros(3, 2),
        "proposal_chunk": torch.zeros(3, 2),
        "next_state_vec": torch.zeros(4),
        "next_proposal_chunk": torch.zeros(3, 2),
        "reward_seq": torch.zeros(3),
        "done": torch.tensor(0.0),
        "actual_steps": torch.tensor(3),
        "bootstrap_mask": torch.tensor(1.0),
    }
    rows = [
        {**copy.deepcopy(template), "critic_mask": torch.tensor(1.0), "actor_q_mask": torch.tensor(0.0)},
        {**copy.deepcopy(template), "critic_mask": torch.tensor(0.0), "actor_q_mask": torch.tensor(1.0)},
    ]
    batch = _collate(rows, torch.tensor([1, 0]), torch.device("cpu"))
    assert batch["critic_mask"].tolist() == [0.0, 1.0]
    assert batch["actor_q_mask"].tolist() == [1.0, 0.0]


@pytest.mark.parametrize("seed", [1, 8])
def test_audit_batch_seed_has_deterministic_order(seed):
    first = torch.randperm(20, generator=torch.Generator().manual_seed(seed))
    second = torch.randperm(20, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(first, second)


def test_matched_config_canonicalizes_equal_legacy_horizon_alias():
    old = {"corrective_risk_horizon_chunks": 3, "actor_q_trust_mode": "fixed"}
    new = {
        **old,
        "corrective_risk_horizon_anchors": 3,
    }
    assert _canonical_policy_config(old) == _canonical_policy_config(new)


def test_three_way_pair_summary_preserves_chunk_and_dimension_diagnostics():
    records = [
        {
            "episode_uid": "e0",
            "delta_action": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "normalized_delta": torch.tensor([0.5, 1.0, 1.5, 2.0]),
            "whole_chunk_l2": 30.0**0.5,
            "q_delta": 0.2,
        },
        {
            "episode_uid": "e1",
            "delta_action": torch.tensor([2.0, 1.0, 4.0, 3.0]),
            "normalized_delta": torch.tensor([1.0, 0.5, 2.0, 1.5]),
            "whole_chunk_l2": 30.0**0.5,
            "q_delta": -0.1,
        },
    ]
    summary = _pair_summary(
        records,
        action_dim=2,
        chunk_length=2,
        seed=3,
        bootstrap_reps=20,
    )
    assert summary["samples"] == 2
    assert len(summary["shift_rmse_per_action_dimension"]) == 2
    assert len(summary["shift_rmse_per_chunk_timestep"]) == 2
    assert summary["gripper_dimension"] == 1
    assert summary["fraction_q_delta_positive"] == 0.5
