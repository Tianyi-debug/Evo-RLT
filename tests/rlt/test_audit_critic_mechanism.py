from __future__ import annotations

import torch
from torch import nn

from evo_rlt.cli.audit_critic_mechanism import (
    _distribution_shift,
    _gradient_audit,
    _pairwise_episode_difference,
    _semantic_group,
)
from evo_rlt.core.actor import ChunkActor


class _LinearTwin(nn.Module):
    def __init__(self, first: torch.Tensor, second: torch.Tensor):
        super().__init__()
        self.register_buffer("first", first)
        self.register_buffer("second", second)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        del state
        return (action * self.first).sum(-1, keepdim=True), (action * self.second).sum(
            -1, keepdim=True
        ) - 0.1


def test_semantic_groups_keep_corrective_boundary_explicit():
    assert (
        _semantic_group(
            {"category": "corrective", "distance_to_corrective_event": 3}, 3
        )
        == "corrective_last_K"
    )
    assert (
        _semantic_group(
            {"category": "corrective", "distance_to_corrective_event": 4}, 3
        )
        == "corrective_earlier"
    )
    assert (
        _semantic_group(
            {"category": "proactive", "distance_to_event_anchors": 2}, 3
        )
        == "proactive_last_K_censored"
    )


def test_pairwise_difference_bootstraps_over_episodes_not_rows():
    left = [
        {"episode_uid": "a", "min_q": 3.0},
        {"episode_uid": "a", "min_q": 1.0},
        {"episode_uid": "b", "min_q": 2.0},
    ]
    right = [
        {"episode_uid": "c", "min_q": 0.0},
        {"episode_uid": "d", "min_q": 0.0},
    ]
    result = _pairwise_episode_difference(
        left, right, key="min_q", seed=7, replicates=200
    )
    assert result["mean_difference"] == 2.0
    assert result["episode_bootstrap_95ci"][0] > 0


def test_distribution_shift_flags_large_shift():
    generator = torch.Generator().manual_seed(2)
    reference_states = torch.randn(128, 4, generator=generator)
    reference_actions = torch.randn(128, 2, generator=generator)
    eval_states = torch.randn(32, 4, generator=generator) + 20.0
    eval_actions = torch.randn(32, 2, generator=generator) + 20.0
    report = _distribution_shift(
        reference_states,
        reference_actions,
        eval_states,
        eval_actions,
        proprio_dim=0,
        state_normalization="none",
        projection_dim=4,
        query_samples=32,
        seed=3,
    )
    assert report["heuristic"]["status"] == "STRONG INPUT-SHIFT SIGNAL"
    assert (
        report["standardized_rms"]["evaluation_fraction_above_reference_p99"] > 0.9
    )


def test_gradient_audit_detects_aligned_twin_local_ordering():
    actor = ChunkActor(
        state_dim=2,
        chunk_dim=2,
        hidden_dim=8,
        num_layers=1,
        action_residual=True,
        delta_scale=0.2,
        proprio_dim=0,
        state_normalization="none",
    )
    critic = _LinearTwin(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]))
    states = torch.zeros(4, 2)
    proposals = torch.zeros(4, 2)
    with torch.no_grad():
        actions, _ = actor(states, proposals)
    result = _gradient_audit(
        actor,
        critic,
        states,
        proposals,
        actions,
        ["autonomous_success"] * 4,
        [f"ep{index}" for index in range(4)],
        fractions=(0.05,),
        device=torch.device("cpu"),
        batch_size=4,
        seed=4,
    )["overall"]
    assert result["twin_gradient_cosine"]["median"] > 0.7
    assert result["eps_0.05_monotonic"]["mean"] == 1.0
    assert result["eps_0.05_finite_difference_vs_autograd_pearson"] is None


def test_gradient_audit_detects_opposed_twin_directions():
    actor = ChunkActor(
        state_dim=2,
        chunk_dim=2,
        hidden_dim=8,
        num_layers=1,
        action_residual=True,
        delta_scale=0.2,
        proprio_dim=0,
        state_normalization="none",
    )
    critic = _LinearTwin(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
    states = torch.zeros(2, 2)
    proposals = torch.zeros(2, 2)
    with torch.no_grad():
        actions, _ = actor(states, proposals)
    result = _gradient_audit(
        actor,
        critic,
        states,
        proposals,
        actions,
        ["corrective_last_K"] * 2,
        ["ep0", "ep1"],
        fractions=(0.05,),
        device=torch.device("cpu"),
        batch_size=2,
        seed=5,
    )["overall"]
    assert result["twin_gradient_cosine"]["median"] == -1.0
    assert result["twin_gradient_negative"]["mean"] == 1.0
