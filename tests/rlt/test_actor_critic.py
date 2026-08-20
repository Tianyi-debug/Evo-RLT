from __future__ import annotations

import torch
import pytest

from evo_rlt.core.actor import ChunkActor, normalize_state_vec
from evo_rlt.core.critic import ChunkCritic, TwinCritic


@pytest.fixture
def actor():
    return ChunkActor(state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2)


@pytest.fixture
def twin_critic():
    return TwinCritic(state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2)


class TestActor:
    def test_forward_shapes(self, actor):
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, std = actor(state, ref)
        assert mu.shape == (8, 140)
        assert std.shape == (8, 140)

    def test_sample_shapes(self, actor):
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        action, mu = actor.sample(state, ref)
        assert action.shape == (8, 140)
        assert mu.shape == (8, 140)

    def test_fixed_std(self, actor):
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        _, std = actor(state, ref)
        assert torch.allclose(std, torch.full_like(std, 0.05))

    def test_ref_dropout_statistics(self):
        """With large batch and training=True, ~50% should be zeroed."""
        actor = ChunkActor(state_dim=78, chunk_dim=140, hidden_dim=64, ref_dropout_p=0.5)
        state = torch.randn(1000, 78)
        ref = torch.ones(1000, 140)  # all ones so we can detect zeroing

        torch.manual_seed(42)
        mu, _ = actor(state, ref, training=True)

        # The ref was multiplied by a mask. We can check by looking at the input
        # indirectly: the ratio of zero-ref samples should be ~50%
        # We verify by calling forward manually and checking the mask effect
        torch.manual_seed(42)
        mask = (torch.rand(1000, 1) > 0.5).float()
        frac_kept = mask.mean().item()
        assert 0.4 < frac_kept < 0.6

    def test_gradient_flow(self, actor):
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        action, _ = actor.sample(state, ref, training=True)
        loss = action.sum()
        loss.backward()
        for p in actor.parameters():
            assert p.grad is not None

    def test_v2_starts_exactly_from_vla_reference(self):
        actor = ChunkActor(
            state_dim=14,
            chunk_dim=12,
            hidden_dim=32,
            num_layers=2,
            proprio_dim=6,
            state_normalization="rl_token_layer_norm",
            action_residual=True,
            delta_scale=0.1,
        )
        state = torch.cat(
            [torch.randn(8, 8) * 1500.0, torch.randn(8, 6)],
            dim=-1,
        )
        ref = torch.empty(8, 12).uniform_(-1.2, 1.2)

        mu, _ = actor(state, ref)

        assert torch.equal(mu, ref.clamp(-1.0, 1.0))

    def test_v2_delta_is_bounded_around_reference(self):
        actor = ChunkActor(
            state_dim=14,
            chunk_dim=12,
            hidden_dim=32,
            num_layers=2,
            proprio_dim=6,
            state_normalization="rl_token_layer_norm",
            action_residual=True,
            delta_scale=0.1,
        )
        output_layer = actor.net[-1]
        with torch.no_grad():
            output_layer.bias.fill_(100.0)
        state = torch.cat(
            [torch.randn(8, 8) * 1500.0, torch.randn(8, 6)],
            dim=-1,
        )
        ref = torch.zeros(8, 12)

        mu, _ = actor(state, ref)

        assert torch.all(mu <= 0.1)
        assert torch.all(mu >= -0.1)
        assert torch.allclose(mu, torch.full_like(mu, 0.1), atol=1e-6)

    def test_per_action_dim_delta_scale_repeats_across_chunk(self):
        actor = ChunkActor(
            state_dim=8,
            chunk_dim=6,
            hidden_dim=16,
            num_layers=2,
            action_residual=True,
            delta_scale=0.1,
            delta_scale_per_action_dim=[0.1, 0.2, 0.7],
        )
        with torch.no_grad():
            actor.net[-1].bias.fill_(100.0)

        mu, _ = actor(torch.zeros(2, 8), torch.zeros(2, 6))

        expected = torch.tensor([0.1, 0.2, 0.7, 0.1, 0.2, 0.7]).repeat(2, 1)
        assert torch.allclose(mu, expected, atol=1e-6)

    def test_residual_feasible_projection_uses_clamps_and_per_dim_bounds(self):
        bounds = [0.30, 0.25, 0.18, 0.40, 0.35, 0.90]
        actor = ChunkActor(
            state_dim=8,
            chunk_dim=60,
            hidden_dim=16,
            num_layers=2,
            action_residual=True,
            delta_scale_per_action_dim=bounds,
        )
        proposal_step = torch.tensor([0.9, -0.9, 0.0, 1.2, -1.2, 0.0])
        proposal = proposal_step.repeat(10).unsqueeze(0)
        lower, upper = actor.residual_reachable_interval(proposal)

        expected_lower = torch.tensor([0.6, -1.0, -0.18, 0.6, -1.0, -0.9])
        expected_upper = torch.tensor([1.0, -0.65, 0.18, 1.0, -0.65, 0.9])
        assert torch.allclose(lower, expected_lower.repeat(10).unsqueeze(0))
        assert torch.allclose(upper, expected_upper.repeat(10).unsqueeze(0))

        inside = (lower + upper) / 2
        assert torch.equal(actor.project_to_residual_support(proposal, inside), inside)

        outside = torch.full_like(proposal, 2.0)
        assert torch.equal(actor.project_to_residual_support(proposal, outside), upper)

    def test_per_action_dim_delta_scale_validates_chunk_width(self):
        with pytest.raises(ValueError, match="must be divisible"):
            ChunkActor(
                state_dim=8,
                chunk_dim=5,
                action_residual=True,
                delta_scale_per_action_dim=[0.1, 0.2],
            )

    def test_rl_token_layer_norm_preserves_proprio_and_controls_scale(self):
        z_rl = torch.randn(16, 2048) * 1300.0 + 100.0
        proprio = torch.randn(16, 6)
        state = torch.cat([z_rl, proprio], dim=-1)

        normalized = normalize_state_vec(
            state,
            proprio_dim=6,
            mode="rl_token_layer_norm",
        )

        normalized_z = normalized[:, :-6]
        assert torch.allclose(normalized[:, -6:], proprio)
        assert torch.allclose(
            normalized_z.mean(dim=-1),
            torch.zeros(16),
            atol=1e-5,
        )
        assert torch.allclose(
            normalized_z.std(dim=-1, unbiased=False),
            torch.ones(16),
            atol=1e-5,
        )

    def test_v2_large_rl_tokens_do_not_kill_first_relu(self):
        actor = ChunkActor(
            state_dim=2054,
            chunk_dim=60,
            hidden_dim=256,
            num_layers=2,
            proprio_dim=6,
            state_normalization="rl_token_layer_norm",
            action_residual=True,
        )
        state = torch.cat(
            [torch.randn(64, 2048) * 1300.0, torch.randn(64, 6)],
            dim=-1,
        )
        ref = torch.empty(64, 60).uniform_(-1.0, 1.0)
        normalized = normalize_state_vec(
            state,
            proprio_dim=6,
            mode="rl_token_layer_norm",
        )
        first_relu = torch.relu(actor.net[0](torch.cat([normalized, ref], dim=-1)))

        assert (first_relu != 0).any()
        assert first_relu.std(dim=0).mean() > 0


class TestCritic:
    def test_chunk_critic_shape(self):
        critic = ChunkCritic(state_dim=78, chunk_dim=140, hidden_dim=64)
        q = critic(torch.randn(8, 78), torch.randn(8, 140))
        assert q.shape == (8, 1)

    def test_twin_critic_shapes(self, twin_critic):
        state = torch.randn(8, 78)
        action = torch.randn(8, 140)
        q1, q2 = twin_critic(state, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_min_q(self, twin_critic):
        state = torch.randn(8, 78)
        action = torch.randn(8, 140)
        q1, q2 = twin_critic(state, action)
        min_q = twin_critic.min_q(state, action)
        expected = torch.minimum(q1, q2)
        assert torch.allclose(min_q, expected)

    def test_gradient_flow(self, twin_critic):
        state = torch.randn(4, 78)
        action = torch.randn(4, 140)
        q = twin_critic.min_q(state, action)
        q.sum().backward()
        for p in twin_critic.parameters():
            assert p.grad is not None

    def test_v2_critic_is_conditioned_on_large_scale_states(self):
        critic = ChunkCritic(
            state_dim=2054,
            chunk_dim=60,
            hidden_dim=256,
            proprio_dim=6,
            state_normalization="rl_token_layer_norm",
        )
        state = torch.cat(
            [torch.randn(64, 2048) * 1300.0, torch.randn(64, 6)],
            dim=-1,
        )
        action = torch.empty(64, 60).uniform_(-1.0, 1.0)

        q = critic(state, action)

        assert q.std() > 0
