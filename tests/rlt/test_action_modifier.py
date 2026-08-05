from __future__ import annotations

import torch
from torch import nn

from evo_rlt.adapters.lerobot.policies.action_modifier import RLTActionModifier
from evo_rlt.core.phase_controller import PhaseController


class _StubRLToken(nn.Module):
    def encode(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        return prefix_tokens[:, 0, :1]


class _StubActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.sample_calls = 0

    def forward(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_calls += 1
        mu = torch.full_like(ref_chunk_flat, 0.25)
        return mu, torch.full_like(mu, 0.02)

    def sample(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.sample_calls += 1
        mu = torch.full_like(ref_chunk_flat, 0.25)
        action = torch.full_like(ref_chunk_flat, 0.75)
        return action, mu


def _make_modifier(*, deterministic: bool) -> tuple[RLTActionModifier, _StubActor]:
    actor = _StubActor()
    phase_ctrl = PhaseController(mode="manual")
    phase_ctrl.trigger_critical()
    modifier = RLTActionModifier(
        rl_token=_StubRLToken(),
        actor=actor,
        phase_ctrl=phase_ctrl,
        chunk_length=2,
        action_dim=2,
        proprio_dim=1,
        deterministic=deterministic,
    )
    return modifier, actor


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vla_chunk = torch.zeros(1, 4, 2)
    proprio = torch.zeros(1, 1)
    prefix_tokens = torch.zeros(1, 1, 1)
    return vla_chunk, proprio, prefix_tokens


def test_action_modifier_deterministic_executes_actor_mean() -> None:
    modifier, actor = _make_modifier(deterministic=True)

    chunk = modifier.compute_chunk(*_inputs())

    assert torch.equal(chunk, torch.full((1, 2, 2), 0.25))
    assert actor.forward_calls == 1
    assert actor.sample_calls == 0


def test_action_modifier_stochastic_executes_actor_sample() -> None:
    modifier, actor = _make_modifier(deterministic=False)

    chunk = modifier.compute_chunk(*_inputs())

    assert torch.equal(chunk, torch.full((1, 2, 2), 0.75))
    assert actor.forward_calls == 0
    assert actor.sample_calls == 1
