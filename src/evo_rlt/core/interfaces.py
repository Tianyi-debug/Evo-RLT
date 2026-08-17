from __future__ import annotations

from dataclasses import dataclass, field

import torch


# ChunkTransition source type IDs (replay-level, distinct from recording-level SOURCE_VLA/RL/HUMAN)
TRANSITION_SOURCE_DEMO = 0
TRANSITION_SOURCE_WARMUP_VLA = 1
TRANSITION_SOURCE_RL_AUTONOMOUS = 2
TRANSITION_SOURCE_HUMAN_OVERRIDE = 3

# Batch dictionary key constants to avoid typos
STATE_VEC = "state_vec"
EXEC_CHUNK_FLAT = "exec_chunk_flat"
# Proposal/target-separated semantics: actor is always conditioned on independently
# generated VLA proposal, while the BC target may be the executed human action.
PROPOSAL_CHUNK_FLAT = "proposal_chunk_flat"
BC_TARGET_CHUNK_FLAT = "bc_target_chunk_flat"
NEXT_PROPOSAL_FLAT = "next_proposal_flat"
# Deprecated aliases kept for old caches, dashboards, and helper scripts.
REF_CHUNK_FLAT = "ref_chunk_flat"
REWARD_SEQ = "reward_seq"
NEXT_STATE_VEC = "next_state_vec"
NEXT_REF_FLAT = "next_ref_flat"
DONE = "done"
ACTUAL_STEPS = "actual_steps"
SOURCE = "source"
EPISODE_ID = "episode_id"
IS_CRITICAL = "is_critical"
INTERVENTION = "intervention"
CRITIC_MASK = "critic_mask"
ACTOR_Q_MASK = "actor_q_mask"


@dataclass
class Observation:
    """Observation from the environment."""

    images: dict[str, torch.Tensor]  # camera_name -> (B, C, H, W)
    proprio: torch.Tensor  # (B, proprio_dim)
    instruction_ids: torch.Tensor | None = None
    timestamp: float | None = None


@dataclass
class VLAOutput:
    """Output from a single VLA forward pass."""

    final_tokens: torch.Tensor  # (B, M, token_dim)
    sampled_action_chunk: torch.Tensor  # (B, H, action_dim)
    extra: dict = field(default_factory=dict)


@dataclass
class ChunkTransition:
    """Single (unbatched) chunk-level transition for replay.

    ``proposal_chunk`` is the VLA action used as actor input and residual base.
    ``exec_chunk`` is the action actually executed and is used by the critic.
    ``bc_target_chunk`` is normally the proposal, but becomes the executed
    human action for intervention chunks. ``ref_chunk`` and ``next_ref_chunk``
    are deprecated compatibility aliases for old caches.
    """

    state_vec: torch.Tensor  # (state_dim,)
    exec_chunk: torch.Tensor  # (C, action_dim)
    ref_chunk: torch.Tensor  # (C, action_dim)
    reward_seq: torch.Tensor  # (C,)
    next_state_vec: torch.Tensor  # (state_dim,)
    next_ref_chunk: torch.Tensor  # (C, action_dim)
    done: torch.Tensor  # scalar
    intervention: torch.Tensor  # scalar, 0/1 flag
    actual_steps: torch.Tensor  # scalar int, steps actually executed (<= C)
    source: torch.Tensor = field(default_factory=lambda: torch.tensor(0))
    episode_id: torch.Tensor = field(default_factory=lambda: torch.tensor(-1))
    is_critical: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    proposal_chunk: torch.Tensor | None = None
    bc_target_chunk: torch.Tensor | None = None
    next_proposal_chunk: torch.Tensor | None = None
    # Per-transition training semantics.  Old caches default to the legacy
    # behavior where every transition participates in critic and actor-Q
    # updates.
    critic_mask: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    actor_q_mask: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0))
    intervention_reason: torch.Tensor = field(default_factory=lambda: torch.tensor(0))

    def __post_init__(self) -> None:
        # Old caches only contain ref_chunk. They remain loadable, but a human
        # chunk whose ref was overwritten cannot recover its original proposal;
        # those caches must be rebuilt for proposal/target-separated training.
        if self.proposal_chunk is None:
            self.proposal_chunk = self.ref_chunk
        if self.bc_target_chunk is None:
            self.bc_target_chunk = self.ref_chunk
        if self.next_proposal_chunk is None:
            self.next_proposal_chunk = self.next_ref_chunk
