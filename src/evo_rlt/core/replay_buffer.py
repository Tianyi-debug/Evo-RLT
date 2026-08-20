from __future__ import annotations

import random
from collections import deque

import torch

from evo_rlt.core.interfaces import (
    ACTOR_BC_MASK,
    ACTOR_Q_MASK,
    ACTUAL_STEPS,
    BOOTSTRAP_MASK,
    CACHE_SEMANTICS_VERSION,
    BC_TARGET_CHUNK_FLAT,
    CRITIC_MASK,
    DONE,
    EPISODE_ID,
    EXEC_CHUNK_FLAT,
    IS_CRITICAL,
    INTERVENTION,
    NEXT_PROPOSAL_FLAT,
    NEXT_REF_FLAT,
    NEXT_STATE_VEC,
    PROPOSAL_CHUNK_FLAT,
    REF_CHUNK_FLAT,
    REWARD_SEQ,
    SOURCE,
    STATE_VEC,
    ChunkTransition,
)


class ReplayBuffer:
    """Deque-based chunk-level replay buffer.

    Stores single (unbatched) ChunkTransition objects and collates them
    into batched dicts at sample time.

    TODO: Replace with tensor-backed circular buffer for performance at scale.
    """

    def __init__(self, capacity: int = 200_000):
        self.buffer: deque[ChunkTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def capacity(self) -> int:
        if self.buffer.maxlen is None:
            raise RuntimeError("Buffer has no capacity limit")
        return self.buffer.maxlen

    def add(self, transition: ChunkTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch and collate into a dict of stacked tensors."""
        n = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), n)
        batch = [self.buffer[i] for i in indices]
        stacked_exec = torch.stack([t.exec_chunk for t in batch])
        stacked_proposal = torch.stack([t.proposal_chunk for t in batch])
        stacked_bc_target = torch.stack([t.bc_target_chunk for t in batch])
        stacked_next_proposal = torch.stack([t.next_proposal_chunk for t in batch])
        proposal_flat = stacked_proposal.flatten(start_dim=-2)
        next_proposal_flat = stacked_next_proposal.flatten(start_dim=-2)
        return {
            STATE_VEC: torch.stack([t.state_vec for t in batch]),
            EXEC_CHUNK_FLAT: stacked_exec.flatten(start_dim=-2),
            PROPOSAL_CHUNK_FLAT: proposal_flat,
            BC_TARGET_CHUNK_FLAT: stacked_bc_target.flatten(start_dim=-2),
            NEXT_PROPOSAL_FLAT: next_proposal_flat,
            # Legacy aliases now consistently mean VLA proposal.
            REF_CHUNK_FLAT: proposal_flat,
            REWARD_SEQ: torch.stack([t.reward_seq for t in batch]),
            NEXT_STATE_VEC: torch.stack([t.next_state_vec for t in batch]),
            NEXT_REF_FLAT: next_proposal_flat,
            DONE: torch.stack([t.done for t in batch]),
            BOOTSTRAP_MASK: torch.stack([t.resolved_bootstrap_mask() for t in batch]),
            ACTUAL_STEPS: torch.stack([t.actual_steps for t in batch]),
            SOURCE: torch.stack([t.source for t in batch]),
            EPISODE_ID: torch.stack([t.episode_id for t in batch]),
            IS_CRITICAL: torch.stack([t.is_critical for t in batch]),
            INTERVENTION: torch.stack([t.intervention for t in batch]),
            CRITIC_MASK: torch.stack([t.critic_mask for t in batch]),
            ACTOR_Q_MASK: torch.stack([t.actor_q_mask for t in batch]),
            ACTOR_BC_MASK: torch.stack([t.actor_bc_mask for t in batch]),
            "intervention_reason": torch.stack([t.intervention_reason for t in batch]),
            CACHE_SEMANTICS_VERSION: torch.stack(
                [t.cache_semantics_version for t in batch]
            ),
            "exec_action_is_actual_sent": torch.stack(
                [t.exec_action_is_actual_sent for t in batch]
            ),
        }
