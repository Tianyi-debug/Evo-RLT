from __future__ import annotations

from evo_rlt.core.interfaces import ChunkTransition, Observation, VLAOutput
from evo_rlt.core.config import RLTConfig, OfflineRLConfig
from evo_rlt.core.vla_adapter import VLAAdapter, DummyVLAAdapter
from evo_rlt.core.rl_token import RLTokenModule
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import ChunkCritic, TwinCritic
from evo_rlt.core.losses import discounted_chunk_return, critic_loss, actor_loss
from evo_rlt.core.replay_buffer import ReplayBuffer
from evo_rlt.core.utils import (
    soft_update,
    flatten_chunk,
    unflatten_chunk,
    compute_discount_vector,
    build_mlp,
    filter_encoder_only,
    infer_actor_architecture,
)
from evo_rlt.core.policy import RLTPolicy
from evo_rlt.core.algorithm import RLTAlgorithm
from evo_rlt.core.collector import Environment, DummyEnvironment, execute_chunk
from evo_rlt.core.rewards import build_reward_seq

__all__ = [
    "ChunkTransition",
    "Observation",
    "VLAOutput",
    "RLTConfig",
    "OfflineRLConfig",
    "VLAAdapter",
    "DummyVLAAdapter",
    "RLTokenModule",
    "ChunkActor",
    "ChunkCritic",
    "TwinCritic",
    "discounted_chunk_return",
    "critic_loss",
    "actor_loss",
    "ReplayBuffer",
    "RLTPolicy",
    "RLTAlgorithm",
    "soft_update",
    "flatten_chunk",
    "unflatten_chunk",
    "compute_discount_vector",
    "build_mlp",
    "filter_encoder_only",
    "infer_actor_architecture",
    "Environment",
    "DummyEnvironment",
    "execute_chunk",
    "build_reward_seq",
]
