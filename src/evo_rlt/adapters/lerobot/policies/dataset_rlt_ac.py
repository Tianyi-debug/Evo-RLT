from __future__ import annotations

import logging
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset


log = logging.getLogger(__name__)


@dataclass
class _EpisodeIndexShim:
    """Minimum LeRobotDataset.meta.episodes shape used by EpisodeAwareSampler."""

    dataset_from_index: list[int]
    dataset_to_index: list[int]


@dataclass
class _MetaShim:
    """Minimum LeRobotDataset.meta surface used by lerobot-train.

    lerobot-train pulls dataset_stats off `meta.stats` and frame ranges off
    `meta.episodes`. Other attributes are duck-typed where possible.
    """

    stats: dict[str, dict[str, torch.Tensor]]
    episodes: _EpisodeIndexShim
    fps: int = 30
    # ChunkTransitionDataset feeds precomputed transitions; policy.input_features /
    # output_features are set by ChunkACPolicyConfig.validate_features, not derived
    # from the dataset. So expose an empty features dict.
    features: dict = field(default_factory=dict)


class ChunkTransitionDataset(Dataset):
    """Dataset over precomputed chunk transitions for ChunkACPolicy training.

    The cache directory must contain `chunk_transitions_{split}.pt`, a Python
    list of dicts with the fields produced by `build_transition_cache.py`:
      state_vec      (state_dim,)
      exec_chunk     (C, action_dim)
      proposal_chunk (C, action_dim), independently generated VLA proposal
      bc_target_chunk (C, action_dim), VLA or executed human target
      reward_seq     (C,)
      next_state_vec (state_dim,)
      next_proposal_chunk (C, action_dim)
      done           ()
      intervention   ()
      actual_steps   ()
      (optional) source, episode_id, is_critical, critic_mask, actor_q_mask,
      actor_bc_mask, intervention_reason

    The stored fields are returned unchanged and a stable `cache_index` is
    injected from the sample position. Flattening for `exec_chunk_flat` /
    flattened proposal/target views is deferred to ChunkACPolicy.forward. Old
    ref-only caches remain readable but cannot recover overwritten proposals.
    """

    def __init__(
        self,
        cache_dir: str | pathlib.Path,
        split: str = "train",
        *,
        training_stage: str = "mixed_ac",
        source_sampling_weights: list[float] | None = None,
        source_sampling_seed: int = 1000,
    ):
        cache = pathlib.Path(cache_dir)
        path = cache / f"chunk_transitions_{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No transition cache at {path}")
        self._transitions: list[dict[str, torch.Tensor]] = torch.load(
            path, weights_only=False, map_location="cpu"
        )
        if not self._transitions:
            raise ValueError(f"empty cache at {path}")
        self.source_counts = self._count_sources(range(len(self._transitions)))
        self._sample_indices = self._build_sample_indices(
            training_stage=training_stage,
            source_sampling_weights=source_sampling_weights,
            source_sampling_seed=source_sampling_seed,
        )
        self.sampling_source_counts = self._count_sources(self._sample_indices)
        self.num_frames = len(self._sample_indices)
        log.info(
            "Loaded transition cache stage=%s raw=%d raw_sources=%s sampled=%d "
            "sampled_sources=%s",
            training_stage,
            len(self._transitions),
            self.source_counts,
            self.num_frames,
            self.sampling_source_counts,
        )
        self.num_episodes = 1  # cache is flattened; treat as one mega-episode.
        self.fps = 30

        # Build minimal meta shim. ACTION/STATE stats are absent (the cache is
        # already encoded); pi05 preprocessor pipeline at deploy-time will use
        # the stats embedded in the policy's own saved processor JSONs.
        self.meta = _MetaShim(
            stats={},
            episodes=_EpisodeIndexShim(
                dataset_from_index=[0],
                dataset_to_index=[self.num_frames],
            ),
            fps=self.fps,
        )
        # Inferred from the first sample.
        sample = self._transitions[0]
        self.features: dict[str, dict[str, Any]] = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                self.features[k] = {"shape": tuple(v.shape), "dtype": str(v.dtype)}

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cache_index = self._sample_indices[idx]
        sample = dict(self._transitions[cache_index])
        sample["cache_index"] = torch.tensor(cache_index, dtype=torch.long)
        return sample

    @staticmethod
    def _source_id(sample: dict[str, Any]) -> int | None:
        value = sample.get("source")
        if value is None:
            return None
        value = torch.as_tensor(value)
        if value.numel() != 1:
            raise ValueError(f"transition source must be scalar, got shape {tuple(value.shape)}")
        return int(value.item())

    def _count_sources(self, indices) -> dict[int, int]:
        counts: dict[int, int] = {}
        for index in indices:
            source = self._source_id(self._transitions[index])
            if source is not None:
                counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def _build_sample_indices(
        self,
        *,
        training_stage: str,
        source_sampling_weights: list[float] | None,
        source_sampling_seed: int,
    ) -> list[int]:
        if training_stage not in ("mixed_ac", "human_bc", "critic_only"):
            raise ValueError(
                "training_stage must be 'mixed_ac', 'human_bc', or 'critic_only', "
                f"got {training_stage!r}"
            )
        if training_stage == "human_bc":
            human = [
                index
                for index, sample in enumerate(self._transitions)
                if self._source_id(sample) == 3
            ]
            if not human:
                raise ValueError("human_bc requires source=3 transitions in the cache")
            return human

        if source_sampling_weights is None:
            return list(range(len(self._transitions)))
        if len(source_sampling_weights) != 4:
            raise ValueError(
                "source_sampling_weights must contain four values ordered as "
                "[demo, VLA, RL autonomous, human]"
            )
        if any(weight < 0 for weight in source_sampling_weights):
            raise ValueError("source_sampling_weights must be non-negative")
        weight_sum = float(sum(source_sampling_weights))
        if weight_sum <= 0:
            raise ValueError("source_sampling_weights must have a positive sum")

        pools: dict[int, list[int]] = {source: [] for source in range(4)}
        for index, sample in enumerate(self._transitions):
            source = self._source_id(sample)
            if source is None:
                raise ValueError("source-balanced sampling requires every transition to have source")
            if source not in pools:
                raise ValueError(f"unsupported transition source id {source}; expected 0..3")
            pools[source].append(index)

        normalized = [float(weight) / weight_sum for weight in source_sampling_weights]
        unavailable = [source for source, weight in enumerate(normalized) if weight > 0 and not pools[source]]
        if unavailable:
            raise ValueError(
                "source_sampling_weights requests absent cache sources: "
                f"{unavailable}; available counts={self.source_counts}"
            )

        total = len(self._transitions)
        exact = [weight * total for weight in normalized]
        targets = [int(value) for value in exact]
        remainder = total - sum(targets)
        order = sorted(range(4), key=lambda source: (exact[source] - targets[source], -source), reverse=True)
        for source in order[:remainder]:
            targets[source] += 1

        rng = random.Random(source_sampling_seed)
        selected: list[int] = []
        for source, target in enumerate(targets):
            pool = list(pools[source])
            while target > 0:
                rng.shuffle(pool)
                take = min(target, len(pool))
                selected.extend(pool[:take])
                target -= take
        rng.shuffle(selected)
        return selected
