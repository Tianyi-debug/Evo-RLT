"""Build episode-safe future-K actor-trust datasets from transition caches.

The input caches are expected to come from online HIL datasets and to retain
typed intervention provenance.  This builder never treats autonomous failure
as a negative takeover label: those anchors are retained for audit purposes but
masked out of the classifier loss.  Likewise, the final K anchors before a
proactive takeover are censored rather than labelled positive or negative.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from evo_rlt.core.interfaces import (
    TRANSITION_SOURCE_DEMO,
    TRANSITION_SOURCE_HUMAN_OVERRIDE,
    TRANSITION_SOURCE_RL_AUTONOMOUS,
    TRANSITION_SOURCE_WARMUP_VLA,
)


AUTONOMOUS_SUCCESS = "autonomous_success"
AUTONOMOUS_FAILURE = "autonomous_failure"
CORRECTIVE = "corrective"
PROACTIVE = "proactive"

CORRECTIVE_REASON = 1
PROACTIVE_REASON = 2


@dataclass(frozen=True)
class EpisodeGroup:
    uid: str
    cache_id: str
    episode_id: int
    category: str
    autonomous_rows: tuple[dict[str, Any], ...]


def _scalar(row: dict[str, Any], key: str, default: int | float | None = None) -> int | float:
    value = row.get(key, default)
    if value is None:
        raise KeyError(f"transition is missing required scalar field {key!r}")
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"transition field {key!r} must be scalar, got {tuple(tensor.shape)}")
    return tensor.item()


def _load_cache_rows(root: Path) -> list[dict[str, Any]]:
    paths = [root / "chunk_transitions_train.pt", root / "chunk_transitions_val.pt"]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing transition cache files: {missing}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, list):
            raise TypeError(f"{path} must contain a list of transition dictionaries")
        rows.extend(loaded)
    return rows


def _classify_episode(rows: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    sources = [int(_scalar(row, "source")) for row in rows]
    if set(sources) == {TRANSITION_SOURCE_DEMO}:
        return None, []
    if TRANSITION_SOURCE_DEMO in sources:
        raise ValueError("one episode mixes demo and online transition sources")

    allowed = {
        TRANSITION_SOURCE_WARMUP_VLA,
        TRANSITION_SOURCE_RL_AUTONOMOUS,
        TRANSITION_SOURCE_HUMAN_OVERRIDE,
    }
    unsupported = set(sources) - allowed
    if unsupported:
        raise ValueError(f"unsupported transition sources in online episode: {sorted(unsupported)}")

    autonomous = [
        row
        for row, source in zip(rows, sources, strict=True)
        if source in {TRANSITION_SOURCE_WARMUP_VLA, TRANSITION_SOURCE_RL_AUTONOMOUS}
    ]
    if not autonomous:
        raise ValueError("online episode contains no autonomous anchors")

    human_indices = [
        index for index, source in enumerate(sources) if source == TRANSITION_SOURCE_HUMAN_OVERRIDE
    ]
    nonzero_reasons = {
        int(_scalar(row, "intervention_reason", 0))
        for row in rows
        if int(_scalar(row, "intervention_reason", 0)) != 0
    }

    if human_indices:
        first_human = human_indices[0]
        if any(
            source in {TRANSITION_SOURCE_WARMUP_VLA, TRANSITION_SOURCE_RL_AUTONOMOUS}
            for source in sources[first_human:]
        ):
            raise ValueError("episode returns to autonomous policy after human takeover")
        if len(nonzero_reasons) != 1:
            raise ValueError(
                "assisted episode must have exactly one typed intervention reason, "
                f"got {sorted(nonzero_reasons)}"
            )
        reason = next(iter(nonzero_reasons))
        if reason == CORRECTIVE_REASON:
            return CORRECTIVE, autonomous
        if reason == PROACTIVE_REASON:
            return PROACTIVE, autonomous
        raise ValueError(f"unsupported intervention reason {reason}")

    if nonzero_reasons:
        raise ValueError(
            "unassisted episode has nonzero intervention reason "
            f"{sorted(nonzero_reasons)}"
        )
    bc_masks = {int(float(_scalar(row, "actor_bc_mask")) > 0.5) for row in autonomous}
    if bc_masks == {1}:
        return AUTONOMOUS_SUCCESS, autonomous
    if bc_masks == {0}:
        return AUTONOMOUS_FAILURE, autonomous
    raise ValueError(
        "unassisted episode has mixed actor_bc_mask values; rebuild the input cache "
        "with --actor-bc-mode outcome-aware"
    )


def load_episode_groups(cache_roots: list[Path]) -> list[EpisodeGroup]:
    groups: list[EpisodeGroup] = []
    seen_uids: set[str] = set()
    for cache_index, root in enumerate(cache_roots):
        cache_id = f"cache{cache_index}:{root.name}"
        by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in _load_cache_rows(root):
            by_episode[int(_scalar(row, "episode_id"))].append(row)
        for episode_id, rows in by_episode.items():
            category, autonomous = _classify_episode(rows)
            if category is None:
                continue
            uid = f"{cache_id}:episode{episode_id}"
            if uid in seen_uids:
                raise ValueError(f"duplicate episode uid {uid}")
            seen_uids.add(uid)
            groups.append(
                EpisodeGroup(
                    uid=uid,
                    cache_id=cache_id,
                    episode_id=episode_id,
                    category=category,
                    autonomous_rows=tuple(autonomous),
                )
            )
    if not groups:
        raise ValueError("no online episodes found in input caches")
    return groups


def split_episode_groups(
    groups: list[EpisodeGroup],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[EpisodeGroup], list[EpisodeGroup]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    strata: dict[tuple[str, str], list[EpisodeGroup]] = defaultdict(list)
    for group in groups:
        strata[(group.cache_id, group.category)].append(group)

    rng = random.Random(seed)
    train: list[EpisodeGroup] = []
    val: list[EpisodeGroup] = []
    for key in sorted(strata):
        stratum = sorted(strata[key], key=lambda item: item.uid)
        rng.shuffle(stratum)
        n_val = round(len(stratum) * val_fraction)
        if val_fraction > 0 and len(stratum) > 1:
            n_val = min(max(n_val, 1), len(stratum) - 1)
        else:
            n_val = 0
        val.extend(stratum[:n_val])
        train.extend(stratum[n_val:])
    return sorted(train, key=lambda item: item.uid), sorted(val, key=lambda item: item.uid)


def _clone_float_tensor(row: dict[str, Any], key: str) -> Tensor:
    if key not in row:
        raise KeyError(f"transition is missing required tensor field {key!r}")
    tensor = torch.as_tensor(row[key]).detach().cpu().to(torch.float32).clone()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"transition field {key!r} contains non-finite values")
    return tensor


def build_samples(
    groups: list[EpisodeGroup],
    *,
    ks: list[int],
    primary_k: int,
    proprio_dim: int,
) -> list[dict[str, Any]]:
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("all future-K values must be positive")
    ks = sorted(set(ks))
    if primary_k not in ks:
        raise ValueError("primary_k must be included in ks")
    if proprio_dim <= 0:
        raise ValueError("proprio_dim must be positive")

    samples: list[dict[str, Any]] = []
    state_dim: int | None = None
    proposal_shape: tuple[int, ...] | None = None
    for group in groups:
        anchor_count = len(group.autonomous_rows)
        for anchor_index, transition in enumerate(group.autonomous_rows):
            state_vec = _clone_float_tensor(transition, "state_vec").reshape(-1)
            proposal = _clone_float_tensor(transition, "proposal_chunk")
            exec_chunk = _clone_float_tensor(transition, "exec_chunk")
            if state_vec.numel() <= proprio_dim:
                raise ValueError(
                    f"state_vec dim {state_vec.numel()} must exceed proprio_dim {proprio_dim}"
                )
            if state_dim is None:
                state_dim = state_vec.numel()
                proposal_shape = tuple(proposal.shape)
            if state_vec.numel() != state_dim:
                raise ValueError(
                    f"state_vec dimension mismatch: expected {state_dim}, got {state_vec.numel()}"
                )
            if tuple(proposal.shape) != proposal_shape or exec_chunk.shape != proposal.shape:
                raise ValueError("proposal/exec chunk shapes are inconsistent across caches")

            distance = anchor_count - anchor_index
            sample: dict[str, Any] = {
                "state_vec": state_vec,
                "z_rl": state_vec[:-proprio_dim].clone(),
                "proprio": state_vec[-proprio_dim:].clone(),
                "proposal_chunk": proposal,
                "exec_chunk": exec_chunk,
                "episode_uid": group.uid,
                "cache_id": group.cache_id,
                "episode_id": group.episode_id,
                "category": group.category,
                "anchor_index": anchor_index,
                "prefix_anchor_count": anchor_count,
                "distance_to_event_anchors": distance if group.category in {CORRECTIVE, PROACTIVE} else -1,
                "source": int(_scalar(transition, "source")),
                "exec_action_is_actual_sent": float(
                    _scalar(transition, "exec_action_is_actual_sent", 0.0)
                ),
            }
            for k in ks:
                inside_window = distance <= k
                if group.category == CORRECTIVE:
                    label = float(inside_window)
                    mask = 1.0
                    censored = 0.0
                elif group.category == PROACTIVE:
                    label = 0.0
                    mask = float(not inside_window)
                    censored = float(inside_window)
                elif group.category == AUTONOMOUS_SUCCESS:
                    label = 0.0
                    mask = 1.0
                    censored = 0.0
                else:
                    label = 0.0
                    mask = 0.0
                    censored = 0.0
                sample[f"label_k{k}"] = torch.tensor(label, dtype=torch.float32)
                sample[f"label_mask_k{k}"] = torch.tensor(mask, dtype=torch.float32)
                sample[f"censored_k{k}"] = torch.tensor(censored, dtype=torch.float32)
            sample["label"] = sample[f"label_k{primary_k}"].clone()
            sample["label_mask"] = sample[f"label_mask_k{primary_k}"].clone()
            sample["censored"] = sample[f"censored_k{primary_k}"].clone()
            samples.append(sample)
    return samples


def _sample_counts(samples: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(samples),
        "episodes": len({sample["episode_uid"] for sample in samples}),
        "categories": dict(Counter(sample["category"] for sample in samples)),
        "actual_sent_samples": sum(
            sample["exec_action_is_actual_sent"] > 0.5 for sample in samples
        ),
    }
    for k in ks:
        if samples:
            labels = torch.stack([sample[f"label_k{k}"] for sample in samples])
            masks = torch.stack([sample[f"label_mask_k{k}"] for sample in samples])
            censored = torch.stack([sample[f"censored_k{k}"] for sample in samples])
        else:
            labels = torch.empty(0)
            masks = torch.empty(0)
            censored = torch.empty(0)
        result[f"k{k}"] = {
            "positive": int(((labels > 0.5) & (masks > 0.5)).sum().item()),
            "negative": int(((labels <= 0.5) & (masks > 0.5)).sum().item()),
            "masked": int((masks <= 0.5).sum().item()),
            "censored": int((censored > 0.5).sum().item()),
        }
    return result


def build_actor_trust_dataset(
    *,
    cache_roots: list[Path],
    output_dir: Path,
    ks: list[int],
    primary_k: int,
    proprio_dim: int,
    chunk_length: int,
    frame_stride: int,
    val_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if chunk_length <= 0 or frame_stride <= 0:
        raise ValueError("chunk_length and frame_stride must be positive")
    groups = load_episode_groups(cache_roots)
    train_groups, val_groups = split_episode_groups(
        groups,
        val_fraction=val_fraction,
        seed=split_seed,
    )
    train_samples = build_samples(
        train_groups,
        ks=ks,
        primary_k=primary_k,
        proprio_dim=proprio_dim,
    )
    val_samples = build_samples(
        val_groups,
        ks=ks,
        primary_k=primary_k,
        proprio_dim=proprio_dim,
    )
    train_uids = {group.uid for group in train_groups}
    val_uids = {group.uid for group in val_groups}
    if train_uids & val_uids:
        raise AssertionError("episode leakage between actor-trust train and validation splits")

    ks = sorted(set(ks))
    all_samples = train_samples + val_samples
    if not all_samples:
        raise ValueError("actor-trust dataset contains no autonomous anchors")
    observed_chunk_length = int(all_samples[0]["proposal_chunk"].shape[0])
    if observed_chunk_length != chunk_length:
        raise ValueError(
            f"cache chunk length is {observed_chunk_length}, expected {chunk_length}"
        )
    metadata: dict[str, Any] = {
        "format_version": 1,
        "inputs": [str(path.resolve()) for path in cache_roots],
        "semantics": {
            "primary_future_k": primary_k,
            "future_k_values": ks,
            "chunk_length": chunk_length,
            "frame_stride": frame_stride,
            "fps": 30,
            "positive": "last K eligible autonomous anchors before corrective takeover",
            "negative": "earlier corrective/proactive prefix anchors and autonomous-success anchors",
            "proactive": "last K anchors are censored with label_mask=0",
            "autonomous_failure": "all anchors retained but excluded with label_mask=0",
            "human": "human-controlled anchors are excluded",
            "split": f"episode-level stratified by cache and category, val_fraction={val_fraction}",
        },
        "dimensions": {
            "state_dim": int(all_samples[0]["state_vec"].numel()),
            "z_rl_dim": int(all_samples[0]["z_rl"].numel()),
            "proprio_dim": proprio_dim,
            "proposal_chunk_shape": list(all_samples[0]["proposal_chunk"].shape),
        },
        "episodes": {
            "total": len(groups),
            "categories": dict(Counter(group.category for group in groups)),
            "train": len(train_groups),
            "val": len(val_groups),
            "train_uids": sorted(train_uids),
            "val_uids": sorted(val_uids),
        },
        "train": _sample_counts(train_samples, ks),
        "val": _sample_counts(val_samples, ks),
        "all": _sample_counts(all_samples, ks),
    }

    output_dir.mkdir(parents=True)
    torch.save(train_samples, output_dir / "actor_trust_train.pt")
    torch.save(val_samples, output_dir / "actor_trust_val.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--future-k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--primary-k", type=int, default=3)
    parser.add_argument("--proprio-dim", type=int, default=6)
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_actor_trust_dataset(
        cache_roots=args.cache_root,
        output_dir=args.output_dir,
        ks=args.future_k,
        primary_k=args.primary_k,
        proprio_dim=args.proprio_dim,
        chunk_length=args.chunk_length,
        frame_stride=args.frame_stride,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
