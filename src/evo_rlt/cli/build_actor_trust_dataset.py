"""Build episode-safe future-K actor-trust datasets from transition caches.

The input caches are expected to come from online HIL datasets and to retain
typed intervention provenance.  This builder never treats autonomous failure
as a negative takeover label: those anchors are retained for audit purposes but
masked out of the classifier loss.  Likewise, the final K anchors before a
proactive takeover are censored rather than labelled positive or negative.
"""

from __future__ import annotations

import argparse
import hashlib
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
    human_rows: tuple[dict[str, Any], ...]


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


def _classify_episode(
    rows: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    sources = [int(_scalar(row, "source")) for row in rows]
    if set(sources) == {TRANSITION_SOURCE_DEMO}:
        return None, [], []
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
    human = [
        row
        for row, source in zip(rows, sources, strict=True)
        if source == TRANSITION_SOURCE_HUMAN_OVERRIDE
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
            return CORRECTIVE, autonomous, human
        if reason == PROACTIVE_REASON:
            return PROACTIVE, autonomous, human
        raise ValueError(f"unsupported intervention reason {reason}")

    if nonzero_reasons:
        raise ValueError(
            "unassisted episode has nonzero intervention reason "
            f"{sorted(nonzero_reasons)}"
        )
    bc_masks = {int(float(_scalar(row, "actor_bc_mask")) > 0.5) for row in autonomous}
    if bc_masks == {1}:
        return AUTONOMOUS_SUCCESS, autonomous, human
    if bc_masks == {0}:
        return AUTONOMOUS_FAILURE, autonomous, human
    raise ValueError(
        "unassisted episode has mixed actor_bc_mask values; rebuild the input cache "
        "with --actor-bc-mode outcome-aware"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_descriptor(root: Path) -> dict[str, str]:
    resolved = root.expanduser().resolve(strict=True)
    train = resolved / "chunk_transitions_train.pt"
    val = resolved / "chunk_transitions_val.pt"
    missing = [path for path in (train, val) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing transition cache files: {missing}")
    train_hash = _sha256_file(train)
    val_hash = _sha256_file(val)
    combined = hashlib.sha256(f"{train_hash}:{val_hash}".encode()).hexdigest()
    return {
        "resolved_path": str(resolved),
        "train_sha256": train_hash,
        "val_sha256": val_hash,
        "cache_sha256": combined,
    }


def _cache_descriptors(cache_roots: list[Path]) -> list[dict[str, str]]:
    descriptors = [_cache_descriptor(root) for root in cache_roots]
    paths = [item["resolved_path"] for item in descriptors]
    if len(paths) != len(set(paths)):
        duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
        raise ValueError(f"duplicate resolved cache roots are forbidden: {duplicates}")
    hashes = [item["cache_sha256"] for item in descriptors]
    if len(hashes) != len(set(hashes)):
        duplicates = sorted(value for value, count in Counter(hashes).items() if count > 1)
        raise ValueError(
            "duplicate transition-cache contents are forbidden even under different paths: "
            f"{duplicates}"
        )
    return descriptors


def _validated_temporal_scalar(row: dict[str, Any], key: str) -> int | float:
    if key not in row:
        raise KeyError(
            f"transition is missing {key!r}; rebuild it with the current "
            "build_transition_cache_v2 before constructing actor-trust data"
        )
    return _scalar(row, key)


def load_episode_groups(
    cache_roots: list[Path],
    *,
    cache_descriptors: list[dict[str, str]] | None = None,
) -> list[EpisodeGroup]:
    descriptors = cache_descriptors or _cache_descriptors(cache_roots)
    if len(descriptors) != len(cache_roots):
        raise ValueError("cache descriptor count does not match cache roots")
    groups: list[EpisodeGroup] = []
    seen_uids: set[str] = set()
    for cache_index, (root, descriptor) in enumerate(zip(cache_roots, descriptors, strict=True)):
        resolved = Path(descriptor["resolved_path"])
        cache_id = f"cache{cache_index}:{resolved.name}:{descriptor['cache_sha256'][:12]}"
        by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in _load_cache_rows(resolved):
            by_episode[int(_scalar(row, "episode_id"))].append(row)
        for episode_id, rows in by_episode.items():
            rows.sort(key=lambda row: int(_validated_temporal_scalar(row, "anchor_start_frame")))
            strides = {int(_validated_temporal_scalar(row, "frame_stride")) for row in rows}
            frame_rates = {float(_validated_temporal_scalar(row, "fps")) for row in rows}
            if len(strides) != 1 or next(iter(strides)) <= 0:
                raise ValueError(f"episode {episode_id} has invalid/mixed frame_stride {strides}")
            if len(frame_rates) != 1 or next(iter(frame_rates)) <= 0:
                raise ValueError(f"episode {episode_id} has invalid/mixed fps {frame_rates}")
            category, autonomous, human = _classify_episode(rows)
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
                    human_rows=tuple(human),
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
            anchor_start_frame = int(_validated_temporal_scalar(transition, "anchor_start_frame"))
            frame_stride = int(_validated_temporal_scalar(transition, "frame_stride"))
            fps = float(_validated_temporal_scalar(transition, "fps"))
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
                # The classifier is trained on the autonomous behavior policy's
                # composite action, never on the VLA proposal or human suffix.
                "action_chunk": exec_chunk.clone(),
                "episode_uid": group.uid,
                "cache_id": group.cache_id,
                "episode_id": group.episode_id,
                "category": group.category,
                "anchor_index": anchor_index,
                "anchor_start_frame": anchor_start_frame,
                "prefix_anchor_count": anchor_count,
                "distance_to_event_anchors": distance if group.category in {CORRECTIVE, PROACTIVE} else -1,
                "distance_to_corrective_event": distance if group.category == CORRECTIVE else -1,
                "frame_stride": frame_stride,
                "fps": fps,
                "source": int(_scalar(transition, "source")),
                "intervention_reason": int(_scalar(transition, "intervention_reason", 0)),
                "exec_action_is_actual_sent": float(
                    _scalar(transition, "exec_action_is_actual_sent", 0.0)
                ),
            }
            sample["action_semantics"] = (
                "actual_sent"
                if sample["exec_action_is_actual_sent"] > 0.5
                else "requested_or_pre_clipping_legacy"
            )
            actor_bc_mask = float(_scalar(transition, "actor_bc_mask", 0.0))
            sample["bc_target_chunk"] = _clone_float_tensor(
                transition,
                "bc_target_chunk",
            )
            sample["bc_target_valid"] = actor_bc_mask
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


def build_human_audit_samples(
    groups: list[EpisodeGroup],
    *,
    proprio_dim: int,
) -> list[dict[str, Any]]:
    """Retain held-out human rows for drift diagnostics, never risk training."""
    samples: list[dict[str, Any]] = []
    for group in groups:
        for transition in group.human_rows:
            state_vec = _clone_float_tensor(transition, "state_vec").reshape(-1)
            if state_vec.numel() <= proprio_dim:
                raise ValueError("human audit state_vec must contain RL-token and proprio")
            proposal = _clone_float_tensor(transition, "proposal_chunk")
            exec_chunk = _clone_float_tensor(transition, "exec_chunk")
            actual_sent = float(_scalar(transition, "exec_action_is_actual_sent", 0.0))
            samples.append(
                {
                    "state_vec": state_vec,
                    "z_rl": state_vec[:-proprio_dim].clone(),
                    "proprio": state_vec[-proprio_dim:].clone(),
                    "proposal_chunk": proposal,
                    "exec_chunk": exec_chunk,
                    "action_chunk": exec_chunk.clone(),
                    "bc_target_chunk": _clone_float_tensor(transition, "bc_target_chunk"),
                    "bc_target_valid": 1.0,
                    "episode_uid": group.uid,
                    "cache_id": group.cache_id,
                    "episode_id": group.episode_id,
                    "category": "human",
                    "anchor_start_frame": int(
                        _validated_temporal_scalar(transition, "anchor_start_frame")
                    ),
                    "source": int(_scalar(transition, "source")),
                    "intervention_reason": int(
                        _scalar(transition, "intervention_reason", 0)
                    ),
                    "frame_stride": int(
                        _validated_temporal_scalar(transition, "frame_stride")
                    ),
                    "fps": float(_validated_temporal_scalar(transition, "fps")),
                    "exec_action_is_actual_sent": actual_sent,
                    "action_semantics": (
                        "actual_sent" if actual_sent > 0.5 else "requested_or_pre_clipping_legacy"
                    ),
                    "label": torch.tensor(0.0),
                    "label_mask": torch.tensor(0.0),
                    "censored": torch.tensor(1.0),
                }
            )
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
    val_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    descriptors = _cache_descriptors(cache_roots)
    groups = load_episode_groups(cache_roots, cache_descriptors=descriptors)
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
    train_human_samples = build_human_audit_samples(train_groups, proprio_dim=proprio_dim)
    val_human_samples = build_human_audit_samples(val_groups, proprio_dim=proprio_dim)
    train_uids = {group.uid for group in train_groups}
    val_uids = {group.uid for group in val_groups}
    if train_uids & val_uids:
        raise AssertionError("episode leakage between actor-trust train and validation splits")

    ks = sorted(set(ks))
    all_samples = train_samples + val_samples
    if not all_samples:
        raise ValueError("actor-trust dataset contains no autonomous anchors")
    chunk_shapes = {tuple(sample["action_chunk"].shape) for sample in all_samples}
    frame_strides = {int(sample["frame_stride"]) for sample in all_samples}
    frame_rates = {float(sample["fps"]) for sample in all_samples}
    if len(chunk_shapes) != 1:
        raise ValueError(f"mixed action chunk shapes are unsupported: {sorted(chunk_shapes)}")
    if len(frame_strides) != 1:
        raise ValueError(f"mixed frame_stride values are unsupported: {sorted(frame_strides)}")
    if len(frame_rates) != 1:
        raise ValueError(f"mixed fps values are unsupported: {sorted(frame_rates)}")
    action_chunk_shape = next(iter(chunk_shapes))
    frame_stride = next(iter(frame_strides))
    fps = next(iter(frame_rates))
    action_semantics_values = sorted({sample["action_semantics"] for sample in all_samples})
    action_semantics = (
        action_semantics_values[0]
        if len(action_semantics_values) == 1
        else "mixed:" + ",".join(action_semantics_values)
    )
    horizon_seconds = {f"k{k}": k * frame_stride / fps for k in ks}
    metadata: dict[str, Any] = {
        "format_version": 2,
        "inputs": descriptors,
        "semantics": {
            "primary_future_k_anchor_horizon": primary_k,
            "future_k_anchor_horizons": ks,
            "anchor_stride_frames": frame_stride,
            "future_k_anchor_horizon_seconds": horizon_seconds,
            "future_k_anchor_horizon_formula": (
                "future_k_anchor_horizon * anchor_stride_frames / fps"
            ),
            "anchor_horizon_interpretation": (
                "K counts overlapping cache anchors, not non-overlapping executed action chunks"
            ),
            # Backward-compatible aliases used by existing checkpoints and CLIs.
            "primary_future_k": primary_k,
            "future_k_values": ks,
            "chunk_length": action_chunk_shape[0],
            "frame_stride": frame_stride,
            "fps": fps,
            "horizon_seconds": horizon_seconds,
            "horizon_formula": "future_k * frame_stride / fps (cache-anchor units)",
            "risk_action_input": "autonomous composite exec_chunk stored by the behavior policy",
            "risk_action_semantics": action_semantics,
            "positive": "last K eligible autonomous anchors before corrective takeover",
            "negative": "earlier corrective/proactive prefix anchors and autonomous-success anchors",
            "proactive": "last K anchors are censored with label_mask=0",
            "autonomous_failure": "all anchors retained but excluded with label_mask=0",
            "human": "excluded from risk loss; retained separately for drift-only audit",
            "split": f"episode-level stratified by cache and category, val_fraction={val_fraction}",
        },
        "dimensions": {
            "state_dim": int(all_samples[0]["state_vec"].numel()),
            "z_rl_dim": int(all_samples[0]["z_rl"].numel()),
            "proprio_dim": proprio_dim,
            "proposal_chunk_shape": list(all_samples[0]["proposal_chunk"].shape),
            "action_chunk_shape": list(action_chunk_shape),
            "action_flat_dim": int(all_samples[0]["action_chunk"].numel()),
        },
        "episodes": {
            "total": len(groups),
            "categories": dict(Counter(group.category for group in groups)),
            "train": len(train_groups),
            "val": len(val_groups),
            "train_categories": dict(Counter(group.category for group in train_groups)),
            "val_categories": dict(Counter(group.category for group in val_groups)),
            "train_uids": sorted(train_uids),
            "val_uids": sorted(val_uids),
        },
        "human_audit": {
            "train_samples": len(train_human_samples),
            "val_samples": len(val_human_samples),
            "used_for_risk_training": False,
        },
        "train": _sample_counts(train_samples, ks),
        "val": _sample_counts(val_samples, ks),
        "all": _sample_counts(all_samples, ks),
    }

    output_dir.mkdir(parents=True)
    torch.save(train_samples, output_dir / "actor_trust_train.pt")
    torch.save(val_samples, output_dir / "actor_trust_val.pt")
    torch.save(train_human_samples, output_dir / "human_audit_train.pt")
    torch.save(val_human_samples, output_dir / "human_audit_val.pt")
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
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
