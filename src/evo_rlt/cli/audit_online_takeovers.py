"""Audit typed intervention events and future-K labels in online datasets."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import pyarrow as pa
import pyarrow.parquet as pq


POLICY_STAGE = 0
CORRECTIVE = 1
PROACTIVE = 2


def _read_tables(paths: list[Path]) -> pa.Table:
    if not paths:
        raise FileNotFoundError("no parquet files found")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def _scalars(table: pa.Table, key: str) -> list[float | int]:
    values = table[key].combine_chunks().to_pylist()
    return [value[0] if isinstance(value, list) else value for value in values]


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values) if values else 0.0,
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.50),
        "mean": mean(values) if values else 0.0,
        "p75": _percentile(values, 0.75),
        "max": max(values) if values else 0.0,
    }


def _anchor_count(frame_count: int, chunk_length: int, stride: int) -> int:
    """Count anchors whose action chunk and bootstrap state remain in-range."""
    last_anchor = frame_count - 1 - chunk_length
    return 0 if last_anchor < 0 else last_anchor // stride + 1


@dataclass(frozen=True)
class EpisodeAudit:
    uid: str
    dataset: str
    episode_index: int
    success: bool
    frame_count: int
    assisted: bool
    event_count: int
    reason: int
    prefix_frames: int
    prefix_anchors: int
    return_to_policy: bool

    @property
    def category(self) -> str:
        if self.assisted:
            return "corrective" if self.reason == CORRECTIVE else "proactive"
        return "autonomous_success" if self.success else "autonomous_failure"


def audit_dataset(root: Path, chunk_length: int, stride: int) -> list[EpisodeAudit]:
    info = json.loads((root / "meta/info.json").read_text())
    frames = _read_tables(sorted((root / "data").rglob("*.parquet")))
    episodes = _read_tables(sorted((root / "meta/episodes").rglob("*.parquet")))
    required = {
        "episode_index",
        "complementary_info.is_intervention",
        "complementary_info.intervention_stage",
        "complementary_info.intervention_reason",
    }
    missing = required - set(frames.column_names)
    if missing:
        raise KeyError(f"{root}: missing typed intervention columns {sorted(missing)}")

    frame_episode = [int(value) for value in _scalars(frames, "episode_index")]
    intervention = [float(value) for value in _scalars(frames, "complementary_info.is_intervention")]
    stage = [int(round(float(value))) for value in _scalars(frames, "complementary_info.intervention_stage")]
    reason = [
        int(round(float(value)))
        for value in _scalars(frames, "complementary_info.intervention_reason")
    ]
    episode_ids = [int(value) for value in _scalars(episodes, "episode_index")]
    outcomes = [str(value).lower() for value in episodes["episode_success"].combine_chunks().to_pylist()]
    dataset_name = root.parent.name

    by_episode: dict[int, list[int]] = defaultdict(list)
    for row, episode_id in enumerate(frame_episode):
        by_episode[episode_id].append(row)

    result: list[EpisodeAudit] = []
    for episode_id, outcome in zip(episode_ids, outcomes, strict=True):
        rows = by_episode[episode_id]
        authority = [
            intervention[row] > 0.5 or stage[row] != POLICY_STAGE
            for row in rows
        ]
        event_starts = [
            index
            for index, active in enumerate(authority)
            if active and (index == 0 or not authority[index - 1])
        ]
        assisted = bool(event_starts)
        prefix_frames = event_starts[0] if assisted else len(rows)
        suffix_reasons = {
            reason[rows[index]]
            for index in range(prefix_frames, len(rows))
            if reason[rows[index]] != 0
        }
        if assisted and len(suffix_reasons) != 1:
            raise ValueError(
                f"{root} episode {episode_id}: expected one typed reason, got "
                f"{sorted(suffix_reasons)}"
            )
        typed_reason = next(iter(suffix_reasons), 0)
        if typed_reason not in (0, CORRECTIVE, PROACTIVE):
            raise ValueError(f"unsupported intervention reason {typed_reason}")
        return_to_policy = assisted and any(not active for active in authority[prefix_frames:])
        result.append(
            EpisodeAudit(
                uid=f"{dataset_name}:{episode_id}",
                dataset=dataset_name,
                episode_index=episode_id,
                success=outcome == "success",
                frame_count=len(rows),
                assisted=assisted,
                event_count=len(event_starts),
                reason=typed_reason,
                prefix_frames=prefix_frames,
                prefix_anchors=_anchor_count(prefix_frames, chunk_length, stride),
                return_to_policy=return_to_policy,
            )
        )
    if len(result) != int(info["total_episodes"]):
        raise ValueError(
            f"{root}: audited {len(result)} episodes, metadata says {info['total_episodes']}"
        )
    return result


def future_k_counts(episodes: list[EpisodeAudit], k: int) -> dict[str, int | float]:
    counts = Counter()
    for episode in episodes:
        anchors = episode.prefix_anchors
        if episode.category == "corrective":
            counts["positive"] += min(k, anchors)
            counts["negative"] += max(0, anchors - k)
        elif episode.category == "proactive":
            counts["censored"] += min(k, anchors)
            counts["negative"] += max(0, anchors - k)
        elif episode.category == "autonomous_success":
            counts["negative"] += anchors
        else:
            counts["autonomous_failure_excluded"] += anchors
    positive = counts["positive"]
    negative = counts["negative"]
    return {
        "positive": positive,
        "negative": negative,
        "censored": counts["censored"],
        "autonomous_failure_excluded": counts["autonomous_failure_excluded"],
        "positive_to_negative": positive / negative if negative else math.inf,
    }


def episode_split(
    episodes: list[EpisodeAudit],
    val_fraction: float,
    seed: int,
) -> tuple[list[EpisodeAudit], list[EpisodeAudit]]:
    grouped: dict[str, list[EpisodeAudit]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.category].append(episode)
    rng = random.Random(seed)
    train: list[EpisodeAudit] = []
    val: list[EpisodeAudit] = []
    for category in sorted(grouped):
        group = sorted(grouped[category], key=lambda item: item.uid)
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        if len(group) > 1:
            n_val = min(max(n_val, 1), len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return sorted(train, key=lambda item: item.uid), sorted(val, key=lambda item: item.uid)


def build_report(
    roots: list[Path],
    chunk_length: int,
    stride: int,
    ks: list[int],
    split_seed: int,
) -> dict:
    episodes = [
        episode
        for root in roots
        for episode in audit_dataset(root, chunk_length, stride)
    ]
    category_counts = Counter(episode.category for episode in episodes)
    corrective = [episode for episode in episodes if episode.category == "corrective"]
    proactive = [episode for episode in episodes if episode.category == "proactive"]
    fps_by_root = {
        root.parent.name: float(json.loads((root / "meta/info.json").read_text())["fps"])
        for root in roots
    }

    def prefix_stats(selected: list[EpisodeAudit]) -> dict[str, dict[str, float]]:
        frames = [float(episode.prefix_frames) for episode in selected]
        seconds = [
            episode.prefix_frames / fps_by_root[episode.dataset]
            for episode in selected
        ]
        anchors = [float(episode.prefix_anchors) for episode in selected]
        return {
            "frames": _summary(frames),
            "seconds": _summary(seconds),
            "chunk_anchors": _summary(anchors),
        }

    train, val = episode_split(episodes, val_fraction=0.2, seed=split_seed)
    return {
        "inputs": [str(root) for root in roots],
        "assumptions": {
            "chunk_length": chunk_length,
            "frame_stride": stride,
            "anchor_requires_action_and_bootstrap_before_boundary": True,
            "split": f"episode-level stratified 80/20 seed={split_seed}",
        },
        "episodes": {
            "total": len(episodes),
            "fully_autonomous_success": category_counts["autonomous_success"],
            "fully_autonomous_failure": category_counts["autonomous_failure"],
            "assisted": category_counts["corrective"] + category_counts["proactive"],
            "corrective_events": sum(
                episode.event_count for episode in corrective
            ),
            "proactive_events": sum(episode.event_count for episode in proactive),
            "episodes_with_more_than_one_takeover": sum(
                episode.event_count > 1 for episode in episodes
            ),
            "episodes_returning_to_policy": sum(
                episode.return_to_policy for episode in episodes
            ),
            "per_dataset": {
                root.parent.name: dict(
                    Counter(
                        episode.category
                        for episode in episodes
                        if episode.dataset == root.parent.name
                    )
                )
                for root in roots
            },
        },
        "corrective_prefix": prefix_stats(corrective),
        "proactive_prefix": prefix_stats(proactive),
        "future_k": {str(k): future_k_counts(episodes, k) for k in ks},
        "split": {
            "train_episodes": len(train),
            "val_episodes": len(val),
            "train_category_counts": dict(Counter(item.category for item in train)),
            "val_category_counts": dict(Counter(item.category for item in val)),
            "future_k": {
                str(k): {
                    "train": future_k_counts(train, k),
                    "val": future_k_counts(val, k),
                }
                for k in ks
            },
            "train_episode_uids": [item.uid for item in train],
            "val_episode_uids": [item.uid for item in val],
        },
        "episode_rows": [asdict(episode) | {"category": episode.category} for episode in episodes],
        "pose_metadata": (
            "UNAVAILABLE: schema contains one task_index but no reliable grasp-side, "
            "clip-pose, or placement-point metadata"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", required=True, type=Path)
    parser.add_argument("--chunk-length", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--future-k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--split-seed", type=int, default=1000)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        roots=args.dataset_root,
        chunk_length=args.chunk_length,
        stride=args.frame_stride,
        ks=args.future_k,
        split_seed=args.split_seed,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
