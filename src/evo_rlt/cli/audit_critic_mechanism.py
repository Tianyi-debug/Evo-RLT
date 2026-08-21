"""Read-only critic mechanism audit on actor-trust data.

The primary outcome audit scores the autonomous behavior action that was
actually sent.  Actor-gradient diagnostics instead start from the frozen
checkpoint actor's current composite action and stay inside its residual
reachable set.  Human suffixes are never used as same-state actor evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from evo_rlt.cli.audit_actor_q_mechanism import _load_heads
from evo_rlt.core.actor import ChunkActor, normalize_state_vec
from evo_rlt.core.critic import TwinCritic


GROUPS = (
    "corrective_last_K",
    "corrective_earlier",
    "autonomous_success",
    "autonomous_failure",
    "proactive_earlier",
    "proactive_last_K_censored",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise TypeError(f"{path} must contain a list of dictionaries")
    return value


def _scalar(row: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in row:
        if default is None:
            raise KeyError(key)
        return default
    value = torch.as_tensor(row[key])
    if value.numel() != 1:
        raise ValueError(f"{key} must be scalar, got {tuple(value.shape)}")
    return float(value.item())


def _semantic_group(row: dict[str, Any], primary_k: int) -> str:
    category = str(row["category"])
    if category == "corrective":
        return (
            "corrective_last_K"
            if int(row["distance_to_corrective_event"]) <= primary_k
            else "corrective_earlier"
        )
    if category == "proactive":
        return (
            "proactive_last_K_censored"
            if int(row["distance_to_event_anchors"]) <= primary_k
            else "proactive_earlier"
        )
    if category in {"autonomous_success", "autonomous_failure"}:
        return category
    raise ValueError(f"unsupported actor-trust category {category!r}")


def _quantiles(values: Tensor) -> dict[str, float | None]:
    values = values.detach().float().reshape(-1).cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {key: None for key in ("mean", "p05", "p25", "median", "p75", "p95")}
    q = torch.quantile(values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "mean": float(values.mean().item()),
        "p05": float(q[0].item()),
        "p25": float(q[1].item()),
        "median": float(q[2].item()),
        "p75": float(q[3].item()),
        "p95": float(q[4].item()),
    }


def _episode_means(records: list[dict[str, Any]], key: str) -> list[float]:
    by_episode: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record[key])
        if math.isfinite(value):
            by_episode[str(record["episode_uid"])].append(value)
    return [sum(values) / len(values) for values in by_episode.values() if values]


def _bootstrap_mean_ci(
    values: list[float], *, seed: int, replicates: int
) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    )
    lo = max(0, int(0.025 * replicates))
    hi = min(replicates - 1, int(0.975 * replicates))
    return [samples[lo], samples[hi]]


def _pairwise_episode_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    key: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    left_values = _episode_means(left, key)
    right_values = _episode_means(right, key)
    result: dict[str, Any] = {
        "left_episodes": len(left_values),
        "right_episodes": len(right_values),
        "mean_difference": (
            sum(left_values) / len(left_values) - sum(right_values) / len(right_values)
            if left_values and right_values
            else None
        ),
        "episode_bootstrap_95ci": None,
    }
    if len(left_values) < 2 or len(right_values) < 2:
        return result
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        left_mean = sum(left_values[rng.randrange(len(left_values))] for _ in left_values) / len(
            left_values
        )
        right_mean = sum(
            right_values[rng.randrange(len(right_values))] for _ in right_values
        ) / len(right_values)
        samples.append(left_mean - right_mean)
    samples.sort()
    result["episode_bootstrap_95ci"] = [
        samples[max(0, int(0.025 * replicates))],
        samples[min(replicates - 1, int(0.975 * replicates))],
    ]
    return result


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += float(pos > neg) + 0.5 * float(pos == neg)
    return wins / (len(positive) * len(negative))


def _summarize_score_records(
    records: list[dict[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(records),
        "episodes": len({record["episode_uid"] for record in records}),
    }
    for key in ("q1", "q2", "min_q", "twin_abs_disagreement"):
        values = torch.tensor([float(record[key]) for record in records])
        result[key] = _quantiles(values)
        episode_values = _episode_means(records, key)
        result[key]["episode_mean"] = (
            sum(episode_values) / len(episode_values) if episode_values else None
        )
        result[key]["episode_bootstrap_95ci"] = _bootstrap_mean_ci(
            episode_values, seed=seed, replicates=replicates
        )
    return result


def _score_actions(
    critic: TwinCritic,
    states: Tensor,
    actions: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Tensor]:
    outputs: dict[str, list[Tensor]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            state = states[start : start + batch_size].to(device)
            action = actions[start : start + batch_size].to(device)
            q1, q2 = critic(state, action)
            q1, q2 = q1.squeeze(-1), q2.squeeze(-1)
            outputs["q1"].append(q1.cpu())
            outputs["q2"].append(q2.cpu())
            outputs["min_q"].append(torch.minimum(q1, q2).cpu())
            outputs["twin_abs_disagreement"].append((q1 - q2).abs().cpu())
    return {key: torch.cat(values) for key, values in outputs.items()}


def _actor_actions(
    actor: ChunkActor,
    states: Tensor,
    proposals: Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            action, _ = actor(
                states[start : start + batch_size].to(device),
                proposals[start : start + batch_size].to(device),
                training=False,
            )
            values.append(action.cpu())
    return torch.cat(values)


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x, y = torch.tensor(left), torch.tensor(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denominator <= 0:
        return None
    return float((x * y).sum().div(denominator).item())


def _gradient_audit(
    actor: ChunkActor,
    critic: nn.Module,
    states: Tensor,
    proposals: Tensor,
    actor_actions: Tensor,
    groups: list[str],
    episode_uids: list[str],
    *,
    fractions: tuple[float, ...],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    random_generator = torch.Generator().manual_seed(seed)
    random_signs = torch.where(
        torch.randn(actor_actions.shape, generator=random_generator) >= 0,
        torch.ones_like(actor_actions),
        -torch.ones_like(actor_actions),
    )
    for start in range(0, len(states), batch_size):
        stop = min(len(states), start + batch_size)
        state = states[start:stop].to(device)
        proposal = proposals[start:stop].to(device)
        action = actor_actions[start:stop].to(device).detach().requires_grad_(True)
        q1_raw, q2_raw = critic(state, action)
        q1, q2 = q1_raw.squeeze(-1), q2_raw.squeeze(-1)
        g1 = torch.autograd.grad(q1.sum(), action, retain_graph=True)[0]
        g2 = torch.autograd.grad(q2.sum(), action)[0]
        select_q1 = q1 <= q2
        gmin = torch.where(select_q1[:, None], g1, g2)
        min_q = torch.minimum(q1, q2).detach()
        cosine = torch.nn.functional.cosine_similarity(g1, g2, dim=-1, eps=1e-12)
        if actor.action_residual:
            lower, upper = actor.residual_reachable_interval(proposal)
            bound = actor.residual_delta_bound(action)
            if bound.numel() == 1:
                bound = bound.expand_as(action)
            else:
                bound = bound.expand_as(action)
        else:
            lower, upper = torch.full_like(action, -1.0), torch.full_like(action, 1.0)
            bound = torch.ones_like(action)

        batch_metrics: dict[str, Tensor] = {
            "q1_grad_norm": g1.norm(dim=-1),
            "q2_grad_norm": g2.norm(dim=-1),
            "min_q_grad_norm": gmin.norm(dim=-1),
            "twin_gradient_cosine": cosine,
            "twin_gradient_negative": (cosine < 0).float(),
            "min_head_is_q1": select_q1.float(),
        }
        for fraction in fractions:
            tag = f"eps_{fraction:g}"
            delta = fraction * bound * gmin.sign()
            plus = torch.maximum(torch.minimum(action.detach() + delta, upper), lower)
            minus = torch.maximum(torch.minimum(action.detach() - delta, upper), lower)
            random_delta = fraction * bound * random_signs[start:stop].to(device)
            random_action = torch.maximum(
                torch.minimum(action.detach() + random_delta, upper), lower
            )
            g1_action = torch.maximum(
                torch.minimum(action.detach() + fraction * bound * g1.sign(), upper), lower
            )
            g2_action = torch.maximum(
                torch.minimum(action.detach() + fraction * bound * g2.sign(), upper), lower
            )
            with torch.no_grad():
                plus_q1, plus_q2 = critic(state, plus)
                minus_q1, minus_q2 = critic(state, minus)
                random_q1, random_q2 = critic(state, random_action)
                cross_q1, _ = critic(state, g2_action)
                _, cross_q2 = critic(state, g1_action)
                plus_min = torch.minimum(plus_q1, plus_q2).squeeze(-1)
                minus_min = torch.minimum(minus_q1, minus_q2).squeeze(-1)
                random_min = torch.minimum(random_q1, random_q2).squeeze(-1)
            actual_gain = plus_min - min_q
            predicted_gain = (gmin * (plus - action.detach())).sum(dim=-1)
            batch_metrics[f"{tag}_plus_gain"] = actual_gain
            batch_metrics[f"{tag}_minus_gain"] = minus_min - min_q
            batch_metrics[f"{tag}_monotonic"] = (
                (plus_min > min_q) & (min_q > minus_min)
            ).float()
            batch_metrics[f"{tag}_gradient_beats_random"] = (
                actual_gain > (random_min - min_q)
            ).float()
            batch_metrics[f"{tag}_predicted_gain"] = predicted_gain
            batch_metrics[f"{tag}_q1_ascent_improves_q2"] = (
                cross_q2.squeeze(-1) > q2.detach()
            ).float()
            batch_metrics[f"{tag}_q2_ascent_improves_q1"] = (
                cross_q1.squeeze(-1) > q1.detach()
            ).float()

        for offset in range(stop - start):
            record = {
                "episode_uid": episode_uids[start + offset],
                "group": groups[start + offset],
            }
            record.update(
                {key: float(value[offset].detach().cpu().item()) for key, value in batch_metrics.items()}
            )
            records.append(record)

    scalar_keys = [key for key in records[0] if key not in {"episode_uid", "group"}]

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "samples": len(selected),
            "episodes": len({row["episode_uid"] for row in selected}),
        }
        if not selected:
            return summary
        for key in scalar_keys:
            values = torch.tensor([row[key] for row in selected])
            summary[key] = _quantiles(values)
        for fraction in fractions:
            tag = f"eps_{fraction:g}"
            summary[f"{tag}_finite_difference_vs_autograd_pearson"] = _pearson(
                [row[f"{tag}_predicted_gain"] for row in selected],
                [row[f"{tag}_plus_gain"] for row in selected],
            )
        return summary

    return {
        "overall": summarize(records),
        "by_group": {
            group: summarize([record for record in records if record["group"] == group])
            for group in GROUPS
        },
    }


def _nearest_distances(
    query: Tensor,
    reference: Tensor,
    *,
    exclude_reference_indices: Tensor | None = None,
    batch_size: int = 256,
) -> Tensor:
    nearest = []
    for start in range(0, len(query), batch_size):
        stop = min(len(query), start + batch_size)
        distances = torch.cdist(query[start:stop], reference)
        if exclude_reference_indices is not None:
            local_rows = torch.arange(stop - start)
            distances[local_rows, exclude_reference_indices[start:stop]] = float("inf")
        nearest.append(distances.min(dim=1).values)
    return torch.cat(nearest)


def _distribution_shift(
    reference_states: Tensor,
    reference_actions: Tensor,
    eval_states: Tensor,
    eval_actions: Tensor,
    *,
    proprio_dim: int,
    state_normalization: str,
    projection_dim: int,
    query_samples: int,
    seed: int,
) -> dict[str, Any]:
    reference_states = normalize_state_vec(
        reference_states, proprio_dim=proprio_dim, mode=state_normalization
    )
    eval_states = normalize_state_vec(
        eval_states, proprio_dim=proprio_dim, mode=state_normalization
    )
    reference = torch.cat([reference_states, reference_actions], dim=-1).float()
    evaluation = torch.cat([eval_states, eval_actions], dim=-1).float()
    mean = reference.mean(dim=0)
    std = reference.std(dim=0, unbiased=False).clamp_min(1e-6)
    reference_z = (reference - mean) / std
    evaluation_z = (evaluation - mean) / std
    state_dim = reference_states.shape[-1]

    def rms(values: Tensor, start: int, stop: int) -> Tensor:
        return values[:, start:stop].square().mean(dim=-1).sqrt()

    reference_rms = rms(reference_z, 0, reference_z.shape[-1])
    eval_rms = rms(evaluation_z, 0, evaluation_z.shape[-1])
    reference_state_rms = rms(reference_z, 0, state_dim)
    eval_state_rms = rms(evaluation_z, 0, state_dim)
    reference_action_rms = rms(reference_z, state_dim, reference_z.shape[-1])
    eval_action_rms = rms(evaluation_z, state_dim, evaluation_z.shape[-1])

    generator = torch.Generator().manual_seed(seed)
    projection_dim = min(projection_dim, reference_z.shape[-1])
    projection = torch.randn(
        reference_z.shape[-1], projection_dim, generator=generator
    ) / math.sqrt(reference_z.shape[-1])
    reference_projected = reference_z @ projection
    eval_projected = evaluation_z @ projection
    query_count = min(query_samples, len(reference_projected))
    query_indices = torch.randperm(len(reference_projected), generator=generator)[:query_count]
    reference_nn = _nearest_distances(
        reference_projected[query_indices],
        reference_projected,
        exclude_reference_indices=query_indices,
    )
    eval_nn = _nearest_distances(eval_projected, reference_projected)
    reference_p99 = float(torch.quantile(reference_rms, 0.99).item())
    reference_nn_p95 = float(torch.quantile(reference_nn, 0.95).item())
    eval_rms_exceeds_p99 = float((eval_rms > reference_p99).float().mean().item())
    eval_nn_median = float(torch.quantile(eval_nn, 0.5).item())
    strong_signal = eval_rms_exceeds_p99 > 0.20 or eval_nn_median > reference_nn_p95
    return {
        "reference_samples": len(reference),
        "evaluation_samples": len(evaluation),
        "standardized_rms": {
            "reference_joint": _quantiles(reference_rms),
            "evaluation_joint": _quantiles(eval_rms),
            "reference_state": _quantiles(reference_state_rms),
            "evaluation_state": _quantiles(eval_state_rms),
            "reference_action": _quantiles(reference_action_rms),
            "evaluation_action": _quantiles(eval_action_rms),
            "evaluation_fraction_above_reference_p99": eval_rms_exceeds_p99,
        },
        "random_projection_knn": {
            "projection_dim": projection_dim,
            "reference_leave_one_out": _quantiles(reference_nn),
            "evaluation_to_reference": _quantiles(eval_nn),
        },
        "heuristic": {
            "status": (
                "STRONG INPUT-SHIFT SIGNAL" if strong_signal else "NO STRONG INPUT-SHIFT SIGNAL"
            ),
            "rule": (
                "evaluation joint RMS exceeds reference p99 for >20% of samples, or "
                "evaluation projected-kNN median exceeds reference leave-one-out p95"
            ),
            "not_a_density_or_calibrated_ood_test": True,
        },
    }


def _outcome_report(
    records: list[dict[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    by_group = {group: [row for row in records if row["group"] == group] for group in GROUPS}
    comparisons = {
        "autonomous_success_minus_autonomous_failure": ("autonomous_success", "autonomous_failure"),
        "autonomous_success_minus_corrective_last_K": ("autonomous_success", "corrective_last_K"),
        "corrective_earlier_minus_corrective_last_K": ("corrective_earlier", "corrective_last_K"),
    }
    pairwise = {
        name: _pairwise_episode_difference(
            by_group[left],
            by_group[right],
            key="min_q",
            seed=seed + index,
            replicates=replicates,
        )
        for index, (name, (left, right)) in enumerate(comparisons.items())
    }
    episode_scores = {
        group: _episode_means(rows, "min_q") for group, rows in by_group.items()
    }
    return {
        "groups": {
            group: _summarize_score_records(rows, seed=seed, replicates=replicates)
            for group, rows in by_group.items()
        },
        "expected_positive_pairwise_differences": pairwise,
        "episode_level_auc": {
            "success_over_failure": _auc(
                episode_scores["autonomous_success"], episode_scores["autonomous_failure"]
            ),
            "success_over_corrective_last_K": _auc(
                episode_scores["autonomous_success"], episode_scores["corrective_last_K"]
            ),
            "corrective_earlier_over_corrective_last_K": _auc(
                episode_scores["corrective_earlier"], episode_scores["corrective_last_K"]
            ),
        },
    }


def run_critic_mechanism_audit(
    *,
    checkpoint: Path,
    actor_trust_dataset: Path,
    critic_train_cache: Path,
    output_path: Path,
    device_name: str = "cpu",
    batch_size: int = 256,
    bootstrap_replicates: int = 2000,
    perturbation_fractions: tuple[float, ...] = (0.01, 0.05, 0.1),
    max_reference_samples: int = 4096,
    reference_query_samples: int = 1024,
    projection_dim: int = 64,
    seed: int = 1000,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    actor_trust_dataset = actor_trust_dataset.expanduser().resolve(strict=True)
    critic_train_cache = critic_train_cache.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve()
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    if not perturbation_fractions or any(value <= 0 or value > 1 for value in perturbation_fractions):
        raise ValueError("perturbation fractions must lie in (0, 1]")

    metadata_path = actor_trust_dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    primary_k = int(metadata["semantics"]["primary_future_k"])
    trust_paths = [
        actor_trust_dataset / "actor_trust_train.pt",
        actor_trust_dataset / "actor_trust_val.pt",
    ]
    rows = _load_list(trust_paths[0]) + _load_list(trust_paths[1])
    if not rows:
        raise ValueError("actor-trust dataset is empty")
    legacy_count = sum(_scalar(row, "exec_action_is_actual_sent", 0.0) <= 0.5 for row in rows)
    if legacy_count:
        raise ValueError(
            f"behavior-action outcome audit requires actual-sent semantics; {legacy_count} rows are legacy"
        )

    config, actor, critic = _load_heads(checkpoint)
    device = torch.device(device_name)
    actor, critic = actor.to(device), critic.to(device)
    states = torch.stack(
        [torch.as_tensor(row["state_vec"], dtype=torch.float32).reshape(-1) for row in rows]
    )
    proposals = torch.stack(
        [torch.as_tensor(row["proposal_chunk"], dtype=torch.float32).reshape(-1) for row in rows]
    )
    behavior_actions = torch.stack(
        [torch.as_tensor(row["exec_chunk"], dtype=torch.float32).reshape(-1) for row in rows]
    )
    expected_state_dim = int(config["rl_token_dim"] + config["proprio_dim"])
    expected_chunk_dim = int(config["chunk_length"] * config["action_dim"])
    if states.shape[1] != expected_state_dim or proposals.shape[1] != expected_chunk_dim:
        raise ValueError(
            f"checkpoint/data shape mismatch: state {states.shape[1]} vs {expected_state_dim}, "
            f"action {proposals.shape[1]} vs {expected_chunk_dim}"
        )
    actor_actions = _actor_actions(
        actor, states, proposals, device=device, batch_size=batch_size
    )
    groups = [_semantic_group(row, primary_k) for row in rows]
    episode_uids = [str(row["episode_uid"]) for row in rows]

    outcome_by_action: dict[str, Any] = {}
    for action_name, actions in (
        ("behavior_actual_sent_primary", behavior_actions),
        ("checkpoint_actor_secondary", actor_actions),
    ):
        scores = _score_actions(
            critic, states, actions, device=device, batch_size=batch_size
        )
        records = [
            {
                "episode_uid": episode_uids[index],
                "group": groups[index],
                **{key: float(value[index].item()) for key, value in scores.items()},
            }
            for index in range(len(rows))
        ]
        outcome_by_action[action_name] = _outcome_report(
            records, seed=seed, replicates=bootstrap_replicates
        )

    reference_paths = [
        critic_train_cache / "chunk_transitions_train.pt",
        critic_train_cache / "chunk_transitions_val.pt",
    ]
    reference_all = _load_list(reference_paths[0]) + _load_list(reference_paths[1])
    reference_rows = [row for row in reference_all if _scalar(row, "critic_mask", 0.0) > 0.5]
    if len(reference_rows) < 2:
        raise ValueError("critic training cache has fewer than two critic_mask=1 rows")
    generator = torch.Generator().manual_seed(seed)
    if max_reference_samples > 0 and len(reference_rows) > max_reference_samples:
        indices = torch.randperm(len(reference_rows), generator=generator)[:max_reference_samples]
        reference_rows = [reference_rows[index] for index in indices.tolist()]
    reference_states = torch.stack(
        [torch.as_tensor(row["state_vec"], dtype=torch.float32).reshape(-1) for row in reference_rows]
    )
    reference_actions = torch.stack(
        [torch.as_tensor(row["exec_chunk"], dtype=torch.float32).reshape(-1) for row in reference_rows]
    )
    if (
        reference_states.shape[1:] != states.shape[1:]
        or reference_actions.shape[1:] != behavior_actions.shape[1:]
    ):
        raise ValueError("critic reference cache dimensions do not match actor-trust data")

    distribution_shift = {
        "behavior_actual_sent": _distribution_shift(
            reference_states,
            reference_actions,
            states,
            behavior_actions,
            proprio_dim=int(config["proprio_dim"]),
            state_normalization=config.get("state_normalization", "rl_token_layer_norm"),
            projection_dim=projection_dim,
            query_samples=reference_query_samples,
            seed=seed,
        ),
        "checkpoint_actor_action": _distribution_shift(
            reference_states,
            reference_actions,
            states,
            actor_actions,
            proprio_dim=int(config["proprio_dim"]),
            state_normalization=config.get("state_normalization", "rl_token_layer_norm"),
            projection_dim=projection_dim,
            query_samples=reference_query_samples,
            seed=seed,
        ),
    }
    gradient = _gradient_audit(
        actor,
        critic,
        states,
        proposals,
        actor_actions,
        groups,
        episode_uids,
        fractions=perturbation_fractions,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )

    primary_pairwise = outcome_by_action["behavior_actual_sent_primary"][
        "expected_positive_pairwise_differences"
    ]
    supported = []
    inconclusive = []
    contradicted = []
    for name, result in primary_pairwise.items():
        interval = result["episode_bootstrap_95ci"]
        if interval is None:
            inconclusive.append(name)
        elif interval[0] > 0:
            supported.append(name)
        elif interval[1] < 0:
            contradicted.append(name)
        else:
            inconclusive.append(name)
    smallest_tag = f"eps_{min(perturbation_fractions):g}"
    gradient_overall = gradient["overall"]
    cosine_median = gradient_overall["twin_gradient_cosine"]["median"]
    negative_fraction = gradient_overall["twin_gradient_negative"]["mean"]
    monotonic_fraction = gradient_overall[f"{smallest_tag}_monotonic"]["mean"]
    input_shift = any(
        section["heuristic"]["status"] == "STRONG INPUT-SHIFT SIGNAL"
        for section in distribution_shift.values()
    )
    report = {
        "schema_version": 1,
        "status": "COMPLETE",
        "checkpoint": {
            "path": str(checkpoint),
            "model_safetensors_sha256": _sha256_file(checkpoint / "model.safetensors"),
        },
        "actor_trust_dataset": {
            "path": str(actor_trust_dataset),
            "metadata_sha256": _sha256_file(metadata_path),
            "train_sha256": _sha256_file(trust_paths[0]),
            "val_sha256": _sha256_file(trust_paths[1]),
            "samples": len(rows),
            "episodes": len(set(episode_uids)),
            "group_counts": {group: groups.count(group) for group in GROUPS},
            "primary_future_k": primary_k,
            "all_actions_actual_sent": True,
        },
        "critic_training_reference": {
            "path": str(critic_train_cache),
            "train_sha256": _sha256_file(reference_paths[0]),
            "val_sha256": _sha256_file(reference_paths[1]),
            "all_rows": len(reference_all),
            "critic_mask_one_rows_before_sampling": sum(
                _scalar(row, "critic_mask", 0.0) > 0.5 for row in reference_all
            ),
            "reference_rows_used": len(reference_rows),
        },
        "semantics": {
            "outcome_ordering_primary": "Q(s, autonomous actual-sent behavior action)",
            "actor_action_ordering_secondary": "Q(s, frozen checkpoint actor action)",
            "gradient_start": "frozen checkpoint actor action",
            "perturbation_support": "clipped to actor residual reachable interval",
            "human_evidence_role": "excluded; never same-state autonomous Q alignment",
            "all_online48_splits_used_reason": (
                "the audited critic predates all online48 episodes; train/val here only split the risk head"
            ),
        },
        "outcome_ordering": outcome_by_action,
        "local_action_perturbation_and_twin_gradients": gradient,
        "input_distribution_shift": distribution_shift,
        "audit_flags": {
            "outcome_pairwise_supported": supported,
            "outcome_pairwise_inconclusive": inconclusive,
            "outcome_pairwise_contradicted": contradicted,
            "twin_gradient_coherent_heuristic": bool(
                cosine_median is not None
                and cosine_median >= 0.5
                and negative_fraction is not None
                and negative_fraction <= 0.1
            ),
            "smallest_step_local_monotonic_fraction": monotonic_fraction,
            "strong_input_shift_signal": input_shift,
            "no_single_metric_is_a_scientific_go_decision": True,
        },
        "limitations": [
            "Outcome groups differ in temporal progress and episode length; score separation is associative.",
            (
                "Random-projection kNN and diagonal standardized RMS are shift diagnostics, "
                "not calibrated density tests."
            ),
            "Local gradient ascent validates critic geometry, not real-robot improvement or safety.",
            "No proactive episodes exist in the current online48 dataset, so proactive groups may be empty.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--actor-trust-dataset", type=Path, required=True)
    parser.add_argument("--critic-train-cache", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--perturbation-fractions", type=float, nargs="+", default=[0.01, 0.05, 0.1]
    )
    parser.add_argument("--max-reference-samples", type=int, default=4096)
    parser.add_argument("--reference-query-samples", type=int, default=1024)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_critic_mechanism_audit(
        checkpoint=args.checkpoint,
        actor_trust_dataset=args.actor_trust_dataset,
        critic_train_cache=args.critic_train_cache,
        output_path=args.output_path,
        device_name=args.device,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap_replicates,
        perturbation_fractions=tuple(args.perturbation_fractions),
        max_reference_samples=args.max_reference_samples,
        reference_query_samples=args.reference_query_samples,
        projection_dim=args.projection_dim,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_path": str(args.output_path.expanduser().resolve()),
                "audit_flags": report["audit_flags"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
