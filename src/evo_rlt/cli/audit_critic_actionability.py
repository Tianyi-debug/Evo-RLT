"""Offline critic qualification and cross-fitted reachable-improvement audit.

The audit is update-relative: it evaluates a trained critic together with an
actor, a concrete one-step direct-Q optimizer update, and the actor's residual
reachable set.  RIR is an offline reliability proxy, not evidence that either
critic is environmentally correct.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor, nn

from evo_rlt.cli.audit_actor_q_mechanism import _construct_heads, _load_heads
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic
from evo_rlt.core.losses import discounted_chunk_return, resolve_td_bootstrap_mask


SCHEMA_VERSION = 1
ZERO_GAIN_EPSILON = 1e-8


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _cache_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = [
            candidate
            for name in ("chunk_transitions_train.pt", "chunk_transitions_val.pt")
            if (candidate := path / name).is_file()
        ]
        if files:
            return files
    raise FileNotFoundError(
        "validation cache must be a .pt file or a directory containing "
        f"chunk_transitions_train.pt/val.pt: {path}"
    )


def _cache_metadata(path: Path) -> dict[str, Any]:
    files = _cache_files(path)
    file_records = [
        {"path": str(file), "sha256": _sha256_file(file)} for file in files
    ]
    digest = hashlib.sha256()
    for record in file_records:
        digest.update(Path(record["path"]).name.encode())
        digest.update(record["sha256"].encode())
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "files": file_records,
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard, file in enumerate(_cache_files(path)):
        value = torch.load(file, map_location="cpu", weights_only=False)
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise TypeError(f"expected list[dict] transition cache, got {type(value)!r}")
        for row in value:
            copied = dict(row)
            copied["_audit_cache_shard"] = shard
            rows.append(copied)
    if not rows:
        raise ValueError(f"transition cache is empty: {path}")
    return rows


def _tensor(row: dict[str, Any], key: str, dtype: torch.dtype = torch.float32) -> Tensor:
    if key not in row:
        raise KeyError(f"transition row is missing {key!r}")
    return torch.as_tensor(row[key], dtype=dtype)


def _scalar(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    if key not in row:
        return default
    return float(torch.as_tensor(row[key]).reshape(-1)[0].item())


def _episode_id(row: dict[str, Any]) -> str:
    return f"{int(row.get('_audit_cache_shard', 0))}:{int(_scalar(row, 'episode_id', -1))}"


def _stack(rows: list[dict[str, Any]], key: str, *, flatten_chunk: bool = False) -> Tensor:
    values = [_tensor(row, key) for row in rows]
    result = torch.stack(values)
    return result.flatten(start_dim=-2) if flatten_chunk else result


def _quantiles(values: Tensor) -> dict[str, float | None]:
    values = values.detach().float().reshape(-1).cpu()
    values = values[torch.isfinite(values)]
    if not len(values):
        return {key: None for key in ("mean", "p10", "p25", "median", "p75", "p90")}
    qs = torch.quantile(values, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return {
        "mean": float(values.mean().item()),
        "p10": float(qs[0].item()),
        "p25": float(qs[1].item()),
        "median": float(qs[2].item()),
        "p75": float(qs[3].item()),
        "p90": float(qs[4].item()),
    }


def _iqr_summary(values: list[float]) -> dict[str, float | None]:
    summary = _quantiles(torch.tensor(values))
    return {
        "mean": summary["mean"],
        "median": summary["median"],
        "iqr": (
            [summary["p25"], summary["p75"]]
            if summary["p25"] is not None
            else None
        ),
    }


def _auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += float(pos > neg) + 0.5 * float(pos == neg)
    return wins / (len(positive) * len(negative))


def _rankdata(values: Tensor) -> Tensor:
    values = values.detach().float().reshape(-1)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(x: Tensor, y: Tensor, *, rank: bool = False) -> float | None:
    x = x.detach().float().reshape(-1).cpu()
    y = y.detach().float().reshape(-1).cpu()
    valid = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return None
    if rank:
        x, y = _rankdata(x), _rankdata(y)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denominator <= 0:
        return None
    return float((x * y).sum().div(denominator).item())


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _bootstrap_statistic(
    records: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[str(record["episode_uid"])].append(record)
    episode_ids = sorted(by_episode)
    if len(episode_ids) < 2:
        return {"replicates": 0, "median": None, "ci95": None}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled: list[dict[str, Any]] = []
        for _ in episode_ids:
            sampled.extend(by_episode[episode_ids[rng.randrange(len(episode_ids))]])
        estimate = statistic(sampled)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return {"replicates": 0, "median": None, "ci95": None}
    return {
        "replicates": len(estimates),
        "median": _percentile(estimates, 0.5),
        "ci95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
    }


def _episode_metadata(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], Tensor]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_episode_id(row)].append(index)
    groups = ["unknown"] * len(rows)
    progress = torch.zeros(len(rows), dtype=torch.float32)
    for episode, indices in grouped.items():
        episode_rows = [rows[index] for index in indices]
        sources = {int(_scalar(row, "source")) for row in episode_rows}
        reasons = {int(_scalar(row, "intervention_reason")) for row in episode_rows}
        reward = sum(float(_tensor(row, "reward_seq").sum().item()) for row in episode_rows)
        terminal = any(_scalar(row, "done") > 0.5 for row in episode_rows)
        if sources == {0}:
            group = "demonstration_success"
        elif 1 in reasons:
            group = "corrective_associated"
        elif 2 in reasons:
            group = "proactive_associated"
        elif sources == {2} and terminal:
            group = "autonomous_success" if reward > 0 else "autonomous_failure"
        else:
            group = "other"
        denominator = max(len(indices) - 1, 1)
        for position, index in enumerate(indices):
            groups[index] = group
            progress[index] = position / denominator
    return [_episode_id(row) for row in rows], groups, progress


def _collate(rows: list[dict[str, Any]], indices: Tensor, device: torch.device) -> dict[str, Tensor]:
    selected = [rows[int(index)] for index in indices.tolist()]
    result = {
        "state_vec": _stack(selected, "state_vec").to(device),
        "exec_chunk_flat": _stack(selected, "exec_chunk", flatten_chunk=True).to(device),
        "proposal_chunk_flat": _stack(selected, "proposal_chunk", flatten_chunk=True).to(device),
        "next_state_vec": _stack(selected, "next_state_vec").to(device),
        "next_proposal_flat": _stack(
            selected, "next_proposal_chunk", flatten_chunk=True
        ).to(device),
        "reward_seq": _stack(selected, "reward_seq").to(device),
        "done": torch.tensor([_scalar(row, "done") for row in selected], device=device),
        "actual_steps": torch.tensor(
            [int(_scalar(row, "actual_steps", len(_tensor(row, "reward_seq")))) for row in selected],
            device=device,
        ),
        "bootstrap_mask": torch.tensor(
            [_scalar(row, "bootstrap_mask", 1.0 - _scalar(row, "done")) for row in selected],
            device=device,
        ),
        "critic_mask": torch.tensor(
            [_scalar(row, "critic_mask", 1.0) for row in selected], device=device
        ),
        "actor_q_mask": torch.tensor(
            [_scalar(row, "actor_q_mask", 1.0) for row in selected], device=device
        ),
    }
    return result


def _load_target_critic(checkpoint: Path, config: dict[str, Any]) -> TwinCritic:
    _, target = _construct_heads(config)
    state = load_safetensors_file(str(checkpoint / "model.safetensors"), device="cpu")
    prefix = "target_critic."
    values = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    if not values:
        raise KeyError(f"checkpoint has no target_critic tensors: {checkpoint}")
    target.load_state_dict(values, strict=True)
    target.eval()
    return target


def _score_critic(
    critic: TwinCritic,
    states: Tensor,
    actions: Tensor,
    *,
    mode: str,
) -> Tensor:
    q1, q2 = critic(states, actions)
    if mode == "min":
        return torch.minimum(q1, q2).reshape(-1)
    if mode == "q1":
        return q1.reshape(-1)
    if mode == "q2":
        return q2.reshape(-1)
    raise ValueError(f"unsupported critic score mode: {mode}")


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    train_config_path = path / "train_config.json"
    config = json.loads(config_path.read_text())
    train = json.loads(train_config_path.read_text()) if train_config_path.is_file() else {}
    policy_train = train.get("policy", {})
    provenance_path = path / "independent_critic_fit.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    dataset = train.get("dataset", {}).get("repo_id")
    return {
        "path": str(path),
        "model_sha256": _sha256_file(path / "model.safetensors"),
        "config_sha256": _sha256_file(config_path),
        "train_config_sha256": (
            _sha256_file(train_config_path) if train_config_path.is_file() else None
        ),
        "seed": train.get("seed"),
        "dataset": dataset,
        "dataset_name": Path(dataset).name if dataset else None,
        "pretrained_path": config.get("pretrained_path", policy_train.get("pretrained_path")),
        "optimizer": train.get("optimizer"),
        "critic_bootstrap_mode": config.get(
            "critic_bootstrap_mode", policy_train.get("critic_bootstrap_mode", "none")
        ),
        "critic_bootstrap_seed": config.get(
            "critic_bootstrap_seed", policy_train.get("critic_bootstrap_seed")
        ),
        "independent_fit_provenance": provenance,
    }


def _artifact_path_identity(path_value: str | None) -> str | None:
    if not path_value:
        return None
    marker = "/outputs/"
    normalized = str(Path(path_value).expanduser())
    return normalized.split(marker, 1)[1] if marker in normalized else normalized


def _independence_audit(
    critic_a: Path, critic_b: Path, *, twin_head_approximation: bool
) -> dict[str, Any]:
    meta_a, meta_b = _checkpoint_metadata(critic_a), _checkpoint_metadata(critic_b)
    if twin_head_approximation:
        status = "TWIN_HEAD_APPROXIMATION"
        limitations = [
            "q1/q2 were trained in one TwinCritic module",
            "the heads share optimizer steps and training samples",
            "critic_bootstrap_mode=none does not create independent data subsets",
        ]
    else:
        distinct_weights = meta_a["model_sha256"] != meta_b["model_sha256"]
        distinct_seed = meta_a["seed"] is not None and meta_a["seed"] != meta_b["seed"]
        start_a = meta_a.get("pretrained_path")
        start_b = meta_b.get("pretrained_path")
        shared_start = bool(
            start_a
            and start_b
            and _artifact_path_identity(start_a) == _artifact_path_identity(start_b)
        )
        if distinct_weights and distinct_seed and shared_start:
            status = "SEPARATE_SEED_FITS_SHARED_START"
            limitations = [
                "the fits use different stochastic batch orders but continue from a shared pretrained actor/critic checkpoint",
                "the fits use the same cached training distribution rather than disjoint transition folds",
            ]
        elif distinct_weights and distinct_seed:
            status = "DISTINCT_SEEDS_DECLARED_START_UNVERIFIED"
            limitations = ["the two train configs do not document a matching pretrained start"]
        else:
            status = "INDEPENDENCE_UNVERIFIED"
            limitations = ["checkpoints do not document distinct training seeds and weights"]
    return {
        "status": status,
        "critic_a": meta_a,
        "critic_b": meta_b,
        "same_training_dataset_declared": meta_a["dataset_name"] == meta_b["dataset_name"],
        "limitations": limitations,
    }


def run_qualification(
    *,
    checkpoint: Path,
    validation_cache: Path,
    device: torch.device,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    rows = _load_rows(validation_cache)
    config, actor, critic = _load_heads(checkpoint)
    target_critic = _load_target_critic(checkpoint, config)
    actor, critic, target_critic = actor.to(device), critic.to(device), target_critic.to(device)
    episode_uids, groups, progress = _episode_metadata(rows)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(rows)))
            batch = _collate(rows, indices, device)
            q1, q2 = critic(batch["state_vec"], batch["exec_chunk_flat"])
            next_action, _ = actor(
                batch["next_state_vec"], batch["next_proposal_flat"], training=False
            )
            next_action = next_action.clamp(-1.0, 1.0)
            q_next = target_critic.min_q(batch["next_state_vec"], next_action)
            target_clip = config.get("target_q_clip", 100.0)
            if target_clip is not None and float(target_clip) > 0:
                q_next = q_next.clamp(-float(target_clip), float(target_clip))
            reward = discounted_chunk_return(
                batch["reward_seq"], float(config["gamma"]), batch["actual_steps"]
            )
            bootstrap_gate = resolve_td_bootstrap_mask(batch, batch["done"])
            exponent = batch["actual_steps"].unsqueeze(-1).float()
            target = reward + (float(config["gamma"]) ** exponent) * bootstrap_gate.unsqueeze(-1) * q_next
            for offset, index in enumerate(indices.tolist()):
                if batch["critic_mask"][offset] <= 0.5:
                    continue
                q1_value, q2_value = float(q1[offset]), float(q2[offset])
                target_value = float(target[offset])
                records.append(
                    {
                        "cache_index": index,
                        "episode_uid": episode_uids[index],
                        "group": groups[index],
                        "progress": float(progress[index]),
                        "q1": q1_value,
                        "q2": q2_value,
                        "min_q": min(q1_value, q2_value),
                        "target": target_value,
                        "abs_td_error": 0.5 * (
                            abs(q1_value - target_value) + abs(q2_value - target_value)
                        ),
                    }
                )
    if not records:
        raise ValueError("validation cache has no critic_mask-valid rows")

    outcome = [row for row in records if row["group"] in {"autonomous_success", "autonomous_failure"}]
    episode_q: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in outcome:
        episode_q[(record["group"], record["episode_uid"])].append(record["min_q"])
    success = [sum(values) / len(values) for (group, _), values in episode_q.items() if group == "autonomous_success"]
    failure = [sum(values) / len(values) for (group, _), values in episode_q.items() if group == "autonomous_failure"]

    def outcome_difference(sample: list[dict[str, Any]]) -> float | None:
        by_group: dict[str, list[float]] = defaultdict(list)
        for row in sample:
            by_group[row["group"]].append(row["min_q"])
        if not by_group["autonomous_success"] or not by_group["autonomous_failure"]:
            return None
        return sum(by_group["autonomous_success"]) / len(by_group["autonomous_success"]) - sum(by_group["autonomous_failure"]) / len(by_group["autonomous_failure"])

    progress_rows = outcome
    progress_q = torch.tensor([row["min_q"] for row in progress_rows])
    progress_values = torch.tensor([row["progress"] for row in progress_rows])
    bins: dict[str, Any] = {}
    for bin_index in range(5):
        selected = [
            row for row in outcome
            if min(int(row["progress"] * 5), 4) == bin_index
        ]
        success_values = [row["min_q"] for row in selected if row["group"] == "autonomous_success"]
        failure_values = [row["min_q"] for row in selected if row["group"] == "autonomous_failure"]
        bins[str(bin_index)] = {
            "success": _iqr_summary(success_values),
            "failure": _iqr_summary(failure_values),
            "success_minus_failure_mean": (
                sum(success_values) / len(success_values) - sum(failure_values) / len(failure_values)
                if success_values and failure_values else None
            ),
            "success_minus_failure_episode_bootstrap": _bootstrap_statistic(
                selected,
                outcome_difference,
                seed=seed + 50 + bin_index,
                replicates=bootstrap_replicates,
            ),
            "success_episodes": len({row["episode_uid"] for row in selected if row["group"] == "autonomous_success"}),
            "failure_episodes": len({row["episode_uid"] for row in selected if row["group"] == "autonomous_failure"}),
        }

    td_values = torch.tensor([row["abs_td_error"] for row in records])
    q_values = torch.tensor([row["min_q"] for row in records])
    target_values = torch.tensor([row["target"] for row in records])
    auc = _auc(success, failure)
    spearman = _correlation(progress_values, progress_q, rank=True)
    finite = bool(torch.isfinite(td_values).all() and torch.isfinite(q_values).all())
    outcome_signal = auc is not None and auc > 0.5
    progress_signal = spearman is not None and abs(spearman) >= 0.10
    outcome_sample_sufficient = len(success) >= 3 and len(failure) >= 3
    if finite and outcome_signal and progress_signal and outcome_sample_sufficient:
        verdict = "VALUE_COMPETENCE_PASS"
    elif finite and (outcome_signal or progress_signal):
        verdict = "VALUE_COMPETENCE_WEAK"
    else:
        verdict = "VALUE_COMPETENCE_FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": _checkpoint_metadata(checkpoint),
        "validation_cache": {
            **_cache_metadata(validation_cache),
            "rows": len(rows),
            "critic_valid_rows": len(records),
        },
        "held_out_td_behavior": {
            "absolute_td_error": _quantiles(td_values),
            "q_scale": _quantiles(q_values),
            "target_scale": _quantiles(target_values),
            "note": "Sparse-reward absolute TD error is scale- and target-policy-dependent; low error alone does not establish critic correctness.",
        },
        "autonomous_outcome_discrimination": {
            "success_episodes": len(success),
            "failure_episodes": len(failure),
            "success_q": _iqr_summary(success),
            "failure_q": _iqr_summary(failure),
            "episode_level_auroc": auc,
            "success_minus_failure_episode_bootstrap": _bootstrap_statistic(
                outcome, outcome_difference, seed=seed, replicates=bootstrap_replicates
            ),
        },
        "progress_dependence": {
            "rows": len(progress_rows),
            "pearson_q_vs_normalized_progress": _correlation(progress_values, progress_q),
            "spearman_q_vs_normalized_progress": spearman,
            "interpretation": "trajectory/value structure only; not actionability evidence",
        },
        "progress_controlled_outcome_ranking": bins,
        "twin_head_value_consistency": {
            "pearson": _correlation(
                torch.tensor([row["q1"] for row in records]),
                torch.tensor([row["q2"] for row in records]),
            ),
            "spearman": _correlation(
                torch.tensor([row["q1"] for row in records]),
                torch.tensor([row["q2"] for row in records]),
                rank=True,
            ),
            "not_independent_replicates": True,
        },
        "verdict_rule": {
            "finite_td_and_q": finite,
            "outcome_auroc_above_chance": outcome_signal,
            "at_least_three_outcome_episodes_per_class": outcome_sample_sufficient,
            "absolute_progress_spearman_at_least_0.10": progress_signal,
        },
        "value_competence_verdict": verdict,
    }


def _actor_actions(actor: ChunkActor, states: Tensor, proposals: Tensor) -> Tensor:
    actor.eval()
    actions, _ = actor(states, proposals, training=False)
    return actions.clamp(-1.0, 1.0)


def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _virtual_q_update(
    *,
    actor: ChunkActor,
    critic: TwinCritic,
    score_mode: str,
    states: Tensor,
    proposals: Tensor,
    lambda_q: float,
    actor_lr: float,
) -> tuple[ChunkActor, dict[str, Any]]:
    updated = copy.deepcopy(actor).train()
    critic_digest_before = _state_digest(critic)
    critic_requires_grad = [parameter.requires_grad for parameter in critic.parameters()]
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(updated.parameters(), lr=actor_lr, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    actions, _ = updated(states, proposals, training=False)
    q = _score_critic(critic, states, actions, mode=score_mode)
    loss = -lambda_q * q.mean()
    loss.backward()
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(updated.parameters(), 1.0)
    optimizer.step()
    for parameter, requires_grad in zip(critic.parameters(), critic_requires_grad, strict=True):
        parameter.requires_grad_(requires_grad)
    if _state_digest(critic) != critic_digest_before:
        raise AssertionError("virtual actor update modified critic parameters")
    updated.eval()
    return updated, {
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "actor_lr": actor_lr,
        "lambda_q": lambda_q,
        "loss": float(loss.detach().item()),
        "pre_clip_gradient_norm": float(pre_clip_norm.detach().item()),
        "optimizer_steps": 1,
        "q_channel_only": True,
    }


def _gain_summary(values: Tensor) -> dict[str, Any]:
    result = _quantiles(values)
    result.update(
        {
            "fraction_positive": float((values > 0).float().mean().item()),
            "fraction_approximately_zero": float(
                (values.abs() <= ZERO_GAIN_EPSILON).float().mean().item()
            ),
            "approximately_zero_epsilon": ZERO_GAIN_EPSILON,
        }
    )
    return result


def _rir_stat(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return sum(float(record[key] > 0) for record in records) / len(records)


def _rir_combined(records: list[dict[str, Any]]) -> float | None:
    first, second = _rir_stat(records, "cross_gain_1_to_2"), _rir_stat(records, "cross_gain_2_to_1")
    return None if first is None or second is None else 0.5 * (first + second)


def _breakdown(records: list[dict[str, Any]], *, seed: int, replicates: int) -> dict[str, Any]:
    if not records:
        return {"samples": 0, "episodes": 0}
    return {
        "samples": len(records),
        "episodes": len({row["episode_uid"] for row in records}),
        "rir": _rir_combined(records),
        "rir_1_to_2": _rir_stat(records, "cross_gain_1_to_2"),
        "rir_2_to_1": _rir_stat(records, "cross_gain_2_to_1"),
        "episode_bootstrap": _bootstrap_statistic(
            records, _rir_combined, seed=seed, replicates=replicates
        ),
    }


def run_rir(
    *,
    actor_checkpoint: Path,
    critic_a_checkpoint: Path,
    critic_b_checkpoint: Path,
    validation_cache: Path,
    device: torch.device,
    batch_size: int,
    lambda_q: float,
    actor_lr_override: float | None,
    bootstrap_replicates: int,
    seed: int,
    twin_head_approximation: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if lambda_q <= 0 or batch_size <= 0:
        raise ValueError("lambda_q and batch_size must be positive")
    if twin_head_approximation and critic_a_checkpoint != critic_b_checkpoint:
        raise ValueError("twin-head approximation requires the same critic checkpoint")
    rows = _load_rows(validation_cache)
    actor_config, actor, _ = _load_heads(actor_checkpoint)
    critic_a_config, _, critic_a = _load_heads(critic_a_checkpoint)
    critic_b_config, _, critic_b = _load_heads(critic_b_checkpoint)
    shape_keys = ("rl_token_dim", "proprio_dim", "chunk_length", "action_dim")
    for key in shape_keys:
        if actor_config[key] != critic_a_config[key] or actor_config[key] != critic_b_config[key]:
            raise ValueError(f"actor/critic {key} mismatch")
    actor_lr = float(
        actor_lr_override
        if actor_lr_override is not None
        else actor_config.get("actor_lr") or actor_config.get("training_lr")
    )
    actor, critic_a, critic_b = actor.to(device), critic_a.to(device), critic_b.to(device)
    actor.eval(), critic_a.eval(), critic_b.eval()
    episode_uids, groups, progress = _episode_metadata(rows)
    valid_indices = torch.tensor(
        [index for index, row in enumerate(rows) if _scalar(row, "actor_q_mask", 1.0) > 0.5],
        dtype=torch.long,
    )
    if not len(valid_indices):
        raise ValueError("validation cache has no actor_q_mask-valid rows")
    generator = torch.Generator().manual_seed(seed)
    permutation = valid_indices[torch.randperm(len(valid_indices), generator=generator)]
    update_indices = permutation[: min(batch_size, len(permutation))]
    update_batch = _collate(rows, update_indices, device)
    states = _stack([rows[int(index)] for index in valid_indices], "state_vec").to(device)
    proposals = _stack(
        [rows[int(index)] for index in valid_indices], "proposal_chunk", flatten_chunk=True
    ).to(device)
    actor_digest_before = _state_digest(actor)
    critic_a_digest_before, critic_b_digest_before = _state_digest(critic_a), _state_digest(critic_b)
    mode_a, mode_b = ("q1", "q2") if twin_head_approximation else ("min", "min")
    actor_1, update_1 = _virtual_q_update(
        actor=actor,
        critic=critic_a,
        score_mode=mode_a,
        states=update_batch["state_vec"],
        proposals=update_batch["proposal_chunk_flat"],
        lambda_q=lambda_q,
        actor_lr=actor_lr,
    )
    actor_2, update_2 = _virtual_q_update(
        actor=actor,
        critic=critic_b,
        score_mode=mode_b,
        states=update_batch["state_vec"],
        proposals=update_batch["proposal_chunk_flat"],
        lambda_q=lambda_q,
        actor_lr=actor_lr,
    )
    if _state_digest(actor) != actor_digest_before:
        raise AssertionError("original actor changed during virtual updates")
    if _state_digest(critic_a) != critic_a_digest_before or _state_digest(critic_b) != critic_b_digest_before:
        raise AssertionError("critic changed during RIR")
    with torch.no_grad():
        base_action = _actor_actions(actor, states, proposals)
        action_1 = _actor_actions(actor_1, states, proposals)
        action_2 = _actor_actions(actor_2, states, proposals)
        q1_base = _score_critic(critic_a, states, base_action, mode=mode_a)
        q1_action_1 = _score_critic(critic_a, states, action_1, mode=mode_a)
        q1_action_2 = _score_critic(critic_a, states, action_2, mode=mode_a)
        q2_base = _score_critic(critic_b, states, base_action, mode=mode_b)
        q2_action_1 = _score_critic(critic_b, states, action_1, mode=mode_b)
        q2_action_2 = _score_critic(critic_b, states, action_2, mode=mode_b)
    self_1, self_2 = q1_action_1 - q1_base, q2_action_2 - q2_base
    cross_1_to_2, cross_2_to_1 = q2_action_1 - q2_base, q1_action_2 - q1_base
    delta_1, delta_2 = action_1 - base_action, action_2 - base_action
    lower, upper = actor.residual_reachable_interval(proposals) if actor.action_residual else (
        torch.full_like(base_action, -1.0), torch.full_like(base_action, 1.0)
    )
    if (action_1 < lower - 1e-6).any() or (action_1 > upper + 1e-6).any() or (action_2 < lower - 1e-6).any() or (action_2 > upper + 1e-6).any():
        raise AssertionError("virtual update produced actions outside actor residual bounds")
    bound = actor.residual_delta_bound(base_action).expand_as(base_action) if actor.action_residual else torch.ones_like(base_action)
    chunk_length, action_dim = int(actor_config["chunk_length"]), int(actor_config["action_dim"])
    if base_action.shape[1] != chunk_length * action_dim:
        raise AssertionError("actor action chunk shape changed")
    records: list[dict[str, Any]] = []
    for offset, cache_index in enumerate(valid_indices.tolist()):
        records.append(
            {
                "cache_index": cache_index,
                "episode_uid": episode_uids[cache_index],
                "group": groups[cache_index],
                "progress": float(progress[cache_index]),
                "progress_bin": min(int(float(progress[cache_index]) * 5), 4),
                "self_gain_1": float(self_1[offset]),
                "self_gain_2": float(self_2[offset]),
                "cross_gain_1_to_2": float(cross_1_to_2[offset]),
                "cross_gain_2_to_1": float(cross_2_to_1[offset]),
                "delta_action_1": delta_1[offset].detach().cpu(),
                "delta_action_2": delta_2[offset].detach().cpu(),
            }
        )
    overall = _breakdown(records, seed=seed, replicates=bootstrap_replicates)
    by_group = {
        group: _breakdown(
            [row for row in records if row["group"] == group],
            seed=seed + index + 1,
            replicates=bootstrap_replicates,
        )
        for index, group in enumerate(sorted(set(groups)))
    }
    by_progress = {
        str(index): _breakdown(
            [row for row in records if row["progress_bin"] == index],
            seed=seed + 20 + index,
            replicates=bootstrap_replicates,
        )
        for index in range(5)
    }
    delta_mean = 0.5 * (delta_1.abs() + delta_2.abs())
    normalized_delta = delta_mean / bound.clamp_min(1e-12)
    chunk_delta = delta_mean.reshape(-1, chunk_length, action_dim)
    chunk_diagnostics = {
        "whole_chunk_l2": _quantiles(
            0.5 * (delta_1.norm(dim=-1) + delta_2.norm(dim=-1))
        ),
        "normalized_by_residual_bounds_rms": _quantiles(
            normalized_delta.square().mean(dim=-1).sqrt()
        ),
        "mean_absolute_shift_per_chunk_timestep": chunk_delta.mean(dim=(0, 2)).tolist(),
        "mean_absolute_shift_per_action_dimension": chunk_delta.mean(dim=(0, 1)).tolist(),
        "gripper_dimension": action_dim - 1,
        "gripper_mean_absolute_shift": float(chunk_delta[:, :, action_dim - 1].mean()),
        "translation_related_dimensions": list(range(min(3, action_dim))),
        "translation_mean_absolute_shift": float(chunk_delta[:, :, : min(3, action_dim)].mean()),
    }
    update_fingerprint = hashlib.sha256(
        json.dumps(update_indices.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema_version": SCHEMA_VERSION,
        "method": "Cross-Fitted Reachable Improvement Rate",
        "interpretation": "Offline reliability proxy: an actor-reachable update induced by one critic is locally supported by another critic. It does not prove environmental correctness.",
        "actor_checkpoint": _checkpoint_metadata(actor_checkpoint),
        "critic_independence": _independence_audit(
            critic_a_checkpoint, critic_b_checkpoint,
            twin_head_approximation=twin_head_approximation,
        ),
        "validation_cache": {
            **_cache_metadata(validation_cache),
            "rows": len(rows),
            "actor_q_valid_rows": len(valid_indices),
        },
        "update_operator": {
            "definition": "one real actor-parameter AdamW update using only -lambda_Q * mean(Q), actor_q_mask-valid states, and the checkpoint actor parameterization",
            "direction_1": update_1,
            "direction_2": update_2,
            "same_theta0": True,
            "same_update_batch": True,
            "update_batch_indices": update_indices.tolist(),
            "audit_batch_fingerprint": update_fingerprint,
            "q_normalization": "masked arithmetic mean; no additional Q normalization exists in actor_refine",
        },
        "overall": overall,
        "gain_diagnostics": {
            "self_1": _gain_summary(self_1),
            "self_2": _gain_summary(self_2),
            "cross_1_to_2": _gain_summary(cross_1_to_2),
            "cross_2_to_1": _gain_summary(cross_2_to_1),
            "self_cross_gap_mean": float(
                0.5 * (self_1.mean() + self_2.mean())
                - 0.5 * (cross_1_to_2.mean() + cross_2_to_1.mean())
            ),
        },
        "replicate_value_consistency_on_actor_actions": {
            "pearson": _correlation(q1_base, q2_base),
            "spearman": _correlation(q1_base, q2_base, rank=True),
            "note": "Coarse value-ordering consistency only; this is not update-direction evidence.",
        },
        "semantic_breakdown": by_group,
        "progress_breakdown": by_progress,
        "chunk_diagnostics": chunk_diagnostics,
        "pollution_checks": {
            "original_actor_bit_identical": _state_digest(actor) == actor_digest_before,
            "critic_a_bit_identical": _state_digest(critic_a) == critic_a_digest_before,
            "critic_b_bit_identical": _state_digest(critic_b) == critic_b_digest_before,
            "residual_bounds_preserved": True,
            "chunk_shape_preserved": True,
        },
    }
    bootstrap = {
        "unit": "episode",
        "seed": seed,
        "requested_replicates": bootstrap_replicates,
        "overall": overall["episode_bootstrap"],
        "directions": {
            "rir_1_to_2": _bootstrap_statistic(
                records,
                lambda sample: _rir_stat(sample, "cross_gain_1_to_2"),
                seed=seed + 100,
                replicates=bootstrap_replicates,
            ),
            "rir_2_to_1": _bootstrap_statistic(
                records,
                lambda sample: _rir_stat(sample, "cross_gain_2_to_1"),
                seed=seed + 101,
                replicates=bootstrap_replicates,
            ),
        },
    }
    return report, records, bootstrap


def _markdown(qualification: dict[str, Any], rir: dict[str, Any]) -> str:
    outcome = qualification["autonomous_outcome_discrimination"]
    overall = rir["overall"]
    bootstrap = overall["episode_bootstrap"]
    return "\n".join(
        [
            "# Critic suitability / actionability audit",
            "",
            f"- Value competence verdict: `{qualification['value_competence_verdict']}`",
            f"- Autonomous outcome AUROC: `{outcome['episode_level_auroc']}`",
            f"- Independence status: `{rir['critic_independence']['status']}`",
            f"- RIR: `{overall['rir']}`",
            f"- RIR 1->2: `{overall['rir_1_to_2']}`",
            f"- RIR 2->1: `{overall['rir_2_to_1']}`",
            f"- Episode-bootstrap 95% CI: `{bootstrap['ci95']}`",
            "",
            "RIR is an offline reliability proxy. It measures cross-critic local support for an actual actor-parameter update; it is not ground-truth policy improvement.",
            "",
        ]
    )


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    actor_checkpoint = args.actor_checkpoint.expanduser().resolve(strict=True)
    critic_a = args.critic_a_checkpoint.expanduser().resolve(strict=True)
    critic_b = (args.critic_b_checkpoint or args.critic_a_checkpoint).expanduser().resolve(strict=True)
    validation_cache = args.validation_cache.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    qualification = run_qualification(
        checkpoint=critic_a,
        validation_cache=validation_cache,
        device=device,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap_reps,
        seed=args.seed,
    )
    rir, records, bootstrap = run_rir(
        actor_checkpoint=actor_checkpoint,
        critic_a_checkpoint=critic_a,
        critic_b_checkpoint=critic_b,
        validation_cache=validation_cache,
        device=device,
        batch_size=args.batch_size,
        lambda_q=args.lambda_q,
        actor_lr_override=args.actor_lr,
        bootstrap_replicates=args.bootstrap_reps,
        seed=args.seed,
        twin_head_approximation=args.twin_head_approximation,
    )
    resolved = {
        "git_commit": _git_commit(),
        "actor_checkpoint": str(actor_checkpoint),
        "critic_a_checkpoint": str(critic_a),
        "critic_b_checkpoint": str(critic_b),
        "validation_cache": str(validation_cache),
        "lambda_q": args.lambda_q,
        "actor_lr_override": args.actor_lr,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "bootstrap_reps": args.bootstrap_reps,
        "device": args.device,
        "twin_head_approximation": args.twin_head_approximation,
    }
    (output_dir / "qualification_report.json").write_text(json.dumps(qualification, indent=2) + "\n")
    (output_dir / "rir_report.json").write_text(json.dumps(rir, indent=2) + "\n")
    (output_dir / "actionability_report.json").write_text(
        json.dumps({"qualification": qualification, "rir": rir}, indent=2) + "\n"
    )
    (output_dir / "actionability_report.md").write_text(_markdown(qualification, rir))
    torch.save(records, output_dir / "rir_samples.pt")
    (output_dir / "bootstrap_summary.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    (output_dir / "config_resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
    return {"qualification": qualification, "rir": rir, "bootstrap": bootstrap}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-a-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-b-checkpoint", type=Path)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--lambda-q", type=float, default=0.25)
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--twin-head-approximation",
        action="store_true",
        help="Treat q1/q2 from one TwinCritic as the weaker cross-fit approximation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_audit(args)
    print(json.dumps({
        "value_competence_verdict": result["qualification"]["value_competence_verdict"],
        "critic_independence": result["rir"]["critic_independence"]["status"],
        "rir": result["rir"]["overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
