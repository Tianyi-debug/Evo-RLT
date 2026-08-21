"""Progress-matched, fixed-state action-sensitivity audit for a frozen critic.

This is a read-only follow-up to ``audit_critic_mechanism``.  Progress matching
tests whether outcome ordering survives obvious episode-time confounding.
Fixed-state action substitution tests how much the critic changes inside the
actor's reachable action set.  Substituted actions have no counterfactual
return label and are never presented as evidence that one action is better.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from evo_rlt.cli.audit_actor_q_mechanism import _load_heads
from evo_rlt.cli.audit_critic_mechanism import (
    GROUPS,
    _actor_actions,
    _bootstrap_mean_ci,
    _episode_means,
    _load_list,
    _pairwise_episode_difference,
    _quantiles,
    _scalar,
    _score_actions,
    _semantic_group,
    _sha256_file,
)
from evo_rlt.core.actor import ChunkActor


def _progress_coordinates(
    row: dict[str, Any], *, normalized_bins: int, elapsed_bin_seconds: float
) -> dict[str, int | float]:
    anchor_index = int(row["anchor_index"])
    prefix_count = int(row["prefix_anchor_count"])
    frame_stride = int(row["frame_stride"])
    fps = float(row["fps"])
    if anchor_index < 0 or prefix_count <= 0 or anchor_index >= prefix_count:
        raise ValueError(
            f"invalid anchor progress: index={anchor_index}, prefix_count={prefix_count}"
        )
    if frame_stride <= 0 or fps <= 0:
        raise ValueError(f"frame_stride and fps must be positive, got {frame_stride}, {fps}")
    normalized = anchor_index / max(prefix_count - 1, 1)
    normalized_bin = min(int(normalized * normalized_bins), normalized_bins - 1)
    elapsed_seconds = anchor_index * frame_stride / fps
    elapsed_bin = int(math.floor(elapsed_seconds / elapsed_bin_seconds))
    return {
        "normalized_progress": normalized,
        "normalized_progress_bin": normalized_bin,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_seconds_bin": elapsed_bin,
    }


def _paired_episode_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    key: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    def means_by_episode(records: list[dict[str, Any]]) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for record in records:
            values[str(record["episode_uid"])].append(float(record[key]))
        return {
            episode: sum(items) / len(items)
            for episode, items in values.items()
            if items
        }

    left_means = means_by_episode(left)
    right_means = means_by_episode(right)
    common = sorted(set(left_means) & set(right_means))
    differences = [left_means[episode] - right_means[episode] for episode in common]
    return {
        "paired_episodes": len(common),
        "mean_difference": (
            sum(differences) / len(differences) if differences else None
        ),
        "episode_paired_bootstrap_95ci": _bootstrap_mean_ci(
            differences, seed=seed, replicates=replicates
        ),
    }


def _score_summary(
    records: list[dict[str, Any]], *, key: str, seed: int, replicates: int
) -> dict[str, Any]:
    episode_values = _episode_means(records, key)
    values = torch.tensor([float(record[key]) for record in records])
    result = {
        "samples": len(records),
        "episodes": len(episode_values),
        "sample_distribution": _quantiles(values),
        "episode_mean": (
            sum(episode_values) / len(episode_values) if episode_values else None
        ),
        "episode_bootstrap_95ci": _bootstrap_mean_ci(
            episode_values, seed=seed, replicates=replicates
        ),
    }
    return result


def _progress_bin_report(
    records: list[dict[str, Any]],
    *,
    bin_key: str,
    key: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    bins = sorted({int(record[bin_key]) for record in records})
    report: dict[str, Any] = {}
    for bin_index in bins:
        selected = [record for record in records if int(record[bin_key]) == bin_index]
        by_group = {
            group: [record for record in selected if record["group"] == group]
            for group in GROUPS
        }
        comparisons = {
            "autonomous_success_minus_autonomous_failure": _pairwise_episode_difference(
                by_group["autonomous_success"],
                by_group["autonomous_failure"],
                key=key,
                seed=seed + bin_index * 10,
                replicates=replicates,
            ),
            "autonomous_success_minus_corrective_last_K": _pairwise_episode_difference(
                by_group["autonomous_success"],
                by_group["corrective_last_K"],
                key=key,
                seed=seed + bin_index * 10 + 1,
                replicates=replicates,
            ),
            "corrective_earlier_minus_corrective_last_K_paired": (
                _paired_episode_difference(
                    by_group["corrective_earlier"],
                    by_group["corrective_last_K"],
                    key=key,
                    seed=seed + bin_index * 10 + 2,
                    replicates=replicates,
                )
            ),
        }
        report[str(bin_index)] = {
            "samples": len(selected),
            "episodes": len({record["episode_uid"] for record in selected}),
            "groups": {
                group: _score_summary(
                    group_records,
                    key=key,
                    seed=seed + bin_index,
                    replicates=replicates,
                )
                for group, group_records in by_group.items()
            },
            "comparisons": comparisons,
        }
    return report


def _progress_flags(progress_report: dict[str, Any]) -> dict[str, Any]:
    supported_success_failure = []
    supported_corrective_reversal = []
    for bin_name, report in progress_report.items():
        success_failure = report["comparisons"][
            "autonomous_success_minus_autonomous_failure"
        ]
        interval = success_failure["episode_bootstrap_95ci"]
        if interval is not None and interval[0] > 0:
            supported_success_failure.append(bin_name)
        corrective = report["comparisons"][
            "corrective_earlier_minus_corrective_last_K_paired"
        ]
        interval = corrective["episode_paired_bootstrap_95ci"]
        if interval is not None and interval[1] < 0:
            supported_corrective_reversal.append(bin_name)
    return {
        "success_over_failure_supported_bins": supported_success_failure,
        "corrective_earlier_below_last_K_supported_bins": supported_corrective_reversal,
    }


def _select_progress_matched_donors(
    normalized_bins: list[int], episode_uids: list[str], *, seed: int
) -> tuple[list[int], int]:
    pools: dict[int, list[int]] = defaultdict(list)
    for index, bin_index in enumerate(normalized_bins):
        pools[bin_index].append(index)
    rng = random.Random(seed)
    donors = []
    same_episode_fallbacks = 0
    for index, bin_index in enumerate(normalized_bins):
        candidates = [
            candidate
            for candidate in pools[bin_index]
            if episode_uids[candidate] != episode_uids[index]
        ]
        if not candidates:
            candidates = [candidate for candidate in pools[bin_index] if candidate != index]
            same_episode_fallbacks += 1
        if not candidates:
            raise ValueError(
                f"normalized progress bin {bin_index} has no shuffled-action donor"
            )
        donors.append(candidates[rng.randrange(len(candidates))])
    return donors, same_episode_fallbacks


def _project_to_actor_support(
    actor: ChunkActor, proposals: Tensor, candidate_actions: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    if actor.action_residual:
        lower, upper = actor.residual_reachable_interval(proposals)
        projected = torch.maximum(torch.minimum(candidate_actions.clamp(-1.0, 1.0), upper), lower)
    else:
        lower, upper = torch.full_like(candidate_actions, -1.0), torch.full_like(
            candidate_actions, 1.0
        )
        projected = candidate_actions.clamp(-1.0, 1.0)
    return projected, lower, upper


def _scalar_record_summary(
    records: list[dict[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(records),
        "episodes": len({record["episode_uid"] for record in records}),
    }
    if not records:
        return result
    excluded = {"episode_uid", "group", "normalized_progress_bin"}
    keys = [key for key in records[0] if key not in excluded]
    for key in keys:
        values = torch.tensor([float(record[key]) for record in records])
        episode_values = _episode_means(records, key)
        result[key] = _quantiles(values)
        result[key]["episode_mean"] = sum(episode_values) / len(episode_values)
        result[key]["episode_bootstrap_95ci"] = _bootstrap_mean_ci(
            episode_values, seed=seed, replicates=replicates
        )
    return result


def _fixed_state_action_report(
    *,
    actor: ChunkActor,
    critic: nn.Module,
    states: Tensor,
    proposals: Tensor,
    behavior_actions: Tensor,
    actor_actions: Tensor,
    groups: list[str],
    episode_uids: list[str],
    normalized_bins: list[int],
    device: torch.device,
    batch_size: int,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    donor_indices, fallback_count = _select_progress_matched_donors(
        normalized_bins, episode_uids, seed=seed
    )
    donor_actions = behavior_actions[torch.tensor(donor_indices)]
    shuffled_actions, lower, upper = _project_to_actor_support(
        actor, proposals, donor_actions
    )
    proposal_actions = proposals.clamp(-1.0, 1.0)
    candidates = {
        "behavior_actual_sent": behavior_actions,
        "checkpoint_actor": actor_actions,
        "vla_proposal_clamped": proposal_actions,
        "progress_matched_shuffled_projected": shuffled_actions,
    }
    scores = {
        name: _score_actions(
            critic, states, actions, device=device, batch_size=batch_size
        )["min_q"]
        for name, actions in candidates.items()
    }
    score_stack = torch.stack(list(scores.values()), dim=1)
    behavior_projected, _, _ = _project_to_actor_support(
        actor, proposals, behavior_actions
    )
    behavior_support_projection_rmse = (
        behavior_projected - behavior_actions
    ).square().mean(dim=1).sqrt()
    actor_to_behavior_rmse = (actor_actions - behavior_actions).square().mean(dim=1).sqrt()
    proposal_to_actor_rmse = (proposal_actions - actor_actions).square().mean(dim=1).sqrt()
    projection_rmse = (shuffled_actions - donor_actions).square().mean(dim=1).sqrt()
    shuffled_to_actor_rmse = (shuffled_actions - actor_actions).square().mean(dim=1).sqrt()
    at_boundary = (
        (shuffled_actions - lower).abs() <= 1e-6
    ) | ((shuffled_actions - upper).abs() <= 1e-6)
    records = []
    for index in range(len(states)):
        record = {
            "episode_uid": episode_uids[index],
            "group": groups[index],
            "normalized_progress_bin": normalized_bins[index],
            "q_behavior": float(scores["behavior_actual_sent"][index].item()),
            "q_actor": float(scores["checkpoint_actor"][index].item()),
            "q_proposal": float(scores["vla_proposal_clamped"][index].item()),
            "q_shuffled": float(
                scores["progress_matched_shuffled_projected"][index].item()
            ),
            "q_actor_minus_behavior": float(
                (scores["checkpoint_actor"][index] - scores["behavior_actual_sent"][index]).item()
            ),
            "q_proposal_minus_actor": float(
                (scores["vla_proposal_clamped"][index] - scores["checkpoint_actor"][index]).item()
            ),
            "q_shuffled_minus_actor": float(
                (
                    scores["progress_matched_shuffled_projected"][index]
                    - scores["checkpoint_actor"][index]
                ).item()
            ),
            "fixed_state_candidate_q_span": float(
                (score_stack[index].max() - score_stack[index].min()).item()
            ),
            "behavior_support_projection_rmse": float(
                behavior_support_projection_rmse[index].item()
            ),
            "actor_to_behavior_action_rmse": float(
                actor_to_behavior_rmse[index].item()
            ),
            "proposal_to_actor_action_rmse": float(
                proposal_to_actor_rmse[index].item()
            ),
            "shuffled_projection_rmse": float(projection_rmse[index].item()),
            "shuffled_to_actor_rmse": float(shuffled_to_actor_rmse[index].item()),
            "shuffled_boundary_element_fraction": float(
                at_boundary[index].float().mean().item()
            ),
        }
        records.append(record)

    by_bin = {}
    for bin_index in sorted(set(normalized_bins)):
        selected = [
            record
            for record in records
            if record["normalized_progress_bin"] == bin_index
        ]
        actor_q = torch.tensor([record["q_actor"] for record in selected])
        span = torch.tensor(
            [record["fixed_state_candidate_q_span"] for record in selected]
        )
        state_iqr = float(
            (torch.quantile(actor_q, 0.75) - torch.quantile(actor_q, 0.25)).item()
        )
        median_span = float(torch.quantile(span, 0.5).item())
        by_bin[str(bin_index)] = {
            "summary": _scalar_record_summary(
                selected, seed=seed + bin_index, replicates=replicates
            ),
            "checkpoint_actor_q_state_iqr": state_iqr,
            "median_fixed_state_candidate_q_span": median_span,
            "candidate_span_to_state_iqr_ratio": (
                median_span / state_iqr if state_iqr > 0 else None
            ),
        }
    return {
        "candidate_definitions": {
            "behavior_actual_sent": "recorded autonomous action actually sent",
            "checkpoint_actor": "frozen checkpoint actor composite action",
            "vla_proposal_clamped": "recorded VLA proposal clamped to [-1,1]",
            "progress_matched_shuffled_projected": (
                "behavior action from another episode in the same normalized-progress bin, "
                "projected into the recipient actor's residual reachable set"
            ),
        },
        "same_episode_donor_fallbacks": fallback_count,
        "overall": _scalar_record_summary(
            records, seed=seed, replicates=replicates
        ),
        "by_group": {
            group: _scalar_record_summary(
                [record for record in records if record["group"] == group],
                seed=seed,
                replicates=replicates,
            )
            for group in GROUPS
        },
        "by_normalized_progress_bin": by_bin,
        "counterfactual_action_quality_label_available": False,
    }


def _crossed_reachable_variance(
    *,
    actor: ChunkActor,
    critic: nn.Module,
    states: Tensor,
    proposals: Tensor,
    donor_actions: Tensor,
    normalized_bins: list[int],
    device: torch.device,
    batch_size: int,
    samples_per_bin: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    by_bin: dict[str, Any] = {}
    all_action_fractions = []
    for bin_index in sorted(set(normalized_bins)):
        indices = torch.tensor(
            [index for index, value in enumerate(normalized_bins) if value == bin_index]
        )
        episode_count = len(indices)
        selected_count = min(samples_per_bin, episode_count)
        if selected_count < 4:
            by_bin[str(bin_index)] = {
                "samples": episode_count,
                "status": "INSUFFICIENT (<4 rows)",
            }
            continue
        metrics: dict[str, list[float]] = defaultdict(list)
        for _ in range(repeats):
            state_indices = indices[
                torch.randperm(len(indices), generator=generator)[:selected_count]
            ]
            action_indices = indices[
                torch.randperm(len(indices), generator=generator)[:selected_count]
            ]
            state_grid = states[state_indices].repeat_interleave(selected_count, dim=0)
            proposal_grid = proposals[state_indices].repeat_interleave(
                selected_count, dim=0
            )
            raw_action_grid = donor_actions[action_indices].repeat(selected_count, 1)
            projected, lower, upper = _project_to_actor_support(
                actor, proposal_grid, raw_action_grid
            )
            q_matrix = _score_actions(
                critic,
                state_grid,
                projected,
                device=device,
                batch_size=batch_size,
            )["min_q"].reshape(selected_count, selected_count)
            state_variance = q_matrix.mean(dim=1).var(unbiased=False)
            action_variance = q_matrix.var(dim=1, unbiased=False).mean()
            denominator = state_variance + action_variance
            action_fraction = (
                action_variance / denominator if denominator > 0 else denominator.new_zeros(())
            )
            boundary = ((projected - lower).abs() <= 1e-6) | (
                (projected - upper).abs() <= 1e-6
            )
            metrics["between_state_mean_q_variance"].append(float(state_variance.item()))
            metrics["within_state_action_q_variance"].append(float(action_variance.item()))
            metrics["action_fraction_of_state_plus_action_variance"].append(
                float(action_fraction.item())
            )
            metrics["mean_within_state_q_range"].append(
                float((q_matrix.max(1).values - q_matrix.min(1).values).mean().item())
            )
            metrics["projection_rmse"].append(
                float((projected - raw_action_grid).square().mean().sqrt().item())
            )
            metrics["projected_boundary_element_fraction"].append(
                float(boundary.float().mean().item())
            )
        all_action_fractions.extend(
            metrics["action_fraction_of_state_plus_action_variance"]
        )
        by_bin[str(bin_index)] = {
            "samples_available": episode_count,
            "samples_per_repeat": selected_count,
            "repeats": repeats,
            **{
                key: _quantiles(torch.tensor(values)) for key, values in metrics.items()
            },
        }
    overall = _quantiles(torch.tensor(all_action_fractions)) if all_action_fractions else None
    return {
        "definition": (
            "Within each normalized-progress bin, cross sampled states with sampled behavior "
            "actions after projecting every action into each recipient state's residual support."
        ),
        "by_normalized_progress_bin": by_bin,
        "action_fraction_across_bin_repeats": overall,
        "not_two_way_anova_due_to_state_dependent_action_projection": True,
    }


def run_progress_action_audit(
    *,
    checkpoint: Path,
    actor_trust_dataset: Path,
    output_path: Path,
    device_name: str = "cpu",
    batch_size: int = 256,
    bootstrap_replicates: int = 2000,
    normalized_progress_bins: int = 5,
    elapsed_bin_seconds: float = 2.0,
    crossed_samples_per_bin: int = 64,
    crossed_repeats: int = 20,
    state_dominance_threshold: float = 0.1,
    seed: int = 1000,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    actor_trust_dataset = actor_trust_dataset.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve()
    if normalized_progress_bins < 2:
        raise ValueError("normalized_progress_bins must be at least 2")
    if elapsed_bin_seconds <= 0:
        raise ValueError("elapsed_bin_seconds must be positive")
    if bootstrap_replicates < 100 or crossed_repeats < 1:
        raise ValueError("bootstrap_replicates >=100 and crossed_repeats >=1 are required")
    if crossed_samples_per_bin < 4:
        raise ValueError("crossed_samples_per_bin must be at least 4")
    if not 0 < state_dominance_threshold < 1:
        raise ValueError("state_dominance_threshold must lie in (0,1)")

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
        raise ValueError(f"fixed-state audit requires actual-sent actions; legacy={legacy_count}")

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
    expected_action_dim = int(config["chunk_length"] * config["action_dim"])
    if states.shape[1] != expected_state_dim or proposals.shape[1] != expected_action_dim:
        raise ValueError(
            f"checkpoint/data shape mismatch: state={states.shape[1]}/{expected_state_dim}, "
            f"action={proposals.shape[1]}/{expected_action_dim}"
        )
    groups = [_semantic_group(row, primary_k) for row in rows]
    episode_uids = [str(row["episode_uid"]) for row in rows]
    progress = [
        _progress_coordinates(
            row,
            normalized_bins=normalized_progress_bins,
            elapsed_bin_seconds=elapsed_bin_seconds,
        )
        for row in rows
    ]
    normalized_bins = [int(value["normalized_progress_bin"]) for value in progress]
    actor_actions = _actor_actions(
        actor, states, proposals, device=device, batch_size=batch_size
    )
    behavior_scores = _score_actions(
        critic, states, behavior_actions, device=device, batch_size=batch_size
    )["min_q"]
    actor_scores = _score_actions(
        critic, states, actor_actions, device=device, batch_size=batch_size
    )["min_q"]
    progress_records = [
        {
            "episode_uid": episode_uids[index],
            "group": groups[index],
            **progress[index],
            "behavior_min_q": float(behavior_scores[index].item()),
            "actor_min_q": float(actor_scores[index].item()),
        }
        for index in range(len(rows))
    ]
    progress_reports = {}
    progress_flags = {}
    for score_name in ("behavior_min_q", "actor_min_q"):
        normalized = _progress_bin_report(
            progress_records,
            bin_key="normalized_progress_bin",
            key=score_name,
            seed=seed,
            replicates=bootstrap_replicates,
        )
        elapsed = _progress_bin_report(
            progress_records,
            bin_key="elapsed_seconds_bin",
            key=score_name,
            seed=seed + 100,
            replicates=bootstrap_replicates,
        )
        progress_reports[score_name] = {
            "normalized_progress_bins": normalized,
            "elapsed_seconds_bins": elapsed,
        }
        progress_flags[score_name] = {
            "normalized_progress": _progress_flags(normalized),
            "elapsed_seconds": _progress_flags(elapsed),
        }

    fixed_state = _fixed_state_action_report(
        actor=actor,
        critic=critic,
        states=states,
        proposals=proposals,
        behavior_actions=behavior_actions,
        actor_actions=actor_actions,
        groups=groups,
        episode_uids=episode_uids,
        normalized_bins=normalized_bins,
        device=device,
        batch_size=batch_size,
        seed=seed,
        replicates=bootstrap_replicates,
    )
    crossed = _crossed_reachable_variance(
        actor=actor,
        critic=critic,
        states=states,
        proposals=proposals,
        donor_actions=behavior_actions,
        normalized_bins=normalized_bins,
        device=device,
        batch_size=batch_size,
        samples_per_bin=crossed_samples_per_bin,
        repeats=crossed_repeats,
        seed=seed,
    )
    action_fraction = crossed["action_fraction_across_bin_repeats"]
    action_fraction_median = action_fraction["median"] if action_fraction else None
    state_dominance = (
        action_fraction_median is not None
        and action_fraction_median < state_dominance_threshold
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
            "all_actions_actual_sent": True,
        },
        "progress_definitions": {
            "normalized_progress": "anchor_index / max(prefix_anchor_count - 1, 1)",
            "normalized_bin_count": normalized_progress_bins,
            "elapsed_seconds": "anchor_index * frame_stride / fps",
            "elapsed_bin_seconds": elapsed_bin_seconds,
            "corrective_comparison": (
                "earlier minus last-K, paired by episode when both occur in one bin"
            ),
            "other_comparisons": "independent episode-clustered bootstrap within each bin",
        },
        "progress_matched_outcome_ordering": progress_reports,
        "progress_matched_flags": progress_flags,
        "fixed_state_action_sensitivity": fixed_state,
        "crossed_reachable_state_action_variance": crossed,
        "decision_support": {
            "state_dominance_threshold": state_dominance_threshold,
            "crossed_action_fraction_median": action_fraction_median,
            "state_dominance_heuristic": state_dominance,
            "counterfactual_action_correctness_established": False,
            "maximum_positive_verdict_from_this_audit_alone": "WEAK GO",
            "stop_rethink_if": (
                "progress-matched corrective reversal persists, or reachable action variance "
                "is negligible relative to between-state variance"
            ),
        },
        "limitations": [
            "Normalized progress aligns relative position, not physical task state or goal distance.",
            "Elapsed-time bins do not control visual or proprioceptive task progress.",
            "Shuffled actions have no observed counterfactual return and cannot validate correctness.",
            "Projection into residual support makes the crossed diagnostic state-dependent, not ANOVA.",
            "No proactive episodes exist in online48, so proactive conclusions remain unavailable.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--actor-trust-dataset", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--normalized-progress-bins", type=int, default=5)
    parser.add_argument("--elapsed-bin-seconds", type=float, default=2.0)
    parser.add_argument("--crossed-samples-per-bin", type=int, default=64)
    parser.add_argument("--crossed-repeats", type=int, default=20)
    parser.add_argument("--state-dominance-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_progress_action_audit(
        checkpoint=args.checkpoint,
        actor_trust_dataset=args.actor_trust_dataset,
        output_path=args.output_path,
        device_name=args.device,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap_replicates,
        normalized_progress_bins=args.normalized_progress_bins,
        elapsed_bin_seconds=args.elapsed_bin_seconds,
        crossed_samples_per_bin=args.crossed_samples_per_bin,
        crossed_repeats=args.crossed_repeats,
        state_dominance_threshold=args.state_dominance_threshold,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_path": str(args.output_path.expanduser().resolve()),
                "decision_support": report["decision_support"],
                "progress_matched_flags": report["progress_matched_flags"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
