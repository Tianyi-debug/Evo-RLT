"""Offline three-way audit for strictly matched Q0/Q0.05/Q0.25 actors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from evo_rlt.cli.audit_actor_q_mechanism import (
    _load_heads,
    validate_matched_actor_refine,
)
from evo_rlt.cli.audit_critic_actionability import (
    _bootstrap_statistic,
    _cache_metadata,
    _checkpoint_metadata,
    _episode_metadata,
    _git_commit,
    _load_rows,
    _quantiles,
    _scalar,
    _stack,
)


def _method_summary(
    values: dict[str, Tensor],
    mask: Tensor,
) -> dict[str, Any]:
    selected = mask.reshape(-1)
    result: dict[str, Any] = {"samples": int(selected.sum().item())}
    if not bool(selected.any()):
        result.update({"teacher_drift_rmse": None, "human_target_rmse": None})
        return result
    if "teacher_squared" in values:
        result["teacher_drift_rmse"] = float(
            values["teacher_squared"][selected].mean().sqrt().item()
        )
    if "human_squared" in values:
        result["human_target_rmse"] = float(
            values["human_squared"][selected].mean().sqrt().item()
        )
    if "q" in values:
        result["critic_q"] = _quantiles(values["q"][selected])
    return result


def _pair_summary(
    records: list[dict[str, Any]],
    *,
    action_dim: int,
    chunk_length: int,
    seed: int,
    bootstrap_reps: int,
) -> dict[str, Any]:
    if not records:
        return {"samples": 0, "episodes": 0}
    delta = torch.stack([record["delta_action"] for record in records])
    normalized = torch.stack([record["normalized_delta"] for record in records])
    q_delta = torch.tensor([record["q_delta"] for record in records])
    whole_l2 = delta.norm(dim=-1)
    normalized_rms = normalized.square().mean(dim=-1).sqrt()
    shaped = delta.reshape(-1, chunk_length, action_dim)

    def mean_metric(sample: list[dict[str, Any]], key: str) -> float:
        return sum(float(record[key]) for record in sample) / len(sample)

    result = {
        "samples": len(records),
        "episodes": len({record["episode_uid"] for record in records}),
        "whole_chunk_l2": _quantiles(whole_l2),
        "normalized_by_residual_bounds_rms": _quantiles(normalized_rms),
        "critic_q_delta": _quantiles(q_delta),
        "fraction_q_delta_positive": float((q_delta > 0).float().mean().item()),
        "shift_rmse_per_action_dimension": shaped.square().mean(dim=(0, 1)).sqrt().tolist(),
        "shift_rmse_per_chunk_timestep": shaped.square().mean(dim=(0, 2)).sqrt().tolist(),
        "gripper_dimension": action_dim - 1,
        "gripper_shift_rmse": float(shaped[:, :, action_dim - 1].square().mean().sqrt()),
        "translation_related_dimensions": list(range(min(3, action_dim))),
        "translation_shift_rmse": float(
            shaped[:, :, : min(3, action_dim)].square().mean().sqrt()
        ),
        "episode_bootstrap": {
            "whole_chunk_l2_mean": _bootstrap_statistic(
                records,
                lambda sample: mean_metric(sample, "whole_chunk_l2"),
                seed=seed,
                replicates=bootstrap_reps,
            ),
            "critic_q_delta_mean": _bootstrap_statistic(
                records,
                lambda sample: mean_metric(sample, "q_delta"),
                seed=seed + 1,
                replicates=bootstrap_reps,
            ),
        },
    }
    return result


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "q0": args.q0_checkpoint.expanduser().resolve(strict=True),
        "q005": args.q005_checkpoint.expanduser().resolve(strict=True),
        "q025": args.q025_checkpoint.expanduser().resolve(strict=True),
        "teacher": args.teacher_checkpoint.expanduser().resolve(strict=True),
    }
    cache = args.validation_cache.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    matches = {
        "q0_vs_q005": validate_matched_actor_refine(paths["q0"], paths["q005"]),
        "q0_vs_q025": validate_matched_actor_refine(paths["q0"], paths["q025"]),
    }
    if any(value["status"] != "MATCHED" for value in matches.values()):
        raise ValueError(f"three-way actor audit requires matched checkpoints: {matches}")

    configs: dict[str, dict[str, Any]] = {}
    actors = {}
    critic = None
    for name in ("q0", "q005", "q025"):
        configs[name], actors[name], loaded_critic = _load_heads(paths[name])
        if name == "q0":
            critic = loaded_critic
    teacher_config, teacher, _ = _load_heads(paths["teacher"])
    assert critic is not None
    shape_keys = ("rl_token_dim", "proprio_dim", "chunk_length", "action_dim")
    for key in shape_keys:
        expected = configs["q0"][key]
        if any(configs[name][key] != expected for name in ("q005", "q025")):
            raise ValueError(f"actor checkpoint {key} mismatch")
        if teacher_config[key] != expected:
            raise ValueError(f"teacher checkpoint {key} mismatch")
    device = torch.device(args.device)
    actors = {name: actor.to(device).eval() for name, actor in actors.items()}
    teacher, critic = teacher.to(device).eval(), critic.to(device).eval()
    rows = _load_rows(cache)
    episode_uids, groups, progress = _episode_metadata(rows)
    states = _stack(rows, "state_vec").to(device)
    proposals = _stack(rows, "proposal_chunk", flatten_chunk=True).to(device)
    bc_targets = _stack(rows, "bc_target_chunk", flatten_chunk=True).to(device)
    source = torch.tensor([int(_scalar(row, "source", -1)) for row in rows], device=device)
    reason = torch.tensor(
        [int(_scalar(row, "intervention_reason", 0)) for row in rows], device=device
    )
    actor_bc = torch.tensor([_scalar(row, "actor_bc_mask", 0.0) for row in rows], device=device) > 0.5
    actor_q = torch.tensor([_scalar(row, "actor_q_mask", 0.0) for row in rows], device=device) > 0.5
    autonomous = (source == 1) | (source == 2)
    teacher_mask = (source == 0) | (autonomous & actor_bc) | (autonomous & (~actor_bc) & (reason == 2))
    human_mask = source == 3

    all_actions: dict[str, list[Tensor]] = defaultdict(list)
    all_q: dict[str, list[Tensor]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            batch_states, batch_proposals = states[start:stop], proposals[start:stop]
            for name, actor in actors.items():
                action, _ = actor(batch_states, batch_proposals, training=False)
                action = action.clamp(-1.0, 1.0)
                all_actions[name].append(action.cpu())
                all_q[name].append(critic.min_q(batch_states, action).reshape(-1).cpu())
            teacher_action, _ = teacher(batch_states, batch_proposals, training=False)
            all_actions["teacher"].append(teacher_action.clamp(-1.0, 1.0).cpu())
    actions = {name: torch.cat(parts) for name, parts in all_actions.items()}
    q_values = {name: torch.cat(parts) for name, parts in all_q.items()}
    bc_targets = bc_targets.cpu()
    teacher_mask, human_mask, actor_q = teacher_mask.cpu(), human_mask.cpu(), actor_q.cpu()
    action_dim = int(configs["q0"]["action_dim"])
    chunk_length = int(configs["q0"]["chunk_length"])
    proposal_cpu = proposals.cpu()
    lower, upper = actors["q0"].cpu().residual_reachable_interval(proposal_cpu)
    bound = actors["q0"].residual_delta_bound(actions["q0"]).expand_as(actions["q0"])
    bounds_ok = all(
        bool(((actions[name] >= lower - 1e-6) & (actions[name] <= upper + 1e-6)).all())
        for name in ("q0", "q005", "q025")
    )

    absolute: dict[str, Any] = {}
    for name in ("q0", "q005", "q025"):
        values = {
            "teacher_squared": (actions[name] - actions["teacher"]).square(),
            "human_squared": (actions[name] - bc_targets).square(),
            "q": q_values[name],
        }
        absolute[name] = {
            "teacher_supervised": _method_summary(
                {key: values[key] for key in ("teacher_squared", "q")},
                teacher_mask,
            ),
            "human": _method_summary(
                {"human_squared": values["human_squared"]},
                human_mask,
            ),
            "actor_q_valid": _method_summary({"q": values["q"]}, actor_q),
        }

    pair_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name in ("q005", "q025"):
        delta = actions[name] - actions["q0"]
        normalized = delta / bound.clamp_min(1e-12)
        q_delta = q_values[name] - q_values["q0"]
        for index in range(len(rows)):
            if not actor_q[index]:
                continue
            pair_records[name].append(
                {
                    "cache_index": index,
                    "episode_uid": episode_uids[index],
                    "group": groups[index],
                    "progress": float(progress[index]),
                    "delta_action": delta[index],
                    "normalized_delta": normalized[index],
                    "whole_chunk_l2": float(delta[index].norm()),
                    "q_delta": float(q_delta[index]),
                }
            )
    pairwise: dict[str, Any] = {}
    for pair_index, name in enumerate(("q005", "q025")):
        records = pair_records[name]
        pairwise[f"{name}_minus_q0"] = {
            "overall": _pair_summary(
                records,
                action_dim=action_dim,
                chunk_length=chunk_length,
                seed=args.seed + pair_index * 100,
                bootstrap_reps=args.bootstrap_reps,
            ),
            "semantic_breakdown": {
                group: _pair_summary(
                    [record for record in records if record["group"] == group],
                    action_dim=action_dim,
                    chunk_length=chunk_length,
                    seed=args.seed + pair_index * 100 + group_index + 1,
                    bootstrap_reps=args.bootstrap_reps,
                )
                for group_index, group in enumerate(sorted(set(groups)))
            },
        }
    report = {
        "schema_version": 1,
        "method": "strictly matched three-way direct-Q actor refinement audit",
        "git_commit": _git_commit(),
        "match_validation": matches,
        "checkpoints": {name: _checkpoint_metadata(path) for name, path in paths.items()},
        "validation_cache": {**_cache_metadata(cache), "rows": len(rows)},
        "masks": {
            "teacher_supervised_rows": int(teacher_mask.sum()),
            "human_rows": int(human_mask.sum()),
            "actor_q_valid_rows": int(actor_q.sum()),
        },
        "absolute_fit": absolute,
        "human_evidence_role": (
            "human-controlled rows are used only for imitation-target drift; "
            "they are not treated as current-policy Bellman actions or Q evidence"
        ),
        "pairwise_q_induced_change": pairwise,
        "invariants": {
            "residual_bounds_preserved": bounds_ok,
            "chunk_shape_preserved": all(
                value.shape[1] == chunk_length * action_dim
                for value in actions.values()
            ),
            "same_actor_initialization_and_batch_order": True,
            "only_algorithmic_difference": "actor_q_weight_max",
        },
    }
    (output_dir / "matched_actor_refinement_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    torch.save(pair_records, output_dir / "per_sample.pt")
    resolved = {
        "git_commit": report["git_commit"],
        "q0_checkpoint": str(paths["q0"]),
        "q005_checkpoint": str(paths["q005"]),
        "q025_checkpoint": str(paths["q025"]),
        "teacher_checkpoint": str(paths["teacher"]),
        "validation_cache": str(cache),
        "seed": args.seed,
        "bootstrap_reps": args.bootstrap_reps,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    (output_dir / "config_resolved.json").write_text(json.dumps(resolved, indent=2) + "\n")
    q005 = pairwise["q005_minus_q0"]["overall"]
    q025 = pairwise["q025_minus_q0"]["overall"]
    markdown = "\n".join(
        [
            "# Task1 strictly matched actor refinement audit",
            "",
            "- Q0 vs Q=0.05: `MATCHED`",
            "- Q0 vs Q=0.25: `MATCHED`",
            f"- Q=0.05 whole-chunk shift mean: `{q005['whole_chunk_l2']['mean']}`",
            f"- Q=0.25 whole-chunk shift mean: `{q025['whole_chunk_l2']['mean']}`",
            f"- Q=0.05 critic-Q delta mean: `{q005['critic_q_delta']['mean']}`",
            f"- Q=0.25 critic-Q delta mean: `{q025['critic_q_delta']['mean']}`",
            "",
            "These are offline actor-output and frozen-critic diagnostics, not real-world policy improvement.",
            "",
        ]
    )
    (output_dir / "matched_actor_refinement_report.md").write_text(markdown)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q0-checkpoint", type=Path, required=True)
    parser.add_argument("--q005-checkpoint", type=Path, required=True)
    parser.add_argument("--q025-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    report = run_audit(build_parser().parse_args())
    print(json.dumps({
        "match_validation": {
            key: value["status"] for key, value in report["match_validation"].items()
        },
        "pairwise_q_induced_change": {
            key: value["overall"] for key, value in report["pairwise_q_induced_change"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
