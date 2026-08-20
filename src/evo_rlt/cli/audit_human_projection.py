"""Head-only offline audit for frozen-teacher human-target projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from evo_rlt.core.actor import ChunkActor


def _make_actor(checkpoint: Path) -> tuple[ChunkActor, dict]:
    config = json.loads((checkpoint / "config.json").read_text())
    actor = ChunkActor(
        state_dim=int(config["rl_token_dim"]) + int(config["proprio_dim"]),
        chunk_dim=int(config["chunk_length"]) * int(config["action_dim"]),
        hidden_dim=int(config["actor_hidden_dim"]),
        num_layers=int(config["actor_num_layers"]),
        fixed_std=float(config["actor_fixed_std"]),
        ref_dropout_p=float(config["actor_ref_dropout_p"]),
        activation=config["actor_activation"],
        layer_norm=bool(config["actor_layer_norm"]),
        residual=bool(config["actor_residual"]),
        proprio_dim=int(config["proprio_dim"]),
        state_normalization=config["state_normalization"],
        action_residual=bool(config["actor_action_residual"]),
        delta_scale=float(config["actor_delta_scale"]),
        delta_scale_per_action_dim=config.get("actor_delta_scale_per_action_dim"),
    )
    full_state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    actor.load_state_dict(
        {
            key.removeprefix("actor."): value
            for key, value in full_state.items()
            if key.startswith("actor.")
        },
        strict=True,
    )
    actor.eval()
    return actor, config


def _rmse(student: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float((student[mask] - target[mask]).square().mean().sqrt().item())


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q).item()) if values.numel() else 0.0


def audit(
    cache: Path,
    student_checkpoint: Path,
    teacher_checkpoint: Path,
    batch_size: int,
) -> dict:
    student, config = _make_actor(student_checkpoint)
    teacher, teacher_config = _make_actor(teacher_checkpoint)
    architecture_keys = (
        "rl_token_dim",
        "proprio_dim",
        "chunk_length",
        "action_dim",
        "actor_hidden_dim",
        "actor_num_layers",
        "actor_activation",
        "actor_layer_norm",
        "actor_action_residual",
        "actor_delta_scale_per_action_dim",
    )
    if any(config.get(key) != teacher_config.get(key) for key in architecture_keys):
        raise ValueError("student and teacher actor architectures differ")

    rows = []
    split_counts = {}
    for split in ("train", "val"):
        split_rows = torch.load(
            cache / f"chunk_transitions_{split}.pt",
            map_location="cpu",
            weights_only=False,
        )
        split_counts[split] = len(split_rows)
        rows.extend(split_rows)

    states = torch.stack([row["state_vec"] for row in rows])
    proposals = torch.stack(
        [row.get("proposal_chunk", row["ref_chunk"]) for row in rows]
    ).flatten(start_dim=-2)
    targets = torch.stack(
        [row.get("bc_target_chunk", row["ref_chunk"]) for row in rows]
    ).flatten(start_dim=-2).clamp(-1.0, 1.0)
    source = torch.stack([torch.as_tensor(row.get("source", 0)) for row in rows]).long()
    actor_bc = torch.stack(
        [torch.as_tensor(row.get("actor_bc_mask", 1.0)) for row in rows]
    ).float() > 0.5
    reason = torch.stack(
        [torch.as_tensor(row.get("intervention_reason", 0)) for row in rows]
    ).long()

    student_outputs = []
    teacher_outputs = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            stop = start + batch_size
            student_outputs.append(student(states[start:stop], proposals[start:stop])[0])
            teacher_outputs.append(teacher(states[start:stop], proposals[start:stop])[0])
    student_mu = torch.cat(student_outputs)
    teacher_mu = torch.cat(teacher_outputs)
    feasible = student.project_to_residual_support(proposals, targets)

    autonomous = (source == 1) | (source == 2)
    masks = {
        "demo": source == 0,
        "autonomous_success": autonomous & actor_bc,
        "autonomous_failure": autonomous & (~actor_bc) & (reason == 0),
        "corrective_prefix": autonomous & (~actor_bc) & (reason == 1),
        "proactive_prefix": autonomous & (~actor_bc) & (reason == 2),
        "human": source == 3,
    }
    human = masks["human"]
    projection = (targets - feasible).abs()
    projection_norm = projection.square().sum(dim=-1).sqrt()[human]
    action_dim = int(config["action_dim"])
    changed = (projection > 1e-6).reshape(len(rows), -1, action_dim)
    human_changed = changed[human]

    return {
        "cache": str(cache),
        "student_checkpoint": str(student_checkpoint),
        "teacher_checkpoint": str(teacher_checkpoint),
        "split_counts": split_counts,
        "human_samples": int(human.sum().item()),
        "configured_human_bc_target_mode": config.get("human_bc_target_mode", "raw"),
        "teacher_drift_rmse": {
            category: _rmse(student_mu, teacher_mu, mask)
            for category, mask in masks.items()
        },
        "human_raw_target_rmse": _rmse(student_mu, targets, human),
        "human_feasible_target_rmse": _rmse(student_mu, feasible, human),
        "projection_error": {
            "mean": float(projection_norm.mean().item()) if projection_norm.numel() else 0.0,
            "p50": _quantile(projection_norm, 0.50),
            "p95": _quantile(projection_norm, 0.95),
        },
        "projection_fraction": (
            float(human_changed.any(dim=(1, 2)).float().mean().item())
            if human_changed.numel()
            else 0.0
        ),
        "outside_step_fraction": (
            float(human_changed.any(dim=-1).float().mean().item())
            if human_changed.numel()
            else 0.0
        ),
        "outside_chunk_fraction": (
            float(human_changed.any(dim=(1, 2)).float().mean().item())
            if human_changed.numel()
            else 0.0
        ),
        "per_action_dimension_projection_frequency": [
            float(human_changed[:, :, dim].float().mean().item())
            if human_changed.numel()
            else 0.0
            for dim in range(action_dim)
        ],
        "gripper_projection_frequency": (
            float(human_changed[:, :, -1].float().mean().item())
            if human_changed.numel()
            else 0.0
        ),
        "mask_counts": {
            category: int(mask.sum().item()) for category, mask in masks.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        cache=args.cache,
        student_checkpoint=args.student_checkpoint,
        teacher_checkpoint=args.teacher_checkpoint,
        batch_size=args.batch_size,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
