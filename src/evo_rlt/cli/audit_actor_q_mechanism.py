"""Matched offline mechanism audit for actor_refine Q=0 versus fixed-Q."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor

from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic


PROVENANCE_FILE = "actor_refine_provenance.json"
REQUIRED_PROVENANCE_KEYS = {
    "actor_initialization_sha256",
    "teacher_sha256",
    "critic_sha256",
    "training_data_sha256",
    "updates",
    "optimizer",
    "seed",
    "batch_order_sha256",
}
ALLOWED_CONFIG_DIFFERENCES = {
    "actor_q_weight_max",
    "diagnostics_jsonl_path",
    "repo_id",
}
ALLOWED_PROVENANCE_DIFFERENCES = {
    "actor_q_weight_max",
    "checkpoint",
    "output_dir",
    "run_name",
}


def _checkpoint_files(root: Path) -> tuple[Path, Path]:
    root = root.expanduser().resolve()
    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"checkpoint must contain config.json and model.safetensors: {root}"
        )
    return config_path, weights_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _without_keys(value: dict[str, Any], ignored: set[str]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in ignored}


def _normalized_train_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    for key in (
        "output_dir",
        "job_name",
        "checkpoint_path",
        "eval_freq",
        "log_freq",
        "save_freq",
        "wandb",
    ):
        normalized.pop(key, None)
    policy = normalized.get("policy", {})
    for key in ALLOWED_CONFIG_DIFFERENCES:
        policy.pop(key, None)
    return normalized


def _artifact_hash(path_value: str, filename: str) -> str:
    path = Path(path_value).expanduser()
    artifact = path if path.is_file() else path / filename
    if not artifact.is_file():
        raise FileNotFoundError(f"matched-run artifact unavailable: {artifact}")
    return _sha256_file(artifact)


def validate_matched_actor_refine(
    checkpoint_a: Path,
    checkpoint_b: Path,
) -> dict[str, Any]:
    mismatches: list[str] = []
    a_config_path, a_weights_path = _checkpoint_files(checkpoint_a)
    b_config_path, b_weights_path = _checkpoint_files(checkpoint_b)
    config_a = json.loads(a_config_path.read_text())
    config_b = json.loads(b_config_path.read_text())
    if config_a.get("training_stage") != "actor_refine":
        mismatches.append("checkpoint A training_stage is not actor_refine")
    if config_b.get("training_stage") != "actor_refine":
        mismatches.append("checkpoint B training_stage is not actor_refine")
    if float(config_a.get("actor_q_weight_max", -1)) != 0.0:
        mismatches.append("checkpoint A is not Q=0")
    if float(config_b.get("actor_q_weight_max", 0)) <= 0.0:
        mismatches.append("checkpoint B does not have positive fixed-Q weight")
    if config_a.get("actor_q_trust_mode", "fixed") != "fixed":
        mismatches.append("checkpoint A trust mode is not fixed")
    if config_b.get("actor_q_trust_mode", "fixed") != "fixed":
        mismatches.append("checkpoint B trust mode is not fixed")
    config_differences = {
        key: (config_a.get(key), config_b.get(key))
        for key in sorted(set(config_a) | set(config_b))
        if key not in ALLOWED_CONFIG_DIFFERENCES and config_a.get(key) != config_b.get(key)
    }
    if config_differences:
        mismatches.append(f"algorithm/training config differs: {config_differences}")

    state_a = load_safetensors_file(str(a_weights_path), device="cpu")
    state_b = load_safetensors_file(str(b_weights_path), device="cpu")
    critic_keys_a = {key for key in state_a if key.startswith("critic.")}
    critic_keys_b = {key for key in state_b if key.startswith("critic.")}
    if not critic_keys_a or critic_keys_a != critic_keys_b:
        mismatches.append("critic tensor keys are absent or differ")
    else:
        changed_critic = [key for key in sorted(critic_keys_a) if not torch.equal(state_a[key], state_b[key])]
        if changed_critic:
            mismatches.append(f"frozen critic tensors differ: {changed_critic[:8]}")

    audit_buffer_keys = ("_actor_refine_step", "_actor_refine_batch_fingerprint")
    for key in audit_buffer_keys:
        if key not in state_a or key not in state_b:
            mismatches.append(f"missing persistent matched-run audit buffer {key}")
        elif not torch.equal(state_a[key], state_b[key]):
            mismatches.append(
                f"matched-run audit buffer differs: {key}="
                f"{state_a[key].item()} vs {state_b[key].item()}"
            )

    train_config_paths = [checkpoint_a / "train_config.json", checkpoint_b / "train_config.json"]
    training_artifacts: dict[str, Any] = {}
    if not all(path.is_file() for path in train_config_paths):
        mismatches.append(
            "missing train_config.json; cannot verify dataset, optimizer, updates, seed, and initialization"
        )
    else:
        train_a = json.loads(train_config_paths[0].read_text())
        train_b = json.loads(train_config_paths[1].read_text())
        normalized_a = _normalized_train_config(train_a)
        normalized_b = _normalized_train_config(train_b)
        if normalized_a != normalized_b:
            mismatches.append("train_config differs beyond actor_q_weight_max and output-only fields")
        try:
            policy_config = train_a["policy"]
            dataset_root = train_a["dataset"]["repo_id"]
            training_artifacts = {
                "initial_model_sha256": _artifact_hash(
                    policy_config["pretrained_path"], "model.safetensors"
                ),
                "teacher_model_sha256": _artifact_hash(
                    policy_config["actor_teacher_pretrained_path"], "model.safetensors"
                ),
                "train_cache_sha256": _artifact_hash(
                    dataset_root, "chunk_transitions_train.pt"
                ),
            }
            policy_config_b = train_b["policy"]
            corresponding_b = {
                "initial_model_sha256": _artifact_hash(
                    policy_config_b["pretrained_path"], "model.safetensors"
                ),
                "teacher_model_sha256": _artifact_hash(
                    policy_config_b["actor_teacher_pretrained_path"], "model.safetensors"
                ),
                "train_cache_sha256": _artifact_hash(
                    train_b["dataset"]["repo_id"], "chunk_transitions_train.pt"
                ),
            }
            if training_artifacts != corresponding_b:
                mismatches.append("initialization, teacher, or training-cache content hashes differ")
        except (KeyError, FileNotFoundError) as error:
            mismatches.append(str(error))

    provenance: dict[str, Any] = {}
    provenance_paths = [checkpoint_a / PROVENANCE_FILE, checkpoint_b / PROVENANCE_FILE]
    if any(path.is_file() for path in provenance_paths) and not all(
        path.is_file() for path in provenance_paths
    ):
        mismatches.append(f"{PROVENANCE_FILE} exists for only one checkpoint")
    elif all(path.is_file() for path in provenance_paths):
        provenance_a = json.loads(provenance_paths[0].read_text())
        provenance_b = json.loads(provenance_paths[1].read_text())
        missing_a = sorted(REQUIRED_PROVENANCE_KEYS - set(provenance_a))
        missing_b = sorted(REQUIRED_PROVENANCE_KEYS - set(provenance_b))
        if missing_a or missing_b:
            mismatches.append(f"provenance missing required keys: A={missing_a}, B={missing_b}")
        provenance_differences = {
            key: (provenance_a.get(key), provenance_b.get(key))
            for key in sorted(set(provenance_a) | set(provenance_b))
            if key not in ALLOWED_PROVENANCE_DIFFERENCES
            and provenance_a.get(key) != provenance_b.get(key)
        }
        if provenance_differences:
            mismatches.append(f"actor_refine provenance differs: {provenance_differences}")
        provenance = {
            "a_sha256": _sha256_file(provenance_paths[0]),
            "b_sha256": _sha256_file(provenance_paths[1]),
        }
    return {
        "status": "MATCHED" if not mismatches else "NOT MATCHED",
        "mismatches": mismatches,
        "allowed_config_difference": "actor_q_weight_max (plus output-only paths)",
        "q_weight_a": config_a.get("actor_q_weight_max"),
        "q_weight_b": config_b.get("actor_q_weight_max"),
        "model_safetensors_sha256_a": _sha256_file(a_weights_path),
        "model_safetensors_sha256_b": _sha256_file(b_weights_path),
        "training_artifacts": training_artifacts,
        "provenance": provenance,
    }


def _construct_heads(config: dict[str, Any]) -> tuple[ChunkActor, TwinCritic]:
    state_dim = int(config["rl_token_dim"] + config["proprio_dim"])
    chunk_dim = int(config["chunk_length"] * config["action_dim"])
    actor = ChunkActor(
        state_dim=state_dim,
        chunk_dim=chunk_dim,
        hidden_dim=int(config.get("actor_hidden_dim", 256)),
        num_layers=int(config.get("actor_num_layers", 2)),
        fixed_std=float(config.get("actor_fixed_std", 0.05)),
        ref_dropout_p=float(config.get("actor_ref_dropout_p", 0.0)),
        activation=config.get("actor_activation", "relu"),
        layer_norm=bool(config.get("actor_layer_norm", False)),
        residual=bool(config.get("actor_residual", False)),
        proprio_dim=int(config["proprio_dim"]),
        state_normalization=config.get("state_normalization", "rl_token_layer_norm"),
        action_residual=bool(config.get("actor_action_residual", True)),
        delta_scale=float(config.get("actor_delta_scale", 0.1)),
        delta_scale_per_action_dim=config.get("actor_delta_scale_per_action_dim"),
    )
    critic = TwinCritic(
        state_dim=state_dim,
        chunk_dim=chunk_dim,
        hidden_dim=int(config.get("critic_hidden_dim", 256)),
        num_layers=int(config.get("critic_num_layers", 2)),
        activation=config.get("critic_activation", "relu"),
        layer_norm=bool(config.get("critic_layer_norm", False)),
        residual=bool(config.get("critic_residual", False)),
        proprio_dim=int(config["proprio_dim"]),
        state_normalization=config.get("state_normalization", "rl_token_layer_norm"),
    )
    return actor, critic


def _load_heads(checkpoint: Path) -> tuple[dict[str, Any], ChunkActor, TwinCritic]:
    config_path, weights_path = _checkpoint_files(checkpoint)
    config = json.loads(config_path.read_text())
    actor, critic = _construct_heads(config)
    state = load_safetensors_file(str(weights_path), device="cpu")
    actor.load_state_dict(
        {key.removeprefix("actor."): value for key, value in state.items() if key.startswith("actor.")},
        strict=True,
    )
    critic.load_state_dict(
        {key.removeprefix("critic."): value for key, value in state.items() if key.startswith("critic.")},
        strict=True,
    )
    actor.eval()
    critic.eval()
    return config, actor, critic


def _load_teacher(config: dict[str, Any], actor_template: ChunkActor) -> ChunkActor:
    teacher_path = Path(config["actor_teacher_pretrained_path"]).expanduser()
    weights_path = teacher_path if teacher_path.is_file() else teacher_path / "model.safetensors"
    state = load_safetensors_file(str(weights_path), device="cpu")
    teacher = copy.deepcopy(actor_template)
    teacher.load_state_dict(
        {key.removeprefix("actor."): value for key, value in state.items() if key.startswith("actor.")},
        strict=True,
    )
    teacher.eval()
    return teacher


def _semantic_group(row: dict[str, Any], primary_k: int) -> str | None:
    category = row["category"]
    if category == "corrective":
        return (
            "corrective_last_K"
            if int(row["distance_to_corrective_event"]) <= primary_k
            else "corrective_earlier"
        )
    if category == "autonomous_success":
        return "autonomous_success"
    if category == "proactive":
        return "proactive_earlier" if int(row["distance_to_event_anchors"]) > primary_k else None
    if category == "autonomous_failure":
        return "autonomous_failure_audit"
    if category == "human":
        return "human"
    return None


def _episode_bootstrap_ci(values: list[tuple[str, float]], seed: int = 1000) -> list[float] | None:
    by_episode: dict[str, list[float]] = defaultdict(list)
    for episode, value in values:
        by_episode[episode].append(value)
    episode_means = [sum(items) / len(items) for items in by_episode.values()]
    if len(episode_means) < 2:
        return None
    rng = random.Random(seed)
    replicates = []
    for _ in range(2000):
        sampled = [episode_means[rng.randrange(len(episode_means))] for _ in episode_means]
        replicates.append(sum(sampled) / len(sampled))
    replicates.sort()
    return [replicates[49], replicates[1949]]


def _summarize_group(records: list[dict[str, Any]], action_dim: int) -> dict[str, Any]:
    if not records:
        return {"samples": 0, "episodes": 0}
    scalar_keys = (
        "shift_rmse",
        "normalized_shift_rmse",
        "teacher_deviation_delta",
        "bc_distance_delta",
        "q_value_delta",
        "q_grad_norm_a",
        "q_grad_norm_b",
    )
    result: dict[str, Any] = {
        "samples": len(records),
        "episodes": len({record["episode_uid"] for record in records}),
    }
    for key in scalar_keys:
        selected = [record for record in records if record[key] is not None]
        result[key] = sum(float(record[key]) for record in selected) / len(selected) if selected else None
        result[f"{key}_episode_bootstrap_95ci"] = _episode_bootstrap_ci(
            [(record["episode_uid"], float(record[key])) for record in selected]
        )
    per_dimension = (
        torch.stack([record["shift_per_action_dim"] for record in records])
        .square()
        .mean(0)
        .sqrt()
    )
    result["shift_rmse_per_action_dim"] = per_dimension.tolist()
    result["gripper_shift_rmse"] = float(per_dimension[action_dim - 1].item())
    return result


def run_mechanistic_diagnostic(
    *,
    checkpoint_a: Path,
    checkpoint_b: Path,
    dataset_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    match = validate_matched_actor_refine(checkpoint_a, checkpoint_b)
    if match["status"] != "MATCHED":
        report = {"status": "NOT MATCHED", "match_validation": match, "scientific_verdict": None}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return report

    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    rows = torch.load(dataset_dir / "actor_trust_val.pt", map_location="cpu", weights_only=False)
    human_path = dataset_dir / "human_audit_val.pt"
    if human_path.is_file():
        rows += torch.load(human_path, map_location="cpu", weights_only=False)
    config_a, actor_a, critic_a = _load_heads(checkpoint_a)
    _, actor_b, _ = _load_heads(checkpoint_b)
    teacher = _load_teacher(config_a, actor_a)
    action_dim = int(config_a["action_dim"])
    primary_k = int(metadata["semantics"]["primary_future_k"])
    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = _semantic_group(row, primary_k)
        if group is None:
            continue
        state = torch.as_tensor(row["state_vec"], dtype=torch.float32).reshape(1, -1)
        proposal = torch.as_tensor(row["proposal_chunk"], dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            action_a, _ = actor_a(state, proposal, training=False)
            action_b, _ = actor_b(state, proposal, training=False)
            teacher_action, _ = teacher(state, proposal, training=False)
        action_a_for_grad = action_a.detach().requires_grad_(True)
        action_b_for_grad = action_b.detach().requires_grad_(True)
        q_a = critic_a.min_q(state, action_a_for_grad).squeeze()
        q_b = critic_a.min_q(state, action_b_for_grad).squeeze()
        grad_a = torch.autograd.grad(q_a, action_a_for_grad)[0]
        grad_b = torch.autograd.grad(q_b, action_b_for_grad)[0]
        shift = (action_b - action_a).reshape(-1)
        bound = actor_b.residual_delta_bound(shift).reshape(-1)
        if bound.numel() == 1:
            bound = bound.expand_as(shift)
        bc_delta: float | None = None
        if float(row.get("bc_target_valid", 0.0)) > 0.5:
            target = torch.as_tensor(row["bc_target_chunk"], dtype=torch.float32).reshape(1, -1)
            distance_b = (action_b - target).square().mean().sqrt()
            distance_a = (action_a - target).square().mean().sqrt()
            bc_delta = float((distance_b - distance_a).item())
        records_by_group[group].append(
            {
                "episode_uid": row["episode_uid"],
                "shift_rmse": float(shift.square().mean().sqrt().item()),
                "normalized_shift_rmse": float((shift / bound).square().mean().sqrt().item()),
                "shift_per_action_dim": shift.reshape(-1, action_dim).square().mean(0).sqrt(),
                "teacher_deviation_delta": float((
                    (action_b - teacher_action).square().mean().sqrt()
                    - (action_a - teacher_action).square().mean().sqrt()
                ).item()),
                "bc_distance_delta": bc_delta,
                "q_value_delta": float((q_b - q_a).detach().item()),
                "q_grad_norm_a": float(grad_a.norm().item()),
                "q_grad_norm_b": float(grad_b.norm().item()),
            }
        )
    groups = {
        name: _summarize_group(records_by_group.get(name, []), action_dim)
        for name in (
            "corrective_last_K",
            "corrective_earlier",
            "autonomous_success",
            "proactive_earlier",
            "autonomous_failure_audit",
            "human",
        )
    }
    high = groups["corrective_last_K"]
    low = groups["autonomous_success"]
    if high.get("samples", 0) == 0 or low.get("samples", 0) == 0:
        verdict = "STOP / RETHINK"
        rationale = "required high-risk or low-risk validation group is empty"
    elif max(high["normalized_shift_rmse"], low["normalized_shift_rmse"]) < 1e-4:
        verdict = "STOP / RETHINK"
        rationale = "Q-induced actor shift is negligible in both primary regions"
    else:
        high_harm = high["teacher_deviation_delta"] > low["teacher_deviation_delta"]
        low_useful = (
            low["bc_distance_delta"] is not None and low["bc_distance_delta"] < 0
        )
        if high_harm and low_useful:
            verdict = "GO"
            rationale = (
                "low-risk target association improves while high-risk teacher "
                "deviation increases more"
            )
        elif high_harm:
            verdict = "WEAK GO"
            rationale = "high-risk harm trend exists, but low-risk positive usefulness is limited"
        else:
            verdict = "STOP / RETHINK"
            rationale = "no region-dependent evidence that fixed Q is more useful in low-risk states"
    report = {
        "status": "MATCHED",
        "match_validation": match,
        "dataset": str(dataset_dir.expanduser().resolve()),
        "groups": groups,
        "human_evidence_role": "drift-only; never treated as same-state autonomous Q alignment",
        "delayed_state_mismatched_alignment_proxy": "NOT COMPUTED",
        "scientific_verdict": verdict,
        "verdict_rule_rationale": rationale,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_mechanistic_diagnostic(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
