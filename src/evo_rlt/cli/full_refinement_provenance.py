"""Fail-closed provenance checks for fixed, multi-step TD3+BC actors.

This deliberately does not relax the historical matched-run validator. Here
the inducing critic is allowed to differ, but the actor initialization and
the entire observed prefix of batch fingerprints must agree. No model update
or dataset sampling is performed by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from evo_rlt.cli.audit_actor_q_mechanism import _construct_heads
from evo_rlt.core.interfaces import (
    SPARSE_TERMINAL_SUCCESS_REWARD_SEMANTICS,
    TRANSITION_CACHE_SEMANTICS_VERSION,
    validate_transition_cache_semantics,
)


class ProvenanceError(ValueError):
    """Required evidence is missing or the proposed runs are not matched."""


@dataclass(frozen=True)
class Candidate:
    q_weight: float
    inducing_critic: int
    checkpoint: Path

    @property
    def key(self) -> str:
        return f"q{self.q_weight:g}_c{self.inducing_critic}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(state: dict[str, torch.Tensor], prefix: str = "") -> str:
    """Hash names, dtypes, shapes and bytes, not just a concatenation of values."""
    digest = hashlib.sha256()
    selected = [(k, v) for k, v in sorted(state.items()) if k.startswith(prefix)]
    require(bool(selected), f"No tensors for prefix {prefix!r}")
    for key, tensor in selected:
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def resolve_artifact(path: str | Path) -> Path:
    """Resolve an existing path, or the same repo-relative /save mirror only."""
    original = Path(path).expanduser()
    if original.exists():
        return original.resolve()
    server = Path("/save/zhangtianyi/catkin_ws/src/Evo-RLT")
    try:
        mirror = Path(__file__).resolve().parents[3] / original.relative_to(server)
    except ValueError:
        mirror = original
    if mirror.exists():
        return mirror.resolve()
    raise ProvenanceError(f"Missing artifact: {original}")


class Evidence:
    """Read-once input registry; rehash before publishing any result."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.states: dict[str, dict[str, torch.Tensor]] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {}

    def file(self, path: str | Path) -> Path:
        result = resolve_artifact(path)
        require(result.is_file(), f"Expected a file, not a directory: {result}")
        key = str(result)
        if key not in self.files:
            self.files[key] = sha256_file(result)
        return result

    def json(self, path: str | Path) -> Any:
        return json.loads(self.file(path).read_text())

    def state(self, path: str | Path) -> dict[str, torch.Tensor]:
        key = str(self.file(path))
        if key not in self.states:
            state = load_file(key, device="cpu")
            require(all(bool(torch.isfinite(v).all()) for v in state.values()), f"Nonfinite tensors: {key}")
            self.states[key] = state
        return self.states[key]

    def cache(self, path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
        file = self.file(path)
        key = str(file)
        if key not in self.rows:
            rows = torch.load(file, map_location="cpu", weights_only=False)
            require(
                isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows),
                f"Transition cache must be a nonempty list[dict]: {file}",
            )
            self.rows[key] = rows
        return file, self.rows[key]

    def verify_unchanged(self) -> None:
        for path, expected in self.files.items():
            require(sha256_file(Path(path)) == expected, f"Input changed during audit: {path}")


def _differences(a: dict, b: dict) -> dict:
    return {k: [a.get(k), b.get(k)] for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}


def _head_signature(config: dict) -> dict:
    # Actual reconstructed module architecture and non-tensor semantics.
    actor, critic = _construct_heads(config)
    return {
        "actor": repr(actor), "critic": repr(critic),
        **{k: config.get(k) for k in (
            "rl_token_dim", "proprio_dim", "chunk_length", "action_dim",
            "state_normalization", "ac_semantics_version", "actor_action_residual",
            "actor_delta_scale", "actor_delta_scale_per_action_dim", "actor_ref_dropout_p",
        )},
    }


def _load_checkpoint(path: Path, evidence: Evidence) -> dict:
    root = resolve_artifact(path)
    cfg = evidence.json(root / "config.json")
    train = evidence.json(root / "train_config.json")
    require(cfg == train["policy"], f"Saved config and train_config.policy differ: {root}")
    state = evidence.state(root / "model.safetensors")
    return {"root": root, "config": cfg, "train": train, "state": state}


def _training_cache(train: dict, evidence: Evidence) -> Path:
    return evidence.file(Path(train["dataset"]["repo_id"]) / "chunk_transitions_train.pt")


def _scalar(row: dict[str, Any], key: str, *, index: int, cache: Path) -> float:
    require(key in row, f"Missing {key!r} at row {index}: {cache}")
    value = torch.as_tensor(row[key])
    require(value.numel() == 1 and bool(torch.isfinite(value).all()),
            f"Invalid scalar {key!r} at row {index}: {cache}")
    return float(value.item())


def validate_paper_cache(
    path: Path,
    evidence: Evidence,
    *,
    require_actual_sent_for_critic: bool,
) -> dict[str, Any]:
    """Require typed reward/credit semantics and stable row provenance."""
    cache, rows = evidence.cache(path)
    cache_version = validate_transition_cache_semantics(rows, cache_name=str(cache))
    require(cache_version == TRANSITION_CACHE_SEMANTICS_VERSION,
            f"Paper audit requires semantic-v{TRANSITION_CACHE_SEMANTICS_VERSION}: {cache}")
    reward_versions = {row.get("reward_semantics_version") for row in rows}
    require(reward_versions == {SPARSE_TERMINAL_SUCCESS_REWARD_SEMANTICS},
            f"Missing/mixed reward semantics in paper cache {cache}: {sorted(map(str, reward_versions))}")

    episode_uids: set[str] = set()
    transition_uids: list[str] = []
    critic_valid = 0
    actual_sent_checked = 0
    actual_sent = 0
    for index, row in enumerate(rows):
        uid = row.get("episode_uid")
        require(isinstance(uid, str) and bool(uid.strip()),
                f"Paper audit requires stable episode_uid at row {index}: {cache}")
        episode_uids.add(uid)
        transition_uid = row.get("transition_uid")
        if transition_uid is not None:
            require(isinstance(transition_uid, str) and bool(transition_uid.strip()),
                    f"Invalid transition_uid at row {index}: {cache}")
            transition_uids.append(transition_uid)
        mask = _scalar(row, "critic_mask", index=index, cache=cache)
        require(mask in (0.0, 1.0), f"critic_mask must be binary at row {index}: {cache}")
        if mask == 1.0:
            critic_valid += 1
            if "exec_action_is_actual_sent" in row:
                sent = _scalar(row, "exec_action_is_actual_sent", index=index, cache=cache)
                require(sent in (0.0, 1.0),
                        f"exec_action_is_actual_sent must be binary at row {index}: {cache}")
                actual_sent_checked += 1
                actual_sent += int(sent == 1.0)
                if require_actual_sent_for_critic:
                    require(sent == 1.0,
                            f"Critic-valid row {index} is not actual-sent (value={sent}): {cache}")
            else:
                require(
                    not require_actual_sent_for_critic,
                    f"Critic-valid row {index} has legacy/unknown executed-action semantics: {cache}",
                )
    require(not transition_uids or len(transition_uids) == len(rows),
            f"transition_uid is present for only some rows: {cache}")
    require(len(set(transition_uids)) == len(transition_uids),
            f"Duplicate transition_uid within cache: {cache}")
    require(not require_actual_sent_for_critic or critic_valid > 0,
            f"Paper critic cache contains no critic-valid rows: {cache}")
    return {
        "path": str(cache),
        "sha256": evidence.files[str(cache)],
        "rows": len(rows),
        "episodes": len(episode_uids),
        "episode_uids": sorted(episode_uids),
        "cache_semantics_version": cache_version,
        "reward_semantics_version": SPARSE_TERMINAL_SUCCESS_REWARD_SEMANTICS,
        "critic_valid_rows": critic_valid,
        "critic_valid_actual_sent_checked_rows": actual_sent_checked,
        "critic_valid_actual_sent_rows": actual_sent,
        "critic_valid_actual_sent_fraction": (
            actual_sent / actual_sent_checked if actual_sent_checked else None
        ),
        "transition_uid_available": bool(transition_uids),
        "transition_uids": transition_uids,
    }


def validate_episode_disjointness(train: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    train_uids, audit_uids = set(train["episode_uids"]), set(audit["episode_uids"])
    overlap = sorted(train_uids & audit_uids)
    train_transitions = set(train["transition_uids"])
    audit_transitions = set(audit["transition_uids"])
    transition_overlap = sorted(train_transitions & audit_transitions)
    report = {
        "train_episodes": len(train_uids),
        "audit_episodes": len(audit_uids),
        "overlapping_episodes": len(overlap),
        "overlapping_episode_uids": overlap,
        "transition_identity_check_available": bool(train_transitions and audit_transitions),
        "overlapping_transitions": len(transition_overlap),
        "overlapping_transition_uids": transition_overlap,
    }
    require(not overlap, f"D_train and D_audit share episodes: {overlap}")
    require(not transition_overlap, f"D_train and D_audit share transitions: {transition_overlap}")
    return report


def _fingerprints(run: dict, updates: int, evidence: Evidence) -> tuple[Path, list[int]]:
    local = run["root"].parents[2] / "diagnostics.jsonl"
    log = evidence.file(local if local.is_file() else run["config"]["diagnostics_jsonl_path"])
    fingerprints = []
    with log.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            step = len(fingerprints) + 1
            require(type(row.get("actor_refine_step")) is int and row["actor_refine_step"] == step,
                    f"Non-contiguous actor steps in {log}: expected {step}")
            require(all(row.get(k) is True for k in ("actor_update", "actor_refine_stage", "actor_refine_td3bc")),
                    f"Not a TD3+BC actor update at step {step}: {log}")
            require(row.get("actor_q_weight") == run["config"]["actor_q_weight_max"],
                    f"Logged Q coefficient differs at step {step}: {log}")
            fp = row.get("actor_refine_batch_fingerprint")
            require(type(fp) is int and 0 <= fp < 9_223_372_036_854_775_783,
                    f"Invalid or float-rounded batch fingerprint at step {step}: {log}")
            fingerprints.append(fp)
            if len(fingerprints) == updates:
                break
    require(len(fingerprints) == updates, f"Missing prefix diagnostics: {log}")
    require(run["state"]["_actor_refine_step"].item() == updates, f"Checkpoint update count differs: {run['root']}")
    require(run["state"]["_actor_refine_batch_fingerprint"].item() == fingerprints[-1],
            f"Checkpoint/log fingerprint mismatch: {run['root']}")
    return log, fingerprints


def _optimizer(run: dict, updates: int, evidence: Evidence) -> list:
    root = run["root"].parent / "training_state"
    require(evidence.json(root / "training_step.json")["step"] == updates, f"Training step differs: {root}")
    groups = evidence.json(root / "optimizer_param_groups.json")
    state = evidence.state(root / "optimizer_state.safetensors")
    require(len(groups) == 1, f"Expected one actor-only optimizer group: {root}")
    actor, _ = _construct_heads(run["config"])
    parameters = list(actor.parameters())
    ids = groups[0]["params"]
    require(len(ids) == len(parameters) and len(set(ids)) == len(ids), f"Optimizer parameter count differs: {root}")
    expected_keys = {f"state/{i}/{key}" for i in ids for key in ("step", "exp_avg", "exp_avg_sq")}
    require(set(state) == expected_keys, f"Unexpected optimizer state keys: {root}")
    for i, parameter in zip(ids, parameters, strict=True):
        require(state[f"state/{i}/step"].item() == updates, f"AdamW step differs for parameter {i}: {root}")
        for key in ("exp_avg", "exp_avg_sq"):
            require(state[f"state/{i}/{key}"].shape == parameter.shape, f"Optimizer shape mismatch: {root}")
    declared = run["train"]["optimizer"]
    expected_lr = run["config"].get("actor_lr")
    if expected_lr is None:
        expected_lr = declared["lr"]
    for key, expected in {"lr": expected_lr, "betas": declared["betas"],
                          "eps": declared["eps"], "weight_decay": declared["weight_decay"]}.items():
        require(groups[0][key] == expected, f"Saved optimizer {key} differs from config: {root}")
    return groups


def _critic_optimizer(run: dict, evidence: Evidence) -> dict[str, Any]:
    state = run["state"]
    require("_critic_step" in state, f"Missing persistent critic update counter: {run['root']}")
    updates = int(state["_critic_step"].item())
    require(updates > 0, f"Critic checkpoint has no realized updates: {run['root']}")
    root = run["root"].parent / "training_state"
    require(evidence.json(root / "training_step.json")["step"] == updates,
            f"Critic checkpoint/training-state update count differs: {root}")
    groups = evidence.json(root / "optimizer_param_groups.json")
    optimizer_state = evidence.state(root / "optimizer_state.safetensors")
    require(len(groups) == 1, f"Expected one critic-only optimizer group: {root}")
    _, critic = _construct_heads(run["config"])
    parameters = list(critic.parameters())
    ids = groups[0]["params"]
    require(len(ids) == len(parameters) and len(set(ids)) == len(ids),
            f"Critic optimizer parameter count differs: {root}")
    expected_keys = {f"state/{i}/{key}" for i in ids for key in ("step", "exp_avg", "exp_avg_sq")}
    require(set(optimizer_state) == expected_keys, f"Unexpected critic optimizer state keys: {root}")
    for index, parameter in zip(ids, parameters, strict=True):
        require(optimizer_state[f"state/{index}/step"].item() == updates,
                f"Critic optimizer step differs for parameter {index}: {root}")
        for moment in ("exp_avg", "exp_avg_sq"):
            require(optimizer_state[f"state/{index}/{moment}"].shape == parameter.shape,
                    f"Critic optimizer shape mismatch: {root}")
    declared = run["train"]["optimizer"]
    require(declared["type"] == "adamw", f"Paper critic fit requires AdamW: {root}")
    expected = {
        "lr": run["config"]["critic_lr"],
        "betas": declared["betas"],
        "eps": declared["eps"],
        "weight_decay": declared["weight_decay"],
    }
    for key, value in expected.items():
        require(groups[0][key] == value, f"Saved critic optimizer {key} differs from config: {root}")
    return {"updates": updates, "groups": groups, "declared": declared}


def _normalized_critic_fit(
    run: dict,
    *,
    initialization_model_sha256: str,
    initialization_actor_sha256: str,
    training_cache_sha256: str,
) -> dict[str, Any]:
    """Remove only seed and non-executing output metadata from a critic fit."""
    result = copy.deepcopy(run["train"])
    for key in ("output_dir", "job_name", "checkpoint_path", "log_freq", "seed"):
        result.pop(key, None)
    wandb = result.pop("wandb", {})
    require(not wandb.get("enable", False), "Paper critic fits cannot differ through an active logger")
    result["dataset"]["repo_id"] = training_cache_sha256
    for key in ("diagnostics_jsonl_path", "repo_id"):
        result["policy"].pop(key, None)
    result["policy"]["pretrained_path"] = {
        "model_sha256": initialization_model_sha256,
        "actor_sha256": initialization_actor_sha256,
    }
    return result


def validate_critic_fits(
    fits: dict[int, dict],
    *,
    audit_cache_sha256: str,
    evidence: Evidence,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify that the two complete critic fits differ only by fit seed/output."""
    require(set(fits) == {1, 2}, "Exactly two critic fits are required")
    records: dict[str, Any] = {}
    normalized: dict[int, dict] = {}
    optimizers: dict[int, dict] = {}
    for direction, fit in fits.items():
        cfg, train, state = fit["config"], fit["train"], fit["state"]
        require(cfg.get("training_stage") == "critic_only", f"Critic {direction} is not critic_only")
        require(type(train.get("seed")) is int, f"Critic {direction} fit seed is missing")
        require(train.get("resume") is False, f"Critic {direction} must start with fresh optimizer state")
        require(train.get("scheduler") is None, f"Critic {direction} scheduler is not supported by paper audit")
        require(train.get("eval_freq") == 0 and train.get("env") is None,
                f"Critic {direction} has training-time evaluation side effects")
        require(not train.get("use_rabc", False), f"Critic {direction} uses an unsupported objective")
        require(train.get("use_policy_training_preset") is True,
                f"Critic {direction} does not use the saved optimizer preset")
        require(math.isfinite(cfg.get("gamma", float("nan"))), f"Critic {direction} gamma is invalid")
        require(math.isfinite(cfg.get("tau", float("nan"))), f"Critic {direction} tau is invalid")
        require(cfg.get("target_q_clip") is None or math.isfinite(cfg["target_q_clip"]),
                f"Critic {direction} target_q_clip is invalid")
        initial = _load_checkpoint(Path(cfg["pretrained_path"]), evidence)
        initial_model = evidence.files[str(initial["root"] / "model.safetensors")]
        initial_actor = tensor_digest(initial["state"], "actor.")
        require(tensor_digest(state, "actor.") == initial_actor,
                f"Critic {direction} modified its actor initialization")
        cache = _training_cache(train, evidence)
        require(evidence.files[str(cache)] != audit_cache_sha256,
                f"Critic {direction} was fitted on D_audit")
        cache_meta = validate_paper_cache(cache, evidence, require_actual_sent_for_critic=True)
        optimizer = _critic_optimizer(fit, evidence)
        require(optimizer["updates"] == int(state["_critic_step"].item()),
                f"Critic {direction} update counter mismatch")
        normalized[direction] = _normalized_critic_fit(
            fit,
            initialization_model_sha256=initial_model,
            initialization_actor_sha256=initial_actor,
            training_cache_sha256=cache_meta["sha256"],
        )
        optimizers[direction] = optimizer
        records[str(direction)] = {
            "checkpoint": str(fit["root"]),
            "training_stage": cfg["training_stage"],
            "fit_seed": train["seed"],
            "critic_sha256": tensor_digest(state, "critic."),
            "target_critic_sha256": tensor_digest(state, "target_critic."),
            "actor_initialization_checkpoint": str(initial["root"]),
            "actor_initialization_model_sha256": initial_model,
            "actor_initialization_sha256": initial_actor,
            "architecture": _head_signature(cfg),
            "training_cache": cache_meta,
            "critic_objective": {
                "name": "masked_twin_critic_chunk_Bellman_MSE",
                "use_rabc": False,
                "validity_mask": "critic_mask",
            },
            "gamma": cfg["gamma"],
            "target_q_clip": cfg.get("target_q_clip"),
            "tau": cfg["tau"],
            "target_critic_update": "Polyak update after every critic optimizer step",
            "critic_bootstrap_mode": cfg.get("critic_bootstrap_mode"),
            "critic_bootstrap_keep_prob": cfg.get("critic_bootstrap_keep_prob"),
            "critic_bootstrap_seed": cfg.get("critic_bootstrap_seed"),
            "optimizer_type": train["optimizer"]["type"],
            "optimizer_configuration": optimizer["declared"],
            "critic_learning_rate": cfg["critic_lr"],
            "batch_size": train["batch_size"],
            "source_sampling_weights": cfg.get("source_sampling_weights"),
            "source_sampling_seed": cfg.get("source_sampling_seed"),
            "critic_updates": optimizer["updates"],
            "normalized_training_config": normalized[direction],
        }
    require(fits[1]["train"]["seed"] != fits[2]["train"]["seed"],
            "Critic fit seeds are not different")
    require(records["1"]["actor_initialization_sha256"] == records["2"]["actor_initialization_sha256"],
            "Initial actor tensors differ across critic fits")
    require(records["1"]["architecture"] == records["2"]["architecture"],
            "Critic architectures differ")
    require(records["1"]["training_cache"]["sha256"] == records["2"]["training_cache"]["sha256"],
            "Critic training-cache hashes differ")
    require(records["1"]["critic_updates"] == records["2"]["critic_updates"],
            "Critic update counts differ")
    require(optimizers[1]["groups"] == optimizers[2]["groups"],
            "Saved critic optimizer groups differ")
    require(normalized[1] == normalized[2],
            f"Critic training configs differ beyond seed/output-only fields: {_differences(normalized[1], normalized[2])}")
    require(records["1"]["critic_sha256"] != records["2"]["critic_sha256"],
            "Both critic arguments contain identical critic tensors")
    fields = (
        "gamma", "target_q_clip", "tau", "critic_bootstrap_mode",
        "critic_bootstrap_keep_prob", "critic_bootstrap_seed", "optimizer_type",
        "critic_learning_rate", "batch_size", "source_sampling_weights",
        "source_sampling_seed", "critic_updates",
    )
    require(all(records["1"][key] == records["2"][key] for key in fields),
            "Explicit critic fitting fields differ")
    return records, records["1"]["training_cache"]


def _effective_config(run: dict, initial_hash: str, cache_hash: str, updates: int) -> dict:
    result = copy.deepcopy(run["train"])
    for key in ("output_dir", "job_name", "checkpoint_path", "log_freq", "wandb"):
        result.pop(key, None)
    # Safe only because the validation below requires no scheduler or other
    # total-step-dependent operator. Keep save_freq to preserve RNG side effects.
    result["steps"] = updates
    result["dataset"]["repo_id"] = cache_hash
    for key in ("actor_q_weight_max", "diagnostics_jsonl_path", "repo_id", "critic_lr"):
        result["policy"].pop(key, None)
    result["policy"]["pretrained_path"] = initial_hash
    return result


def validate_full_refinement(
    *, q0: Path, critics: dict[int, Path], candidates: list[Candidate],
    audit_cache: Path, expected_updates: int = 897, require_bidirectional: bool = True,
    evidence: Evidence | None = None,
) -> tuple[dict, Evidence]:
    """Accept only the agreed fresh-optimizer, fixed-BC, actor-only operator."""
    require(expected_updates > 0, "expected_updates must be positive")
    require(set(critics) == {1, 2}, "Exactly two complete independent critics are required")
    require(bool(candidates), "At least one positive-Q candidate is required")
    for c in candidates:
        require(math.isfinite(c.q_weight) and c.q_weight > 0 and c.inducing_critic in (1, 2),
                f"Invalid candidate: {c}")
    require(len({c.key for c in candidates}) == len(candidates), "Duplicate Q/direction candidate")
    present = {(c.q_weight, c.inducing_critic) for c in candidates}
    missing = [f"q{q:g}_c{k}" for q in sorted({c.q_weight for c in candidates}) for k in (1, 2)
               if (q, k) not in present]
    require(not require_bidirectional or not missing, f"Missing reverse candidates: {missing}")
    evidence = evidence or Evidence()
    audit = evidence.file(audit_cache)  # A directory is deliberately not accepted.
    require(audit.suffix == ".pt", "audit_cache must be an explicit .pt file")
    audit_meta = validate_paper_cache(audit, evidence, require_actual_sent_for_critic=False)
    fits = {k: _load_checkpoint(p, evidence) for k, p in critics.items()}
    critic_records, train_cache_meta = validate_critic_fits(
        fits,
        audit_cache_sha256=audit_meta["sha256"],
        evidence=evidence,
    )
    disjointness = validate_episode_disjointness(train_cache_meta, audit_meta)
    fit_heads = _head_signature(fits[1]["config"])

    run_specs = [("q0", q0, 0.0, None)] + [(c.key, c.checkpoint, c.q_weight, c.inducing_critic) for c in candidates]
    records: dict[str, dict] = {}
    reference = None
    reference_fp = None
    reference_groups = None
    baseline_run = None
    for key, path, q_weight, direction in run_specs:
        run = _load_checkpoint(path, evidence)
        cfg, train, state = run["config"], run["train"], run["state"]
        for field, expected in {
            "training_stage": "actor_refine", "actor_refine_objective": "td3bc",
            "actor_bc_weight_mode": "fixed", "actor_q_trust_mode": "fixed",
            "actor_behavior_preservation_weight": 0.0, "actor_human_weight": 0.0,
            "actor_teacher_weight": 0.0, "actor_teacher_pretrained_path": "",
            "actor_q_weight_max": q_weight, "actor_ref_dropout_p": 0.0,
            "actor_action_residual": True, "utd_ratio": 1, "actor_update_interval": 1,
            "use_amp": False, "use_peft": False,
        }.items():
            require(cfg.get(field) == expected, f"{key}: unsupported {field}={cfg.get(field)!r}")
        require(math.isfinite(cfg["beta"]) and cfg["beta"] > 0, f"{key}: BC coefficient must be positive")
        require(train["scheduler"] is None and train["resume"] is False, f"{key}: requires fresh optimizer, no scheduler")
        require(train["steps"] >= expected_updates and train["eval_freq"] == 0 and train["env"] is None,
                f"{key}: incompatible training length/evaluation schedule")
        require(train["num_workers"] == 0 and not train.get("use_rabc", False), f"{key}: unsupported sampler/operator")
        require(train["optimizer"]["type"] == "adamw" and train["optimizer"]["weight_decay"] == 0
                and train["use_policy_training_preset"] is True, f"{key}: requires actor-only AdamW preset")
        require(not train.get("wandb", {}).get("enable", False), f"{key}: active external logger requires separate RNG audit")
        require(_head_signature(cfg) == fit_heads, f"{key}: head construction differs from critics")
        initial = _load_checkpoint(Path(cfg["pretrained_path"]), evidence)
        fit_options = [fits[direction]] if direction else list(fits.values())
        require(any(evidence.files[str(initial['root'] / 'model.safetensors')] ==
                    evidence.files[str(f['root'] / 'model.safetensors')] for f in fit_options),
                f"{key}: initialization is not its declared inducing critic checkpoint")
        require(initial["state"]["_actor_refine_step"].item() == 0 and
                initial["state"]["_actor_refine_batch_fingerprint"].item() == 0, f"{key}: initialization is already refined")
        initial_actor_hash = tensor_digest(initial["state"], "actor.")
        require(set(initial["state"]) == set(state), f"{key}: checkpoint tensor keys differ from initialization")
        for name in state:
            if not name.startswith("actor.") and name not in ("_actor_refine_step", "_actor_refine_batch_fingerprint"):
                require(torch.equal(state[name], initial["state"][name]), f"{key}: non-actor tensor changed: {name}")
        cache = _training_cache(train, evidence)
        cache_hash = evidence.files[str(cache)]
        require(cache_hash != evidence.files[str(audit)], f"{key}: audit file is actor training cache")
        require(cache_hash == train_cache_meta["sha256"],
                f"{key}: actor refinement D_train differs from critic fitting D_train")
        log, fp = _fingerprints(run, expected_updates, evidence)
        groups = _optimizer(run, expected_updates, evidence)
        effective = _effective_config(run, initial_actor_hash, cache_hash, expected_updates)
        if reference is None:
            reference, reference_fp, reference_groups, baseline_run = effective, fp, groups, run
        else:
            require(effective == reference, f"{key}: effective training configuration mismatch: {_differences(reference, effective)}")
            require(fp == reference_fp, f"{key}: batch fingerprint prefix mismatch (not just final fingerprint)")
            require(groups == reference_groups, f"{key}: saved optimizer groups differ")
        records[key] = {
            "checkpoint": str(run["root"]), "q_weight": q_weight, "inducing_critic": direction,
            "model_sha256": evidence.files[str(run["root"] / "model.safetensors")],
            "actor_sha256": tensor_digest(state, "actor."),
            "critic_sha256": tensor_digest(state, "critic."),
            "actor_initialization_checkpoint": str(initial["root"]),
            "actor_initialization_sha256": initial_actor_hash,
            "training_cache": str(cache), "training_cache_sha256": cache_hash,
            "updates": expected_updates, "planned_steps": train["steps"],
            "seed": train["seed"], "source_sampling_seed": cfg["source_sampling_seed"],
            "diagnostics": str(log), "batch_fingerprints": fp,
            "batch_fingerprint_prefix_sha256": hashlib.sha256(json.dumps(fp).encode()).hexdigest(),
            "optimizer_groups": groups, "optimizer_config": train["optimizer"],
            "raw_policy_differences_from_q0": _differences(baseline_run["config"], cfg),
            "raw_top_level_differences_from_q0": _differences(
                {k: v for k, v in baseline_run["train"].items() if k != "policy"},
                {k: v for k, v in train.items() if k != "policy"}),
        }
    return {
        "status": "MATCHED_EFFECTIVE_OPERATOR", "bidirectional_complete": not missing,
        "missing_candidates": missing, "expected_updates": expected_updates,
        "runs": records,
        "critics": critic_records,
        "critic_fit_match": {
            "allowed_differences": ["seed", "output-only paths", "log frequency"],
            "all_other_saved_training_configuration_identical": True,
            "initial_model_sha256_identical": (
                critic_records["1"]["actor_initialization_model_sha256"]
                == critic_records["2"]["actor_initialization_model_sha256"]
            ),
            "initial_actor_sha256_identical": True,
            "architecture_identical": True,
            "training_cache_sha256_identical": True,
            "reward_and_cache_semantics_identical": True,
            "optimizer_configuration_identical": True,
            "critic_updates_identical": True,
        },
        "training_cache": train_cache_meta,
        "audit_cache": audit_meta,
        "episode_disjointness": disjointness,
        "effective_config": reference,
        "multi_step_update_object": {
            "notation": {
                "control": "theta_BC^(T) = U_0^(T)(theta_0; D_train)",
                "treatment": "theta_lambda^(i,T) = U_lambda^(T)(Q^(i), theta_0; D_train)",
            },
            "T": expected_updates,
            "theta_0_actor_sha256": records["q0"]["actor_initialization_sha256"],
            "theta_0_checkpoint_by_direction": {
                "1": critic_records["1"]["checkpoint"],
                "2": critic_records["2"]["checkpoint"],
            },
            "D_train_sha256": train_cache_meta["sha256"],
            "actor_optimizer_type": records["q0"]["optimizer_config"]["type"],
            "actor_learning_rate": records["q0"]["optimizer_groups"][0]["lr"],
            "beta": reference["policy"]["beta"],
            "batch_size": reference["batch_size"],
            "batch_seed": reference["seed"],
            "source_sampling_seed": reference["policy"]["source_sampling_seed"],
            "source_sampling_weights": reference["policy"]["source_sampling_weights"],
            "batch_fingerprint_prefix_sha256": records["q0"]["batch_fingerprint_prefix_sha256"],
            "Q0_q_weight": 0.0,
            "treatment_q_weights": sorted({candidate.q_weight for candidate in candidates}),
            "critic_parameters_frozen": True,
            "identical_non_Q_actor_configuration": True,
        },
        "critic_td_target_semantics": {
            "reward_padding": "reward_seq is zero-padded beyond actual_steps",
            "bootstrap_exponent": "gamma ** actual_steps",
            "bootstrap_gate": "explicit bootstrap_mask required by semantic-v2",
            "next_action": "deterministic current-actor mean at next state/proposal, clamped to [-1,1]",
            "next_value": "target critic min-Q",
            "target_q_clip": critic_records["1"]["target_q_clip"],
            "stop_gradient": True,
            "separate_target_actor": False,
        },
        "explicit_exceptions": {
            "critic_lr": "Inactive: optimizer contains only actor parameters; frozen critic tensors verified.",
            "planned_steps": "Compare realized prefix; scheduler is None, fixed weights, no training-time evaluation.",
            "pretrained_path": "Inducing critic changes intentionally; initial actor tensors and head semantics must match.",
            "output_metadata": "Output paths/job names/repo IDs/log frequency/inactive wandb settings may differ.",
        },
        "limitations": [
            "Rolling batch fingerprints are observed sequence evidence, not retained raw batch-index lists.",
            "Matching saved evidence does not establish identical historical CUDA/software environments.",
            "Episode disjointness depends on source-dataset-stable episode_uid values stored in every transition.",
        ],
    }, evidence
