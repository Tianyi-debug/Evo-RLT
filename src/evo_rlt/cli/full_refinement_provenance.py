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
    fits = {k: _load_checkpoint(p, evidence) for k, p in critics.items()}
    for fit in fits.values():
        require(fit["config"]["training_stage"] == "critic_only", "Inducing checkpoint is not a critic-only fit")
        require(type(fit["train"]["seed"]) is int, "Critic fit seed is missing")
        cache = _training_cache(fit["train"], evidence)
        require(evidence.files[str(cache)] != evidence.files[str(audit)], "Audit file is critic training cache")
    require(fits[1]["train"]["seed"] != fits[2]["train"]["seed"], "Critic fit seeds are not independent")
    require(tensor_digest(fits[1]["state"], "actor.") == tensor_digest(fits[2]["state"], "actor."),
            "Initial actor tensors differ across the two critic checkpoints")
    require(tensor_digest(fits[1]["state"], "critic.") != tensor_digest(fits[2]["state"], "critic."),
            "Both critic arguments contain the same critic tensors")
    fit_heads = _head_signature(fits[1]["config"])
    require(fit_heads == _head_signature(fits[2]["config"]), "Critic/actor construction differs across fits")

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
        "critics": {str(k): {"checkpoint": str(f["root"]), "fit_seed": f["train"]["seed"],
                               "critic_sha256": tensor_digest(f["state"], "critic.")} for k, f in fits.items()},
        "audit_cache": str(audit), "audit_cache_sha256": evidence.files[str(audit)],
        "effective_config": reference,
        "explicit_exceptions": {
            "critic_lr": "Inactive: optimizer contains only actor parameters; frozen critic tensors verified.",
            "planned_steps": "Compare realized prefix; scheduler is None, fixed weights, no training-time evaluation.",
            "pretrained_path": "Inducing critic changes intentionally; initial actor tensors and head semantics must match.",
            "output_metadata": "Output paths/job names/repo IDs/log frequency/inactive wandb settings may differ.",
        },
        "limitations": [
            "Rolling batch fingerprints are observed sequence evidence, not retained raw batch-index lists.",
            "Matching saved evidence does not establish identical historical CUDA/software environments.",
            "Different cache hashes establish different files, not episode-level disjointness; local episode IDs cannot prove that.",
        ],
    }, evidence
