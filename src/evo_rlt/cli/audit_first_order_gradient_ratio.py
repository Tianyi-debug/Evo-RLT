"""Reproducible D_train-only first-order Q/BC actor-gradient diagnostic.

This is deliberately separate from RIR. It evaluates raw objective gradients at
the shared actor initialization before clipping or AdamW preconditioning and
never takes an optimizer step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from evo_rlt.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset
from evo_rlt.cli.audit_actor_q_mechanism import _construct_heads
from evo_rlt.cli.full_refinement_provenance import (
    Evidence,
    ProvenanceError,
    _effective_config,
    _fingerprints,
    _head_signature,
    _load_checkpoint,
    _optimizer,
    _training_cache,
    require,
    tensor_digest,
    validate_paper_cache,
)
from evo_rlt.core.losses import fixed_td3bc_objective_terms


def _gradient_norm(gradients: tuple[torch.Tensor | None, ...]) -> float:
    terms = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    require(bool(terms), "Objective has no actor gradient")
    return float(torch.stack(terms).sum().sqrt().item())


def compute_gradient_ratio(
    *,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    lambda_q: float,
    beta: float,
    epsilon: float,
) -> dict[str, Any]:
    """Compute lambda*||dLQ||/(beta*||dLBC||+eps), without optimizer state."""
    require(math.isfinite(lambda_q) and lambda_q > 0, "lambda_q must be finite and positive")
    require(math.isfinite(beta) and beta > 0, "beta must be finite and positive")
    require(math.isfinite(epsilon) and epsilon > 0, "epsilon must be finite and positive")
    before_actor = tensor_digest(actor.state_dict())
    before_critic = tensor_digest(critic.state_dict())
    require(all(parameter.grad is None for parameter in actor.parameters()),
            "Actor has populated .grad buffers before gradient diagnostic")
    require(all(parameter.grad is None for parameter in critic.parameters()),
            "Critic has populated .grad buffers before gradient diagnostic")
    actor_requires_grad = [parameter.requires_grad for parameter in actor.parameters()]
    critic_requires_grad = [parameter.requires_grad for parameter in critic.parameters()]
    actor_was_training = actor.training
    critic_was_training = critic.training
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    actor.eval()
    critic.eval()
    parameters = tuple(actor.parameters())
    try:
        # training=True matches actor refinement. The paper validator requires
        # actor_ref_dropout_p=0, so this path is deterministic.
        mu, _ = actor(batch["state_vec"], batch["proposal_chunk_flat"], training=True)
        q1, q2 = critic(batch["state_vec"], mu)
        q = torch.minimum(q1, q2).reshape(-1)
        terms = fixed_td3bc_objective_terms(
            mu=mu,
            q=q,
            batch=batch,
            action_residual=bool(getattr(actor, "action_residual", False)),
        )
        require(bool(terms["actor_q_valid"].any()), "Diagnostic batch has no actor-Q-valid rows")
        require(bool(terms["actor_bc_valid"].any()), "Diagnostic batch has no actor-BC-valid rows")
        q_gradients = torch.autograd.grad(
            terms["q_loss"], parameters, retain_graph=True, allow_unused=True
        )
        bc_gradients = torch.autograd.grad(
            terms["bc_loss"], parameters, allow_unused=True
        )
        q_norm = _gradient_norm(q_gradients)
        bc_norm = _gradient_norm(bc_gradients)
        result = {
            "lambda_q": lambda_q,
            "beta": beta,
            "epsilon": epsilon,
            "q_gradient_norm": q_norm,
            "bc_gradient_norm": bc_norm,
            "weighted_q_gradient_norm": lambda_q * q_norm,
            "weighted_bc_gradient_norm": beta * bc_norm,
            "r_grad": lambda_q * q_norm / (beta * bc_norm + epsilon),
            "actor_q_valid_rows": int(terms["actor_q_valid"].sum().item()),
            "actor_bc_valid_rows": int(terms["actor_bc_valid"].sum().item()),
            "pre_gradient_clipping": True,
            "pre_optimizer_preconditioning": True,
            "optimizer_steps": 0,
            "critic_parameters_frozen": True,
            "parameter_grad_buffers_populated": False,
            "actor_and_critic_tensors_bit_identical_after_diagnostic": True,
            "bc_target_semantics": (
                "a_BC_bar = clip(a_BC_raw, -1, 1) for residual actor; "
                "no projection into residual reachable interval"
            ),
        }
    finally:
        for parameter, original in zip(actor.parameters(), actor_requires_grad, strict=True):
            parameter.requires_grad_(original)
        for parameter, original in zip(critic.parameters(), critic_requires_grad, strict=True):
            parameter.requires_grad_(original)
        actor.train(actor_was_training)
        critic.train(critic_was_training)
    require(tensor_digest(actor.state_dict()) == before_actor, "Actor tensors changed in gradient diagnostic")
    require(tensor_digest(critic.state_dict()) == before_critic, "Critic tensors changed in gradient diagnostic")
    require(all(parameter.grad is None for parameter in actor.parameters()),
            "Gradient diagnostic populated actor .grad buffers")
    require(all(parameter.grad is None for parameter in critic.parameters()),
            "Gradient diagnostic populated critic .grad buffers")
    return result


def deterministic_training_minibatch(
    *,
    cache: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    batch_size: int,
    minibatch_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Select theta_0's diagnostic batch by a declared, replayable rule."""
    require(cache.name == "chunk_transitions_train.pt", "Gradient diagnostic must use D_train")
    require(batch_size > 0, "batch_size must be positive")
    dataset = ChunkTransitionDataset(
        cache.parent,
        split="train",
        training_stage="actor_refine",
        source_sampling_weights=config.get("source_sampling_weights"),
        source_sampling_seed=int(config["source_sampling_seed"]),
    )
    generator = torch.Generator().manual_seed(minibatch_seed)
    positions = torch.randperm(len(dataset), generator=generator)[: min(batch_size, len(dataset))].tolist()
    indices = [dataset.sample_indices[position] for position in positions]
    require(bool(indices), "No D_train rows selected")
    selected = [rows[index] for index in indices]

    def stack(key: str) -> torch.Tensor:
        require(all(key in row for row in selected), f"D_train batch is missing {key!r}")
        return torch.stack([torch.as_tensor(row[key]) for row in selected])

    proposals = stack("proposal_chunk").float().flatten(1)
    batch = {
        "state_vec": stack("state_vec").float(),
        "proposal_chunk_flat": proposals,
        "bc_target_chunk_flat": stack("bc_target_chunk").float().flatten(1),
        "actor_q_mask": stack("actor_q_mask").float(),
        "actor_bc_mask": stack("actor_bc_mask").float(),
    }
    fingerprint = 0
    modulus = 9_223_372_036_854_775_783
    for index in indices:
        fingerprint = (fingerprint * 1_000_003 + index + 1) % modulus
    return batch, {
        "rule": (
            "build source-balanced actor_refine sample map with saved source_sampling_seed; "
            "torch.randperm over sample-map positions with minibatch_seed; take first batch_size"
        ),
        "source_sampling_seed": config["source_sampling_seed"],
        "source_sampling_weights": config.get("source_sampling_weights"),
        "minibatch_seed": minibatch_seed,
        "sample_map_positions": positions,
        "raw_cache_indices": indices,
        "raw_cache_indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
        "batch_fingerprint": fingerprint,
    }


def _write_json(path: Path, value: Any) -> None:
    require(not path.exists(), f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _validate_paper_actor_run(run: dict[str, Any], *, q_weight: float, label: str) -> None:
    config, train = run["config"], run["train"]
    for field, expected in {
        "training_stage": "actor_refine",
        "actor_refine_objective": "td3bc",
        "actor_bc_weight_mode": "fixed",
        "actor_q_trust_mode": "fixed",
        "actor_behavior_preservation_weight": 0.0,
        "actor_teacher_weight": 0.0,
        "actor_human_weight": 0.0,
        "actor_ref_dropout_p": 0.0,
        "actor_action_residual": True,
        "actor_q_weight_max": q_weight,
        "use_amp": False,
        "use_peft": False,
    }.items():
        require(config.get(field) == expected, f"{label}: unsupported {field}={config.get(field)!r}")
    require(math.isfinite(config.get("beta", float("nan"))) and config["beta"] > 0,
            f"{label}: beta must be finite and positive")
    require(train.get("scheduler") is None and train.get("resume") is False,
            f"{label}: requires fresh optimizer and no scheduler")
    require(train.get("num_workers") == 0 and not train.get("use_rabc", False),
            f"{label}: unsupported sampler/operator")
    require(train["optimizer"]["type"] == "adamw" and train["optimizer"]["weight_decay"] == 0,
            f"{label}: requires actor-only AdamW without weight decay")


def run(args: argparse.Namespace) -> dict[str, Any]:
    evidence = Evidence()
    candidate = _load_checkpoint(args.candidate_checkpoint, evidence)
    control = _load_checkpoint(args.q0_checkpoint, evidence)
    cfg, train = candidate["config"], candidate["train"]
    lambda_q = float(cfg["actor_q_weight_max"])
    beta = float(cfg["beta"])
    require(lambda_q > 0, "Candidate Q weight must be positive")
    control_cfg, control_train = control["config"], control["train"]
    _validate_paper_actor_run(candidate, q_weight=lambda_q, label="candidate")
    _validate_paper_actor_run(control, q_weight=0.0, label="Q0")
    candidate_updates = int(candidate["state"]["_actor_refine_step"].item())
    control_updates = int(control["state"]["_actor_refine_step"].item())
    require(candidate_updates > 0 and candidate_updates == control_updates,
            "Candidate and Q0 realized actor-refinement step counts differ")
    initial = _load_checkpoint(Path(cfg["pretrained_path"]), evidence)
    control_initial = _load_checkpoint(Path(control_cfg["pretrained_path"]), evidence)
    require(initial["config"].get("training_stage") == "critic_only",
            "Candidate initialization is not an inducing critic checkpoint")
    require(control_initial["config"].get("training_stage") == "critic_only",
            "Q0 initialization is not an inducing critic checkpoint")
    require(
        tensor_digest(initial["state"], "actor.")
        == tensor_digest(control_initial["state"], "actor."),
        "Candidate and Q0 do not share actor initialization theta_0",
    )
    require(_head_signature(cfg) == _head_signature(control_cfg),
            "Candidate and Q0 actor/critic head semantics differ")
    cache = _training_cache(train, evidence)
    control_cache = _training_cache(control_train, evidence)
    require(evidence.files[str(cache)] == evidence.files[str(control_cache)],
            "Candidate and Q0 D_train hashes differ")
    inducing_cache = _training_cache(initial["train"], evidence)
    require(evidence.files[str(cache)] == evidence.files[str(inducing_cache)],
            "Gradient D_train differs from the inducing critic's D_train")
    control_inducing_cache = _training_cache(control_initial["train"], evidence)
    require(evidence.files[str(cache)] == evidence.files[str(control_inducing_cache)],
            "Gradient D_train differs from the Q0 inducing critic's D_train")
    candidate_effective = _effective_config(
        candidate,
        tensor_digest(initial["state"], "actor."),
        evidence.files[str(cache)],
        candidate_updates,
    )
    control_effective = _effective_config(
        control,
        tensor_digest(control_initial["state"], "actor."),
        evidence.files[str(control_cache)],
        control_updates,
    )
    require(candidate_effective == control_effective,
            "Candidate and Q0 differ in non-Q actor-refinement configuration")
    _, candidate_fingerprints = _fingerprints(candidate, candidate_updates, evidence)
    _, control_fingerprints = _fingerprints(control, control_updates, evidence)
    require(candidate_fingerprints == control_fingerprints,
            "Candidate and Q0 batch-fingerprint sequences differ")
    require(
        _optimizer(candidate, candidate_updates, evidence)
        == _optimizer(control, control_updates, evidence),
        "Candidate and Q0 actor optimizer groups differ",
    )
    cache_meta = validate_paper_cache(cache, evidence, require_actual_sent_for_critic=True)
    _, rows = evidence.cache(cache)
    requested_batch_size = args.batch_size if args.batch_size is not None else int(train["batch_size"])
    minibatch_seed = args.minibatch_seed if args.minibatch_seed is not None else int(train["seed"])
    batch, selection = deterministic_training_minibatch(
        cache=cache,
        rows=rows,
        config=cfg,
        batch_size=requested_batch_size,
        minibatch_seed=minibatch_seed,
    )
    actor, critic = _construct_heads(initial["config"])
    state = initial["state"]
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in state.items()
                           if key.startswith("actor.")}, strict=True)
    critic.load_state_dict({key.removeprefix("critic."): value for key, value in state.items()
                            if key.startswith("critic.")}, strict=True)
    batch = {key: value.to(args.device) for key, value in batch.items()}
    actor, critic = actor.to(args.device), critic.to(args.device)
    diagnostic = compute_gradient_ratio(
        actor=actor,
        critic=critic,
        batch=batch,
        lambda_q=lambda_q,
        beta=beta,
        epsilon=args.epsilon,
    )
    evidence.verify_unchanged()
    report = {
        "schema_version": 1,
        "method": "first_order_Q_BC_gradient_ratio",
        "separate_from_RIR": True,
        "theta_t": "shared actor initialization theta_0",
        "theta_checkpoint": str(initial["root"]),
        "theta_actor_sha256": tensor_digest(initial["state"], "actor."),
        "matched_q0_checkpoint": str(control["root"]),
        "matched_q0_theta_checkpoint": str(control_initial["root"]),
        "matched_q0_theta_actor_sha256": tensor_digest(control_initial["state"], "actor."),
        "matched_actor_refinement_steps": candidate_updates,
        "matched_actor_batch_fingerprint_prefix_sha256": hashlib.sha256(
            json.dumps(candidate_fingerprints).encode()
        ).hexdigest(),
        "critic_checkpoint": str(initial["root"]),
        "critic_sha256": tensor_digest(initial["state"], "critic."),
        "candidate_configuration_checkpoint": str(candidate["root"]),
        "D_train": cache_meta,
        "minibatch": selection,
        "diagnostic": diagnostic,
        "input_files_sha256": evidence.files,
        "input_files_unchanged": True,
        "runtime": {"argv": sys.argv, "python": sys.version, "torch": torch.__version__, "device": args.device},
    }
    _write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"output": str(args.output.expanduser().resolve()), **diagnostic}, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--q0-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--minibatch-seed", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ProvenanceError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"Gradient diagnostic refused: {error}\n")


if __name__ == "__main__":
    main()
