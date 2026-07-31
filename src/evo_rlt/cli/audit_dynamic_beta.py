#!/usr/bin/env python
"""Calibrate and audit disagreement-weighted VLA-BC without loading the VLA."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor

from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--calibration-cache-dir", required=True)
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--training-cache-dir", required=True)
    parser.add_argument("--training-split", default="train")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--kappa", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-batches", type=int, default=8)
    parser.add_argument("--max-calibration-samples", type=int, default=None)
    parser.add_argument("--max-training-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load_config(policy_path: str | Path) -> dict:
    path = Path(policy_path) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"AC config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_heads(config: dict) -> tuple[ChunkActor, TwinCritic]:
    state_dim = int(config["rl_token_dim"]) + int(config["proprio_dim"])
    chunk_dim = int(config["chunk_length"]) * int(config["action_dim"])
    actor = ChunkActor(
        state_dim=state_dim,
        chunk_dim=chunk_dim,
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
    )
    critic = TwinCritic(
        state_dim=state_dim,
        chunk_dim=chunk_dim,
        hidden_dim=int(config["critic_hidden_dim"]),
        num_layers=int(config["critic_num_layers"]),
        activation=config["critic_activation"],
        layer_norm=bool(config["critic_layer_norm"]),
        residual=bool(config["critic_residual"]),
        proprio_dim=int(config["proprio_dim"]),
        state_normalization=config["state_normalization"],
    )
    return actor, critic


def load_heads(
    policy_path: str | Path,
    device: str,
) -> tuple[ChunkActor, TwinCritic, dict]:
    config = _load_config(policy_path)
    actor, critic = _build_heads(config)
    weights_path = Path(policy_path) / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"AC weights not found: {weights_path}")
    weights = load_file(weights_path, device="cpu")
    actor.load_state_dict(
        {key.removeprefix("actor."): value for key, value in weights.items() if key.startswith("actor.")}
    )
    critic.load_state_dict(
        {key.removeprefix("critic."): value for key, value in weights.items() if key.startswith("critic.")}
    )
    actor.to(device).eval()
    critic.to(device).eval()
    return actor, critic, config


def _load_samples(
    cache_dir: str | Path,
    split: str,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Tensor]]:
    path = Path(cache_dir) / f"chunk_transitions_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Transition cache not found: {path}")
    samples = torch.load(path, map_location="cpu", weights_only=False)
    if not samples:
        raise ValueError(f"Transition cache is empty: {path}")
    if max_samples is not None and len(samples) > max_samples:
        indices = random.Random(seed).sample(range(len(samples)), max_samples)
        samples = [samples[index] for index in sorted(indices)]
    return samples


def _batch(samples: list[dict[str, Tensor]], device: str) -> dict[str, Tensor]:
    return {
        "state_vec": torch.stack([sample["state_vec"] for sample in samples]).to(device),
        "ref_chunk_flat": torch.stack([sample["ref_chunk"] for sample in samples])
        .flatten(start_dim=-2)
        .to(device),
        "source": torch.stack(
            [sample.get("source", torch.tensor(0)) for sample in samples]
        ).reshape(-1).to(device),
    }


def _iter_batches(
    samples: list[dict[str, Tensor]],
    batch_size: int,
    device: str,
):
    for start in range(0, len(samples), batch_size):
        yield _batch(samples[start : start + batch_size], device)


def _quantiles(values: Tensor) -> dict[str, float]:
    values = values.float().cpu()
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
    }


@torch.no_grad()
def score_samples(
    actor: ChunkActor,
    critic: TwinCritic,
    samples: list[dict[str, Tensor]],
    batch_size: int,
    device: str,
    beta: float,
    kappa: float,
    tau_low: float | None = None,
    tau_high: float | None = None,
) -> dict:
    disagreements: list[Tensor] = []
    bc_values: list[Tensor] = []
    ref_delta_values: list[Tensor] = []
    q_values: list[Tensor] = []
    sources: list[Tensor] = []
    for batch in _iter_batches(samples, batch_size, device):
        state = batch["state_vec"]
        ref = batch["ref_chunk_flat"]
        mu, _ = actor(state, ref, training=False)
        q1, q2 = critic(state, mu)
        disagreements.append(((q1 - q2).abs() * 0.5).reshape(-1).cpu())
        q_values.append(torch.minimum(q1, q2).reshape(-1).cpu())
        bc_target = ref.clamp(-1.0, 1.0) if actor.action_residual else ref
        bc_values.append(((mu - bc_target) ** 2).sum(dim=-1).cpu())
        ref_delta_values.append((mu - bc_target).abs().flatten().cpu())
        sources.append(batch["source"].cpu())

    disagreement = torch.cat(disagreements)
    bc = torch.cat(bc_values)
    ref_delta = torch.cat(ref_delta_values)
    q = torch.cat(q_values)
    source = torch.cat(sources).long()
    result = {
        "count": int(disagreement.numel()),
        "disagreement": _quantiles(disagreement),
        "bc_per_sample": _quantiles(bc),
        "actor_ref_delta_abs": _quantiles(ref_delta),
        "q_min": _quantiles(q),
        "source": {},
    }
    for source_id in range(4):
        mask = source == source_id
        source_result = {"count": int(mask.sum()), "fraction": float(mask.float().mean())}
        if bool(mask.any()):
            source_result["disagreement"] = _quantiles(disagreement[mask])
        result["source"][str(source_id)] = source_result

    if tau_low is not None and tau_high is not None:
        if tau_high <= tau_low:
            raise ValueError(f"tau_high must be greater than tau_low, got {tau_high} <= {tau_low}")
        rho = ((disagreement - tau_low) / (tau_high - tau_low)).clamp(0.0, 1.0)
        beta_values = beta * (1.0 + kappa * rho)
        result["rho"] = _quantiles(rho)
        result["rho"]["zero_frac"] = float((rho <= 0.0).float().mean())
        result["rho"]["one_frac"] = float((rho >= 1.0).float().mean())
        result["beta"] = _quantiles(beta_values)
        result["beta_matched"] = float(beta * (1.0 + kappa * rho.mean()))
        for source_id in range(4):
            mask = source == source_id
            if bool(mask.any()):
                result["source"][str(source_id)]["rho_mean"] = float(rho[mask].mean())
                result["source"][str(source_id)]["beta_mean"] = float(beta_values[mask].mean())
    return result


def _grad_norm(grads: tuple[Tensor | None, ...]) -> float:
    squared = sum(
        float(grad.detach().float().pow(2).sum())
        for grad in grads
        if grad is not None
    )
    return math.sqrt(squared)


def gradient_audit(
    actor: ChunkActor,
    critic: TwinCritic,
    samples: list[dict[str, Tensor]],
    batch_size: int,
    num_batches: int,
    device: str,
    beta: float,
    kappa: float,
    tau_low: float,
    tau_high: float,
) -> dict[str, float]:
    params = tuple(parameter for parameter in actor.parameters() if parameter.requires_grad)
    q_norms: list[float] = []
    bc_raw_norms: list[float] = []
    bc_weighted_norms: list[float] = []
    for batch_index, batch in enumerate(_iter_batches(samples, batch_size, device)):
        if batch_index >= num_batches:
            break
        state = batch["state_vec"]
        ref = batch["ref_chunk_flat"]
        mu, _ = actor(state, ref, training=False)
        q1, q2 = critic(state, mu)
        q_loss = -torch.minimum(q1, q2).mean()
        disagreement = ((q1.detach() - q2.detach()).abs() * 0.5).reshape(-1)
        rho = ((disagreement - tau_low) / (tau_high - tau_low)).clamp(0.0, 1.0)
        beta_values = (beta * (1.0 + kappa * rho)).detach()
        bc_target = ref.clamp(-1.0, 1.0) if actor.action_residual else ref
        bc_per_sample = ((mu - bc_target) ** 2).sum(dim=-1)
        bc_raw = bc_per_sample.mean()
        bc_weighted = (beta_values * bc_per_sample).mean()

        q_norms.append(_grad_norm(torch.autograd.grad(q_loss, params, retain_graph=True, allow_unused=True)))
        bc_raw_norms.append(
            _grad_norm(torch.autograd.grad(bc_raw, params, retain_graph=True, allow_unused=True))
        )
        bc_weighted_norms.append(
            _grad_norm(torch.autograd.grad(bc_weighted, params, allow_unused=True))
        )

    if not q_norms:
        raise ValueError("No batches were available for gradient audit")
    return {
        "num_batches": len(q_norms),
        "q_grad_norm_mean": sum(q_norms) / len(q_norms),
        "bc_raw_grad_norm_mean": sum(bc_raw_norms) / len(bc_raw_norms),
        "bc_weighted_grad_norm_mean": sum(bc_weighted_norms) / len(bc_weighted_norms),
    }


def build_report(args: argparse.Namespace) -> dict:
    if args.beta < 0:
        raise ValueError(f"beta must be non-negative, got {args.beta}")
    if args.kappa < 0:
        raise ValueError(f"kappa must be non-negative, got {args.kappa}")
    actor, critic, config = load_heads(args.policy_path, args.device)
    calibration_samples = _load_samples(
        args.calibration_cache_dir,
        args.calibration_split,
        args.max_calibration_samples,
        args.seed,
    )
    training_samples = _load_samples(
        args.training_cache_dir,
        args.training_split,
        args.max_training_samples,
        args.seed + 1,
    )
    calibration = score_samples(
        actor,
        critic,
        calibration_samples,
        args.batch_size,
        args.device,
        args.beta,
        args.kappa,
    )
    tau_low = calibration["disagreement"]["p50"]
    tau_high = calibration["disagreement"]["p95"]
    training = score_samples(
        actor,
        critic,
        training_samples,
        args.batch_size,
        args.device,
        args.beta,
        args.kappa,
        tau_low=tau_low,
        tau_high=tau_high,
    )
    actor_residual_bound = (
        float(config["actor_delta_scale"])
        if bool(config["actor_action_residual"])
        else None
    )
    gradients = gradient_audit(
        actor,
        critic,
        training_samples,
        args.batch_size,
        args.gradient_batches,
        args.device,
        args.beta,
        args.kappa,
        tau_low,
        tau_high,
    )
    return {
        "policy_path": str(Path(args.policy_path).resolve()),
        "calibration_cache": str(Path(args.calibration_cache_dir).resolve()),
        "training_cache": str(Path(args.training_cache_dir).resolve()),
        "ac_semantics_version": config.get("ac_semantics_version"),
        "base_beta": args.beta,
        "kappa": args.kappa,
        "tau_low": tau_low,
        "tau_high": tau_high,
        "calibration": calibration,
        "training": training,
        "gradients": gradients,
        "checks": {
            "all_finite": all(
                math.isfinite(value)
                for value in (
                    tau_low,
                    tau_high,
                    training["beta"]["min"],
                    training["beta"]["max"],
                    gradients["q_grad_norm_mean"],
                    gradients["bc_weighted_grad_norm_mean"],
                )
            ),
            "beta_within_expected_range": (
                training["beta"]["min"] >= args.beta - 1e-7
                and training["beta"]["max"] <= args.beta * (1.0 + args.kappa) + 1e-7
            ),
            "actor_ref_delta_within_bound": (
                actor_residual_bound is None
                or training["actor_ref_delta_abs"]["max"] <= actor_residual_bound + 1e-6
            ),
        },
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    report = build_report(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved audit report to {output_path}")
    failed_checks = [name for name, passed in report["checks"].items() if not passed]
    if failed_checks:
        raise SystemExit(f"Dynamic-beta audit failed checks: {', '.join(failed_checks)}")


if __name__ == "__main__":
    main()
