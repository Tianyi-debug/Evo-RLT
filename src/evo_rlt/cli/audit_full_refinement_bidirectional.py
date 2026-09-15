"""Evaluate realized BC+Q actors against one matched multi-step BC-only actor.

Only fixed forward passes are permitted here: no optimizer, backward pass,
virtual actor update, checkpoint selection, or real-robot success import.
Run --preflight-only with the existing direction before reverse actors exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from evo_rlt.cli.audit_actor_q_mechanism import _construct_heads
from evo_rlt.cli.full_refinement_provenance import (
    Candidate, Evidence, ProvenanceError, require, sha256_file, tensor_digest,
    validate_full_refinement,
)


@torch.inference_mode()
def evaluate_fixed_actors(
    *, control: torch.nn.Module, candidates: dict[float, dict[int, torch.nn.Module]],
    critics: dict[int, torch.nn.Module], states: torch.Tensor, proposals: torch.Tensor,
    batch_size: int = 256, device: str = "cpu",
) -> tuple[dict[float, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Compute both metrics from the exact same cached action predictions."""
    require(batch_size > 0 and len(states) > 0, "Positive batch size and nonempty states required")
    require(states.ndim == proposals.ndim == 2 and len(states) == len(proposals), "Input shape mismatch")
    require(bool(torch.isfinite(states).all() and torch.isfinite(proposals).all()), "Nonfinite audit input")
    require(set(critics) == {1, 2}, "Two independent critic modules required")
    require(bool(candidates) and all(set(pair) == {1, 2} for pair in candidates.values()),
            "Both fixed candidates are required for each Q coefficient")
    models = {"q0": control, "critic1": critics[1], "critic2": critics[2]}
    for q, pair in candidates.items():
        models.update({f"q{q:g}_c{k}": actor for k, actor in pair.items()})
    before = {key: tensor_digest(model.state_dict()) for key, model in models.items()}
    for model in models.values():
        model.to(device).eval().requires_grad_(False)

    chunks: dict[float, dict[str, list[torch.Tensor]]] = {q: {} for q in candidates}
    actions: dict[str, list[torch.Tensor]] = {k: [] for k in models if not k.startswith("critic")}

    def predict(actor, x, p):
        result, _ = actor(x, p, training=False)
        require(result.shape == p.shape and bool(torch.isfinite(result).all()), "Invalid actor output")
        return result.clamp(-1.0, 1.0)

    def score(critic, x, a):
        first, second = critic(x, a)
        result = torch.minimum(first, second).reshape(-1)
        require(len(result) == len(x) and bool(torch.isfinite(result).all()), "Invalid critic output")
        return result

    for start in range(0, len(states), batch_size):
        x = states[start:start + batch_size].to(device)
        p = proposals[start:start + batch_size].to(device)
        a0 = predict(control, x, p)
        actions["q0"].append(a0.cpu())
        bound = control.residual_delta_bound(a0).expand_as(a0)
        require(bool(torch.isfinite(bound).all() and (bound > 0).all()), "Residual bounds must be finite/positive")
        v0 = {k: score(critic, x, a0) for k, critic in critics.items()}
        for q, pair in candidates.items():
            values = {"q0_value_c1": v0[1], "q0_value_c2": v0[2]}
            for k, actor in pair.items():
                a = predict(actor, x, p)
                require(torch.equal(actor.residual_delta_bound(a).expand_as(a), bound), "Candidate residual bounds differ")
                actions[f"q{q:g}_c{k}"].append(a.cpu())
                delta = a - a0
                values[f"normalized_shift_percent_{k}"] = 100 * (delta / bound).square().mean(-1).sqrt()
                values[f"whole_chunk_l2_{k}"] = delta.norm(dim=-1)
                values[f"self_gain_{k}"] = score(critics[k], x, a) - v0[k]
                other = 3 - k
                values[f"cross_gain_{k}_to_{other}"] = score(critics[other], x, a) - v0[other]
                values[f"rir_{k}_to_{other}"] = (values[f"cross_gain_{k}_to_{other}"] > 0).double()
            # Average scalar per-direction RMS, never average action vectors first.
            values["normalized_shift_percent"] = 0.5 * (
                values["normalized_shift_percent_1"] + values["normalized_shift_percent_2"])
            values["rir"] = 0.5 * (values["rir_1_to_2"] + values["rir_2_to_1"])
            values["mean_cross_gain"] = 0.5 * (values["cross_gain_1_to_2"] + values["cross_gain_2_to_1"])
            values["self_minus_cross_gap"] = 0.5 * (values["self_gain_1"] + values["self_gain_2"]) - values["mean_cross_gain"]
            for key, value in values.items():
                require(bool(torch.isfinite(value).all()), f"Nonfinite metric: {key}")
                chunks[q].setdefault(key, []).append(value.cpu())
    after = {key: tensor_digest(model.state_dict()) for key, model in models.items()}
    require(before == after, "Model tensors changed during forward-only evaluation")
    return (
        {q: {key: torch.cat(value) for key, value in values.items()} for q, values in chunks.items()},
        {key: torch.cat(value).contiguous() for key, value in actions.items()},
    )


def summarize_metrics(
    values: dict[str, torch.Tensor], episode_uids: list[str], *, seed: int, replicates: int,
) -> dict[str, Any]:
    """Joint episode bootstrap; keep directions/states paired within each draw."""
    require(replicates >= 0, "Bootstrap replicates cannot be negative")
    keys = sorted(values)
    array = np.column_stack([values[k].double().numpy() for k in keys])
    require(len(array) == len(episode_uids) and len(array) > 0, "Episode/metric length mismatch")
    require(bool(np.isfinite(array).all()), "Nonfinite summary values")
    episodes = sorted(set(episode_uids))
    counts = np.array([episode_uids.count(uid) for uid in episodes], dtype=np.int64)
    sums = np.stack([array[np.array(episode_uids) == uid].sum(0) for uid in episodes])
    estimates = []
    if len(episodes) >= 2:
        rng = random.Random(seed)
        for _ in range(replicates):
            indices = [rng.randrange(len(episodes)) for _ in episodes]
            estimates.append(sums[indices].sum(0) / counts[indices].sum())
    boot = np.array(estimates)
    result = {"states": len(array), "episodes": len(episodes), "metrics": {}}
    for index, key in enumerate(keys):
        result["metrics"][key] = {
            "mean": float(array[:, index].mean()),
            "episode_bootstrap": {
                "replicates": len(estimates),
                "median": float(np.median(boot[:, index])) if len(estimates) else None,
                "ci95": np.quantile(boot[:, index], [0.025, 0.975]).tolist() if len(estimates) else None,
            },
        }
    for direction in ("1_to_2", "2_to_1"):
        gain = values[f"cross_gain_{direction}"]
        result[f"near_zero_fraction_{direction}"] = float((gain.abs() <= 1e-8).double().mean())
    return result


def _load_models(match: dict, evidence: Evidence):
    def load(path):
        root = Path(path)
        cfg = evidence.json(root / "config.json")
        actor, critic = _construct_heads(cfg)
        state = evidence.state(root / "model.safetensors")
        for prefix, module in (("actor.", actor), ("critic.", critic)):
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        return actor, critic

    control, _ = load(match["runs"]["q0"]["checkpoint"])
    critics = {int(k): load(fit["checkpoint"])[1] for k, fit in match["critics"].items()}
    candidates: dict[float, dict] = {}
    for key, run in match["runs"].items():
        if key != "q0":
            candidates.setdefault(run["q_weight"], {})[run["inducing_critic"]] = load(run["checkpoint"])[0]
    return control, candidates, critics


def _audit_inputs(path: Path, config: dict):
    # Exactly one explicitly supplied audit file; no directory concatenation.
    rows = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(rows, list) and bool(rows), "Audit cache must be a nonempty list of transitions")
    selected, episodes = [], []
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and all(k in row for k in ("actor_q_mask", "episode_id", "state_vec", "proposal_chunk")),
                f"Missing audit schema at row {index}")
        mask = torch.as_tensor(row["actor_q_mask"])
        require(mask.numel() == 1 and mask.item() in (0, 1), f"Invalid actor_q_mask at row {index}")
        episode = torch.as_tensor(row["episode_id"])
        require(episode.numel() == 1 and math_is_nonnegative_integer(episode.item()), f"Invalid episode_id at row {index}")
        if mask.item() == 1:
            selected.append(index)
            episodes.append(f"audit:{int(episode.item())}")
    require(bool(selected), "Audit cache contains no Q-valid states")
    states = torch.stack([torch.as_tensor(rows[i]["state_vec"], dtype=torch.float32) for i in selected])
    proposals = torch.stack([torch.as_tensor(rows[i]["proposal_chunk"], dtype=torch.float32) for i in selected])
    require(states.shape == (len(selected), config["rl_token_dim"] + config["proprio_dim"]), "Audit state shape differs from checkpoint")
    require(proposals.shape == (len(selected), config["chunk_length"], config["action_dim"]), "Audit proposal shape differs from checkpoint")
    return states, proposals.flatten(1), selected, episodes, len(rows)


def math_is_nonnegative_integer(value) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value) and value >= 0 and int(value) == value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args: argparse.Namespace) -> dict:
    candidates = [Candidate(float(q), int(k), Path(path)) for q, k, path in args.candidate]
    output = args.output_dir.expanduser().resolve()
    require(not output.exists(), f"Refusing to overwrite an existing output directory: {output}")
    match, evidence = validate_full_refinement(
        q0=args.q0_checkpoint, critics={1: args.critic_1_checkpoint, 2: args.critic_2_checkpoint},
        candidates=candidates, audit_cache=args.audit_cache, expected_updates=args.expected_updates,
        require_bidirectional=not args.preflight_only,
    )
    sources = [Path(__file__), Path(__file__).with_name("full_refinement_provenance.py"),
               Path(__file__).with_name("audit_actor_q_mechanism.py"),
               Path(__file__).resolve().parents[1] / "core" / "actor.py",
               Path(__file__).resolve().parents[1] / "core" / "critic.py",
               Path(__file__).resolve().parents[1] / "core" / "utils.py"]
    report = {
        "schema_version": 1, "mode": "preflight_only" if args.preflight_only else "fixed_full_refinement_bidirectional",
        "status": "PREFLIGHT_ONLY" if args.preflight_only else "EVALUATED",
        "match_validation": match, "results": {},
        "protocol": {
            "actor_optimizer_steps_in_this_audit": 0, "audit_cache_is_evaluation_only": True,
            "critic_score": "min of both heads of the complete critic",
            "positive_support": "strict cross_gain > 0; near-zero fraction reported separately at abs(gain)<=1e-8",
            "shift": "100 * mean_states(RMS_action_dimensions((candidate-Q0)/residual_bound)); then mean_directions",
            "rir": "mean_directions(mean_states(cross_gain > 0))",
            "bootstrap": "joint episode resampling with transition-weighted means; no refitting",
            "robot_success": "not computed/imported; deployed C1 actor is distinct from a two-direction aggregate",
            "gradient_diagnostics": "not computed/imported; no gradients on audit states",
        },
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                    "argv": sys.argv, "torch_num_threads": torch.get_num_threads(),
                    "device": args.device, "batch_size": args.batch_size,
                    "bootstrap_seed": args.bootstrap_seed, "bootstrap_reps": args.bootstrap_reps,
                    "source_sha256": {str(p): evidence.files[str(evidence.file(p))] for p in sources}},
    }
    all_records, saved_actions = [], None
    if not args.preflight_only:
        control, actors, critics = _load_models(match, evidence)
        states, proposals, indices, episodes, total = _audit_inputs(
            Path(match["audit_cache"]), match["effective_config"]["policy"])
        values, saved_actions = evaluate_fixed_actors(
            control=control, candidates=actors, critics=critics, states=states, proposals=proposals,
            batch_size=args.batch_size, device=args.device)
        report["audit_rows"] = {"total": total, "selected": len(indices), "excluded_by_q_mask": total - len(indices),
                                "selected_index_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest()}
        for q, metrics in values.items():
            summary = summarize_metrics(metrics, episodes, seed=args.bootstrap_seed, replicates=args.bootstrap_reps)
            summary["update_objects"] = {key: match["runs"][key] for key in ("q0", f"q{q:g}_c1", f"q{q:g}_c2")}
            report["results"][f"q{q:g}"] = summary
            for offset, index in enumerate(indices):
                all_records.append({"q_weight": q, "cache_index": index, "episode_uid": episodes[offset],
                                    **{key: float(value[offset]) for key, value in metrics.items()}})
        saved_actions["audit_cache_indices"] = torch.tensor(indices, dtype=torch.int64)
        saved_actions["residual_bounds"] = control.residual_delta_bound(proposals[:1]).cpu().contiguous()
    evidence.verify_unchanged()
    report["input_files_sha256"] = evidence.files
    report["input_files_unchanged"] = True
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "match_report.json", match)
    if saved_actions is not None:
        save_file(saved_actions, str(output / "fixed_actor_actions.safetensors"))
        with (output / "per_state_values.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
            writer.writeheader()
            writer.writerows(all_records)
        report["raw_exports_sha256"] = {name: sha256_file(output / name) for name in (
            "fixed_actor_actions.safetensors", "per_state_values.csv")}
    _write_json(output / "full_refinement_report.json", report)
    print(json.dumps({"status": report["status"], "bidirectional_complete": match["bidirectional_complete"],
                      "missing_candidates": match["missing_candidates"], "output_dir": str(output),
                      "results": {k: {m: v["metrics"][m]["mean"] for m in ("rir", "normalized_shift_percent")}
                                  for k, v in report["results"].items()}}, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q0-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-1-checkpoint", type=Path, required=True)
    parser.add_argument("--critic-2-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", nargs=3, action="append", required=True, metavar=("Q_WEIGHT", "CRITIC_ID", "CHECKPOINT"))
    parser.add_argument("--audit-cache", type=Path, required=True, help="Exact evaluation-only .pt file; never a directory")
    parser.add_argument("--expected-updates", type=int, default=897)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-seed", type=int, default=1000)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--preflight-only", action="store_true", help="Check saved evidence only; allow missing reverse candidates; emit no metrics")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory (existing paths are refused)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.expected_updates <= 0 or args.bootstrap_reps < 0:
        parser.error("batch-size/expected-updates must be positive; bootstrap-reps must be nonnegative")
    try:
        run(args)
    except (ProvenanceError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"Audit refused: {error}\n")


if __name__ == "__main__":
    main()
