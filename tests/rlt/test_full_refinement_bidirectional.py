"""Synthetic fixtures only: no actor optimizer steps or real experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from evo_rlt.cli.audit_actor_q_mechanism import _construct_heads
from evo_rlt.cli.audit_first_order_gradient_ratio import (
    compute_gradient_ratio,
    run as run_gradient_diagnostic,
)
from evo_rlt.cli.audit_full_refinement_bidirectional import evaluate_fixed_actors, run, summarize_metrics
from evo_rlt.cli.full_refinement_provenance import (
    Candidate, ProvenanceError, tensor_digest, validate_full_refinement,
)
from evo_rlt.core.interfaces import (
    SPARSE_TERMINAL_SUCCESS_REWARD_SEMANTICS,
    TRANSITION_CACHE_SEMANTICS_VERSION,
)


class FixedActor(nn.Module):
    def __init__(self, action, bound=None):
        super().__init__()
        self.action = nn.Parameter(torch.tensor(action, dtype=torch.float32))
        self.register_buffer(
            "bound",
            torch.tensor([1.0] * len(action) if bound is None else bound, dtype=torch.float32),
        )

    def forward(self, states, proposal, training=False):
        assert not training and not torch.is_grad_enabled()
        mu = self.action.expand(len(states), -1)
        return mu, torch.zeros_like(mu)

    def residual_delta_bound(self, like):
        return self.bound.to(like.device).reshape(1, -1)


class LinearCritic(nn.Module):
    def __init__(self, sign):
        super().__init__()
        self.register_buffer("sign", torch.tensor(float(sign)))

    def forward(self, states, actions):
        assert not torch.is_grad_enabled()
        value = self.sign * actions.sum(-1, keepdim=True)
        return value, value + 0.1  # Require min-head scoring, not head mean.


class GradientActor(nn.Module):
    action_residual = True

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        nn.init.constant_(self.linear.weight, 0.1)

    def forward(self, states, proposal, training=True):
        delta = 0.2 * torch.tanh(self.linear(states))
        return proposal + delta, delta


class GradientCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_scale = nn.Parameter(torch.tensor([0.8, -0.3]))

    def forward(self, states, actions):
        q1 = (actions * self.action_scale).sum(-1, keepdim=True)
        return q1, q1 + 0.1


def forbid_updates(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An audit must not create an optimizer or call backward")
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)


def test_exact_same_actions_control_and_rms_before_direction_average(monkeypatch):
    forbid_updates(monkeypatch)
    control = FixedActor([0.1, 0.1])
    actors = {5.0: {1: FixedActor([0.7, 0.1]), 2: FixedActor([0.1, 0.9])}}
    critics = {1: LinearCritic(1), 2: LinearCritic(2)}
    originals = [control, *actors[5.0].values(), *critics.values()]
    hashes = [tensor_digest(m.state_dict()) for m in originals]
    values, actions = evaluate_fixed_actors(
        control=control, candidates=actors, critics=critics,
        states=torch.zeros(3, 4), proposals=torch.zeros(3, 2), batch_size=2)
    metrics = values[5.0]
    assert metrics["rir"].tolist() == [1, 1, 1]
    assert metrics["normalized_shift_percent"].mean().item() == pytest.approx(100 * 0.7 / np.sqrt(2))
    assert metrics["q0_value_c2"].mean().item() == pytest.approx(0.4)
    assert metrics["cross_gain_1_to_2"].mean().item() == pytest.approx(1.2)
    for k in (1, 2):
        recomputed = 100 * (actions[f"q5_c{k}"] - actions["q0"]).square().mean(-1).sqrt()
        torch.testing.assert_close(metrics[f"normalized_shift_percent_{k}"], recomputed)
    assert [tensor_digest(m.state_dict()) for m in originals] == hashes
    assert all(p.grad is None for m in originals for p in m.parameters())


def test_opposite_critics_are_cross_evaluated_not_self_evaluated():
    values, _ = evaluate_fixed_actors(
        control=FixedActor([0, 0]), candidates={25.0: {1: FixedActor([0.2, 0]), 2: FixedActor([-0.2, 0])}},
        critics={1: LinearCritic(1), 2: LinearCritic(-1)}, states=torch.zeros(2, 4), proposals=torch.zeros(2, 2))
    assert values[25.0]["rir"].tolist() == [0, 0]
    assert bool((values[25.0]["self_gain_1"] > 0).all())
    assert bool((values[25.0]["self_gain_2"] > 0).all())


def test_zero_gain_is_not_positive_support():
    values, _ = evaluate_fixed_actors(
        control=FixedActor([0.1, 0.2]), candidates={5.0: {1: FixedActor([0.1, 0.2]), 2: FixedActor([0.1, 0.2])}},
        critics={1: LinearCritic(1), 2: LinearCritic(2)}, states=torch.zeros(2, 4), proposals=torch.zeros(2, 2))
    assert not values[5.0]["rir"].any()
    assert not values[5.0]["normalized_shift_percent"].any()


def test_normalized_shift_uses_elementwise_bounds_then_state_and_direction_means():
    values, _ = evaluate_fixed_actors(
        control=FixedActor([0, 0], bound=[0.5, 2.0]),
        candidates={5.0: {
            1: FixedActor([0.5, 1.0], bound=[0.5, 2.0]),
            2: FixedActor([1.0, 0.0], bound=[0.5, 2.0]),
        }},
        critics={1: LinearCritic(1), 2: LinearCritic(2)},
        states=torch.zeros(3, 4),
        proposals=torch.zeros(3, 2),
    )
    direction_1 = 100 * np.sqrt((1.0**2 + 0.5**2) / 2)
    direction_2 = 100 * np.sqrt((2.0**2 + 0.0**2) / 2)
    assert values[5.0]["normalized_shift_percent"].mean().item() == pytest.approx(
        0.5 * (direction_1 + direction_2)
    )


def test_bootstrap_keeps_directions_paired_and_weights_transitions():
    values = {
        "rir_1_to_2": torch.tensor([1.0, 1, 0]), "rir_2_to_1": torch.tensor([0.0, 0, 1]),
        "rir": torch.tensor([0.5, 0.5, 0.5]),
        "cross_gain_1_to_2": torch.tensor([1.0, 1, -1]), "cross_gain_2_to_1": torch.tensor([-1.0, -1, 1]),
    }
    result = summarize_metrics(values, ["a", "a", "b"], seed=4, replicates=100)
    assert result == summarize_metrics(values, ["a", "a", "b"], seed=4, replicates=100)
    assert result["metrics"]["rir_1_to_2"]["mean"] == pytest.approx(2 / 3)
    assert result["metrics"]["rir"]["episode_bootstrap"]["ci95"] == [0.5, 0.5]
    single = summarize_metrics(values, ["a"] * 3, seed=4, replicates=100)
    assert single["metrics"]["rir"]["episode_bootstrap"]["ci95"] is None


def test_gradient_ratio_is_raw_masked_and_does_not_mutate_models(monkeypatch):
    actor, critic = GradientActor(), GradientCritic()
    actor_before = {key: value.clone() for key, value in actor.state_dict().items()}
    critic_before = {key: value.clone() for key, value in critic.state_dict().items()}

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("Gradient diagnostic must not construct an optimizer")

    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden_optimizer)
    batch = {
        "state_vec": torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]),
        "proposal_chunk_flat": torch.zeros(2, 2),
        "bc_target_chunk_flat": torch.tensor([[0.2, -0.1], [0.1, 0.3]]),
        "actor_q_mask": torch.tensor([1.0, 0.0]),
        "actor_bc_mask": torch.tensor([1.0, 1.0]),
    }
    result = compute_gradient_ratio(
        actor=actor,
        critic=critic,
        batch=batch,
        lambda_q=5.0,
        beta=1.0,
        epsilon=1e-12,
    )
    assert result["r_grad"] == pytest.approx(
        5.0 * result["q_gradient_norm"]
        / (result["bc_gradient_norm"] + 1e-12)
    )
    assert result["actor_q_valid_rows"] == 1
    assert result["actor_bc_valid_rows"] == 2
    assert result["pre_gradient_clipping"] is True
    assert result["pre_optimizer_preconditioning"] is True
    assert result["optimizer_steps"] == 0
    assert all(torch.equal(actor.state_dict()[key], value) for key, value in actor_before.items())
    assert all(torch.equal(critic.state_dict()[key], value) for key, value in critic_before.items())
    assert all(parameter.grad is None for parameter in (*actor.parameters(), *critic.parameters()))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def transition_row(dataset_uid: str, episode: int, anchor: int, *, source: int) -> dict:
    episode_uid = f"{dataset_uid}:episode:{episode}"
    return {
        "state_vec": torch.tensor([episode, anchor, source, 1.0], dtype=torch.float32),
        "exec_chunk": torch.zeros(1, 2),
        "ref_chunk": torch.zeros(1, 2),
        "reward_seq": torch.zeros(1),
        "next_state_vec": torch.tensor([episode, anchor + 1, source, 1.0], dtype=torch.float32),
        "next_ref_chunk": torch.zeros(1, 2),
        "proposal_chunk": torch.zeros(1, 2),
        "bc_target_chunk": torch.full((1, 2), 0.1),
        "next_proposal_chunk": torch.zeros(1, 2),
        "done": torch.tensor(0.0),
        "intervention": torch.tensor(0.0),
        "bootstrap_mask": torch.tensor(1.0),
        "actual_steps": torch.tensor(1),
        "source": torch.tensor(source),
        "episode_id": torch.tensor(episode),
        "episode_uid": episode_uid,
        "transition_uid": f"{episode_uid}:anchor:{anchor}:chunk:1:stride:1",
        "is_critical": torch.tensor(0.0),
        "critic_mask": torch.tensor(1.0),
        "actor_q_mask": torch.tensor(1.0),
        "actor_bc_mask": torch.tensor(1.0),
        "intervention_reason": torch.tensor(0),
        "cache_semantics_version": torch.tensor(TRANSITION_CACHE_SEMANTICS_VERSION),
        "reward_semantics_version": SPARSE_TERMINAL_SUCCESS_REWARD_SEMANTICS,
        "exec_action_is_actual_sent": torch.tensor(1.0),
        "anchor_start_frame": torch.tensor(anchor),
        "frame_stride": torch.tensor(1),
        "fps": torch.tensor(30.0),
    }


def write_optimizer_state(root: Path, module: nn.Module, *, steps: int, lr: float) -> None:
    state_root = root.parent / "training_state"
    write_json(state_root / "training_step.json", {"step": steps})
    parameters = list(module.parameters())
    write_json(
        state_root / "optimizer_param_groups.json",
        [{
            "params": list(range(len(parameters))),
            "lr": lr,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
        }],
    )
    optimizer_state = {}
    for index, parameter in enumerate(parameters):
        optimizer_state[f"state/{index}/step"] = torch.tensor(float(steps))
        optimizer_state[f"state/{index}/exp_avg"] = torch.zeros_like(parameter)
        optimizer_state[f"state/{index}/exp_avg_sq"] = torch.zeros_like(parameter)
    save_file(optimizer_state, str(state_root / "optimizer_state.safetensors"))


@pytest.fixture
def artifacts(tmp_path):
    cfg = {
        "rl_token_dim": 4, "proprio_dim": 0, "chunk_length": 1, "action_dim": 2,
        "actor_hidden_dim": 4, "actor_num_layers": 1, "critic_hidden_dim": 4, "critic_num_layers": 1,
        "state_normalization": "none", "actor_action_residual": True, "actor_delta_scale": 0.1,
        "actor_delta_scale_per_action_dim": [0.2, 0.3], "actor_ref_dropout_p": 0.0,
        "ac_semantics_version": 2, "training_stage": "actor_refine", "actor_refine_objective": "td3bc",
        "actor_bc_weight_mode": "fixed", "actor_q_trust_mode": "fixed", "beta": 1.0,
        "actor_behavior_preservation_weight": 0.0, "actor_human_weight": 0.0, "actor_teacher_weight": 0.0,
        "actor_teacher_pretrained_path": "", "actor_q_weight_max": 0.0,
        "actor_lr": 5e-6, "training_lr": 5e-6, "critic_lr": 1e-4,
        "gamma": 0.99, "tau": 0.005, "target_q_clip": 100.0,
        "critic_bootstrap_mode": "none", "critic_bootstrap_keep_prob": 0.8,
        "critic_bootstrap_seed": 1000,
        "utd_ratio": 1, "actor_update_interval": 1, "use_amp": False, "use_peft": False,
        "source_sampling_seed": 1000, "source_sampling_weights": [0.5, 0, 0.5, 0],
        "pretrained_path": "",
    }
    train_cache = tmp_path / "train" / "chunk_transitions_train.pt"
    train_cache.parent.mkdir()
    torch.save([
        transition_row("train-set", 0, 0, source=0),
        transition_row("train-set", 0, 1, source=0),
        transition_row("train-set", 1, 0, source=2),
        transition_row("train-set", 1, 1, source=2),
    ], train_cache)
    audit = tmp_path / "audit" / "chunk_transitions_val.pt"
    audit.parent.mkdir()
    audit_rows = [transition_row("audit-set", i, 0, source=2) for i in range(3)]
    audit_rows[2]["actor_q_mask"] = torch.tensor(0.0)
    torch.save(audit_rows, audit)
    train_base = {
        "dataset": {"repo_id": str(train_cache.parent)}, "policy": cfg,
        "optimizer": {"type": "adamw", "lr": 5e-6, "weight_decay": 0, "betas": [0.9, 0.999],
                      "eps": 1e-8, "grad_clip_norm": 1},
        "scheduler": None, "resume": False, "steps": 6, "seed": 1000,
        "batch_size": 2, "num_workers": 0, "save_freq": 3, "eval_freq": 0, "log_freq": 1,
        "env": None, "use_policy_training_preset": True, "wandb": {"enable": False},
    }
    torch.manual_seed(17)
    initial_actor, initial_critic = _construct_heads(cfg)
    warmup = tmp_path / "warmup" / "checkpoints" / "000000" / "pretrained_model"
    warmup_config = {**cfg, "training_stage": "mixed_ac"}
    write_json(warmup / "config.json", warmup_config)
    write_json(warmup / "train_config.json", {**train_base, "policy": warmup_config})
    warmup_state = {
        **{f"actor.{name}": value.clone() for name, value in initial_actor.state_dict().items()},
        **{f"critic.{name}": value.clone() for name, value in initial_critic.state_dict().items()},
        **{f"target_critic.{name}": value.clone() for name, value in initial_critic.state_dict().items()},
        "_actor_refine_step": torch.tensor(0),
        "_actor_refine_batch_fingerprint": torch.tensor(0),
        "_critic_step": torch.tensor(0),
    }
    save_file(warmup_state, str(warmup / "model.safetensors"))
    critics, initial_states = {}, {}
    for k in (1, 2):
        torch.manual_seed(10 * k)
        _, critic = _construct_heads(cfg)
        state = {f"actor.{name}": value.clone() for name, value in initial_actor.state_dict().items()}
        for prefix in ("critic.", "target_critic."):
            state.update({prefix + name: value.clone() for name, value in critic.state_dict().items()})
        state.update({"_actor_refine_step": torch.tensor(0),
                      "_actor_refine_batch_fingerprint": torch.tensor(0),
                      "_critic_step": torch.tensor(6)})
        root = tmp_path / f"critic{k}" / "checkpoints" / "000006" / "pretrained_model"
        fit_config = {**cfg, "training_stage": "critic_only", "pretrained_path": str(warmup)}
        write_json(root / "config.json", fit_config)
        write_json(root / "train_config.json", {**train_base, "policy": fit_config, "seed": k * 1000})
        save_file(state, str(root / "model.safetensors"))
        write_optimizer_state(root, critic, steps=6, lr=1e-4)
        critics[k], initial_states[k] = root, state
    paths = {}
    for label, q, direction in (("q0", 0, 1), ("q5_c1", 5, 1), ("q5_c2", 5, 2)):
        root = tmp_path / label / "checkpoints" / "000003" / "pretrained_model"
        policy = {**cfg, "actor_q_weight_max": q, "pretrained_path": str(critics[direction]),
                  "diagnostics_jsonl_path": str(tmp_path / label / "diagnostics.jsonl")}
        write_json(root / "config.json", policy)
        write_json(root / "train_config.json", {**train_base, "policy": policy, "steps": 3 if direction == 2 else 6})
        state = {name: value.clone() for name, value in initial_states[direction].items()}
        # Fixed synthetic checkpoints, not results of any training invocation.
        actor_key = next(name for name in state if name.startswith("actor.") and name.endswith("bias"))
        state[actor_key].add_(0.001 * (1 + q + direction))
        state["_actor_refine_step"] = torch.tensor(3)
        state["_actor_refine_batch_fingerprint"] = torch.tensor(333)
        save_file(state, str(root / "model.safetensors"))
        diagnostics = [{"actor_refine_step": step, "actor_refine_batch_fingerprint": fp,
                        "actor_refine_stage": True, "actor_refine_td3bc": True,
                        "actor_update": True, "actor_q_weight": q} for step, fp in enumerate([111, 222, 333], 1)]
        Path(policy["diagnostics_jsonl_path"]).write_text("\n".join(json.dumps(row) for row in diagnostics))
        write_optimizer_state(root, initial_actor, steps=3, lr=5e-6)
        paths[label] = root
    return {"q0": paths["q0"], "critics": critics, "audit_cache": audit, "expected_updates": 3,
            "candidates": [Candidate(5, k, paths[f"q5_c{k}"]) for k in (1, 2)]}


def patch_policy(path, **changes):
    cfg = json.loads((path / "config.json").read_text())
    cfg.update(changes)
    train = json.loads((path / "train_config.json").read_text())
    train["policy"] = cfg
    write_json(path / "config.json", cfg)
    write_json(path / "train_config.json", train)


def training_cache_for_checkpoint(path: Path) -> Path:
    train = json.loads((path / "train_config.json").read_text())
    return Path(train["dataset"]["repo_id"]) / "chunk_transitions_train.pt"


def test_matched_effective_operator_and_critic_lr_exception(artifacts):
    patch_policy(artifacts["candidates"][1].checkpoint, critic_lr=None)
    match, evidence = validate_full_refinement(**artifacts)
    assert match["status"] == "MATCHED_EFFECTIVE_OPERATOR"
    assert match["bidirectional_complete"]
    assert match["runs"]["q5_c2"]["raw_policy_differences_from_q0"]["critic_lr"] == [1e-4, None]
    assert match["runs"]["q5_c2"]["planned_steps"] == 3
    assert match["runs"]["q0"]["planned_steps"] == 6
    assert match["critics"]["1"]["training_cache"]["sha256"] == match["training_cache"]["sha256"]
    assert match["episode_disjointness"]["overlapping_episodes"] == 0
    assert match["multi_step_update_object"]["T"] == 3
    assert match["training_cache"]["critic_valid_actual_sent_fraction"] == 1.0
    evidence.verify_unchanged()


def test_gradient_diagnostic_cli_path_uses_matched_q0_and_dtrain(artifacts, tmp_path):
    report = run_gradient_diagnostic(argparse.Namespace(
        candidate_checkpoint=artifacts["candidates"][0].checkpoint,
        q0_checkpoint=artifacts["q0"],
        batch_size=2,
        minibatch_seed=123,
        epsilon=1e-12,
        device="cpu",
        output=tmp_path / "gradient_ratio.json",
    ))
    assert report["separate_from_RIR"] is True
    assert report["matched_q0_checkpoint"] == str(artifacts["q0"])
    assert report["D_train"]["path"].endswith("chunk_transitions_train.pt")
    assert report["diagnostic"]["optimizer_steps"] == 0


def test_rejects_critic_training_cache_hash_difference(artifacts, tmp_path):
    source = training_cache_for_checkpoint(artifacts["critics"][1])
    rows = torch.load(source, map_location="cpu", weights_only=False)
    rows[0]["state_vec"] = rows[0]["state_vec"] + 0.25
    alternate = tmp_path / "alternate-train" / "chunk_transitions_train.pt"
    alternate.parent.mkdir()
    torch.save(rows, alternate)
    critic_2_train_path = artifacts["critics"][2] / "train_config.json"
    critic_2_train = json.loads(critic_2_train_path.read_text())
    critic_2_train["dataset"]["repo_id"] = str(alternate.parent)
    write_json(critic_2_train_path, critic_2_train)
    with pytest.raises(ProvenanceError, match="training-cache hashes differ"):
        validate_full_refinement(**artifacts)


def test_rejects_critic_objective_or_config_difference_beyond_seed(artifacts):
    patch_policy(artifacts["critics"][2], gamma=0.95)
    with pytest.raises(ProvenanceError, match="differ beyond seed/output-only"):
        validate_full_refinement(**artifacts)


def test_rejects_train_audit_episode_overlap(artifacts):
    rows = torch.load(artifacts["audit_cache"], map_location="cpu", weights_only=False)
    rows[0]["episode_uid"] = "train-set:episode:0"
    rows[0]["transition_uid"] = "train-set:episode:0:audit-anchor:0"
    torch.save(rows, artifacts["audit_cache"])
    with pytest.raises(ProvenanceError, match="share episodes"):
        validate_full_refinement(**artifacts)


def test_rejects_critic_valid_row_without_actual_sent_action(artifacts):
    train_cache = training_cache_for_checkpoint(artifacts["critics"][1])
    rows = torch.load(train_cache, map_location="cpu", weights_only=False)
    rows[0]["exec_action_is_actual_sent"] = torch.tensor(0.0)
    torch.save(rows, train_cache)
    with pytest.raises(ProvenanceError, match="not actual-sent"):
        validate_full_refinement(**artifacts)


def test_rejects_actor_refinement_step_count_difference(artifacts):
    checkpoint = artifacts["candidates"][1].checkpoint / "model.safetensors"
    state = load_file(str(checkpoint))
    state["_actor_refine_step"] = torch.tensor(2)
    save_file(state, str(checkpoint))
    with pytest.raises(ProvenanceError, match="Checkpoint update count differs"):
        validate_full_refinement(**artifacts)


@pytest.mark.parametrize("field,value", [("beta", 2.0), ("actor_teacher_weight", 1.0),
                                        ("source_sampling_seed", 2000), ("actor_ref_dropout_p", 0.1)])
def test_rejects_operator_differences(artifacts, field, value):
    patch_policy(artifacts["candidates"][1].checkpoint, **{field: value})
    with pytest.raises(ProvenanceError):
        validate_full_refinement(**artifacts)


def test_rejects_interior_fingerprint_difference_even_if_final_matches(artifacts):
    root = artifacts["candidates"][1].checkpoint
    log = root.parents[2] / "diagnostics.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows[1]["actor_refine_batch_fingerprint"] += 1
    log.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ProvenanceError, match="prefix mismatch"):
        validate_full_refinement(**artifacts)


def test_rejects_float_fingerprints(artifacts):
    log = artifacts["candidates"][1].checkpoint.parents[2] / "diagnostics.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    rows[0]["actor_refine_batch_fingerprint"] = 111.0
    log.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ProvenanceError, match="float-rounded"):
        validate_full_refinement(**artifacts)


def test_rejects_wrong_inducing_critic(artifacts):
    patch_policy(artifacts["candidates"][1].checkpoint, pretrained_path=str(artifacts["critics"][1]))
    with pytest.raises(ProvenanceError, match="declared inducing critic"):
        validate_full_refinement(**artifacts)


def test_rejects_mismatched_optimizer_step(artifacts):
    path = artifacts["candidates"][1].checkpoint.parent / "training_state" / "optimizer_state.safetensors"
    state = load_file(str(path))
    state["state/0/step"] = torch.tensor(2.)
    save_file(state, str(path))
    with pytest.raises(ProvenanceError, match="AdamW step differs"):
        validate_full_refinement(**artifacts)


def test_rejects_updated_critic(artifacts):
    path = artifacts["candidates"][1].checkpoint / "model.safetensors"
    state = load_file(str(path))
    state[next(k for k in state if k.startswith("critic."))].add_(1)
    save_file(state, str(path))
    with pytest.raises(ProvenanceError, match="non-actor tensor changed"):
        validate_full_refinement(**artifacts)


def test_rejects_distinct_initial_actor_even_for_preflight(artifacts):
    path = artifacts["critics"][2] / "model.safetensors"
    state = load_file(str(path))
    state[next(k for k in state if k.startswith("actor."))].add_(1)
    save_file(state, str(path))
    with pytest.raises(ProvenanceError, match="actor initialization"):
        validate_full_refinement(**artifacts)


def test_rejects_training_cache_as_audit_file(artifacts):
    config = json.loads((artifacts["q0"] / "train_config.json").read_text())
    artifacts["audit_cache"] = Path(config["dataset"]["repo_id"]) / "chunk_transitions_train.pt"
    with pytest.raises(ProvenanceError, match="fitted on D_audit"):
        validate_full_refinement(**artifacts)


def test_refuses_missing_reverse_but_allows_explicit_preflight(artifacts):
    artifacts["candidates"] = artifacts["candidates"][:1]
    with pytest.raises(ProvenanceError, match="Missing reverse"):
        validate_full_refinement(**artifacts)
    match, _ = validate_full_refinement(**artifacts, require_bidirectional=False)
    assert not match["bidirectional_complete"]
    assert match["missing_candidates"] == ["q5_c2"]


def arguments(artifacts, output, preflight=False):
    return argparse.Namespace(
        q0_checkpoint=artifacts["q0"], critic_1_checkpoint=artifacts["critics"][1],
        critic_2_checkpoint=artifacts["critics"][2],
        candidate=[(str(c.q_weight), str(c.inducing_critic), str(c.checkpoint)) for c in artifacts["candidates"]],
        audit_cache=artifacts["audit_cache"], expected_updates=3, output_dir=output,
        preflight_only=preflight, device="cpu", batch_size=1, bootstrap_seed=1000, bootstrap_reps=10,
    )


def test_end_to_end_exports_only_fixed_actor_results(artifacts, tmp_path, monkeypatch):
    forbid_updates(monkeypatch)
    report = run(arguments(artifacts, tmp_path / "result"))
    assert report["status"] == "EVALUATED" and report["input_files_unchanged"]
    assert report["audit_rows"]["selected"] == 2
    assert report["audit_rows"]["excluded_by_q_mask"] == 1
    assert report["protocol"]["actor_optimizer_steps_in_this_audit"] == 0
    assert (tmp_path / "result" / "per_state_values.csv").is_file()
    actions = load_file(str(tmp_path / "result" / "fixed_actor_actions.safetensors"))
    assert set(actions) == {"q0", "q5_c1", "q5_c2", "audit_cache_indices", "residual_bounds"}
    with pytest.raises(ProvenanceError, match="overwrite"):
        run(arguments(artifacts, tmp_path / "result"))


def test_preflight_validates_provenance_without_evaluating_audit_rows(artifacts, tmp_path):
    artifacts["candidates"] = artifacts["candidates"][:1]
    report = run(arguments(artifacts, tmp_path / "preflight", preflight=True))
    assert report["status"] == "PREFLIGHT_ONLY" and report["results"] == {}
    assert not (tmp_path / "preflight" / "per_state_values.csv").exists()


def test_detects_inputs_changed_after_validation(artifacts):
    _, evidence = validate_full_refinement(**artifacts)
    artifacts["audit_cache"].write_bytes(b"changed")
    with pytest.raises(ProvenanceError, match="Input changed"):
        evidence.verify_unchanged()


@pytest.mark.parametrize("change", ["resume", "scheduler", "dataset", "optimizer", "saved_config", "diagnostics"])
def test_additional_provenance_failures(artifacts, tmp_path, change):
    root = artifacts["candidates"][1].checkpoint
    config_path = root / "train_config.json"
    cfg = json.loads(config_path.read_text())
    if change == "resume":
        cfg["resume"] = True
    elif change == "scheduler":
        cfg["scheduler"] = {"type": "cosine"}
    elif change == "dataset":
        alternate = tmp_path / "other_cache"
        alternate.mkdir()
        torch.save([{"different": True}], alternate / "chunk_transitions_train.pt")
        cfg["dataset"]["repo_id"] = str(alternate)
    elif change == "optimizer":
        cfg["optimizer"]["grad_clip_norm"] = 2.0
    elif change == "saved_config":
        cfg["policy"]["beta"] = 8
    else:
        Path(cfg["policy"]["diagnostics_jsonl_path"]).unlink()
    write_json(config_path, cfg)
    with pytest.raises(ProvenanceError):
        validate_full_refinement(**artifacts)


def test_refuses_nonfinite_actor_output():
    with pytest.raises(ProvenanceError, match="Invalid actor output"):
        evaluate_fixed_actors(
            control=FixedActor([0, 0]), candidates={5.0: {1: FixedActor([float('nan'), 0]), 2: FixedActor([0.2, 0])}},
            critics={1: LinearCritic(1), 2: LinearCritic(2)}, states=torch.zeros(2, 4), proposals=torch.zeros(2, 2))


def test_refuses_forward_that_changes_model_buffers():
    class MutatingCritic(LinearCritic):
        def forward(self, states, actions):
            self.sign.add_(1)
            return super().forward(states, actions)

    with pytest.raises(ProvenanceError, match="Model tensors changed"):
        evaluate_fixed_actors(
            control=FixedActor([0, 0]), candidates={5.0: {1: FixedActor([0.1, 0]), 2: FixedActor([0.2, 0])}},
            critics={1: MutatingCritic(1), 2: LinearCritic(2)}, states=torch.zeros(2, 4), proposals=torch.zeros(2, 2))
