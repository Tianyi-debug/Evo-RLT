from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from evo_rlt.cli.audit_dynamic_beta import _build_heads, build_report, td_correlation


def test_dynamic_beta_audit_builds_head_only_report(tmp_path):
    policy_path = tmp_path / "policy"
    calibration_cache = tmp_path / "calibration"
    training_cache = tmp_path / "training"
    policy_path.mkdir()
    calibration_cache.mkdir()
    training_cache.mkdir()

    config = {
        "rl_token_dim": 4,
        "proprio_dim": 2,
        "chunk_length": 2,
        "action_dim": 2,
        "actor_hidden_dim": 8,
        "actor_num_layers": 2,
        "actor_fixed_std": 0.05,
        "actor_ref_dropout_p": 0.0,
        "actor_activation": "silu",
        "actor_layer_norm": True,
        "actor_residual": False,
        "state_normalization": "rl_token_layer_norm",
        "actor_action_residual": True,
        "actor_delta_scale": 0.1,
        "critic_hidden_dim": 8,
        "critic_num_layers": 2,
        "critic_activation": "silu",
        "critic_layer_norm": True,
        "critic_residual": False,
        "ac_semantics_version": 2,
        "gamma": 0.99,
        "target_q_clip": 100.0,
        "actor_bc_uncertainty_threshold_mode": "ema_quantile",
        "actor_bc_uncertainty_min_gap": 1e-6,
        "critic_bootstrap_mode": "fixed_bernoulli",
        "critic_bootstrap_keep_prob": 0.8,
        "critic_bootstrap_seed": 1000,
    }
    (policy_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    actor, critic = _build_heads(config)
    with torch.no_grad():
        actor.net[-1].bias.fill_(0.1)
        # Avoid an exactly constant disagreement distribution in this synthetic checkpoint.
        critic.q2.net[-1].bias.fill_(0.2)
    weights = {
        **{f"actor.{key}": value for key, value in actor.state_dict().items()},
        **{f"critic.{key}": value for key, value in critic.state_dict().items()},
        **{
            f"target_critic.{key}": value
            for key, value in copy.deepcopy(critic).state_dict().items()
        },
        "_actor_bc_tau_low_ema": torch.tensor(0.01),
        "_actor_bc_tau_high_ema": torch.tensor(0.2),
    }
    save_file(weights, policy_path / "model.safetensors")

    def samples(count):
        result = []
        generator = torch.Generator().manual_seed(7)
        for index in range(count):
            result.append(
                {
                    "state_vec": torch.randn(6, generator=generator),
                    "exec_chunk": torch.randn(2, 2, generator=generator),
                    "ref_chunk": torch.randn(2, 2, generator=generator),
                    "reward_seq": torch.zeros(2),
                    "next_state_vec": torch.randn(6, generator=generator),
                    "next_ref_chunk": torch.randn(2, 2, generator=generator),
                    "done": torch.tensor(0.0),
                    "intervention": torch.tensor(float(index % 3 == 0)),
                    "actual_steps": torch.tensor(2),
                    "source": torch.tensor(index % 4),
                    "episode_id": torch.tensor(index // 4),
                    "is_critical": torch.tensor(1.0),
                }
            )
        return result

    torch.save(samples(24), calibration_cache / "chunk_transitions_val.pt")
    torch.save(samples(32), training_cache / "chunk_transitions_train.pt")
    args = SimpleNamespace(
        policy_path=str(policy_path),
        calibration_cache_dir=str(calibration_cache),
        calibration_split="val",
        training_cache_dir=str(training_cache),
        training_split="train",
        beta=0.3,
        kappa=3.0,
        batch_size=8,
        gradient_batches=2,
        max_calibration_samples=None,
        max_training_samples=None,
        seed=1000,
        device="cpu",
        output_json=str(tmp_path / "audit.json"),
    )

    report = build_report(args)

    assert report["checks"]["all_finite"] is True
    assert report["checks"]["beta_within_expected_range"] is True
    assert report["checks"]["actor_ref_delta_within_bound"] is True
    assert report["tau_high"] > report["tau_low"]
    assert report["tau_low"] == pytest.approx(0.01)
    assert report["tau_high"] == pytest.approx(0.2)
    assert report["threshold_mode"] == "ema_quantile"
    assert report["critic_bootstrap_mode"] == "fixed_bernoulli"
    assert 0.3 <= report["training"]["beta_matched"] <= 1.2
    assert report["training"]["actor_ref_delta_abs"]["max"] <= 0.1 + 1e-6
    assert report["training"]["source"]["3"]["count"] > 0
    assert report["gradients"]["q_grad_norm_mean"] > 0.0
    assert report["gradients"]["bc_raw_grad_norm_mean"] > 0.0
    assert report["gradients"]["bc_weighted_grad_norm_mean"] > 0.0
    assert report["td_correlation"]["overall"]["count"] == 24
    assert report["td_correlation"]["primary_metric"] == (
        "proposal_disagreement_vs_td_center_abs"
    )
    assert report["td_correlation"]["circular_diagnostic"] == (
        "proposal_disagreement_vs_td_avg_abs"
    )
    assert torch.isfinite(torch.tensor(report["td_correlation"]["overall"]["spearman"]))
    assert 0.0 <= report["td_correlation"]["overall"]["target_between_heads_frac"] <= 1.0


def test_td_correlation_uses_non_circular_center_residual_as_primary():
    class StubActor:
        def __call__(self, state, reference, training=False):
            return torch.zeros_like(reference), torch.zeros_like(reference)

    class StubCritic:
        def __call__(self, state, action):
            return state[:, :1], state[:, 1:2]

    class StubTargetCritic:
        def min_q(self, state, action):
            return torch.zeros(state.shape[0], 1)

    # Q centers are [3, 2, 1] while disagreements are [1, 2, 3]. With target=0,
    # the primary center residual is perfectly anti-correlated with disagreement.
    states = [torch.tensor([4.0, 2.0]), torch.tensor([4.0, 0.0]), torch.tensor([4.0, -2.0])]
    samples = [
        {
            "state_vec": state,
            "exec_chunk": torch.zeros(1, 1),
            "ref_chunk": torch.zeros(1, 1),
            "reward_seq": torch.zeros(1),
            "next_state_vec": torch.zeros(2),
            "next_ref_chunk": torch.zeros(1, 1),
            "done": torch.tensor(1.0),
            "actual_steps": torch.tensor(1),
            "source": torch.tensor(0),
        }
        for state in states
    ]

    result = td_correlation(
        StubActor(),
        StubCritic(),
        StubTargetCritic(),
        samples,
        batch_size=3,
        device="cpu",
        gamma=0.99,
        target_q_clip=100.0,
    )
    overall = result["overall"]

    assert overall["spearman"] == pytest.approx(-1.0)
    assert overall["pearson"] == pytest.approx(-1.0)
    assert overall["td_abs_mean"] == pytest.approx(2.0)
    assert overall["td_center_abs_mean"] == pytest.approx(2.0)
    assert overall["td_avg_abs_circular_spearman"] != pytest.approx(-1.0)
    assert overall["target_between_heads_frac"] == pytest.approx(2 / 3)
