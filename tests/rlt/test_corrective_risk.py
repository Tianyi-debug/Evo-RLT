import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

from evo_rlt.cli.audit_actor_q_mechanism import validate_matched_actor_refine
from evo_rlt.cli.train_corrective_risk import binary_classification_metrics
from evo_rlt.core.corrective_risk import (
    CorrectiveTakeoverRiskMLP,
    load_corrective_risk_checkpoint,
    masked_risk_bce_with_logits,
    save_corrective_risk_checkpoint,
)


def test_masked_risk_bce_ignores_censored_rows():
    logits = torch.tensor([0.0, 1.0, -2.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    first = masked_risk_bce_with_logits(logits, labels, mask, pos_weight=1.5)
    changed = logits.detach().clone()
    changed[2] = 1000.0
    second = masked_risk_bce_with_logits(changed, labels, mask, pos_weight=1.5)
    assert first.item() == pytest.approx(second.item())
    first.backward()
    assert logits.grad[2].item() == 0.0


def test_corrective_risk_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(7)
    model = CorrectiveTakeoverRiskMLP(6, 4, hidden_dims=(8, 4))
    state = torch.randn(5, 6)
    action = torch.randn(5, 2, 2)
    expected = model(state, action).detach()
    path = tmp_path / "risk.pt"
    metadata = {"primary_future_k": 3, "normalization": "none"}
    save_corrective_risk_checkpoint(path, model, metadata)

    loaded, loaded_metadata = load_corrective_risk_checkpoint(path)
    assert torch.equal(expected, loaded(state, action))
    assert loaded_metadata == metadata
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_state_only_risk_head_has_no_action_input():
    model = CorrectiveTakeoverRiskMLP(6, 0, hidden_dims=(8, 4))
    state = torch.randn(5, 6)
    assert model(state).shape == (5,)
    with pytest.raises(ValueError, match="does not accept action"):
        model(state, torch.randn(5, 1))


def test_binary_risk_metrics_are_exact_for_perfect_ranking():
    metrics = binary_classification_metrics(
        torch.tensor([-4.0, -2.0, 2.0, 4.0]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["accuracy_at_0_5"] == pytest.approx(1.0)
    assert metrics["false_positive_rate_at_0_5"] == pytest.approx(0.0)


def _write_checkpoint(root: Path, *, q_weight: float, critic_value: float) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "training_stage": "actor_refine",
                "actor_q_weight_max": q_weight,
                "actor_q_trust_mode": "fixed",
                "actor_teacher_pretrained_path": "/tmp/teacher",
            }
        )
    )
    save_safetensors_file(
        {
            "actor.weight": torch.ones(1),
            "critic.q1.weight": torch.tensor([critic_value]),
        },
        root / "model.safetensors",
    )


def test_diagnostic_reports_not_matched_before_interpretation(tmp_path: Path):
    checkpoint_a = tmp_path / "a"
    checkpoint_b = tmp_path / "b"
    _write_checkpoint(checkpoint_a, q_weight=0.0, critic_value=1.0)
    _write_checkpoint(checkpoint_b, q_weight=0.25, critic_value=2.0)

    result = validate_matched_actor_refine(checkpoint_a, checkpoint_b)

    assert result["status"] == "NOT MATCHED"
    assert any("critic tensors differ" in mismatch for mismatch in result["mismatches"])
    assert any("missing train_config.json" in mismatch for mismatch in result["mismatches"])
