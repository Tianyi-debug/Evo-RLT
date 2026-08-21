from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from evo_rlt.cli.audit_risk_action_dependence import (
    FULL,
    SHUFFLED,
    STATE_ONLY,
    make_shuffled_action_rows,
    paired_episode_bootstrap,
    run_action_dependence_ablation,
)


def _row(
    episode_uid: str,
    *,
    category: str,
    label: float,
    value: float,
    anchor_index: int,
) -> dict:
    return {
        "state_vec": torch.tensor([value, value + 0.1, value + 0.2]),
        "action_chunk": torch.tensor([[value + 1.0, value + 2.0]]),
        "label": torch.tensor(label),
        "label_mask": torch.tensor(1.0),
        "episode_uid": episode_uid,
        "category": category,
        "anchor_index": anchor_index,
    }


def _split(prefix: str) -> list[dict]:
    rows: list[dict] = []
    for episode in range(2):
        uid = f"{prefix}-corrective-{episode}"
        rows.append(
            _row(uid, category="corrective", label=0.0, value=episode + 0.1, anchor_index=0)
        )
        rows.append(
            _row(uid, category="corrective", label=1.0, value=episode + 1.1, anchor_index=1)
        )
    for episode in range(2):
        uid = f"{prefix}-success-{episode}"
        rows.append(
            _row(
                uid,
                category="autonomous_success",
                label=0.0,
                value=episode + 3.1,
                anchor_index=0,
            )
        )
    return rows


def test_shuffled_actions_are_deterministic_deranged_and_split_local():
    rows = _split("train")
    shuffled_a, mapping_a = make_shuffled_action_rows(rows, seed=123)
    shuffled_b, mapping_b = make_shuffled_action_rows(rows, seed=123)
    assert mapping_a == mapping_b
    assert all(item["recipient_index"] != item["donor_index"] for item in mapping_a)
    for original, shuffled, mapping in zip(rows, shuffled_a, mapping_a, strict=True):
        assert torch.equal(original["state_vec"], shuffled["state_vec"])
        assert torch.equal(original["label"], shuffled["label"])
        assert torch.equal(
            shuffled["action_chunk"], rows[mapping["donor_index"]]["action_chunk"]
        )


def test_paired_bootstrap_uses_episode_units_and_reports_deltas():
    rows = _split("validation")
    labels = torch.tensor([float(row["label"].item()) for row in rows])
    full = torch.tensor([-3.0, 3.0, -2.0, 2.0, -3.0, -3.0])
    weak = torch.zeros_like(full)
    report = paired_episode_bootstrap(
        {FULL: full, STATE_ONLY: weak, SHUFFLED: weak},
        labels,
        rows,
        replicates=100,
        seed=7,
    )
    assert report["unit"] == "episode"
    assert report["full_minus_state_only"]["average_precision"]["median"] > 0
    assert report["full_minus_shuffled_action"]["auroc"]["p_gt_zero"] > 0.9


def test_end_to_end_action_ablation_writes_three_checkpoints(tmp_path: Path):
    torch_rng_state = torch.random.get_rng_state()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train_rows = _split("train")
    val_rows = _split("validation")
    torch.save(train_rows, dataset / "actor_trust_train.pt")
    torch.save(val_rows, dataset / "actor_trust_val.pt")
    metadata = {
        "semantics": {
            "primary_future_k": 3,
            "frame_stride": 2,
            "fps": 30.0,
            "risk_action_semantics": "actual_sent",
        },
        "dimensions": {"state_dim": 3, "action_flat_dim": 2},
    }
    (dataset / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "output"

    try:
        report = run_action_dependence_ablation(
            dataset_dir=dataset,
            output_dir=output,
            epochs=1,
            batch_size=3,
            learning_rate=3e-4,
            weight_decay=1e-4,
            seed=1000,
            bootstrap_replicates=20,
            device_name="cpu",
        )
    finally:
        torch.random.set_rng_state(torch_rng_state)

    assert report["status"] == "COMPLETE"
    assert report["anchor_horizon"]["horizon_seconds"] == pytest.approx(0.2)
    assert report["shuffled_action_mapping"]["cross_split_shuffle"] is False
    assert set(report["variants"]) == {FULL, STATE_ONLY, SHUFFLED}
    for variant in (FULL, STATE_ONLY, SHUFFLED):
        assert (output / variant / "risk.pt").is_file()
    assert (output / "action_dependence_report.json").is_file()
