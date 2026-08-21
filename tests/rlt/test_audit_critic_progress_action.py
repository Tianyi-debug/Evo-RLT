from __future__ import annotations

import json

import torch
from safetensors.torch import save_file as save_safetensors_file
from torch import nn

from evo_rlt.cli.audit_critic_progress_action import (
    _crossed_reachable_variance,
    _paired_episode_difference,
    _progress_coordinates,
    _select_progress_matched_donors,
    run_progress_action_audit,
)
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic


class _ActionOnlyTwin(nn.Module):
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        del state
        value = action.sum(-1, keepdim=True)
        return value, value


class _StateOnlyTwin(nn.Module):
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        del action
        value = state[:, :1]
        return value, value


def _actor() -> ChunkActor:
    return ChunkActor(
        state_dim=2,
        chunk_dim=2,
        hidden_dim=8,
        num_layers=1,
        action_residual=True,
        delta_scale=0.5,
        proprio_dim=0,
        state_normalization="none",
    )


def test_progress_coordinates_use_anchor_stride_not_chunk_length():
    result = _progress_coordinates(
        {
            "anchor_index": 3,
            "prefix_anchor_count": 7,
            "frame_stride": 2,
            "fps": 30.0,
        },
        normalized_bins=5,
        elapsed_bin_seconds=0.1,
    )
    assert result["normalized_progress"] == 0.5
    assert result["normalized_progress_bin"] == 2
    assert result["elapsed_seconds"] == 0.2
    assert result["elapsed_seconds_bin"] == 2


def test_corrective_progress_comparison_is_episode_paired():
    left = [
        {"episode_uid": "a", "q": 1.0},
        {"episode_uid": "b", "q": 2.0},
        {"episode_uid": "left-only", "q": 100.0},
    ]
    right = [
        {"episode_uid": "a", "q": 3.0},
        {"episode_uid": "b", "q": 4.0},
        {"episode_uid": "right-only", "q": -100.0},
    ]
    result = _paired_episode_difference(
        left, right, key="q", seed=4, replicates=200
    )
    assert result["paired_episodes"] == 2
    assert result["mean_difference"] == -2.0
    assert result["episode_paired_bootstrap_95ci"] == [-2.0, -2.0]


def test_progress_matched_donors_come_from_another_episode_in_same_bin():
    bins = [0, 0, 1, 1]
    episodes = ["a", "b", "c", "d"]
    donors, fallback_count = _select_progress_matched_donors(
        bins, episodes, seed=5
    )
    assert fallback_count == 0
    for index, donor in enumerate(donors):
        assert bins[donor] == bins[index]
        assert episodes[donor] != episodes[index]


def test_crossed_variance_distinguishes_action_only_from_state_only_critic():
    actor = _actor()
    states = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    )
    proposals = torch.zeros(5, 2)
    actions = torch.tensor(
        [[-0.4, -0.2], [-0.2, 0.0], [0.0, 0.2], [0.2, 0.3], [0.4, 0.4]]
    )
    common = dict(
        actor=actor,
        states=states,
        proposals=proposals,
        donor_actions=actions,
        normalized_bins=[0] * 5,
        device=torch.device("cpu"),
        batch_size=64,
        samples_per_bin=5,
        repeats=3,
        seed=6,
    )
    action_report = _crossed_reachable_variance(
        critic=_ActionOnlyTwin(), **common
    )
    state_report = _crossed_reachable_variance(
        critic=_StateOnlyTwin(), **common
    )
    assert action_report["action_fraction_across_bin_repeats"]["median"] > 0.99
    assert state_report["action_fraction_across_bin_repeats"]["median"] == 0.0


def test_progress_action_audit_runs_from_checkpoint_and_actor_trust_files(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    dataset = tmp_path / "actor_trust"
    checkpoint.mkdir()
    dataset.mkdir()
    config = {
        "rl_token_dim": 1,
        "proprio_dim": 1,
        "chunk_length": 1,
        "action_dim": 2,
        "actor_hidden_dim": 8,
        "actor_num_layers": 1,
        "critic_hidden_dim": 8,
        "critic_num_layers": 1,
        "state_normalization": "none",
        "actor_action_residual": True,
        "actor_delta_scale": 0.5,
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    actor = ChunkActor(
        state_dim=2,
        chunk_dim=2,
        hidden_dim=8,
        num_layers=1,
        action_residual=True,
        delta_scale=0.5,
        proprio_dim=1,
        state_normalization="none",
    )
    critic = TwinCritic(
        state_dim=2,
        chunk_dim=2,
        hidden_dim=8,
        num_layers=1,
        proprio_dim=1,
        state_normalization="none",
    )
    weights = {
        **{f"actor.{key}": value for key, value in actor.state_dict().items()},
        **{f"critic.{key}": value for key, value in critic.state_dict().items()},
    }
    save_safetensors_file(weights, checkpoint / "model.safetensors")
    (dataset / "metadata.json").write_text(
        json.dumps({"semantics": {"primary_future_k": 1}})
    )

    categories = [
        "autonomous_success",
        "autonomous_failure",
        "corrective",
        "autonomous_success",
    ]
    rows = []
    for episode, category in enumerate(categories):
        for anchor in range(2):
            rows.append(
                {
                    "state_vec": torch.tensor([float(episode), float(anchor)]),
                    "proposal_chunk": torch.zeros(1, 2),
                    "exec_chunk": torch.tensor(
                        [[0.05 * (episode + 1), 0.05 * (anchor + 1)]]
                    ),
                    "episode_uid": f"ep{episode}",
                    "category": category,
                    "anchor_index": anchor,
                    "prefix_anchor_count": 2,
                    "anchor_start_frame": anchor * 2,
                    "frame_stride": 2,
                    "fps": 30.0,
                    "distance_to_corrective_event": (
                        2 - anchor if category == "corrective" else -1
                    ),
                    "distance_to_event_anchors": (
                        2 - anchor if category == "corrective" else -1
                    ),
                    "exec_action_is_actual_sent": 1.0,
                }
            )
    torch.save(rows[:4], dataset / "actor_trust_train.pt")
    torch.save(rows[4:], dataset / "actor_trust_val.pt")
    output = tmp_path / "report.json"
    report = run_progress_action_audit(
        checkpoint=checkpoint,
        actor_trust_dataset=dataset,
        output_path=output,
        bootstrap_replicates=100,
        normalized_progress_bins=2,
        elapsed_bin_seconds=0.05,
        crossed_samples_per_bin=4,
        crossed_repeats=1,
        seed=7,
    )
    assert report["status"] == "COMPLETE"
    assert report["actor_trust_dataset"]["samples"] == 8
    assert report["fixed_state_action_sensitivity"][
        "counterfactual_action_quality_label_available"
    ] is False
    assert output.is_file()
