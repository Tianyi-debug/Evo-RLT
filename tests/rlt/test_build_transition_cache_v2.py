from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch


def test_transition_cache_v2_config_overrides_apply_before_policy_load(tmp_path):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    module.RLTokenPolicyConfig.ensure_registered()
    module.RLTokenPolicyConfig(
        vla_pretrained_path="/tmp/stale-vla",
        device="cpu",
    ).save_pretrained(tmp_path)

    loaded = module.PreTrainedConfig.from_pretrained(
        tmp_path,
        cli_overrides=[
            "--vla_pretrained_path=/tmp/vla",
            "--vla_type=smolvla",
            "--tokenizer_path=/tmp/tokenizer",
            "--norm_stats_path=/tmp/norm-stats.pt",
        ],
    )

    assert loaded.vla_pretrained_path == "/tmp/vla"
    assert loaded.vla_type == "smolvla"
    assert loaded.tokenizer_path == "/tmp/tokenizer"
    assert loaded.norm_stats_path == "/tmp/norm-stats.pt"


def test_transition_cache_v2_passes_video_backend(monkeypatch, tmp_path):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_episodes = 0
            self.meta = SimpleNamespace(episodes=None)

    class FakeMetadata:
        fps = 20

        def __init__(self, **kwargs):
            captured["metadata_kwargs"] = kwargs

    class FakePolicy:
        config = SimpleNamespace(
            action_dim=12,
            chunk_size=50,
            image_only=False,
            proprio_dim=12,
            token_pool_size=0,
            vla_pretrained_path=None,
        )
        _num_image_tokens = 0
        _pi05 = object()
        rl_token = object()

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeCapture:
        def __init__(self, **kwargs):
            pass

        def attach(self, pi05):
            pass

        def detach(self):
            pass

    monkeypatch.setattr(module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(module, "LeRobotDatasetMetadata", FakeMetadata)

    def load_config(path, cli_overrides):
        captured["config_path"] = path
        captured["config_overrides"] = cli_overrides
        return FakePolicy.config

    monkeypatch.setattr(module.PreTrainedConfig, "from_pretrained", load_config)
    monkeypatch.setattr(
        module.RLTokenPolicy,
        "from_pretrained",
        lambda path, config: FakePolicy(),
    )
    monkeypatch.setattr(module, "PrefixOutputCapture", FakeCapture)
    monkeypatch.setattr(module, "make_rlt_token_pre_post_processors", lambda config: (object(), object()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_transition_cache_v2.py",
            "--demo-dataset-repo-id",
            "local/demo",
            "--demo-dataset-root",
            "/tmp/demo",
            "--rl-token-policy-path",
            "/tmp/rl-token",
            "--vla-pretrained-path",
            "/tmp/vla",
            "--vla-type",
            "smolvla",
            "--tokenizer-path",
            "/tmp/tokenizer",
            "--norm-stats-path",
            "/tmp/norm-stats.pt",
            "--output-dir",
            str(tmp_path),
            "--max-episodes",
            "0",
            "--video-backend",
            "video_reader",
        ],
    )

    module.main()

    assert captured["video_backend"] == "video_reader"
    assert captured["delta_timestamps"]["action"][1] == pytest.approx(0.05)
    assert captured["metadata_kwargs"] == {
        "repo_id": "local/demo",
        "root": "/tmp/demo",
    }
    assert captured["config_path"] == "/tmp/rl-token"
    assert captured["config_overrides"] == [
        "--vla_pretrained_path=/tmp/vla",
        "--vla_type=smolvla",
        "--tokenizer_path=/tmp/tokenizer",
        "--norm_stats_path=/tmp/norm-stats.pt",
    ]


def test_transition_cache_v2_semantic_builder_uses_exec_action_c_step_and_reward():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    C = 3
    action_dim = 2
    frame_indices = [0, 1, 2, 3, 4]
    state_vecs = torch.arange(len(frame_indices) * 4, dtype=torch.float32).view(len(frame_indices), 4)
    ref_chunks = torch.full((len(frame_indices), C, action_dim), 10.0)
    exec_chunks = torch.full((len(frame_indices), C, action_dim), -5.0)
    exec_chunks[0] = 100.0
    exec_chunks[1] = 200.0

    transitions = module._encoded_episode_to_transitions(
        state_vecs=state_vecs,
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=frame_indices,
        episode_last_frame=4,
        chunk_length=C,
        frame_stride=1,
        episode_success=True,
        ep_id=7,
        fps=25.0,
    )

    assert len(transitions) == 2
    assert torch.equal(transitions[0].exec_chunk, exec_chunks[0])
    assert torch.equal(transitions[0].ref_chunk, ref_chunks[0])
    assert torch.equal(transitions[0].proposal_chunk, ref_chunks[0])
    assert torch.equal(transitions[0].bc_target_chunk, ref_chunks[0])
    assert not torch.equal(transitions[0].exec_chunk, transitions[0].ref_chunk)
    assert torch.equal(transitions[0].next_state_vec, state_vecs[3])
    assert torch.equal(transitions[1].next_state_vec, state_vecs[4])
    assert transitions[0].done.item() == 0.0
    assert transitions[1].done.item() == 1.0
    assert transitions[0].bootstrap_mask.item() == 1.0
    assert transitions[1].bootstrap_mask.item() == 0.0
    assert all(t.critic_mask.item() == 1.0 for t in transitions)
    assert torch.equal(transitions[0].reward_seq, torch.zeros(C))
    assert torch.equal(transitions[1].reward_seq, torch.tensor([0.0, 0.0, 1.0]))
    assert transitions[1].episode_id.item() == 7
    assert transitions[0].anchor_start_frame.item() == 0
    assert transitions[0].frame_stride.item() == 1
    assert transitions[0].fps.item() == pytest.approx(25.0)


def test_transition_cache_v2_demo_executed_mode_uses_expert_bc_target():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    chunk_length = 2
    frame_indices = [0, 1, 2]
    ref_chunks = torch.full((3, chunk_length, 2), 0.25)
    exec_chunks = torch.full((3, chunk_length, 2), -0.5)

    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.randn(3, 4),
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=frame_indices,
        episode_last_frame=2,
        chunk_length=chunk_length,
        frame_stride=1,
        episode_success=True,
        ep_id=8,
        demo_reference_mode="executed",
    )

    assert len(transitions) == 1
    assert transitions[0].source.item() == module.TRANSITION_SOURCE_DEMO
    assert torch.equal(transitions[0].proposal_chunk, ref_chunks[0])
    assert torch.equal(transitions[0].ref_chunk, ref_chunks[0])
    assert torch.equal(transitions[0].exec_chunk, exec_chunks[0])
    assert torch.equal(transitions[0].bc_target_chunk, exec_chunks[0])


def test_transition_cache_marks_actual_sent_exec_and_human_bc_uses_it():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    chunk_length = 2
    frame_indices = [0, 1, 2]
    requested = torch.full((3, chunk_length, 2), 0.8)
    actual_sent = torch.full((3, chunk_length, 2), 0.3)
    provenance = module.FrameProvenance(
        is_intervention=torch.ones(3),
        collector_policy_id=torch.full((3,), 3),
        intervention_stage=torch.full((3,), 2.0),
        intervention_reason=torch.full((3,), 1),
    )

    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.randn(3, 4),
        ref_chunks=requested,
        exec_chunks=actual_sent,
        frame_indices=frame_indices,
        episode_last_frame=2,
        chunk_length=chunk_length,
        frame_stride=1,
        episode_success=True,
        ep_id=9,
        provenance=provenance,
        human_reference_mode="executed",
        exec_action_is_actual_sent=True,
    )

    assert len(transitions) == 1
    assert transitions[0].exec_action_is_actual_sent.item() == 1.0
    assert torch.equal(transitions[0].exec_chunk, actual_sent[0])
    assert torch.equal(transitions[0].bc_target_chunk, actual_sent[0])
    assert not torch.equal(transitions[0].exec_chunk, requested[0])


def test_transition_cache_v2_demo_executed_mode_with_zero_provenance():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    chunk_length = 2
    frame_indices = [0, 1, 2]
    ref_chunks = torch.full((3, chunk_length, 2), 0.25)
    exec_chunks = torch.full((3, chunk_length, 2), -0.5)
    provenance = module.FrameProvenance(
        is_intervention=torch.zeros(3),
        collector_policy_id=torch.zeros(3),
        intervention_stage=torch.zeros(3),
    )

    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.randn(3, 4),
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=frame_indices,
        episode_last_frame=2,
        chunk_length=chunk_length,
        frame_stride=1,
        episode_success=True,
        ep_id=9,
        provenance=provenance,
        demo_reference_mode="executed",
    )

    assert len(transitions) == 1
    assert transitions[0].source.item() == module.TRANSITION_SOURCE_DEMO
    assert transitions[0].intervention.item() == 0.0
    assert torch.equal(transitions[0].proposal_chunk, ref_chunks[0])
    assert torch.equal(transitions[0].bc_target_chunk, exec_chunks[0])


def test_outcome_aware_actor_bc_uses_executed_demo_target():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    ref_chunks = torch.full((3, 2, 2), 0.25)
    exec_chunks = torch.full((3, 2, 2), -0.5)
    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.randn(3, 4),
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=[0, 1, 2],
        episode_last_frame=2,
        chunk_length=2,
        frame_stride=1,
        episode_success=True,
        ep_id=10,
        actor_bc_mode="outcome-aware",
    )

    assert len(transitions) == 1
    assert transitions[0].source.item() == module.TRANSITION_SOURCE_DEMO
    assert transitions[0].actor_bc_mask.item() == 1.0
    assert torch.equal(transitions[0].proposal_chunk, ref_chunks[0])
    assert torch.equal(transitions[0].bc_target_chunk, exec_chunks[0])


@pytest.mark.parametrize(
    ("episode_success", "expected_mask"),
    [(True, 1.0), (False, 0.0)],
)
def test_outcome_aware_actor_bc_clones_only_successful_autonomous_episodes(
    episode_success,
    expected_mask,
):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    frame_indices = [0, 1, 2]
    ref_chunks = torch.full((3, 2, 2), 0.25)
    exec_chunks = torch.full((3, 2, 2), -0.5)
    provenance = module.FrameProvenance(
        is_intervention=torch.zeros(3),
        collector_policy_id=torch.full((3,), 2),
        intervention_stage=torch.zeros(3),
        intervention_reason=torch.zeros(3, dtype=torch.long),
    )
    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.randn(3, 4),
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=frame_indices,
        episode_last_frame=2,
        chunk_length=2,
        frame_stride=1,
        episode_success=episode_success,
        ep_id=11,
        provenance=provenance,
        actor_bc_mode="outcome-aware",
    )

    assert len(transitions) == 1
    assert transitions[0].source.item() == module.TRANSITION_SOURCE_RL_AUTONOMOUS
    assert transitions[0].actor_bc_mask.item() == expected_mask
    expected_target = exec_chunks[0] if episode_success else ref_chunks[0]
    assert torch.equal(transitions[0].bc_target_chunk, expected_target)


def test_outcome_aware_actor_bc_masks_policy_prefix_in_assisted_success():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    frame_indices = list(range(12))
    ref_chunks = torch.full((12, 2, 2), 0.25)
    exec_chunks = torch.full((12, 2, 2), -0.5)
    provenance = module.FrameProvenance(
        is_intervention=torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0], dtype=torch.float32
        ),
        collector_policy_id=torch.tensor([2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2]),
        intervention_stage=torch.tensor(
            [0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 3], dtype=torch.float32
        ),
        intervention_reason=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.randn(12, 4),
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=frame_indices,
        episode_last_frame=11,
        chunk_length=2,
        frame_stride=1,
        episode_success=True,
        ep_id=12,
        provenance=provenance,
        actor_bc_mode="outcome-aware",
    )

    policy = [
        transition
        for transition in transitions
        if int(transition.source.item()) == module.TRANSITION_SOURCE_RL_AUTONOMOUS
    ]
    human = [
        transition
        for transition in transitions
        if int(transition.source.item()) == module.TRANSITION_SOURCE_HUMAN_OVERRIDE
    ]
    assert policy and human
    assert all(transition.actor_bc_mask.item() == 0.0 for transition in policy)
    assert all(transition.actor_bc_mask.item() == 1.0 for transition in human)
    assert all(
        torch.equal(transition.bc_target_chunk, transition.exec_chunk)
        for transition in human
    )


def test_transition_cache_v2_semantic_builder_zero_reward_on_failure():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    C = 3
    frame_indices = [0, 1, 2, 3, 4]
    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.randn(len(frame_indices), 4),
        ref_chunks=torch.randn(len(frame_indices), C, 2),
        exec_chunks=torch.randn(len(frame_indices), C, 2),
        frame_indices=frame_indices,
        episode_last_frame=4,
        chunk_length=C,
        frame_stride=1,
        episode_success=False,
        ep_id=0,
    )

    assert transitions[-1].done.item() == 1.0
    assert transitions[-1].bootstrap_mask.item() == 0.0
    assert all(t.critic_mask.item() == 1.0 for t in transitions)
    assert all(t.reward_seq.sum().item() == pytest.approx(0.0) for t in transitions)


def test_transition_cache_v2_separates_human_bc_target_from_vla_proposal():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    chunk_length = 2
    frame_indices = [0, 1, 2, 3, 4]
    ref_chunks = torch.arange(5 * 2 * 2, dtype=torch.float32).view(5, 2, 2)
    exec_chunks = -ref_chunks - 1
    provenance = module.FrameProvenance(
        is_intervention=torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
        collector_policy_id=torch.tensor([2, 2, 0, 0, 0]),
    )

    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.randn(5, 4),
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=frame_indices,
        episode_last_frame=4,
        chunk_length=chunk_length,
        frame_stride=1,
        episode_success=True,
        ep_id=60,
        provenance=provenance,
        demo_reference_mode="executed",
    )

    assert len(transitions) == 3
    assert transitions[0].source.item() == module.TRANSITION_SOURCE_RL_AUTONOMOUS
    assert transitions[0].intervention.item() == 0.0
    assert torch.equal(transitions[0].ref_chunk, ref_chunks[0])
    assert transitions[1].source.item() == module.TRANSITION_SOURCE_HUMAN_OVERRIDE
    assert transitions[1].intervention.item() == 1.0
    assert torch.equal(transitions[1].proposal_chunk, ref_chunks[1])
    assert torch.equal(transitions[1].ref_chunk, ref_chunks[1])
    assert torch.equal(transitions[1].bc_target_chunk, exec_chunks[1])
    assert torch.equal(transitions[0].next_proposal_chunk, ref_chunks[2])
    assert torch.equal(transitions[0].next_ref_chunk, ref_chunks[2])


def test_transition_cache_v2_vla_mode_preserves_human_reference():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    chunk_length = 2
    frame_indices = [0, 1, 2, 3, 4]
    ref_chunks = torch.arange(5 * 2 * 2, dtype=torch.float32).view(5, 2, 2)
    exec_chunks = -ref_chunks - 1
    provenance = module.FrameProvenance(
        is_intervention=torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
        collector_policy_id=torch.tensor([2, 2, 0, 0, 0]),
    )

    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.randn(5, 4),
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=frame_indices,
        episode_last_frame=4,
        chunk_length=chunk_length,
        frame_stride=1,
        episode_success=True,
        ep_id=60,
        provenance=provenance,
        human_reference_mode="vla",
    )

    assert transitions[1].source.item() == module.TRANSITION_SOURCE_HUMAN_OVERRIDE
    assert transitions[1].intervention.item() == 1.0
    assert torch.equal(transitions[1].exec_chunk, exec_chunks[1])
    assert torch.equal(transitions[1].proposal_chunk, ref_chunks[1])
    assert torch.equal(transitions[1].bc_target_chunk, ref_chunks[1])
    assert torch.equal(transitions[0].next_ref_chunk, ref_chunks[2])


def test_transition_cache_v2_summary_audits_residual_reachability():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    transition = module.ChunkTransition(
        state_vec=torch.zeros(2),
        exec_chunk=torch.full((1, 2), 0.5),
        ref_chunk=torch.zeros(1, 2),
        reward_seq=torch.zeros(1),
        next_state_vec=torch.zeros(2),
        next_ref_chunk=torch.zeros(1, 2),
        done=torch.tensor(0.0),
        intervention=torch.tensor(1.0),
        actual_steps=torch.tensor(1),
        source=torch.tensor(module.TRANSITION_SOURCE_HUMAN_OVERRIDE),
        proposal_chunk=torch.zeros(1, 2),
        bc_target_chunk=torch.full((1, 2), 0.5),
        next_proposal_chunk=torch.zeros(1, 2),
    )

    summary = module._transition_summary([transition], residual_delta_scale=0.1)

    assert "human_vla_action_rmse=0.500000" in summary
    assert "human_target_outside_residual_bound_frac=1.000000" in summary


def test_transition_cache_v2_summary_audits_demo_expert_reachability():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    transition = module.ChunkTransition(
        state_vec=torch.zeros(2),
        exec_chunk=torch.full((1, 2), 0.5),
        ref_chunk=torch.zeros(1, 2),
        reward_seq=torch.zeros(1),
        next_state_vec=torch.zeros(2),
        next_ref_chunk=torch.zeros(1, 2),
        done=torch.tensor(0.0),
        intervention=torch.tensor(0.0),
        actual_steps=torch.tensor(1),
        source=torch.tensor(module.TRANSITION_SOURCE_DEMO),
        proposal_chunk=torch.zeros(1, 2),
        # The audit must compare expert execution with the proposal even when
        # a legacy cache uses the proposal itself as the demo BC target.
        bc_target_chunk=torch.zeros(1, 2),
        next_proposal_chunk=torch.zeros(1, 2),
    )

    summary = module._transition_summary([transition], residual_delta_scale=0.1)

    assert "demo_vla_action_rmse=0.500000" in summary
    assert "demo_vla_action_abs_max=0.500000" in summary
    assert "demo_target_outside_residual_bound_frac=1.000000" in summary
    assert "actor_bc_valid=1/1" in summary
    assert "actor_bc_valid_sources={0: 1}" in summary


def test_transition_cache_v2_drop_legacy_handoff_and_invalid_bootstrap():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    chunk_length = 2
    frame_indices = list(range(7))
    provenance = module.FrameProvenance(
        is_intervention=torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.float32),
        collector_policy_id=torch.tensor([2, 2, 2, 0, 0, 0, 0]),
    )
    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.arange(7 * 4, dtype=torch.float32).view(7, 4),
        ref_chunks=torch.randn(7, chunk_length, 2),
        exec_chunks=torch.randn(7, chunk_length, 2),
        frame_indices=frame_indices,
        episode_last_frame=6,
        chunk_length=chunk_length,
        frame_stride=1,
        episode_success=True,
        ep_id=62,
        provenance=provenance,
        human_reference_mode="vla",
        legacy_handoff_policy="drop",
    )

    # start=2 crosses the handoff; start=0 bootstraps into that invalid anchor.
    assert [int(transition.state_vec[0].item() // 4) for transition in transitions] == [1, 3, 4]
    assert transitions[0].source.item() == module.TRANSITION_SOURCE_RL_AUTONOMOUS
    assert all(
        transition.source.item() == module.TRANSITION_SOURCE_HUMAN_OVERRIDE
        for transition in transitions[1:]
    )


def test_transition_cache_v2_excludes_two_stage_hold_and_handoff_chunks():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    chunk_length = 2
    frame_indices = list(range(9))
    ref_chunks = torch.arange(9 * 2 * 2, dtype=torch.float32).view(9, 2, 2)
    exec_chunks = -ref_chunks - 1
    provenance = module.FrameProvenance(
        is_intervention=torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 0], dtype=torch.float32),
        collector_policy_id=torch.tensor([2, 2, 0, 0, 0, 0, 0, 0, 2]),
        intervention_stage=torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 3], dtype=torch.float32),
    )

    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.arange(9 * 4, dtype=torch.float32).view(9, 4),
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=frame_indices,
        episode_last_frame=8,
        chunk_length=chunk_length,
        frame_stride=1,
        episode_success=True,
        ep_id=61,
        provenance=provenance,
    )

    assert len(transitions) == 3
    assert [int(transition.state_vec[0].item() // 4) for transition in transitions] == [4, 5, 6]
    assert all(
        transition.source.item() == module.TRANSITION_SOURCE_HUMAN_OVERRIDE
        for transition in transitions
    )
    assert all(transition.intervention.item() == 1.0 for transition in transitions)
    assert all(torch.equal(transition.proposal_chunk, transition.ref_chunk) for transition in transitions)
    assert all(torch.equal(transition.bc_target_chunk, transition.exec_chunk) for transition in transitions)


@pytest.mark.parametrize("reason", [1, 2], ids=["corrective", "proactive"])
def test_transition_cache_v2_censors_typed_authority_boundary(reason):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    chunk_length = 2
    frame_indices = list(range(0, 17, 2))
    provenance = module.FrameProvenance(
        is_intervention=torch.tensor(
            [0] * 10 + [1] * 6 + [0],
            dtype=torch.float32,
        ),
        collector_policy_id=torch.tensor(
            [2] * 10 + [0] * 6 + [2]
        ),
        intervention_stage=torch.tensor(
            [0] * 10 + [1, 1] + [2] * 4 + [3],
            dtype=torch.float32,
        ),
        intervention_reason=torch.tensor(
            [0] * 10 + [reason] * 7
        ),
    )

    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.tensor(frame_indices, dtype=torch.float32)
        .unsqueeze(1)
        .repeat(1, 4),
        ref_chunks=torch.randn(len(frame_indices), chunk_length, 2),
        exec_chunks=torch.randn(len(frame_indices), chunk_length, 2),
        frame_indices=frame_indices,
        episode_last_frame=16,
        chunk_length=chunk_length,
        frame_stride=2,
        episode_success=True,
        ep_id=70,
        provenance=provenance,
    )

    policy = [t for t in transitions if int(t.source.item()) == module.TRANSITION_SOURCE_RL_AUTONOMOUS]
    human = [t for t in transitions if int(t.source.item()) == module.TRANSITION_SOURCE_HUMAN_OVERRIDE]
    assert policy
    assert human
    assert [int(t.state_vec[0].item()) for t in policy] == [0, 2, 4, 6, 8]
    assert all(t.critic_mask.item() == 0.0 for t in human)
    assert all(t.actor_q_mask.item() == 0.0 for t in human)
    assert all(t.bootstrap_mask.item() == 0.0 for t in human)
    assert all(torch.equal(t.bc_target_chunk, t.exec_chunk) for t in human)

    boundary = [t for t in policy if t.bootstrap_mask.item() == 0.0]
    assert len(boundary) == 1
    boundary = boundary[0]
    assert boundary.intervention_reason.item() == reason
    assert boundary.done.item() == 0.0
    assert boundary.reward_seq.sum().item() == 0.0
    assert boundary.critic_mask.item() == 0.0
    assert boundary.actor_q_mask.item() == 0.0

    # The earlier autonomous prefix remains a normal Bellman chain. In
    # particular, corrective censoring is no longer encoded as done=1/reward=0.
    earlier = [t for t in policy if t is not boundary]
    assert len(earlier) == 4
    assert all(t.done.item() == 0.0 for t in earlier)
    assert all(t.bootstrap_mask.item() == 1.0 for t in earlier)
    assert all(t.critic_mask.item() == 1.0 for t in earlier)
    assert all(t.actor_q_mask.item() == 1.0 for t in earlier)


def test_transition_cache_v2_resumed_autonomy_reenters_td_and_actor_q():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    chunk_length = 2
    frame_indices = list(range(17))
    provenance = module.FrameProvenance(
        is_intervention=torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32,
        ),
        collector_policy_id=torch.tensor(
            [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2]
        ),
        intervention_stage=torch.tensor(
            [0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 0, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32,
        ),
        intervention_reason=torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        ),
    )

    transitions = module._encoded_episode_to_transitions(
        state_vecs=torch.arange(17 * 4, dtype=torch.float32).view(17, 4),
        ref_chunks=torch.randn(17, chunk_length, 2),
        exec_chunks=torch.randn(17, chunk_length, 2),
        frame_indices=frame_indices,
        episode_last_frame=16,
        chunk_length=chunk_length,
        frame_stride=1,
        episode_success=True,
        ep_id=71,
        provenance=provenance,
    )

    resumed = [
        t
        for t in transitions
        if int(t.source.item()) == module.TRANSITION_SOURCE_RL_AUTONOMOUS
        and int(t.state_vec[0].item() // 4) >= 10
    ]
    assert resumed
    assert all(t.intervention_reason.item() == 0 for t in resumed)
    assert all(t.critic_mask.item() == 1.0 for t in resumed)
    assert all(t.actor_q_mask.item() == 1.0 for t in resumed)
    assert any(t.done.item() == 0.0 and t.bootstrap_mask.item() == 1.0 for t in resumed)
    terminal = [t for t in resumed if t.done.item() == 1.0]
    assert len(terminal) == 1
    assert terminal[0].bootstrap_mask.item() == 0.0
    assert terminal[0].reward_seq.sum().item() == 1.0


def test_transition_cache_v2_stratifies_source_and_outcome_groups():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    class Episodes:
        def __getitem__(self, key):
            values = {
                "dataset_from_index": [0, 2, 4, 6, 8, 10],
                "dataset_to_index": [2, 4, 6, 8, 10, 12],
                "episode_success": [
                    "success",
                    "success",
                    "success",
                    "failure",
                    "success",
                    "failure",
                ],
            }
            return values[key]

    dataset = SimpleNamespace(meta=SimpleNamespace(episodes=Episodes()))
    provenance = module.FrameProvenance(
        is_intervention=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
        collector_policy_id=torch.tensor([0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2]),
    )

    train, val, summary = module._split_episode_indices(
        dataset=dataset,
        n_episodes=6,
        train_ratio=0.5,
        seed=42,
        missing_episode_success="error",
        provenance=provenance,
        stratify_provenance=True,
    )

    assert sorted(train + val) == list(range(6))
    assert set(train).isdisjoint(val)
    assert summary == {
        "demo/success": {"total": 2, "train": 1, "val": 1},
        "online_rl_autonomous/failure": {"total": 2, "train": 1, "val": 1},
        "online_rl_autonomous/success": {"total": 1, "train": 1, "val": 0},
        "online_rl_intervention/success": {"total": 1, "train": 1, "val": 0},
    }


def test_transition_cache_v2_stratification_honors_selected_episode_ids():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    class Episodes:
        def __getitem__(self, key):
            values = {
                "dataset_from_index": [0, 2, 4, 6, 8, 10],
                "dataset_to_index": [2, 4, 6, 8, 10, 12],
                "episode_success": [
                    "success",
                    "success",
                    "success",
                    "failure",
                    "success",
                    "failure",
                ],
            }
            return values[key]

    dataset = SimpleNamespace(meta=SimpleNamespace(episodes=Episodes()))
    provenance = module.FrameProvenance(
        is_intervention=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
        collector_policy_id=torch.tensor([0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2]),
    )

    train, val, summary = module._split_episode_indices(
        dataset=dataset,
        n_episodes=6,
        train_ratio=0.5,
        seed=42,
        missing_episode_success="error",
        provenance=provenance,
        stratify_provenance=True,
        episode_ids=[0, 1, 3, 5],
    )

    assert sorted(train + val) == [0, 1, 3, 5]
    assert set(train).isdisjoint(val)
    assert summary == {
        "demo/success": {"total": 2, "train": 1, "val": 1},
        "online_rl_autonomous/failure": {"total": 2, "train": 1, "val": 1},
    }


def test_transition_cache_v2_extracts_preprocessed_exec_action():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    action = torch.arange(2 * 5 * 4, dtype=torch.float32).view(2, 5, 4)
    exec_chunk = module._extract_exec_chunk({"action": action}, chunk_length=3, action_dim=2)

    assert torch.equal(exec_chunk, action[:, :3, :2])


def test_transition_cache_v2_reads_episode_success_metadata():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    class FakeEpisodes:
        def __init__(self, labels):
            self.labels = labels

        def __getitem__(self, key):
            if key != "episode_success":
                raise KeyError(key)
            return self.labels

    dataset = SimpleNamespace(meta=SimpleNamespace(episodes=FakeEpisodes(["success", "failure"])))

    assert module._episode_success_from_metadata(dataset, 0, "error") is True
    assert module._episode_success_from_metadata(dataset, 1, "error") is False


def test_transition_cache_v2_rejects_missing_episode_success_by_default(monkeypatch):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_transition_cache_v2.py",
            "--demo-dataset-repo-id",
            "local/demo",
            "--demo-dataset-root",
            "/tmp/demo",
            "--rl-token-policy-path",
            "/tmp/rl-token",
            "--vla-pretrained-path",
            "/tmp/vla",
            "--output-dir",
            "/tmp/cache",
        ],
    )

    args = module.parse_args()

    assert args.missing_episode_success == "error"
    assert args.trim_leading_idle is False
    assert args.trim_leading_idle_threshold == pytest.approx(1.0)
    assert args.trim_leading_idle_hold_frames == 5
    assert args.trim_leading_idle_pre_roll_frames is None
    assert args.trim_leading_idle_action_dims == 5


def test_transition_cache_v2_detects_sustained_motion_and_ignores_gripper_spike():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    actions = torch.zeros(14, 6)
    actions[2, 0] = 3.0  # One-frame arm spike must not count as sustained motion.
    actions[5:, 5] = 20.0  # Gripper is outside the default five onset dimensions.
    actions[8:, 1] = 2.0

    onset = module._find_leading_motion_onset(
        actions,
        threshold=1.0,
        hold_frames=3,
        action_dims=5,
    )

    assert onset == 8


def test_transition_cache_v2_trims_to_motion_onset_minus_pre_roll():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    all_actions = [torch.zeros(6) for _ in range(120)]
    for frame in range(108, 120):
        all_actions[frame][0] = 2.0

    class FakeHFDataset:
        column_names = ["action"]

        def __getitem__(self, key):
            if key != "action":
                raise KeyError(key)
            return all_actions

    dataset = SimpleNamespace(hf_dataset=FakeHFDataset())

    trimmed_start, onset = module._trimmed_episode_start(
        dataset,
        episode_id=7,
        episode_start=100,
        episode_stop=120,
        threshold=1.0,
        hold_frames=3,
        pre_roll_frames=4,
        action_dims=5,
    )

    assert onset == 108
    assert trimmed_start == 104


def test_transition_cache_v2_rejects_episode_without_sustained_motion():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    class FakeHFDataset:
        column_names = ["action"]

        def __getitem__(self, key):
            if key != "action":
                raise KeyError(key)
            return [torch.zeros(6) for _ in range(12)]

    dataset = SimpleNamespace(hf_dataset=FakeHFDataset())

    with pytest.raises(ValueError, match="Episode 3 has no sustained motion onset"):
        module._trimmed_episode_start(
            dataset,
            episode_id=3,
            episode_start=0,
            episode_stop=12,
            threshold=1.0,
            hold_frames=5,
            pre_roll_frames=10,
            action_dims=5,
        )


def test_transition_cache_v2_uses_cli_chunk_length_for_frame_indices(monkeypatch, tmp_path):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    captured = {}

    class FakeEpisodes:
        def __getitem__(self, key):
            values = {
                "dataset_from_index": [0],
                "dataset_to_index": [14],
                "episode_success": ["success"],
            }
            if key not in values:
                raise KeyError(key)
            return values[key]

    class FakeDataset:
        def __init__(self, **kwargs):
            captured["delta_timestamps"] = kwargs["delta_timestamps"]
            self.num_episodes = 1
            self.meta = SimpleNamespace(episodes=FakeEpisodes())

    class FakeMetadata:
        fps = 25

        def __init__(self, **kwargs):
            pass

    class FakePolicy:
        config = SimpleNamespace(
            action_dim=12,
            chunk_size=50,
            image_only=False,
            proprio_dim=12,
            token_pool_size=0,
            vla_pretrained_path=None,
        )
        _num_image_tokens = 0
        _pi05 = object()
        rl_token = object()

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeCapture:
        def __init__(self, **kwargs):
            pass

        def attach(self, pi05):
            pass

        def detach(self):
            pass

    def fake_encode_episode(**kwargs):
        captured["frame_indices"] = kwargs["frame_indices"]
        captured["demo_reference_mode"] = kwargs["demo_reference_mode"]
        return []

    monkeypatch.setattr(module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(module, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(
        module.PreTrainedConfig,
        "from_pretrained",
        lambda path, cli_overrides: FakePolicy.config,
    )
    monkeypatch.setattr(
        module.RLTokenPolicy,
        "from_pretrained",
        lambda path, config: FakePolicy(),
    )
    monkeypatch.setattr(module, "PrefixOutputCapture", FakeCapture)
    monkeypatch.setattr(module, "make_rlt_token_pre_post_processors", lambda config: (object(), object()))
    monkeypatch.setattr(module, "_encode_episode", fake_encode_episode)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_transition_cache_v2.py",
            "--demo-dataset-repo-id",
            "local/demo",
            "--demo-dataset-root",
            "/tmp/demo",
            "--rl-token-policy-path",
            "/tmp/rl-token",
            "--vla-pretrained-path",
            "/tmp/vla",
            "--output-dir",
            str(tmp_path),
            "--chunk-length",
            "10",
            "--frame-stride",
            "2",
            "--demo-reference-mode",
            "executed",
            "--train-ratio",
            "1.0",
            "--device",
            "cpu",
        ],
    )

    module.main()

    assert captured["delta_timestamps"]["action"] == [i / 25.0 for i in range(10)]
    assert captured["frame_indices"] == [0, 2, 3, 4, 6, 8, 10, 12, 13]
    assert captured["demo_reference_mode"] == "executed"
