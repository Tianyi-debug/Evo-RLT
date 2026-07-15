from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch


def test_transition_cache_v2_passes_video_backend(monkeypatch, tmp_path):
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_episodes = 0
            self.meta = SimpleNamespace(episodes=None)

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
    monkeypatch.setattr(module.RLTokenPolicy, "from_pretrained", lambda path: FakePolicy())
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

    transitions = module._encoded_episode_to_transition_dicts(
        state_vecs=state_vecs,
        ref_chunks=ref_chunks,
        exec_chunks=exec_chunks,
        frame_indices=frame_indices,
        episode_last_frame=4,
        chunk_length=C,
        frame_stride=1,
        episode_success=True,
        ep_id=7,
    )

    assert len(transitions) == 2
    assert torch.equal(transitions[0]["exec_chunk"], exec_chunks[0])
    assert torch.equal(transitions[0]["ref_chunk"], ref_chunks[0])
    assert not torch.equal(transitions[0]["exec_chunk"], transitions[0]["ref_chunk"])
    assert torch.equal(transitions[0]["next_state_vec"], state_vecs[3])
    assert torch.equal(transitions[1]["next_state_vec"], state_vecs[4])
    assert transitions[0]["done"].item() == 0.0
    assert transitions[1]["done"].item() == 1.0
    assert torch.equal(transitions[0]["reward_seq"], torch.zeros(C))
    assert torch.equal(transitions[1]["reward_seq"], torch.tensor([0.0, 0.0, 1.0]))
    assert transitions[1]["episode_id"].item() == 7


def test_transition_cache_v2_semantic_builder_zero_reward_on_failure():
    module = pytest.importorskip("evo_rlt.cli.build_transition_cache_v2")

    C = 3
    frame_indices = [0, 1, 2, 3, 4]
    transitions = module._encoded_episode_to_transition_dicts(
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

    assert transitions[-1]["done"].item() == 1.0
    assert all(t["reward_seq"].sum().item() == pytest.approx(0.0) for t in transitions)


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
        return []

    monkeypatch.setattr(module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(module.RLTokenPolicy, "from_pretrained", lambda path: FakePolicy())
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
            "--train-ratio",
            "1.0",
            "--device",
            "cpu",
        ],
    )

    module.main()

    assert captured["delta_timestamps"]["action"] == [i / 30.0 for i in range(10)]
    assert captured["frame_indices"] == [0, 2, 3, 4, 6, 8, 10, 12, 13]
