from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


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
