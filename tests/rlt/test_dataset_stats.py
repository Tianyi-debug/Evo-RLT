from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evo_rlt.adapters.lerobot.dataset_stats import recompute_numeric_dataset_stats


def _write_test_dataset(root):
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    episode_0 = np.arange(10, dtype=np.float32)
    episode_1 = np.arange(100, 110, dtype=np.float32)
    values = np.concatenate([episode_0, episode_1])
    frame_index = np.tile(np.arange(10, dtype=np.int64), 2)
    episode_index = np.repeat(np.arange(2, dtype=np.int64), 10)
    pd.DataFrame(
        {
            "action": [[value, -value] for value in values],
            "observation.state": [[value + 1, -value - 1] for value in values],
            "timestamp": frame_index.astype(np.float32) / 30.0,
            "frame_index": frame_index,
            "episode_index": episode_index,
            "index": np.arange(20, dtype=np.int64),
            "task_index": np.zeros(20, dtype=np.int64),
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": 20,
        "total_tasks": 1,
        "fps": 30,
        "features": {
            "action": {"dtype": "float32", "shape": [2], "names": None},
            "observation.state": {"dtype": "float32", "shape": [2], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.images.top": {
                "dtype": "video",
                "shape": [8, 8, 3],
                "names": ["height", "width", "channels"],
            },
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    inaccurate_stats = {
        "action": {
            "min": [0.0, -109.0],
            "max": [109.0, 0.0],
            "mean": [54.5, -54.5],
            "std": [50.0, 50.0],
            "count": [20],
            "q01": [50.0, -59.0],
            "q10": [50.0, -59.0],
            "q50": [54.5, -54.5],
            "q90": [59.0, -50.0],
            "q99": [59.0, -50.0],
        },
        "observation.images.top": {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.1]], [[0.1]], [[0.1]]],
            "count": [20],
            "q01": [[[0.0]], [[0.0]], [[0.0]]],
            "q10": [[[0.1]], [[0.1]], [[0.1]]],
            "q50": [[[0.5]], [[0.5]], [[0.5]]],
            "q90": [[[0.9]], [[0.9]], [[0.9]]],
            "q99": [[[1.0]], [[1.0]], [[1.0]]],
        },
    }
    (root / "meta" / "stats.json").write_text(json.dumps(inaccurate_stats))


def test_recompute_uses_all_frames_and_preserves_video_stats(tmp_path):
    root = tmp_path / "dataset"
    _write_test_dataset(root)

    result = recompute_numeric_dataset_stats(root, backup=True, write=True)

    expected_action = np.stack(
        [
            np.concatenate([np.arange(10), np.arange(100, 110)]),
            -np.concatenate([np.arange(10), np.arange(100, 110)]),
        ],
        axis=1,
    )
    assert np.allclose(result.stats["action"]["q01"], np.quantile(expected_action, 0.01, axis=0))
    assert np.allclose(result.stats["action"]["q99"], np.quantile(expected_action, 0.99, axis=0))
    assert result.stats["index"]["max"].tolist() == [19]
    assert result.stats["episode_index"]["max"].tolist() == [1]
    assert result.stats["observation.images.top"]["std"].reshape(-1).tolist() == [0.1, 0.1, 0.1]
    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert np.all(result.validation["action"].outside_q01_q99_pct <= 10.0)


def test_check_only_does_not_replace_stats(tmp_path):
    root = tmp_path / "dataset"
    _write_test_dataset(root)
    original = (root / "meta" / "stats.json").read_bytes()

    result = recompute_numeric_dataset_stats(root, backup=True, write=False)

    assert (root / "meta" / "stats.json").read_bytes() == original
    assert result.backup_path is None
    assert result.stats["index"]["max"].tolist() == [19]

