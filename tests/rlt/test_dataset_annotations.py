from __future__ import annotations

import json

import pandas as pd

from evo_rlt.adapters.lerobot.dataset_annotations import (
    harmonize_collector_policy_id_codebooks,
    repair_collector_policy_id_codebook,
)


def _write_dataset(root, collector_ids, codebook):
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    pd.DataFrame(
        {
            "complementary_info.collector_policy_id": [[value] for value in collector_ids],
            "index": list(range(len(collector_ids))),
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    info = {
        "total_frames": len(collector_ids),
        "features": {
            "complementary_info.collector_policy_id": {
                "dtype": "int64",
                "shape": [1],
                "names": ["collector_policy_id"],
                "info": {"codebook": codebook},
            }
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))


def _read_codebook(root):
    info = json.loads((root / "meta" / "info.json").read_text())
    return info["features"]["complementary_info.collector_policy_id"]["info"]["codebook"]


def test_repair_codebook_adds_observed_rlt_actor(tmp_path):
    root = tmp_path / "online"
    _write_dataset(root, [2, 2, 0], {"0": "human", "1": "pretrained_model"})

    result = repair_collector_policy_id_codebook(root, write=True, backup=True)

    assert result is not None
    assert result.changed is True
    assert result.observed_ids == (0, 2)
    assert result.backup_path is not None
    assert result.backup_path.exists()
    assert _read_codebook(root) == {
        "0": "human",
        "1": "pi0.5",
        "2": "pi_rlt_actor",
    }


def test_harmonize_installs_one_codebook_for_expert_and_online(tmp_path):
    expert = tmp_path / "expert"
    online = tmp_path / "online"
    _write_dataset(expert, [0, 0], {"0": "human"})
    _write_dataset(online, [2, 0], {"0": "human", "1": "pretrained_model"})

    merged = harmonize_collector_policy_id_codebooks([expert, online], backup=False)

    expected = {"0": "human", "1": "pi0.5", "2": "pi_rlt_actor"}
    assert merged == expected
    assert _read_codebook(expert) == expected
    assert _read_codebook(online) == expected
