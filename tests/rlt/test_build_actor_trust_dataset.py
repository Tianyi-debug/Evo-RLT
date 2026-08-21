from pathlib import Path

import pytest
import torch

from evo_rlt.cli.build_actor_trust_dataset import (
    AUTONOMOUS_FAILURE,
    AUTONOMOUS_SUCCESS,
    CORRECTIVE,
    PROACTIVE,
    build_actor_trust_dataset,
    build_samples,
    load_episode_groups,
    split_episode_groups,
)


def _row(
    episode_id: int,
    source: int,
    *,
    reason: int = 0,
    bc_valid: bool = False,
    value: float = 0.0,
) -> dict:
    return {
        "episode_id": torch.tensor(episode_id),
        "source": torch.tensor(source),
        "intervention_reason": torch.tensor(reason),
        "actor_bc_mask": torch.tensor(float(bc_valid)),
        "state_vec": torch.arange(10, dtype=torch.float32) + value,
        "proposal_chunk": torch.full((2, 2), value),
        "exec_chunk": torch.full((2, 2), value + 1),
        "bc_target_chunk": torch.full((2, 2), value + 1),
        "exec_action_is_actual_sent": torch.tensor(float(episode_id % 2 == 0)),
        "anchor_start_frame": torch.tensor(-1),
        "frame_stride": torch.tensor(2),
        "fps": torch.tensor(25.0),
    }


def _write_cache(root: Path) -> None:
    rows = []
    rows += [_row(0, 2, reason=1, value=index) for index in range(8)]
    rows += [_row(0, 3, reason=1, bc_valid=True, value=20 + index) for index in range(4)]
    rows += [_row(1, 2, reason=2, value=40 + index) for index in range(7)]
    rows += [_row(1, 3, reason=2, bc_valid=True, value=60 + index) for index in range(4)]
    rows += [_row(2, 2, bc_valid=True, value=80 + index) for index in range(10)]
    rows += [_row(3, 2, bc_valid=False, value=100 + index) for index in range(9)]
    rows += [_row(4, 0, bc_valid=True, value=120 + index) for index in range(5)]
    episode_offsets: dict[int, int] = {}
    for row in rows:
        episode_id = int(row["episode_id"].item())
        offset = episode_offsets.get(episode_id, 0)
        row["anchor_start_frame"] = torch.tensor(offset * 2)
        episode_offsets[episode_id] = offset + 1
    torch.save(rows[:25], root / "chunk_transitions_train.pt")
    torch.save(rows[25:], root / "chunk_transitions_val.pt")


def test_future_k_samples_respect_typed_takeover_semantics(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache)
    groups = load_episode_groups([cache])
    assert {group.category for group in groups} == {
        CORRECTIVE,
        PROACTIVE,
        AUTONOMOUS_SUCCESS,
        AUTONOMOUS_FAILURE,
    }

    samples = build_samples(groups, ks=[1, 3, 5], primary_k=3, proprio_dim=2)
    by_category = {
        category: [sample for sample in samples if sample["category"] == category]
        for category in {CORRECTIVE, PROACTIVE, AUTONOMOUS_SUCCESS, AUTONOMOUS_FAILURE}
    }
    corrective = by_category[CORRECTIVE]
    assert sum(sample["label"].item() for sample in corrective) == 3
    assert all(sample["label_mask"].item() == 1 for sample in corrective)
    proactive = by_category[PROACTIVE]
    assert sum(sample["censored"].item() for sample in proactive) == 3
    assert sum(sample["label_mask"].item() for sample in proactive) == 4
    success = by_category[AUTONOMOUS_SUCCESS]
    assert all(sample["label"].item() == 0 for sample in success)
    assert all(sample["label_mask"].item() == 1 for sample in success)
    failure = by_category[AUTONOMOUS_FAILURE]
    assert all(sample["label_mask"].item() == 0 for sample in failure)
    assert samples[0]["z_rl"].shape == (8,)
    assert samples[0]["proprio"].shape == (2,)
    assert samples[0]["action_semantics"] in {
        "actual_sent",
        "requested_or_pre_clipping_legacy",
    }


def test_split_is_episode_safe_and_output_refuses_overwrite(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_cache(cache)
    groups = load_episode_groups([cache])
    train, val = split_episode_groups(groups, val_fraction=0.5, seed=1000)
    assert {group.uid for group in train}.isdisjoint(group.uid for group in val)

    output = tmp_path / "actor_trust"
    metadata = build_actor_trust_dataset(
        cache_roots=[cache],
        output_dir=output,
        ks=[1, 3, 5],
        primary_k=3,
        proprio_dim=2,
        val_fraction=0.5,
        split_seed=1000,
    )
    assert metadata["dimensions"]["z_rl_dim"] == 8
    assert metadata["semantics"]["frame_stride"] == 2
    assert metadata["semantics"]["fps"] == pytest.approx(25.0)
    assert metadata["semantics"]["horizon_seconds"]["k3"] == pytest.approx(0.24)
    assert metadata["inputs"][0]["cache_sha256"]
    assert (output / "actor_trust_train.pt").is_file()
    assert (output / "actor_trust_val.pt").is_file()
    assert (output / "human_audit_val.pt").is_file()
    assert (output / "metadata.json").is_file()
    with pytest.raises(FileExistsError):
        build_actor_trust_dataset(
            cache_roots=[cache],
            output_dir=output,
            ks=[3],
            primary_k=3,
            proprio_dim=2,
            val_fraction=0.5,
            split_seed=1000,
        )


def test_duplicate_cache_contents_are_rejected(tmp_path: Path):
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"
    cache_a.mkdir()
    cache_b.mkdir()
    _write_cache(cache_a)
    for filename in ("chunk_transitions_train.pt", "chunk_transitions_val.pt"):
        (cache_b / filename).write_bytes((cache_a / filename).read_bytes())

    with pytest.raises(ValueError, match="duplicate transition-cache contents"):
        load_episode_groups([cache_a, cache_b])
