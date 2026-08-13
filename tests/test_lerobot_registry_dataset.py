from types import SimpleNamespace

import torch

import evo_rlt.adapters.lerobot.registry as registry
from evo_rlt.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset


def test_register_loads_chunk_transition_dataset_from_cache_dir(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save(
        [
            {
                "state_vec": torch.zeros(3),
                "exec_chunk": torch.zeros(2, 1),
            }
        ],
        cache_dir / "chunk_transitions_train.pt",
    )

    import lerobot.datasets.factory as dataset_factory

    registry._REGISTERED = False
    registry.register()

    cfg = SimpleNamespace(dataset=SimpleNamespace(repo_id=str(cache_dir)))
    dataset = dataset_factory.make_dataset(cfg)

    assert isinstance(dataset, ChunkTransitionDataset)
    assert len(dataset) == 1
    assert dataset[0]["state_vec"].shape == (3,)


def test_register_forwards_two_stage_dataset_settings(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save(
        [
            {"state_vec": torch.zeros(3), "source": torch.tensor(1)},
            {"state_vec": torch.ones(3), "source": torch.tensor(3)},
        ],
        cache_dir / "chunk_transitions_train.pt",
    )

    import lerobot.datasets.factory as dataset_factory

    registry._REGISTERED = False
    registry.register()
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(repo_id=str(cache_dir)),
        policy=SimpleNamespace(
            training_stage="human_bc",
            source_sampling_weights=[0.0, 0.0, 0.0, 1.0],
            source_sampling_seed=9,
        ),
    )

    dataset = dataset_factory.make_dataset(cfg)

    assert len(dataset) == 1
    assert dataset[0]["source"].item() == 3
