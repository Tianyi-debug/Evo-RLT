from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from evo_rlt.cli.train_task_weighted import (
    TaskSamplingConfig,
    _TaskWeightedDataLoaderFactory,
    build_frame_sampling_weights,
    parse_task_sampling_weights,
    parse_wrapper_args,
)


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, task_indices: list[int]):
        self.task_indices = task_indices
        self.hf_dataset = {"task_index": task_indices}

    def __len__(self) -> int:
        return len(self.task_indices)

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index, "task_index": self.task_indices[index]}


@dataclass
class _IndexSampler:
    indices: list[int]


def test_parse_task_sampling_weights_normalizes_csv_and_json() -> None:
    assert parse_task_sampling_weights("3,5,2") == pytest.approx((0.3, 0.5, 0.2))
    assert parse_task_sampling_weights("[0.3, 0.5, 0.2]") == pytest.approx((0.3, 0.5, 0.2))


def test_parse_wrapper_args_removes_only_wrapper_options() -> None:
    config, remaining = parse_wrapper_args(
        [
            "--dataset.root=/tmp/data",
            "--task-sampling-weights=0.3,0.5,0.2",
            "--task-sampling-seed",
            "42",
            "--steps=100",
        ]
    )
    assert config.weights == pytest.approx((0.3, 0.5, 0.2))
    assert config.seed == 42
    assert remaining == ["--dataset.root=/tmp/data", "--steps=100"]


def test_frame_weights_give_each_task_requested_probability_mass() -> None:
    dataset = _ToyDataset([0, 0, 0, 0, 1, 1, 1, 2, 2])
    eligible = [0, 1, 2, 4, 5, 7]
    weights, counts = build_frame_sampling_weights(dataset, eligible, (0.3, 0.5, 0.2))

    assert counts == {0: 3, 1: 2, 2: 1}
    assert weights[[0, 1, 2]].sum().item() == pytest.approx(0.3)
    assert weights[[4, 5]].sum().item() == pytest.approx(0.5)
    assert weights[7].item() == pytest.approx(0.2)
    assert weights[[3, 6, 8]].sum().item() == 0


def test_dataloader_factory_respects_eligible_indices_and_sampling_ratio() -> None:
    dataset = _ToyDataset([0] * 20 + [1] * 10 + [2] * 5)
    eligible = list(range(18)) + list(range(20, 29)) + list(range(30, 34))
    factory = _TaskWeightedDataLoaderFactory(
        torch.utils.data.DataLoader,
        TaskSamplingConfig(weights=(0.3, 0.5, 0.2), seed=7, num_samples=20_000),
    )

    loader = factory(
        dataset,
        batch_size=200,
        shuffle=False,
        sampler=_IndexSampler(eligible),
    )
    observed = torch.cat([batch["task_index"] for batch in loader])
    ratios = torch.bincount(observed, minlength=3).float() / observed.numel()

    assert ratios.tolist() == pytest.approx([0.3, 0.5, 0.2], abs=0.015)
    sampled_indices = torch.cat([batch["index"] for batch in loader]).tolist()
    assert set(sampled_indices).issubset(set(eligible))


def test_patched_init_keeps_dataloader_usable_as_isinstance_type() -> None:
    dataset = _ToyDataset([0, 0, 1, 1, 2, 2])
    dataloader_type = torch.utils.data.DataLoader
    original_init = dataloader_type.__init__
    factory = _TaskWeightedDataLoaderFactory(
        dataloader_type,
        TaskSamplingConfig(weights=(0.3, 0.5, 0.2), seed=7),
    )

    dataloader_type.__init__ = factory.make_init_patch()
    try:
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

        # This is the exact type check Accelerate performs in _prepare_one.
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert isinstance(loader.sampler, torch.utils.data.WeightedRandomSampler)
    finally:
        dataloader_type.__init__ = original_init


def test_missing_task_is_rejected_instead_of_silently_renormalized() -> None:
    dataset = _ToyDataset([0, 0, 2, 2])
    with pytest.raises(ValueError, match="exactly match"):
        build_frame_sampling_weights(dataset, range(len(dataset)), (0.3, 0.5, 0.2))
