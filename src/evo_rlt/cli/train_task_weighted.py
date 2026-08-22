"""Run LeRobot training with explicit task-balanced frame sampling.

LeRobot 0.5.1 accepts one dataset per training run and samples its frames
uniformly.  After several task datasets are aggregated, that makes the task
probability proportional to the number of frames in each task.  This wrapper
keeps the upstream training entrypoint unchanged while replacing its sampler
with a :class:`~torch.utils.data.WeightedRandomSampler` whose total probability
mass is specified per ``task_index``.

All arguments other than the ``--task-sampling-*`` options are passed through
to ``lerobot.scripts.lerobot_train`` unchanged.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSamplingConfig:
    weights: tuple[float, ...]
    seed: int = 1000
    num_samples: int | None = None
    audit_path: Path | None = None


def parse_task_sampling_weights(value: str) -> tuple[float, ...]:
    """Parse either ``0.3,0.5,0.2`` or ``[0.3, 0.5, 0.2]`` and normalize it."""

    text = value.strip()
    if not text:
        raise ValueError("task sampling weights must not be empty")

    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON task sampling weights must be a list")
        raw = parsed
    else:
        raw = [item.strip() for item in text.split(",")]

    try:
        weights = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid task sampling weights: {value!r}") from exc

    if not weights:
        raise ValueError("at least one task sampling weight is required")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError(f"all task sampling weights must be finite and positive, got {weights}")

    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"task_index tensor must contain one value, got shape {tuple(value.shape)}")
        return int(value.item())
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(f"task_index sequence must contain one value, got {value!r}")
        value = value[0]
    return int(value)


def _task_indices(dataset: Any) -> list[int]:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        raise TypeError(
            "task-weighted training requires a LeRobotDataset with an hf_dataset task_index column"
        )
    try:
        values = hf_dataset["task_index"]
    except (KeyError, TypeError) as exc:
        raise ValueError("dataset does not contain the required task_index column") from exc
    task_indices = [_as_int(value) for value in values]
    if len(task_indices) != len(dataset):
        raise ValueError(
            f"task_index column length {len(task_indices)} does not match dataset length {len(dataset)}"
        )
    return task_indices


def build_frame_sampling_weights(
    dataset: Any,
    eligible_indices: Sequence[int],
    target_task_weights: Sequence[float],
) -> tuple[torch.Tensor, dict[int, int]]:
    """Build per-frame weights with the requested probability mass per task.

    Frames excluded by an upstream episode-aware sampler receive zero weight,
    so policies that drop terminal frames retain that behavior.
    """

    task_indices = _task_indices(dataset)
    eligible = [int(index) for index in eligible_indices]
    if not eligible:
        raise ValueError("no eligible dataset frames remain for task-weighted sampling")
    if min(eligible) < 0 or max(eligible) >= len(dataset):
        raise IndexError("eligible sampler indices fall outside the dataset")

    counts = Counter(task_indices[index] for index in eligible)
    expected = set(range(len(target_task_weights)))
    observed = set(counts)
    if observed != expected:
        raise ValueError(
            "task_index values must exactly match the weight positions: "
            f"expected {sorted(expected)}, observed {sorted(observed)}"
        )

    weights = torch.zeros(len(dataset), dtype=torch.double)
    for index in eligible:
        task_index = task_indices[index]
        weights[index] = float(target_task_weights[task_index]) / counts[task_index]
    return weights, dict(sorted(counts.items()))


def _pop_option(argv: list[str], name: str) -> tuple[str | None, list[str]]:
    result: list[str] = []
    found: str | None = None
    index = 0
    prefix = f"{name}="
    while index < len(argv):
        argument = argv[index]
        if argument.startswith(prefix):
            if found is not None:
                raise ValueError(f"{name} may only be specified once")
            found = argument[len(prefix) :]
        elif argument == name:
            if found is not None:
                raise ValueError(f"{name} may only be specified once")
            if index + 1 >= len(argv):
                raise ValueError(f"{name} requires a value")
            index += 1
            found = argv[index]
        else:
            result.append(argument)
        index += 1
    return found, result


def parse_wrapper_args(argv: Sequence[str]) -> tuple[TaskSamplingConfig, list[str]]:
    remaining = list(argv)
    weights_value, remaining = _pop_option(remaining, "--task-sampling-weights")
    seed_value, remaining = _pop_option(remaining, "--task-sampling-seed")
    num_samples_value, remaining = _pop_option(remaining, "--task-sampling-num-samples")
    audit_path_value, remaining = _pop_option(remaining, "--task-sampling-audit-path")

    if weights_value is None:
        raise ValueError("--task-sampling-weights is required")

    seed = 1000 if seed_value is None else int(seed_value)
    num_samples = None if num_samples_value is None else int(num_samples_value)
    if num_samples is not None and num_samples <= 0:
        raise ValueError("--task-sampling-num-samples must be positive")

    return (
        TaskSamplingConfig(
            weights=parse_task_sampling_weights(weights_value),
            seed=seed,
            num_samples=num_samples,
            audit_path=None if audit_path_value is None else Path(audit_path_value),
        ),
        remaining,
    )


class _TaskWeightedDataLoaderFactory:
    def __init__(self, original: Any, config: TaskSamplingConfig):
        self.original = original
        self.config = config
        self.applied = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.applied:
            return self.original(*args, **kwargs)

        dataset = args[0] if args else kwargs.get("dataset")
        if dataset is None:
            raise TypeError("DataLoader dataset argument is missing")

        existing_sampler = kwargs.get("sampler")
        if existing_sampler is None:
            eligible_indices = list(range(len(dataset)))
        elif hasattr(existing_sampler, "indices"):
            eligible_indices = list(existing_sampler.indices)
        else:
            raise TypeError(
                "cannot preserve an existing sampler without an indices attribute in task-weighted training"
            )

        frame_weights, counts = build_frame_sampling_weights(
            dataset,
            eligible_indices,
            self.config.weights,
        )
        num_samples = self.config.num_samples or len(eligible_indices)
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            frame_weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )

        kwargs["shuffle"] = False
        kwargs["sampler"] = weighted_sampler
        self.applied = True

        summary = {
            "task_sampling_weights": list(self.config.weights),
            "eligible_frames_per_task": {str(key): value for key, value in counts.items()},
            "eligible_frames_total": len(eligible_indices),
            "sampler_num_samples": num_samples,
            "replacement": True,
            "seed": self.config.seed,
        }
        logger.info("Task-weighted sampler: %s", json.dumps(summary, sort_keys=True))
        if self.config.audit_path is not None:
            self.config.audit_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.audit_path.write_text(json.dumps(summary, indent=2) + "\n")

        return self.original(*args, **kwargs)


def main() -> None:
    config, passthrough = parse_wrapper_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *passthrough]

    from evo_rlt.adapters.lerobot import register

    register()

    from lerobot.scripts import lerobot_train

    original_dataloader = torch.utils.data.DataLoader
    factory = _TaskWeightedDataLoaderFactory(original_dataloader, config)
    torch.utils.data.DataLoader = factory
    try:
        lerobot_train.main()
    finally:
        torch.utils.data.DataLoader = original_dataloader


if __name__ == "__main__":
    main()
