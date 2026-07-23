from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
NON_NUMERIC_DTYPES = {"image", "video", "string", "language"}


@dataclass(frozen=True)
class FeatureStatsValidation:
    key: str
    below_q01_pct: np.ndarray
    above_q99_pct: np.ndarray

    @property
    def outside_q01_q99_pct(self) -> np.ndarray:
        return self.below_q01_pct + self.above_q99_pct


@dataclass(frozen=True)
class DatasetStatsResult:
    root: Path
    total_frames: int
    recomputed_features: tuple[str, ...]
    preserved_features: tuple[str, ...]
    stats: dict[str, dict[str, np.ndarray]]
    validation: dict[str, FeatureStatsValidation]
    backup_path: Path | None = None


def _data_parquet_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")
    return paths


def _numeric_feature_keys(info: dict, parquet_columns: set[str]) -> list[str]:
    keys = []
    for key, feature in info["features"].items():
        if key not in parquet_columns:
            continue
        if feature.get("dtype") in NON_NUMERIC_DTYPES:
            continue
        keys.append(key)
    return keys


def _column_to_matrix(column, *, key: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    values = column.combine_chunks().to_pylist()
    if not values:
        raise ValueError(f"Feature {key!r} has no values")

    array = np.asarray(values)
    if array.dtype == object:
        array = np.stack([np.asarray(value) for value in values])

    expected_width = int(np.prod(expected_shape, dtype=np.int64)) if expected_shape else 1
    try:
        matrix = array.reshape(len(values), expected_width)
    except ValueError as exc:
        raise ValueError(
            f"Feature {key!r} cannot be reshaped from {array.shape} "
            f"to ({len(values)}, {expected_width})"
        ) from exc
    if not np.issubdtype(matrix.dtype, np.number) and matrix.dtype != np.bool_:
        raise TypeError(f"Feature {key!r} is not numeric: dtype={matrix.dtype}")
    return matrix


def _compute_matrix_stats(
    matrix: np.ndarray,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[str, np.ndarray]:
    if matrix.shape[0] < 2:
        raise ValueError("At least two frames are required to compute dataset statistics")

    stats: dict[str, np.ndarray] = {
        "min": np.min(matrix, axis=0),
        "max": np.max(matrix, axis=0),
        "mean": np.mean(matrix, axis=0, dtype=np.float64),
        "std": np.std(matrix, axis=0, dtype=np.float64),
        "count": np.array([matrix.shape[0]], dtype=np.int64),
    }
    quantile_input = matrix.astype(np.float64, copy=False) if matrix.dtype == np.bool_ else matrix
    quantile_values = np.quantile(quantile_input, quantiles, axis=0)
    for quantile, values in zip(quantiles, quantile_values, strict=True):
        stats[f"q{int(round(quantile * 100)):02d}"] = np.asarray(values)
    return stats


def _validate_indices(table, info: dict, total_frames: int) -> None:
    def scalar_column(key: str) -> np.ndarray:
        return np.asarray(table[key].combine_chunks().to_numpy())

    if "index" in table.column_names:
        index = scalar_column("index")
        expected = np.arange(total_frames, dtype=index.dtype)
        if not np.array_equal(index, expected):
            raise ValueError("Dataset `index` is not the contiguous range [0, total_frames)")

    if "episode_index" in table.column_names:
        episode_index = scalar_column("episode_index")
        unique = np.unique(episode_index)
        expected = np.arange(int(info["total_episodes"]), dtype=unique.dtype)
        if not np.array_equal(unique, expected):
            raise ValueError(
                "Dataset `episode_index` does not match the contiguous range "
                f"[0, {info['total_episodes']})"
            )


def _atomic_write_stats(stats: dict[str, dict[str, np.ndarray]], root: Path) -> None:
    from lerobot.datasets.io_utils import write_stats

    temp_root = Path(tempfile.mkdtemp(prefix=".evo-rlt-stats-", dir=root))
    try:
        write_stats(stats, temp_root)
        source = temp_root / "meta" / "stats.json"
        target = root / "meta" / "stats.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _backup_stats(root: Path) -> Path | None:
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = stats_path.with_name(f"stats.before-global-recompute-{timestamp}.json")
    suffix = 1
    while backup_path.exists():
        backup_path = stats_path.with_name(
            f"stats.before-global-recompute-{timestamp}-{suffix}.json"
        )
        suffix += 1
    shutil.copy2(stats_path, backup_path)
    return backup_path


def recompute_numeric_dataset_stats(
    root: str | Path,
    *,
    backup: bool = False,
    write: bool = True,
) -> DatasetStatsResult:
    """Recompute exact full-dataset statistics for every numeric parquet feature.

    LeRobot v0.5.1 computes per-episode quantiles and combines them with a
    count-weighted average. A weighted average of quantiles is not the quantile
    of the combined frames. This function loads the final data parquet shards
    and computes numeric statistics over all final parquet frames.

    Image/video statistics are preserved because decoding every frame is much
    more expensive and the supported Pi0.5/SmolVLA policies use IDENTITY visual
    normalization.
    """
    import pyarrow.parquet as pq
    from lerobot.datasets.io_utils import load_info, load_stats

    dataset_root = Path(root).expanduser().resolve()
    info = load_info(dataset_root)
    paths = _data_parquet_paths(dataset_root)
    parquet_columns = set(pq.read_schema(paths[0]).names)
    numeric_keys = _numeric_feature_keys(info, parquet_columns)
    table = pq.read_table(paths, columns=numeric_keys)
    total_frames = table.num_rows
    if total_frames != int(info["total_frames"]):
        raise ValueError(
            f"info.json total_frames={info['total_frames']} but parquet has {total_frames} rows"
        )
    _validate_indices(table, info, total_frames)

    existing_stats = load_stats(dataset_root) or {}
    new_stats = dict(existing_stats)
    matrices: dict[str, np.ndarray] = {}
    for key in numeric_keys:
        shape = tuple(info["features"][key].get("shape") or ())
        matrix = _column_to_matrix(table[key], key=key, expected_shape=shape)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Feature {key!r} contains NaN or infinity")
        matrices[key] = matrix
        new_stats[key] = _compute_matrix_stats(matrix)

    validation = {}
    for key in ("action", "observation.state"):
        if key not in matrices:
            continue
        matrix = matrices[key]
        q01 = new_stats[key]["q01"]
        q99 = new_stats[key]["q99"]
        validation[key] = FeatureStatsValidation(
            key=key,
            below_q01_pct=np.mean(matrix < q01, axis=0) * 100.0,
            above_q99_pct=np.mean(matrix > q99, axis=0) * 100.0,
        )

    backup_path = _backup_stats(dataset_root) if backup and write else None
    if write:
        _atomic_write_stats(new_stats, dataset_root)
        logging.info(
            "Recomputed global numeric stats for %s (%d frames, features=%s)",
            dataset_root,
            total_frames,
            numeric_keys,
        )

    preserved = tuple(sorted(set(new_stats) - set(numeric_keys)))
    return DatasetStatsResult(
        root=dataset_root,
        total_frames=total_frames,
        recomputed_features=tuple(numeric_keys),
        preserved_features=preserved,
        stats=new_stats,
        validation=validation,
        backup_path=backup_path,
    )
