from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from evo_rlt.adapters.lerobot.record.annotations import RLT_COLLECTOR_POLICY_ID_TO_NAME


COLLECTOR_POLICY_ID_KEY = "complementary_info.collector_policy_id"


@dataclass(frozen=True)
class CollectorCodebookRepairResult:
    root: Path
    observed_ids: tuple[int, ...]
    before: dict[str, str]
    after: dict[str, str]
    changed: bool
    backup_path: Path | None = None


def _load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset info not found: {info_path}")
    return json.loads(info_path.read_text())


def _feature_codebook(info: dict) -> dict[str, str] | None:
    feature = info.get("features", {}).get(COLLECTOR_POLICY_ID_KEY)
    if feature is None:
        return None
    raw = feature.get("info", {}).get("codebook", {})
    return {str(key): str(value) for key, value in raw.items()}


def _observed_collector_ids(root: Path) -> tuple[int, ...]:
    import pyarrow.parquet as pq

    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")
    if COLLECTOR_POLICY_ID_KEY not in pq.read_schema(paths[0]).names:
        return ()
    values = pq.read_table(paths, columns=[COLLECTOR_POLICY_ID_KEY])[COLLECTOR_POLICY_ID_KEY]
    array = np.asarray(values.combine_chunks().to_pylist()).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{COLLECTOR_POLICY_ID_KEY} contains NaN or infinity")
    rounded = np.rint(array)
    if not np.allclose(array, rounded):
        raise ValueError(f"{COLLECTOR_POLICY_ID_KEY} contains non-integer values")
    return tuple(int(value) for value in np.unique(rounded.astype(np.int64)))


def _canonical_codebook(
    before: dict[str, str],
    observed_ids: Iterable[int],
) -> dict[str, str]:
    observed = set(observed_ids)
    after = dict(before)
    if 2 in observed:
        # RLT recordings can emit all three ids depending on phase and human
        # takeover. Include the complete stable codebook so teleop-only expert
        # datasets can be aggregated with policy rollouts after harmonization.
        after.update(
            {str(code): name for code, name in RLT_COLLECTOR_POLICY_ID_TO_NAME.items()}
        )
    else:
        for code in observed:
            key = str(code)
            if code in RLT_COLLECTOR_POLICY_ID_TO_NAME:
                after.setdefault(key, RLT_COLLECTOR_POLICY_ID_TO_NAME[code])
            else:
                after.setdefault(key, f"policy_{code}")
    return dict(sorted(after.items(), key=lambda item: int(item[0])))


def _backup_info(root: Path) -> Path:
    info_path = root / "meta" / "info.json"
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = info_path.with_name(f"info.before-codebook-repair-{timestamp}.json")
    suffix = 1
    while backup_path.exists():
        backup_path = info_path.with_name(
            f"info.before-codebook-repair-{timestamp}-{suffix}.json"
        )
        suffix += 1
    shutil.copy2(info_path, backup_path)
    return backup_path


def _atomic_write_info(root: Path, info: dict) -> None:
    info_path = root / "meta" / "info.json"
    fd, tmp_name = tempfile.mkstemp(prefix=".info-codebook-", suffix=".json", dir=info_path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(info, handle, indent=4)
            handle.write("\n")
        os.replace(tmp_name, info_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def repair_collector_policy_id_codebook(
    root: str | Path,
    *,
    write: bool = True,
    backup: bool = False,
    codebook: dict[str, str] | None = None,
) -> CollectorCodebookRepairResult | None:
    """Repair a dataset codebook from the collector ids actually stored in parquet.

    Returns ``None`` for datasets without the unified collector-policy feature.
    When ``codebook`` is supplied it is installed verbatim after verifying that
    every observed id is represented.
    """

    dataset_root = Path(root).expanduser().resolve()
    info = _load_info(dataset_root)
    before = _feature_codebook(info)
    if before is None:
        return None
    observed_ids = _observed_collector_ids(dataset_root)
    after = (
        {str(key): str(value) for key, value in codebook.items()}
        if codebook is not None
        else _canonical_codebook(before, observed_ids)
    )
    missing = [code for code in observed_ids if str(code) not in after]
    if missing:
        raise ValueError(
            f"Collector codebook for {dataset_root} does not describe observed ids {missing}"
        )
    after = dict(sorted(after.items(), key=lambda item: int(item[0])))
    changed = after != before
    backup_path = None
    if changed and write:
        if backup:
            backup_path = _backup_info(dataset_root)
        feature = info["features"][COLLECTOR_POLICY_ID_KEY]
        feature.setdefault("info", {})["codebook"] = after
        _atomic_write_info(dataset_root, info)
    return CollectorCodebookRepairResult(
        root=dataset_root,
        observed_ids=observed_ids,
        before=before,
        after=after,
        changed=changed,
        backup_path=backup_path,
    )


def harmonize_collector_policy_id_codebooks(
    roots: Iterable[str | Path],
    *,
    backup: bool = True,
) -> dict[str, str] | None:
    """Install one compatible codebook across datasets before aggregation."""

    inspections = [
        repair_collector_policy_id_codebook(root, write=False)
        for root in roots
    ]
    present = [result for result in inspections if result is not None]
    if not present:
        return None
    if len(present) != len(inspections):
        raise ValueError(
            "Cannot aggregate a mix of datasets with and without "
            f"{COLLECTOR_POLICY_ID_KEY}"
        )

    has_rlt_actor = any(2 in result.observed_ids for result in present)
    merged = (
        {str(code): name for code, name in RLT_COLLECTOR_POLICY_ID_TO_NAME.items()}
        if has_rlt_actor
        else {}
    )
    for result in present:
        for key, value in result.after.items():
            if has_rlt_actor and int(key) in RLT_COLLECTOR_POLICY_ID_TO_NAME:
                continue
            previous = merged.get(key)
            if previous is not None and previous != value:
                raise ValueError(
                    f"Conflicting collector codebook label for id {key}: "
                    f"{previous!r} vs {value!r}"
                )
            merged[key] = value
    merged = dict(sorted(merged.items(), key=lambda item: int(item[0])))
    for result in present:
        repair_collector_policy_id_codebook(
            result.root,
            write=True,
            backup=backup,
            codebook=merged,
        )
    return merged
