from __future__ import annotations

import argparse
import logging
from pathlib import Path

from evo_rlt.adapters.lerobot.dataset_stats import recompute_numeric_dataset_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate local LeRobot datasets and recompute exact global numeric statistics."
    )
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help="Local source dataset root. Repeat once per source dataset.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--output-repo-id",
        default=None,
        help="Logical LeRobot repo id. Defaults to local/<output directory name>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source_roots = [Path(path).expanduser().resolve() for path in args.source_root]
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists: {output_root}. Refusing to overwrite an existing dataset."
        )
    output_repo_id = args.output_repo_id or f"local/{output_root.name}"

    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=[f"local/{root.name}" for root in source_roots],
        aggr_repo_id=output_repo_id,
        roots=source_roots,
        aggr_root=output_root,
    )
    result = recompute_numeric_dataset_stats(output_root, backup=False, write=True)
    print(
        f"Aggregated {len(source_roots)} datasets into {output_root}; "
        f"recomputed {len(result.recomputed_features)} numeric features over "
        f"{result.total_frames} frames."
    )


if __name__ == "__main__":
    main()

