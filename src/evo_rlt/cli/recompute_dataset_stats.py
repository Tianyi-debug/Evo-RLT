from __future__ import annotations

import argparse
import logging

import numpy as np

from evo_rlt.adapters.lerobot.dataset_stats import recompute_numeric_dataset_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute exact global numeric statistics from all LeRobot parquet frames. "
            "Image/video statistics are preserved."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Back up the existing meta/stats.json before replacing it.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Compute and validate without writing meta/stats.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = recompute_numeric_dataset_stats(
        args.dataset_root,
        backup=args.backup,
        write=not args.check_only,
    )
    print(f"Dataset: {result.root}")
    print(f"Frames: {result.total_frames}")
    print(f"Recomputed numeric features: {', '.join(result.recomputed_features)}")
    if result.preserved_features:
        print(f"Preserved non-numeric stats: {', '.join(result.preserved_features)}")
    for key, validation in result.validation.items():
        values = validation.outside_q01_q99_pct
        print(
            f"{key} outside q01/q99 (% by dim): "
            f"{np.array2string(values, precision=3, separator=', ')}"
        )
    if result.backup_path is not None:
        print(f"Backup: {result.backup_path}")
    print("Mode: check-only" if args.check_only else "Mode: stats.json updated")


if __name__ == "__main__":
    main()

