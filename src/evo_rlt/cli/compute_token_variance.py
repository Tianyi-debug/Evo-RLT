#!/usr/bin/env python
"""Compute per-dim mean/std of pi0.5 prefix tokens over N batches.

Stats are computed in float32 over the full dataset (no train/val split, since
we only need a global moments estimate). Output is a .pt with keys:
  mean: (D,) tensor
  std:  (D,) tensor
  n:    int (total tokens used)
  config: dict (image_only, active_cameras, num_batches, batch_size)

Used by train_rl_token.py via --norm-stats + --norm-gamma to apply per-dim
weighted MSE in reconstruction_loss (weight = std^{-gamma}).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evo_rlt.cli.common import build_pi05_policy, configure_logging, load_training_config

logger = configure_logging(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument("--demo-dataset-path", default=None)
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    parser.add_argument("--task-instruction", default="screw")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--active-cameras", default=None)
    parser.add_argument("--token-pool-size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from evo_rlt.adapters.lerobot.demo_loader import make_demo_loader

    config = load_training_config(None)
    cams = args.active_cameras.split(",") if args.active_cameras else None
    policy = build_pi05_policy(
        config=config,
        model_path=args.model_path,
        task_instruction=args.task_instruction,
        device=args.device,
        token_pool_size=args.token_pool_size,
        dtype=args.dtype,
        image_only=args.image_only,
        active_cameras=cams,
    )
    policy.eval()

    loader = make_demo_loader(
        dataset_path=args.demo_dataset_path,
        batch_size=args.batch_size,
        chunk_length=config.vla_horizon,
        num_workers=0,
        device=args.device,
    )

    # Welford online stats over flattened (B*M, D)
    n = 0
    mean: torch.Tensor | None = None
    M2: torch.Tensor | None = None
    start = time.time()
    for i, (obs, _) in enumerate(loader):
        if i >= args.num_batches:
            break
        with torch.no_grad():
            vla_out = policy.vla.forward_vla(obs)
        x = vla_out.final_tokens.to(torch.float32).reshape(-1, vla_out.final_tokens.shape[-1])
        bn = x.shape[0]
        if mean is None:
            mean = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
            M2 = torch.zeros_like(mean)
        delta = x - mean
        n += bn
        mean = mean + delta.sum(dim=0) / n
        delta2 = x - mean
        M2 = M2 + (delta * delta2).sum(dim=0)
        if (i + 1) % 10 == 0:
            logger.info("Batch %d/%d, n=%d, elapsed=%.1fs", i + 1, args.num_batches, n, time.time() - start)

    var = M2 / max(n - 1, 1)
    std = var.sqrt()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": mean.cpu(),
        "std": std.cpu(),
        "n": n,
        "config": {
            "image_only": args.image_only,
            "active_cameras": cams,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "model_path": args.model_path,
            "dataset_path": args.demo_dataset_path,
        },
    }
    torch.save(payload, out)

    sorted_std, sorted_idx = std.sort(descending=True)
    summary = {
        "n_tokens": n,
        "feature_dim": int(std.shape[0]),
        "std_min": float(std.min()),
        "std_median": float(std.median()),
        "std_mean": float(std.mean()),
        "std_max": float(std.max()),
        "max_over_median_ratio": float(sorted_std[0] / std.median()),
        "top10_var_share": float(sorted_std[:10].pow(2).sum() / std.pow(2).sum()),
        "top50_var_share": float(sorted_std[:50].pow(2).sum() / std.pow(2).sum()),
        "top200_var_share": float(sorted_std[:200].pow(2).sum() / std.pow(2).sum()),
        "top20_dims": [
            {"dim": int(sorted_idx[k]), "std": float(sorted_std[k]), "mean": float(mean[sorted_idx[k]])}
            for k in range(20)
        ],
    }
    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Per-dim variance summary ===")
    logger.info("n_tokens=%d  D=%d", n, std.shape[0])
    logger.info("std min=%.4f median=%.4f mean=%.4f max=%.4f", summary["std_min"], summary["std_median"], summary["std_mean"], summary["std_max"])
    logger.info("max/median = %.2fx", summary["max_over_median_ratio"])
    logger.info("Top-10 dims explain %.1f%% of total variance", 100 * summary["top10_var_share"])
    logger.info("Top-50 dims explain %.1f%% of total variance", 100 * summary["top50_var_share"])
    logger.info("Top-200 dims explain %.1f%% of total variance", 100 * summary["top200_var_share"])
    logger.info("Top 20 dims:")
    for r in summary["top20_dims"]:
        logger.info("  dim %4d  std=%.4f  mean=%+.4f", r["dim"], r["std"], r["mean"])
    logger.info("Saved stats to %s", out)
    logger.info("Saved summary to %s", out.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
