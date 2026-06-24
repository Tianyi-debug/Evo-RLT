#!/usr/bin/env python
"""Pre-mix bucket sanity check + automatic renormalization.

Combines what used to live in two separate tools:
  - datamix_phase_b.bucket_q01q99_check / flag_divergence (cache-empirical
    detection of normalization drift across buckets)
  - renorm_teleop_cache (HF-authoritative renormalization of one cache to
    another bucket's q01/q99 stats)

into a single CLI that prepares all buckets for downstream mix training.

Workflow:
  1. For each bucket spec ``name=cache_path[:hf_repo_id]``, load the cache and
     compute empirical action q01/q99 for divergence detection.
  2. Compare per-bucket ranges against the reference bucket; flag any dim where
     range ratio exceeds --threshold (default 1.5x).
  3. For each flagged non-reference bucket that has both src and dst HF
     repo_ids available, renormalize action chunks + proprio portion of state
     vectors using authoritative q01/q99 from HF metadata, and write the new
     cache to ``<output_dir>/<name>_renorm/``.
  4. Emit a JSON manifest at ``<output_dir>/mix_manifest.json`` listing the
     final cache path each bucket should be loaded from in downstream training.

Example:
  prepare_mix_buckets.py \\
    --bucket demo_a=/path/to/cache_a:your-org/demo_a \\
    --bucket demo_b=/path/to/cache_b:your-org/demo_b \\
    --bucket demo_c=/path/to/cache_c:your-org/demo_c \\
    --reference demo_a \\
    --output-dir /path/to/cache_mix_ready \\
    --proprio-dim 12 --threshold 1.5
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evo_rlt.cli.common import configure_logging  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from evo_rlt.adapters.lerobot.offline_dataset import load_transition_cache, save_transition_cache  # noqa: E402

logger = configure_logging(__name__)


# ---------------------------------------------------------------------------
# Bucket spec parsing
# ---------------------------------------------------------------------------

@dataclass
class BucketSpec:
    name: str
    cache_path: str
    repo_id: str | None  # HF dataset repo id; None = no authoritative stats


def parse_bucket_spec(spec: str) -> BucketSpec:
    """name=cache_path[:hf_repo_id]"""
    if "=" not in spec:
        raise ValueError(f"Bucket spec missing =: {spec!r}")
    name, rest = spec.split("=", 1)
    if ":" in rest:
        cache_path, repo_id = rest.split(":", 1)
        return BucketSpec(name=name, cache_path=cache_path, repo_id=repo_id)
    return BucketSpec(name=name, cache_path=rest, repo_id=None)


# ---------------------------------------------------------------------------
# Stats fetch (HF authoritative + cache empirical)
# ---------------------------------------------------------------------------

def fetch_repo_stats(repo_id: str) -> dict[str, torch.Tensor]:
    """Pull q01/q99 from HF dataset metadata. Returns action_q01/99 + state_q01/99."""
    md = LeRobotDatasetMetadata(repo_id=repo_id, revision="main")
    s = md.stats
    return {
        "action_q01": torch.as_tensor(s["action"]["q01"], dtype=torch.float32),
        "action_q99": torch.as_tensor(s["action"]["q99"], dtype=torch.float32),
        "state_q01": torch.as_tensor(s["observation.state"]["q01"], dtype=torch.float32),
        "state_q99": torch.as_tensor(s["observation.state"]["q99"], dtype=torch.float32),
    }


def cache_empirical_action_quantiles(cache_dir: str, n_sample: int = 500) -> dict[str, list[float]]:
    """Sample first n_sample exec_chunks from train split; return per-dim q01/q99."""
    buf = load_transition_cache(cache_dir, "train", capacity=200_000)
    sample = list(itertools.islice(buf.buffer, n_sample))
    if not sample:
        raise RuntimeError(f"Cache {cache_dir} train split has 0 transitions")
    stacked = torch.stack([t.exec_chunk.flatten() for t in sample])
    return {
        "action_q01": stacked.quantile(0.01, dim=0).tolist(),
        "action_q99": stacked.quantile(0.99, dim=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Divergence flagging
# ---------------------------------------------------------------------------

def flag_divergence(
    empirical: dict[str, dict[str, list[float]]],
    threshold: float,
) -> list[dict]:
    """Flag dims where any pair of buckets have action-range ratio > threshold."""
    names = list(empirical.keys())
    if len(names) < 2:
        return []
    dim = len(empirical[names[0]]["action_q01"])
    flags = []
    for d in range(dim):
        ranges = {n: empirical[n]["action_q99"][d] - empirical[n]["action_q01"][d] for n in names}
        abs_ranges = {n: abs(r) for n, r in ranges.items()}
        lo = min(abs_ranges.values())
        hi = max(abs_ranges.values())
        if lo <= 1e-6:
            continue
        if hi / lo > threshold:
            flags.append({
                "dim": d,
                "range_min": lo,
                "range_max": hi,
                "ratio": hi / lo,
                "per_bucket": ranges,
            })
    return flags


# ---------------------------------------------------------------------------
# Renormalization (action + proprio portion of state_vec)
# ---------------------------------------------------------------------------

def normalize(raw: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    denom = (q99 - q01).clamp(min=1e-8)
    return (raw - q01) / denom * 2.0 - 1.0


def denormalize(norm: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    return (norm + 1.0) / 2.0 * (q99 - q01) + q01


def renorm_action(chunk: torch.Tensor, src_q01, src_q99, dst_q01, dst_q99) -> torch.Tensor:
    return normalize(denormalize(chunk, src_q01, src_q99), dst_q01, dst_q99)


def renorm_state_vec(vec: torch.Tensor, proprio_dim: int,
                     src_q01, src_q99, dst_q01, dst_q99) -> torch.Tensor:
    """state_vec last proprio_dim dims are normalized proprio; renorm those only."""
    new = vec.clone()
    new[-proprio_dim:] = normalize(
        denormalize(vec[-proprio_dim:], src_q01, src_q99),
        dst_q01, dst_q99,
    )
    return new


def renorm_bucket(
    src_cache: str, dst_cache: str,
    src_stats: dict, dst_stats: dict,
    proprio_dim: int, capacity: int,
) -> int:
    """Read each split, renormalize all transitions, write to dst_cache. Returns total count."""
    Path(dst_cache).mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ("train", "val", "test"):
        split_path = Path(src_cache) / f"chunk_transitions_{split}.pt"
        if not split_path.exists():
            logger.info("[%s] split missing in %s, skipping", split, src_cache)
            continue
        buf = load_transition_cache(src_cache, split, capacity=capacity)
        new_transitions = [
            _renorm_one(t, src_stats, dst_stats, proprio_dim) for t in buf.buffer
        ]
        save_transition_cache(new_transitions, dst_cache, split)
        total += len(new_transitions)
    return total


def _renorm_one(t, src_stats, dst_stats, proprio_dim):
    t.exec_chunk = renorm_action(t.exec_chunk,
                                 src_stats["action_q01"], src_stats["action_q99"],
                                 dst_stats["action_q01"], dst_stats["action_q99"])
    t.ref_chunk = renorm_action(t.ref_chunk,
                                src_stats["action_q01"], src_stats["action_q99"],
                                dst_stats["action_q01"], dst_stats["action_q99"])
    t.next_ref_chunk = renorm_action(t.next_ref_chunk,
                                     src_stats["action_q01"], src_stats["action_q99"],
                                     dst_stats["action_q01"], dst_stats["action_q99"])
    t.state_vec = renorm_state_vec(t.state_vec, proprio_dim,
                                   src_stats["state_q01"], src_stats["state_q99"],
                                   dst_stats["state_q01"], dst_stats["state_q99"])
    t.next_state_vec = renorm_state_vec(t.next_state_vec, proprio_dim,
                                        src_stats["state_q01"], src_stats["state_q99"],
                                        dst_stats["state_q01"], dst_stats["state_q99"])
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", action="append", required=True,
                   help="name=cache_path[:hf_repo_id]; pass multiple times")
    p.add_argument("--reference", required=True,
                   help="Name of the reference bucket whose stats are the renorm target")
    p.add_argument("--output-dir", required=True,
                   help="Directory under which renormalized caches and manifest are written")
    p.add_argument("--proprio-dim", type=int, default=12)
    p.add_argument("--threshold", type=float, default=1.5,
                   help="Per-dim range ratio above which a dim is flagged")
    p.add_argument("--capacity", type=int, default=200_000)
    p.add_argument("--n-sample", type=int, default=500,
                   help="Transitions per bucket sampled for empirical q01/q99")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {s.name: s for s in (parse_bucket_spec(b) for b in args.bucket)}
    if args.reference not in specs:
        raise ValueError(f"--reference {args.reference!r} not among --bucket names {list(specs)}")

    logger.info("Computing empirical action quantiles for %d buckets...", len(specs))
    empirical = {n: cache_empirical_action_quantiles(s.cache_path, args.n_sample)
                 for n, s in specs.items()}

    flags = flag_divergence(empirical, args.threshold)
    if flags:
        logger.warning("NORMALIZATION DIVERGENCE in %d action dims (ratio>%.2f):",
                       len(flags), args.threshold)
        for f in flags:
            logger.warning("  dim %d: ratio=%.2f (%s)", f["dim"], f["ratio"], f["per_bucket"])
    else:
        logger.info("All bucket action ranges within %.2fx — no renormalization needed",
                    args.threshold)

    final_paths: dict[str, str] = {n: s.cache_path for n, s in specs.items()}
    renormalized: dict[str, str] = {}

    if flags:
        ref_spec = specs[args.reference]
        if ref_spec.repo_id is None:
            raise ValueError(
                f"Divergence found but reference bucket {args.reference!r} has no repo_id; "
                "supply name=path:repo_id for the reference to enable authoritative renorm"
            )
        logger.info("Fetching reference stats from %s ...", ref_spec.repo_id)
        dst_stats = fetch_repo_stats(ref_spec.repo_id)

        for name, spec in specs.items():
            if name == args.reference:
                continue
            if spec.repo_id is None:
                logger.warning("Bucket %s has no repo_id; cannot renorm, leaving as-is", name)
                continue
            logger.info("Renormalizing %s (%s -> %s) ...", name, spec.repo_id, ref_spec.repo_id)
            src_stats = fetch_repo_stats(spec.repo_id)
            dst_cache = str(out_dir / f"{name}_renorm")
            n = renorm_bucket(spec.cache_path, dst_cache, src_stats, dst_stats,
                              args.proprio_dim, args.capacity)
            logger.info("  wrote %d transitions to %s", n, dst_cache)
            renormalized[name] = dst_cache
            final_paths[name] = dst_cache

    manifest = {
        "reference": args.reference,
        "threshold": args.threshold,
        "proprio_dim": args.proprio_dim,
        "bucket_empirical_action_quantiles": empirical,
        "divergence_flags": flags,
        "renormalized": renormalized,
        "final_paths": final_paths,
    }
    manifest_path = out_dir / "mix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote manifest to %s", manifest_path)
    logger.info("Final paths for downstream mix training:")
    for name, path in final_paths.items():
        logger.info("  %s -> %s", name, path)


if __name__ == "__main__":
    main()
