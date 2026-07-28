"""Build chunk-transition cache for ChunkACPolicy training.

Encodes each base frame in a LeRobotDataset through VLA + the trained RL
Token encoder into a (state_vec, exec_chunk, ref_chunk, ...) tuple stored on
disk. The cache is consumed by ChunkTransitionDataset at AC training time.

This v2 replaces the legacy custom-load builder. It loads the preprocessor
directly from the SFT VLA ckpt so the cache is byte-aligned with the deploy
normalization. Per-batch progress with elapsed time is written to stdout in
unbuffered mode so a hung run is visible immediately. Each completed episode
is checkpointed to a tmp file so a kill mid-run only forfeits the in-flight
episode.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from evo_rlt.adapters.lerobot.offline_dataset import (
    _encoded_to_transitions,
    build_overlap_frame_indices,
    save_transition_cache,
)
from evo_rlt.adapters.lerobot.policies.action_modifier import PrefixOutputCapture
from evo_rlt.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
from evo_rlt.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy
from evo_rlt.adapters.lerobot.policies.processor_rlt_token import make_rlt_token_pre_post_processors
from evo_rlt.core.interfaces import (
    TRANSITION_SOURCE_DEMO,
    TRANSITION_SOURCE_HUMAN_OVERRIDE,
    TRANSITION_SOURCE_RL_AUTONOMOUS,
    TRANSITION_SOURCE_WARMUP_VLA,
    ChunkTransition,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-dataset-repo-id", required=True)
    p.add_argument("--demo-dataset-root", required=True)
    p.add_argument("--rl-token-policy-path", required=True)
    p.add_argument("--vla-pretrained-path", required=True,
                   help="SFT VLA ckpt dir — preprocessor source. Must match deploy.")
    p.add_argument("--vla-type", default=None, choices=["auto", "pi05", "smolvla"],
                   help="VLA backbone type. Defaults to the type saved in the RL-token checkpoint.")
    p.add_argument("--tokenizer-path", default=None,
                   help="Tokenizer repo id or local snapshot path for the SFT preprocessor.")
    p.add_argument(
        "--norm-stats-path",
        default=None,
        help="RL Token normalization stats override for relocated checkpoints.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-instruction", default="screw")
    p.add_argument("--chunk-length", type=int, default=10)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap on episodes to process (debug).")
    p.add_argument("--video-backend", default="pyav",
                   help="Video decoder backend passed to LeRobotDataset.")
    p.add_argument("--tolerance-s", type=float, default=0.04,
                   help="Timestamp tolerance passed to LeRobotDataset video decoding.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--empty-cache-every", type=int, default=4,
                   help="Call torch.cuda.empty_cache() every N batches.")
    p.add_argument(
        "--missing-episode-success",
        choices=["success", "failure", "error"],
        default="error",
        help=(
            "Policy when dataset metadata lacks episode_success. "
            "The default is strict; use 'success' only for verified all-success legacy datasets."
        ),
    )
    p.add_argument(
        "--provenance-mode",
        choices=["auto", "demo", "mixed"],
        default="auto",
        help=(
            "How to interpret per-frame collector annotations. 'auto' uses "
            "is_intervention + collector_policy_id when both exist, 'demo' "
            "keeps legacy all-demo semantics, and 'mixed' requires annotations."
        ),
    )
    p.add_argument(
        "--stratify-provenance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When mixed provenance is available, split episodes by "
            "(demo/online-vla/online-rl, success/failure) instead of one global shuffle."
        ),
    )
    return p.parse_args()


@dataclass(frozen=True)
class FrameProvenance:
    is_intervention: Tensor
    collector_policy_id: Tensor


def _log(msg: str) -> None:
    """Unbuffered timestamped log line."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _scalar_value(value):
    if isinstance(value, Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _dataset_scalar_column(dataset: LeRobotDataset, key: str) -> list | None:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        return None
    if key not in getattr(hf_dataset, "column_names", []):
        return None
    return [_scalar_value(value) for value in hf_dataset[key]]


def _load_frame_provenance(
    dataset: LeRobotDataset,
    mode: str,
) -> FrameProvenance | None:
    if mode == "demo":
        return None
    intervention = _dataset_scalar_column(dataset, "complementary_info.is_intervention")
    collector = _dataset_scalar_column(dataset, "complementary_info.collector_policy_id")
    if intervention is None or collector is None:
        if mode == "mixed":
            raise KeyError(
                "Mixed provenance requires complementary_info.is_intervention and "
                "complementary_info.collector_policy_id"
            )
        _log("collector provenance columns absent; falling back to all-demo semantics")
        return None

    if len(intervention) != len(dataset) or len(collector) != len(dataset):
        raise ValueError(
            "Collector provenance columns must align with dataset frames: "
            f"intervention={len(intervention)} collector={len(collector)} dataset={len(dataset)}"
        )
    intervention_t = torch.as_tensor(intervention, dtype=torch.float32).reshape(-1)
    collector_float = torch.as_tensor(collector, dtype=torch.float32).reshape(-1)
    collector_t = collector_float.round().to(torch.long)
    if not torch.allclose(collector_float, collector_t.to(torch.float32)):
        raise ValueError("complementary_info.collector_policy_id contains non-integer values")
    _log(
        "collector provenance: "
        f"frames={len(dataset)} intervention_frac={float((intervention_t > 0.5).float().mean()):.3f} "
        f"collector_ids={sorted(int(value) for value in collector_t.unique().tolist())}"
    )
    return FrameProvenance(
        is_intervention=intervention_t,
        collector_policy_id=collector_t,
    )


def _dominant_collector_id(values: Tensor) -> int:
    ids, counts = values.to(torch.long).unique(return_counts=True)
    max_count = int(counts.max().item())
    # Prefer the higher-priority policy id on ties. Human takeover itself is
    # determined separately from is_intervention and therefore is unaffected.
    return max(int(value) for value, count in zip(ids.tolist(), counts.tolist()) if count == max_count)


def _chunk_provenance(
    provenance: FrameProvenance,
    start_frame: int,
    chunk_length: int,
) -> tuple[int, bool]:
    intervention = provenance.is_intervention[start_frame : start_frame + chunk_length]
    if intervention.numel() != chunk_length:
        raise ValueError(
            f"Provenance window at frame {start_frame} has {intervention.numel()} "
            f"values, expected {chunk_length}"
        )
    human_frames = int((intervention > 0.5).sum().item())
    human_override = human_frames >= chunk_length - human_frames
    if human_override:
        return TRANSITION_SOURCE_HUMAN_OVERRIDE, True

    collector = provenance.collector_policy_id[start_frame : start_frame + chunk_length]
    collector_id = _dominant_collector_id(collector)
    source = {
        0: TRANSITION_SOURCE_DEMO,
        1: TRANSITION_SOURCE_WARMUP_VLA,
        2: TRANSITION_SOURCE_RL_AUTONOMOUS,
    }.get(collector_id, TRANSITION_SOURCE_RL_AUTONOMOUS)
    return source, False


def _episode_provenance_group(
    dataset: LeRobotDataset,
    provenance: FrameProvenance,
    ep_id: int,
) -> str:
    ep_meta = dataset.meta.episodes
    ep_from = int(ep_meta["dataset_from_index"][ep_id])
    ep_to = int(ep_meta["dataset_to_index"][ep_id])
    intervention = provenance.is_intervention[ep_from:ep_to]
    collector = provenance.collector_policy_id[ep_from:ep_to]
    if bool((intervention > 0.5).any()):
        return "online_rl_intervention"
    if bool((collector == 2).any()):
        return "online_rl_autonomous"
    if bool((collector == 1).any()):
        return "online_vla"
    return "demo"


def _split_episode_indices(
    dataset: LeRobotDataset,
    n_episodes: int,
    train_ratio: float,
    seed: int,
    missing_episode_success: str,
    provenance: FrameProvenance | None,
    stratify_provenance: bool,
) -> tuple[list[int], list[int], dict[str, dict[str, int]]]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be within [0, 1], got {train_ratio}")
    if provenance is None or not stratify_provenance:
        episode_ids = list(range(n_episodes))
        random.Random(seed).shuffle(episode_ids)
        n_train = int(train_ratio * n_episodes)
        return episode_ids[:n_train], episode_ids[n_train:], {}

    groups: dict[tuple[str, str], list[int]] = {}
    for ep_id in range(n_episodes):
        source_group = _episode_provenance_group(dataset, provenance, ep_id)
        outcome = (
            "success"
            if _episode_success_from_metadata(dataset, ep_id, missing_episode_success)
            else "failure"
        )
        groups.setdefault((source_group, outcome), []).append(ep_id)

    train: list[int] = []
    val: list[int] = []
    summary: dict[str, dict[str, int]] = {}
    for group_index, ((source_group, outcome), episode_ids) in enumerate(sorted(groups.items())):
        shuffled = list(episode_ids)
        random.Random(seed + group_index).shuffle(shuffled)
        if train_ratio <= 0.0:
            n_val = len(shuffled)
        elif train_ratio >= 1.0 or len(shuffled) == 1:
            n_val = 0
        else:
            n_val = max(1, int(round((1.0 - train_ratio) * len(shuffled))))
            n_val = min(n_val, len(shuffled) - 1)
        group_train = shuffled[: len(shuffled) - n_val]
        group_val = shuffled[len(shuffled) - n_val :]
        train.extend(group_train)
        val.extend(group_val)
        summary[f"{source_group}/{outcome}"] = {
            "total": len(shuffled),
            "train": len(group_train),
            "val": len(group_val),
        }
    random.Random(seed + 10_000).shuffle(train)
    random.Random(seed + 20_000).shuffle(val)
    return train, val, summary


def _parse_episode_success(value, ep_id: int) -> bool:
    value = _scalar_value(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "success":
            return True
        if normalized == "failure":
            return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"Unrecognized episode_success value for episode {ep_id}: {value!r}")


def _episode_success_from_metadata(dataset: LeRobotDataset, ep_id: int, missing_policy: str) -> bool:
    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    raw = None
    has_value = False
    if episodes is not None:
        try:
            column = episodes["episode_success"]
            raw = column[ep_id]
            has_value = True
        except (KeyError, IndexError, TypeError, AttributeError):
            has_value = False

    if has_value:
        return _parse_episode_success(raw, ep_id)
    if missing_policy == "error":
        raise KeyError(
            f"dataset metadata has no episode_success for episode {ep_id}; "
            "relabel the dataset or pass --missing-episode-success"
        )
    return missing_policy == "success"


def _extract_exec_chunk(preprocessed: dict, chunk_length: int, action_dim: int) -> Tensor:
    if "action" not in preprocessed:
        raise KeyError(
            "preprocessed batch does not contain 'action'; transition cache v2 needs "
            "dataset action chunks so exec_chunk is the action that was actually recorded"
        )
    action = preprocessed["action"]
    if not isinstance(action, Tensor):
        action = torch.as_tensor(action)
    if action.ndim != 3:
        raise ValueError(f"expected preprocessed action shape (B,H,D), got {tuple(action.shape)}")
    if action.shape[1] < chunk_length:
        raise ValueError(
            f"preprocessed action horizon {action.shape[1]} is shorter than chunk_length={chunk_length}"
        )
    if action.shape[2] < action_dim:
        raise ValueError(
            f"preprocessed action dim {action.shape[2]} is smaller than action_dim={action_dim}"
        )
    return action[:, :chunk_length, :action_dim].detach().to("cpu")


def _encoded_episode_to_transitions(
    state_vecs: Tensor,
    ref_chunks: Tensor,
    exec_chunks: Tensor,
    frame_indices: list[int],
    episode_last_frame: int,
    chunk_length: int,
    frame_stride: int,
    episode_success: bool,
    ep_id: int,
    provenance: FrameProvenance | None = None,
) -> list[ChunkTransition]:
    if not (state_vecs.shape[0] == ref_chunks.shape[0] == exec_chunks.shape[0] == len(frame_indices)):
        raise ValueError(
            "encoded tensors and frame_indices must have matching first dimension: "
            f"state={state_vecs.shape[0]} ref={ref_chunks.shape[0]} "
            f"exec={exec_chunks.shape[0]} frames={len(frame_indices)}"
        )
    encoded = [
        (state_vecs[i], ref_chunks[i], exec_chunks[i])
        for i in range(len(frame_indices))
    ]
    transitions = _encoded_to_transitions(
        encoded=encoded,
        frame_indices=frame_indices,
        episode_last_frame=episode_last_frame,
        chunk_length=chunk_length,
        stride=frame_stride,
        episode_success=episode_success,
        source=TRANSITION_SOURCE_DEMO,
        episode_id=ep_id,
        is_critical=1.0,
    )
    if provenance is None:
        return transitions

    start_anchors = [
        frame
        for frame in frame_indices
        if frame + chunk_length <= episode_last_frame
    ]
    if len(start_anchors) != len(transitions):
        raise RuntimeError(
            f"Episode {ep_id}: start_anchors={len(start_anchors)} "
            f"vs transitions={len(transitions)}"
        )

    for transition, start_frame in zip(transitions, start_anchors, strict=True):
        source, human_override = _chunk_provenance(
            provenance,
            start_frame,
            chunk_length,
        )
        transition.source = torch.tensor(source)
        transition.intervention = torch.tensor(float(human_override))
        if human_override:
            transition.ref_chunk = transition.exec_chunk.clone()

    # A target action is conditioned on x_{t+C}. If that next anchor is a
    # human-override chunk, propagate its repaired ref rather than retaining
    # the original VLA proposal.
    anchor_to_index = {anchor: index for index, anchor in enumerate(start_anchors)}
    for transition, start_frame in zip(transitions, start_anchors, strict=True):
        next_index = anchor_to_index.get(start_frame + chunk_length)
        if next_index is not None:
            transition.next_ref_chunk = transitions[next_index].ref_chunk.clone()
    return transitions


def _encode_episode(
    vla,
    rl_token,
    preprocessor,
    capture: PrefixOutputCapture,
    dataset: LeRobotDataset,
    frame_indices: list[int],
    chunk_length: int,
    action_dim: int,
    proprio_dim: int,
    batch_size: int,
    num_workers: int,
    device: str,
    empty_cache_every: int,
    task_str: str,
    ep_id: int,
    episode_last_frame: int,
    frame_stride: int,
    episode_success: bool,
    provenance: FrameProvenance | None,
) -> list[ChunkTransition]:
    """Encode sampled episode frames and build paper-style C-step transitions."""
    if not frame_indices:
        return []

    loader = DataLoader(
        Subset(dataset, frame_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=False,
    )

    state_vecs: list[Tensor] = []
    ref_chunks: list[Tensor] = []
    exec_chunks: list[Tensor] = []
    t_ep = time.time()
    for batch_i, batch in enumerate(loader):
        t_b = time.time()
        if "task" not in batch:
            batch["task"] = [task_str] * batch["observation.state"].shape[0]
        pre = preprocessor(batch)
        with torch.no_grad():
            vla_chunk = vla.predict_action_chunk(pre)
            prefix = capture.consume()
            z = rl_token.encode(prefix.to(torch.float32))
        if z.dim() == 3:
            z = z.mean(dim=1)
        proprio = pre["observation.state"][:, :proprio_dim].detach().to("cpu")
        state_vec = torch.cat([z.detach().to("cpu"), proprio], dim=-1)
        ref_chunk = vla_chunk[:, :chunk_length, :action_dim].detach().to("cpu")
        exec_chunk = _extract_exec_chunk(pre, chunk_length, action_dim)
        state_vecs.append(state_vec)
        ref_chunks.append(ref_chunk)
        exec_chunks.append(exec_chunk)
        del vla_chunk, prefix, z, pre
        if (batch_i + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()
        if batch_i == 0 or (batch_i + 1) % 4 == 0:
            elapsed = time.time() - t_b
            cum = time.time() - t_ep
            _log(f"    ep{ep_id} batch {batch_i+1}/{len(loader)} bs={batch_size} dt={elapsed:.2f}s cum={cum:.1f}s")

    state_vecs_t = torch.cat(state_vecs, dim=0)
    ref_chunks_t = torch.cat(ref_chunks, dim=0)
    exec_chunks_t = torch.cat(exec_chunks, dim=0)

    return _encoded_episode_to_transitions(
        state_vecs=state_vecs_t,
        ref_chunks=ref_chunks_t,
        exec_chunks=exec_chunks_t,
        frame_indices=frame_indices,
        episode_last_frame=episode_last_frame,
        chunk_length=chunk_length,
        frame_stride=frame_stride,
        episode_success=episode_success,
        ep_id=ep_id,
        provenance=provenance,
    )


def _save_partial(
    out_dir: pathlib.Path,
    split: str,
    transitions: list[ChunkTransition],
    label: str,
) -> None:
    path = out_dir / f"chunk_transitions_{split}.pt"
    save_transition_cache(transitions, out_dir, split)
    _log(f"  [{label}] checkpointed {len(transitions)} transitions -> {path.name}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"args: {vars(args)}")
    _log(f"load RLTokenPolicy from {args.rl_token_policy_path}")
    RLTokenPolicyConfig.ensure_registered()
    config_overrides = [f"--vla_pretrained_path={args.vla_pretrained_path}"]
    if args.vla_type is not None:
        config_overrides.append(f"--vla_type={args.vla_type}")
    if args.tokenizer_path is not None:
        config_overrides.append(f"--tokenizer_path={args.tokenizer_path}")
    if args.norm_stats_path is not None:
        config_overrides.append(f"--norm_stats_path={args.norm_stats_path}")
    cfg = PreTrainedConfig.from_pretrained(
        args.rl_token_policy_path,
        cli_overrides=config_overrides,
    )
    policy = (
        RLTokenPolicy.from_pretrained(
            args.rl_token_policy_path,
            config=cfg,
        )
        .to(args.device)
        .eval()
    )

    _log(f"load preprocessor from SFT VLA dir {args.vla_pretrained_path}")
    preprocessor, _ = make_rlt_token_pre_post_processors(config=cfg)

    _log(f"load dataset {args.demo_dataset_repo_id} root={args.demo_dataset_root}")
    metadata = LeRobotDatasetMetadata(
        repo_id=args.demo_dataset_repo_id,
        root=args.demo_dataset_root,
    )
    delta = {"action": [i / metadata.fps for i in range(args.chunk_length)]}
    dataset = LeRobotDataset(
        repo_id=args.demo_dataset_repo_id,
        root=args.demo_dataset_root,
        delta_timestamps=delta,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
    )
    provenance = _load_frame_provenance(dataset, args.provenance_mode)
    n_episodes = dataset.num_episodes
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)
    _log(f"episodes: {n_episodes} of {dataset.num_episodes}; batch_size={args.batch_size} num_workers={args.num_workers}")

    vla = policy._pi05
    rl_token = policy.rl_token

    capture = PrefixOutputCapture(
        token_pool_size=cfg.token_pool_size,
        image_only=cfg.image_only,
        num_image_tokens=policy._num_image_tokens,
        num_per_camera=getattr(cfg, "num_per_camera", 0),
        active_camera_indices=getattr(cfg, "active_camera_indices", None),
    )
    capture.attach(policy._pi05)
    try:
        train_eps, val_eps, split_summary = _split_episode_indices(
            dataset=dataset,
            n_episodes=n_episodes,
            train_ratio=args.train_ratio,
            seed=args.seed,
            missing_episode_success=args.missing_episode_success,
            provenance=provenance,
            stratify_provenance=args.stratify_provenance,
        )
        _log(f"split: train={len(train_eps)} val={len(val_eps)}")
        for group, counts in split_summary.items():
            _log(
                f"  split group {group}: total={counts['total']} "
                f"train={counts['train']} val={counts['val']}"
            )

        t_start = time.time()
        for split_name, eps in (("train", train_eps), ("val", val_eps)):
            all_tx: list[ChunkTransition] = []
            for k, ep_id in enumerate(eps):
                ep_meta = dataset.meta.episodes
                ep_from = int(ep_meta["dataset_from_index"][ep_id])
                ep_to = int(ep_meta["dataset_to_index"][ep_id])
                frame_indices = build_overlap_frame_indices(
                    episode_start=ep_from,
                    episode_stop=ep_to,
                    chunk_length=args.chunk_length,
                    stride=args.frame_stride,
                )
                episode_success = _episode_success_from_metadata(
                    dataset,
                    ep_id,
                    args.missing_episode_success,
                )
                _log(f"  [{split_name}] ep {k+1}/{len(eps)} id={ep_id} frames={ep_to-ep_from} chunks={len(frame_indices)} (total transitions={len(all_tx)}, wall={time.time()-t_start:.0f}s)")
                ep_tx = _encode_episode(
                    vla=vla,
                    rl_token=rl_token,
                    preprocessor=preprocessor,
                    capture=capture,
                    dataset=dataset,
                    frame_indices=frame_indices,
                    chunk_length=args.chunk_length,
                    action_dim=cfg.action_dim,
                    proprio_dim=cfg.proprio_dim,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=args.device,
                    empty_cache_every=args.empty_cache_every,
                    task_str=args.task_instruction,
                    ep_id=ep_id,
                    episode_last_frame=ep_to - 1,
                    frame_stride=args.frame_stride,
                    episode_success=episode_success,
                    provenance=provenance,
                )
                all_tx.extend(ep_tx)
                if (k + 1) % 5 == 0 or (k + 1) == len(eps):
                    _save_partial(out_dir, split_name, all_tx, f"{split_name} ep {k+1}/{len(eps)}")
    finally:
        capture.detach()

    _log(f"done, total wall {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
