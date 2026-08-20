"""Build chunk-transition cache for ChunkACPolicy training.

Encodes each base frame in a LeRobotDataset through VLA + the trained RL
Token encoder into a proposal/exec/BC-target-separated chunk transition stored on
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
from evo_rlt.adapters.lerobot.record.annotations import (
    INTERVENTION_REASON_CORRECTIVE,
    INTERVENTION_REASON_NONE,
    INTERVENTION_REASON_PROACTIVE,
    INTERVENTION_STAGE_POLICY,
    INTERVENTION_STAGE_TELEOP,
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
    p.add_argument(
        "--trim-leading-idle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Automatically trim each episode's leading idle frames before building "
            "frame indices. Disabled by default, so existing commands are unchanged."
        ),
    )
    p.add_argument(
        "--trim-leading-idle-threshold",
        type=float,
        default=1.0,
        help=(
            "Motion-onset threshold: L2 displacement from the episode's first action "
            "over the selected action dimensions, in raw dataset action units."
        ),
    )
    p.add_argument(
        "--trim-leading-idle-hold-frames",
        type=int,
        default=5,
        help="Number of consecutive above-threshold frames required to confirm motion onset.",
    )
    p.add_argument(
        "--trim-leading-idle-pre-roll-frames",
        type=int,
        default=None,
        help=(
            "Frames retained immediately before detected motion onset. Defaults to "
            "--chunk-length."
        ),
    )
    p.add_argument(
        "--trim-leading-idle-action-dims",
        type=int,
        default=5,
        help=(
            "Number of leading action dimensions used for onset detection. The default "
            "excludes the sixth SO101 gripper dimension."
        ),
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap on episodes to process (debug).")
    p.add_argument(
        "--exclude-episode-indices",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Dataset episode indices to skip before the train/validation split. "
            "The source dataset is not modified."
        ),
    )
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
    p.add_argument(
        "--demo-reference-mode",
        choices=["vla", "executed"],
        default="vla",
        help=(
            "BC target stored for demonstration chunks. 'executed' supervises "
            "the actor with the recorded expert action while preserving the "
            "independently generated VLA proposal as actor input; 'vla' keeps "
            "the legacy behavior."
        ),
    )
    p.add_argument(
        "--human-reference-mode",
        choices=["executed", "vla"],
        default="executed",
        help=(
            "BC target stored for human-override chunks. 'executed' supervises "
            "the actor with the human action while preserving the independently "
            "generated VLA proposal as actor input; 'vla' uses VLA BC everywhere."
        ),
    )
    p.add_argument(
        "--actor-bc-mode",
        choices=["legacy", "outcome-aware"],
        default="legacy",
        help=(
            "Actor BC target/mask semantics stored in the cache. 'legacy' uses "
            "--demo-reference-mode/--human-reference-mode and applies BC to every "
            "transition. 'outcome-aware' uses recorded actions for demo and human "
            "chunks, uses recorded actions only for successful fully autonomous "
            "policy episodes, and masks actor BC on failed or assisted policy "
            "chunks. The VLA proposal remains the actor input/residual base."
        ),
    )
    p.add_argument(
        "--legacy-handoff-policy",
        choices=["majority", "drop"],
        default="majority",
        help=(
            "How to handle old datasets without intervention_stage. 'majority' "
            "preserves existing chunk voting; 'drop' removes chunks crossing a "
            "policy/human boundary and transitions that bootstrap into one."
        ),
    )
    p.add_argument(
        "--missing-intervention-reason",
        choices=["legacy", "error", "corrective", "proactive"],
        default="legacy",
        help=(
            "How to handle mixed datasets without the frame-level "
            "complementary_info.intervention_reason field. 'legacy' preserves "
            "the old credit semantics; 'error' requires explicit labels; "
            "'corrective' or 'proactive' assigns that reason to every handoff."
        ),
    )
    p.add_argument(
        "--residual-delta-scale",
        type=float,
        default=0.1,
        help=(
            "Residual actor correction bound used only for cache audit output. "
            "Set this to the same value as --policy.actor_delta_scale."
        ),
    )
    return p.parse_args()


@dataclass(frozen=True)
class FrameProvenance:
    is_intervention: Tensor
    collector_policy_id: Tensor
    intervention_stage: Tensor | None = None
    intervention_reason: Tensor | None = None


def _apply_actor_bc_semantics(
    transitions: list[ChunkTransition],
    *,
    mode: str,
    episode_success: bool,
    episode_has_human: bool,
) -> list[ChunkTransition]:
    """Select actor BC targets without changing critic or actor-Q credit.

    Outcome-aware semantics preserve expert and human supervision while cloning
    autonomous behavior only when the whole episode completed successfully
    without human assistance. Failed autonomous actions and policy prefixes in
    assisted episodes remain available to the critic but cannot move the actor
    through BC.
    """
    if mode == "legacy":
        return transitions
    if mode != "outcome-aware":
        raise ValueError(f"unsupported actor BC mode: {mode!r}")

    autonomous_sources = {
        TRANSITION_SOURCE_WARMUP_VLA,
        TRANSITION_SOURCE_RL_AUTONOMOUS,
    }
    autonomous_bc_valid = bool(episode_success and not episode_has_human)
    for transition in transitions:
        source = int(transition.source.item())
        if source in {TRANSITION_SOURCE_DEMO, TRANSITION_SOURCE_HUMAN_OVERRIDE}:
            transition.bc_target_chunk = transition.exec_chunk.clone()
            transition.actor_bc_mask = torch.tensor(1.0)
        elif source in autonomous_sources:
            transition.bc_target_chunk = (
                transition.exec_chunk.clone()
                if autonomous_bc_valid
                else transition.proposal_chunk.clone()
            )
            transition.actor_bc_mask = torch.tensor(float(autonomous_bc_valid))
        else:
            raise ValueError(f"unsupported transition source for actor BC: {source}")
    return transitions


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


def _episode_action_tensor(
    dataset: LeRobotDataset,
    episode_start: int,
    episode_stop: int,
) -> Tensor:
    """Read raw per-frame actions for one episode without decoding observations."""
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None or "action" not in getattr(hf_dataset, "column_names", []):
        raise KeyError(
            "--trim-leading-idle requires an 'action' column in dataset.hf_dataset"
        )
    values = hf_dataset["action"][episode_start:episode_stop]
    if len(values) == 0:
        raise ValueError(
            f"Cannot detect motion onset in empty episode range [{episode_start}, {episode_stop})"
        )
    actions = torch.stack(
        [torch.as_tensor(value, dtype=torch.float32).reshape(-1) for value in values]
    )
    if not torch.isfinite(actions).all():
        raise ValueError(
            f"Non-finite action found in episode range [{episode_start}, {episode_stop})"
        )
    return actions


def _find_leading_motion_onset(
    actions: Tensor,
    *,
    threshold: float,
    hold_frames: int,
    action_dims: int,
) -> int | None:
    """Return the first relative frame of sustained motion away from frame zero."""
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape [frames, dims], got {tuple(actions.shape)}")
    if threshold <= 0:
        raise ValueError("--trim-leading-idle-threshold must be > 0")
    if hold_frames <= 0:
        raise ValueError("--trim-leading-idle-hold-frames must be > 0")
    if action_dims <= 0 or action_dims > actions.shape[1]:
        raise ValueError(
            "--trim-leading-idle-action-dims must be in "
            f"[1, {actions.shape[1]}], got {action_dims}"
        )
    if actions.shape[0] < hold_frames:
        return None

    displacement = torch.linalg.vector_norm(
        actions[:, :action_dims] - actions[0, :action_dims],
        dim=1,
    )
    sustained = (displacement > threshold).unfold(0, hold_frames, 1).all(dim=1)
    onset_candidates = torch.nonzero(sustained, as_tuple=False).reshape(-1)
    if onset_candidates.numel() == 0:
        return None
    return int(onset_candidates[0].item())


def _trimmed_episode_start(
    dataset: LeRobotDataset,
    *,
    episode_id: int,
    episode_start: int,
    episode_stop: int,
    threshold: float,
    hold_frames: int,
    pre_roll_frames: int,
    action_dims: int,
) -> tuple[int, int]:
    """Return absolute (trimmed start, detected onset) for one episode."""
    if pre_roll_frames < 0:
        raise ValueError("--trim-leading-idle-pre-roll-frames must be >= 0")
    actions = _episode_action_tensor(dataset, episode_start, episode_stop)
    onset = _find_leading_motion_onset(
        actions,
        threshold=threshold,
        hold_frames=hold_frames,
        action_dims=action_dims,
    )
    if onset is None:
        raise ValueError(
            f"Episode {episode_id} has no sustained motion onset: threshold={threshold}, "
            f"hold_frames={hold_frames}, action_dims={action_dims}. Inspect or exclude this "
            "episode instead of silently retaining an all-idle prefix."
        )
    trimmed_start = episode_start + max(0, onset - pre_roll_frames)
    return trimmed_start, episode_start + onset


def _load_frame_provenance(
    dataset: LeRobotDataset,
    mode: str,
    missing_intervention_reason: str = "legacy",
) -> FrameProvenance | None:
    if mode == "demo":
        return None
    intervention = _dataset_scalar_column(dataset, "complementary_info.is_intervention")
    collector = _dataset_scalar_column(dataset, "complementary_info.collector_policy_id")
    intervention_stage = _dataset_scalar_column(
        dataset,
        "complementary_info.intervention_stage",
    )
    intervention_reason = _dataset_scalar_column(
        dataset,
        "complementary_info.intervention_reason",
    )
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
    intervention_stage_t = None
    if intervention_stage is not None:
        if len(intervention_stage) != len(dataset):
            raise ValueError(
                "Intervention-stage column must align with dataset frames: "
                f"stage={len(intervention_stage)} dataset={len(dataset)}"
            )
        intervention_stage_t = torch.as_tensor(
            intervention_stage,
            dtype=torch.float32,
        ).reshape(-1)
    intervention_reason_t = None
    handoff_mask = intervention_t > 0.5
    if intervention_stage_t is not None:
        handoff_mask = handoff_mask | (intervention_stage_t != INTERVENTION_STAGE_POLICY)
    if intervention_reason is not None:
        if len(intervention_reason) != len(dataset):
            raise ValueError(
                "Intervention-reason column must align with dataset frames: "
                f"reason={len(intervention_reason)} dataset={len(dataset)}"
            )
        reason_float = torch.as_tensor(intervention_reason, dtype=torch.float32).reshape(-1)
        reason_int = reason_float.round().to(torch.long)
        if not torch.allclose(reason_float, reason_int.to(torch.float32)):
            raise ValueError("complementary_info.intervention_reason contains non-integer values")
        allowed = {
            int(INTERVENTION_REASON_NONE),
            int(INTERVENTION_REASON_CORRECTIVE),
            int(INTERVENTION_REASON_PROACTIVE),
        }
        observed = set(int(value) for value in reason_int.unique().tolist())
        if not observed.issubset(allowed):
            raise ValueError(
                "complementary_info.intervention_reason contains unsupported codes: "
                f"{sorted(observed - allowed)}"
            )
        missing_mask = handoff_mask & (reason_int == int(INTERVENTION_REASON_NONE))
        if bool(missing_mask.any()):
            if missing_intervention_reason == "error":
                raise ValueError(
                    "Intervention frames are missing an intervention_reason label; "
                    "relabel the dataset or choose an explicit fallback."
                )
            if missing_intervention_reason == "legacy":
                _log(
                    "intervention_reason is incomplete; falling back to legacy credit semantics"
                )
                reason_int = None
            else:
                fill = (
                    int(INTERVENTION_REASON_CORRECTIVE)
                    if missing_intervention_reason == "corrective"
                    else int(INTERVENTION_REASON_PROACTIVE)
                )
                reason_int[missing_mask] = fill
        if reason_int is not None:
            intervention_reason_t = reason_int
    elif bool(handoff_mask.any()):
        if missing_intervention_reason == "error":
            raise KeyError(
                "Mixed provenance contains interventions but no "
                "complementary_info.intervention_reason column"
            )
        if missing_intervention_reason in {"corrective", "proactive"}:
            fill = (
                int(INTERVENTION_REASON_CORRECTIVE)
                if missing_intervention_reason == "corrective"
                else int(INTERVENTION_REASON_PROACTIVE)
            )
            intervention_reason_t = torch.zeros(len(dataset), dtype=torch.long)
            intervention_reason_t[handoff_mask] = fill
    _log(
        "collector provenance: "
        f"frames={len(dataset)} intervention_frac={float((intervention_t > 0.5).float().mean()):.3f} "
        f"collector_ids={sorted(int(value) for value in collector_t.unique().tolist())} "
        f"intervention_stage={'present' if intervention_stage_t is not None else 'legacy-absent'}"
        f" intervention_reason="
        f"{'present' if intervention_reason_t is not None else 'legacy-absent'}"
    )
    return FrameProvenance(
        is_intervention=intervention_t,
        collector_policy_id=collector_t,
        intervention_stage=intervention_stage_t,
        intervention_reason=intervention_reason_t,
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
    legacy_handoff_policy: str = "majority",
) -> tuple[int, bool, bool]:
    intervention = provenance.is_intervention[start_frame : start_frame + chunk_length]
    if intervention.numel() != chunk_length:
        raise ValueError(
            f"Provenance window at frame {start_frame} has {intervention.numel()} "
            f"values, expected {chunk_length}"
        )
    if provenance.intervention_stage is not None:
        stage = provenance.intervention_stage[start_frame : start_frame + chunk_length]
        if stage.numel() != chunk_length:
            raise ValueError(
                f"Intervention-stage window at frame {start_frame} has {stage.numel()} "
                f"values, expected {chunk_length}"
            )
        is_policy_chunk = bool(torch.all(stage == INTERVENTION_STAGE_POLICY))
        is_teleop_chunk = bool(torch.all(stage == INTERVENTION_STAGE_TELEOP))
        if is_teleop_chunk:
            if not bool(torch.all(intervention > 0.5)):
                raise ValueError(
                    f"Teleop stage at frame {start_frame} is inconsistent with is_intervention"
                )
            return TRANSITION_SOURCE_HUMAN_OVERRIDE, True, True
        if not is_policy_chunk:
            # Any hold/release frame, or a chunk crossing a handoff boundary,
            # is a safety transition rather than policy or human supervision.
            return TRANSITION_SOURCE_DEMO, False, False
    else:
        if legacy_handoff_policy == "drop":
            is_policy_chunk = bool(torch.all(intervention <= 0.5))
            is_teleop_chunk = bool(torch.all(intervention > 0.5))
            if is_teleop_chunk:
                return TRANSITION_SOURCE_HUMAN_OVERRIDE, True, True
            if not is_policy_chunk:
                return TRANSITION_SOURCE_DEMO, False, False
        human_frames = int((intervention > 0.5).sum().item())
        human_override = human_frames >= chunk_length - human_frames
        if human_override:
            return TRANSITION_SOURCE_HUMAN_OVERRIDE, True, True

    collector = provenance.collector_policy_id[start_frame : start_frame + chunk_length]
    collector_id = _dominant_collector_id(collector)
    source = {
        0: TRANSITION_SOURCE_DEMO,
        1: TRANSITION_SOURCE_WARMUP_VLA,
        2: TRANSITION_SOURCE_RL_AUTONOMOUS,
    }.get(collector_id, TRANSITION_SOURCE_RL_AUTONOMOUS)
    return source, False, True


def _intervention_reason_for_chunk(
    provenance: FrameProvenance,
    start_frame: int,
    chunk_length: int,
) -> int:
    if provenance.intervention_reason is None:
        return int(INTERVENTION_REASON_NONE)
    values = provenance.intervention_reason[start_frame : start_frame + chunk_length]
    nonzero = values[values != int(INTERVENTION_REASON_NONE)].unique()
    if nonzero.numel() != 1:
        raise ValueError(
            f"Intervention chunk at frame {start_frame} must have exactly one non-zero "
            f"reason, got {nonzero.tolist()}"
        )
    return int(nonzero.item())


def _policy_run_credit_semantics(
    provenance: FrameProvenance,
    episode_start: int,
    episode_stop: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """Map policy frames to the following takeover's run id and reason.

    The reason is stored on hold/teleop/release frames because it is only known
    when the operator presses a takeover key.  This helper propagates that
    reason backward within the immediately preceding contiguous policy run,
    without crossing an earlier human segment.
    """
    if provenance.intervention_reason is None:
        return {}, {}
    if provenance.intervention_stage is not None:
        policy_mask = (
            provenance.intervention_stage[episode_start:episode_stop]
            == INTERVENTION_STAGE_POLICY
        )
    else:
        policy_mask = provenance.is_intervention[episode_start:episode_stop] <= 0.5

    run_by_frame: dict[int, int] = {}
    reason_by_frame: dict[int, int] = {}
    relative = 0
    run_id = 0
    frame_count = episode_stop - episode_start
    while relative < frame_count:
        if not bool(policy_mask[relative]):
            relative += 1
            continue
        policy_start = relative
        while relative < frame_count and bool(policy_mask[relative]):
            relative += 1
        policy_stop = relative
        handoff_stop = relative
        while handoff_stop < frame_count and not bool(policy_mask[handoff_stop]):
            handoff_stop += 1
        block = provenance.intervention_reason[
            episode_start + policy_stop : episode_start + handoff_stop
        ]
        nonzero = block[block != int(INTERVENTION_REASON_NONE)].unique()
        reason = int(INTERVENTION_REASON_NONE)
        if nonzero.numel() > 1:
            raise ValueError(
                "One handoff segment contains multiple intervention reasons: "
                f"{nonzero.tolist()}"
            )
        if nonzero.numel() == 1:
            reason = int(nonzero.item())
        for offset in range(policy_start, policy_stop):
            frame = episode_start + offset
            run_by_frame[frame] = run_id
            reason_by_frame[frame] = reason
        run_id += 1
    return run_by_frame, reason_by_frame


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
    episode_ids: list[int] | None = None,
) -> tuple[list[int], list[int], dict[str, dict[str, int]]]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be within [0, 1], got {train_ratio}")
    selected_episode_ids = list(range(n_episodes)) if episode_ids is None else list(episode_ids)
    if len(selected_episode_ids) != len(set(selected_episode_ids)):
        raise ValueError("episode_ids contains duplicates")
    invalid_episode_ids = sorted(
        episode_id for episode_id in selected_episode_ids if not 0 <= episode_id < n_episodes
    )
    if invalid_episode_ids:
        raise ValueError(
            f"episode_ids contains indices outside [0, {n_episodes}): {invalid_episode_ids}"
        )
    if provenance is None or not stratify_provenance:
        shuffled_episode_ids = list(selected_episode_ids)
        random.Random(seed).shuffle(shuffled_episode_ids)
        n_train = int(train_ratio * len(shuffled_episode_ids))
        return shuffled_episode_ids[:n_train], shuffled_episode_ids[n_train:], {}

    groups: dict[tuple[str, str], list[int]] = {}
    for ep_id in selected_episode_ids:
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
    demo_reference_mode: str = "vla",
    human_reference_mode: str = "executed",
    legacy_handoff_policy: str = "majority",
    actor_bc_mode: str = "legacy",
    exec_action_is_actual_sent: bool = False,
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
    for transition in transitions:
        transition.exec_action_is_actual_sent = torch.tensor(
            float(exec_action_is_actual_sent)
        )
    if provenance is None:
        if demo_reference_mode == "executed":
            for transition in transitions:
                transition.bc_target_chunk = transition.exec_chunk.clone()
        return _apply_actor_bc_semantics(
            transitions,
            mode=actor_bc_mode,
            episode_success=episode_success,
            episode_has_human=False,
        )

    episode_start_frame = min(frame_indices)
    episode_has_human = bool(
        (provenance.is_intervention[episode_start_frame : episode_last_frame + 1] > 0.5)
        .any()
        .item()
    )

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

    policy_run_by_frame, policy_reason_by_frame = _policy_run_credit_semantics(
        provenance,
        episode_start=min(frame_indices),
        episode_stop=episode_last_frame + 1,
    )
    usable_by_anchor: dict[int, bool] = {}
    policy_run_by_anchor: dict[int, int] = {}
    for transition, start_frame in zip(transitions, start_anchors, strict=True):
        source, human_override, usable = _chunk_provenance(
            provenance,
            start_frame,
            chunk_length,
            legacy_handoff_policy=legacy_handoff_policy,
        )
        usable_by_anchor[start_frame] = usable
        transition.source = torch.tensor(source)
        transition.intervention = torch.tensor(float(human_override))
        reason = int(INTERVENTION_REASON_NONE)
        if human_override:
            # Human-controlled samples never enter either TD or actor-Q. The
            # independent bootstrap gate is also zero even for legacy datasets
            # that have no typed intervention_reason metadata.
            transition.critic_mask = torch.tensor(0.0)
            transition.actor_q_mask = torch.tensor(0.0)
            transition.bootstrap_mask = torch.tensor(0.0)
        if provenance.intervention_reason is not None:
            if human_override:
                reason = _intervention_reason_for_chunk(
                    provenance,
                    start_frame,
                    chunk_length,
                )
            else:
                reason = policy_reason_by_frame.get(
                    start_frame,
                    int(INTERVENTION_REASON_NONE),
                )
                run_id = policy_run_by_frame.get(start_frame)
                if run_id is not None:
                    policy_run_by_anchor[start_frame] = run_id
        transition.intervention_reason = torch.tensor(reason, dtype=torch.long)
        # Never overwrite proposal/ref: training and deployment must condition
        # on the same independently generated VLA action. Only the BC target is
        # replaced on human chunks.
        transition.proposal_chunk = transition.ref_chunk
        use_executed_target = (
            source == TRANSITION_SOURCE_DEMO and demo_reference_mode == "executed"
        ) or (
            human_override and human_reference_mode == "executed"
        )
        transition.bc_target_chunk = (
            transition.exec_chunk.clone()
            if use_executed_target
            else transition.proposal_chunk
        )
        transition.next_proposal_chunk = transition.next_ref_chunk

    anchor_to_index = {anchor: index for index, anchor in enumerate(start_anchors)}

    authority_boundary_by_run: dict[int, int] = {}
    for transition, start_frame in zip(transitions, start_anchors, strict=True):
        if not usable_by_anchor[start_frame]:
            continue
        if int(transition.intervention_reason.item()) not in {
            int(INTERVENTION_REASON_CORRECTIVE),
            int(INTERVENTION_REASON_PROACTIVE),
        }:
            continue
        if int(transition.source.item()) == TRANSITION_SOURCE_HUMAN_OVERRIDE:
            continue
        run_id = policy_run_by_anchor.get(start_frame)
        if run_id is not None:
            authority_boundary_by_run[run_id] = max(
                start_frame,
                authority_boundary_by_run.get(run_id, start_frame),
            )

    should_filter = (
        provenance.intervention_stage is not None
        or legacy_handoff_policy == "drop"
    )
    if not should_filter:
        return _apply_actor_bc_semantics(
            transitions,
            mode=actor_bc_mode,
            episode_success=episode_success,
            episode_has_human=episode_has_human,
        )

    filtered: list[ChunkTransition] = []
    for transition, start_frame in zip(transitions, start_anchors, strict=True):
        if not usable_by_anchor[start_frame]:
            continue
        run_id = policy_run_by_anchor.get(start_frame)
        authority_boundary = (
            run_id is not None
            and authority_boundary_by_run.get(run_id) == start_frame
        )
        if authority_boundary:
            # This is a control-authority censor, not an environment terminal or
            # an observed task failure. Stop bootstrap into the human-controlled
            # next state, and exclude the unknown outcome target entirely. Earlier
            # autonomous transitions in the run retain ordinary TD supervision.
            if bool(transition.done.item()):
                raise RuntimeError(
                    "Authority-boundary transition unexpectedly coincides with a real "
                    f"episode terminal at frame {start_frame}"
                )
            transition.bootstrap_mask = torch.tensor(0.0)
            transition.critic_mask = torch.tensor(0.0)
            transition.actor_q_mask = torch.tensor(0.0)
            filtered.append(transition)
            continue
        next_index = anchor_to_index.get(start_frame + chunk_length)
        if (
            not bool(transition.done.item())
            and next_index is not None
            and not usable_by_anchor[start_anchors[next_index]]
        ):
            # Avoid bootstrapping an otherwise valid chunk into a hold/release
            # handoff state.
            continue
        filtered.append(transition)
    return _apply_actor_bc_semantics(
        filtered,
        mode=actor_bc_mode,
        episode_success=episode_success,
        episode_has_human=episode_has_human,
    )


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
    demo_reference_mode: str,
    human_reference_mode: str,
    legacy_handoff_policy: str,
    actor_bc_mode: str,
    exec_action_is_actual_sent: bool,
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
        demo_reference_mode=demo_reference_mode,
        human_reference_mode=human_reference_mode,
        legacy_handoff_policy=legacy_handoff_policy,
        actor_bc_mode=actor_bc_mode,
        exec_action_is_actual_sent=exec_action_is_actual_sent,
    )


def _transition_summary(
    transitions: list[ChunkTransition], residual_delta_scale: float = 0.1,
) -> str:
    if residual_delta_scale <= 0:
        raise ValueError("residual_delta_scale must be positive")
    source_counts: dict[int, int] = {}
    success_terminal = 0
    failure_terminal = 0
    human_squared_error = 0.0
    human_elements = 0
    human_delta_abs_max = 0.0
    human_outside_bound = 0
    human_samples = 0
    demo_squared_error = 0.0
    demo_elements = 0
    demo_delta_abs_max = 0.0
    demo_outside_bound = 0
    demo_samples = 0
    critic_valid = 0
    actor_q_valid = 0
    actor_bc_valid = 0
    actual_sent_exec = 0
    actor_bc_valid_sources: dict[int, int] = {}
    corrective_transitions = 0
    proactive_transitions = 0
    for transition in transitions:
        source = int(transition.source.item())
        source_counts[source] = source_counts.get(source, 0) + 1
        critic_valid += int(float(transition.critic_mask.item()) > 0.5)
        actor_q_valid += int(float(transition.actor_q_mask.item()) > 0.5)
        bc_is_valid = int(float(transition.actor_bc_mask.item()) > 0.5)
        actor_bc_valid += bc_is_valid
        actual_sent_exec += int(
            float(transition.exec_action_is_actual_sent.item()) > 0.5
        )
        actor_bc_valid_sources[source] = actor_bc_valid_sources.get(source, 0) + bc_is_valid
        reason = int(transition.intervention_reason.item())
        corrective_transitions += int(reason == int(INTERVENTION_REASON_CORRECTIVE))
        proactive_transitions += int(reason == int(INTERVENTION_REASON_PROACTIVE))
        if source == TRANSITION_SOURCE_DEMO:
            # Audit expert reachability independently of which demo BC target
            # mode was selected, so legacy VLA-target caches do not report a
            # misleading zero expert/VLA mismatch.
            delta = (
                transition.exec_chunk.clamp(-1.0, 1.0)
                - transition.proposal_chunk.clamp(-1.0, 1.0)
            )
            demo_squared_error += float(delta.square().sum().item())
            demo_elements += delta.numel()
            demo_delta_abs_max = max(demo_delta_abs_max, float(delta.abs().max().item()))
            demo_outside_bound += int(
                float(delta.abs().max().item()) > residual_delta_scale + 1e-6
            )
            demo_samples += 1
        if source == TRANSITION_SOURCE_HUMAN_OVERRIDE:
            delta = (
                transition.bc_target_chunk.clamp(-1.0, 1.0)
                - transition.proposal_chunk.clamp(-1.0, 1.0)
            )
            human_squared_error += float(delta.square().sum().item())
            human_elements += delta.numel()
            human_delta_abs_max = max(human_delta_abs_max, float(delta.abs().max().item()))
            human_outside_bound += int(
                float(delta.abs().max().item()) > residual_delta_scale + 1e-6
            )
            human_samples += 1
        if bool(transition.done.item()) and float(transition.critic_mask.item()) > 0.5:
            if float(transition.reward_seq.sum().item()) > 0:
                success_terminal += 1
            else:
                failure_terminal += 1
    demo_rmse = (demo_squared_error / max(demo_elements, 1)) ** 0.5
    human_rmse = (human_squared_error / max(human_elements, 1)) ** 0.5
    return (
        f"sources={dict(sorted(source_counts.items()))} "
        f"critic_valid={critic_valid}/{len(transitions)} "
        f"actor_q_valid={actor_q_valid}/{len(transitions)} "
        f"actor_bc_valid={actor_bc_valid}/{len(transitions)} "
        f"exec_action_actual_sent={actual_sent_exec}/{len(transitions)} "
        f"actor_bc_valid_sources={dict(sorted(actor_bc_valid_sources.items()))} "
        f"corrective_labeled={corrective_transitions} "
        f"proactive_labeled={proactive_transitions} "
        f"terminal_success={success_terminal} terminal_failure={failure_terminal} "
        f"demo_vla_action_rmse={demo_rmse:.6f} "
        f"demo_vla_action_abs_max={demo_delta_abs_max:.6f} "
        "demo_target_outside_residual_bound_frac="
        f"{demo_outside_bound / max(demo_samples, 1):.6f} "
        f"human_vla_action_rmse={human_rmse:.6f} "
        f"human_vla_action_abs_max={human_delta_abs_max:.6f} "
        "human_target_outside_residual_bound_frac="
        f"{human_outside_bound / max(human_samples, 1):.6f} "
        f"residual_delta_scale={residual_delta_scale:.6f}"
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
    trim_pre_roll_frames = (
        args.chunk_length
        if args.trim_leading_idle_pre_roll_frames is None
        else args.trim_leading_idle_pre_roll_frames
    )
    if args.trim_leading_idle:
        if args.trim_leading_idle_threshold <= 0:
            raise ValueError("--trim-leading-idle-threshold must be > 0")
        if args.trim_leading_idle_hold_frames <= 0:
            raise ValueError("--trim-leading-idle-hold-frames must be > 0")
        if trim_pre_roll_frames < 0:
            raise ValueError("--trim-leading-idle-pre-roll-frames must be >= 0")
        if args.trim_leading_idle_action_dims <= 0:
            raise ValueError("--trim-leading-idle-action-dims must be > 0")
        _log(
            "leading-idle trim enabled: "
            f"threshold={args.trim_leading_idle_threshold} "
            f"hold_frames={args.trim_leading_idle_hold_frames} "
            f"pre_roll_frames={trim_pre_roll_frames} "
            f"action_dims={args.trim_leading_idle_action_dims}"
        )
    provenance = _load_frame_provenance(
        dataset,
        args.provenance_mode,
        args.missing_intervention_reason,
    )
    exec_action_is_actual_sent = (
        "complementary_info.requested_action"
        in getattr(dataset, "features", {})
    )
    _log(
        "dataset action semantics: "
        + (
            "actual-sent (requested_action provenance present)"
            if exec_action_is_actual_sent
            else "legacy requested/pre-clipping (requested_action provenance absent)"
        )
    )
    _log(
        f"actor BC mode={args.actor_bc_mode}; proposal remains the actor input/residual base"
    )
    n_episodes = dataset.num_episodes
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)
    excluded_episode_ids = sorted(set(args.exclude_episode_indices))
    invalid_excluded_ids = [
        episode_id for episode_id in excluded_episode_ids if not 0 <= episode_id < n_episodes
    ]
    if invalid_excluded_ids:
        raise ValueError(
            f"--exclude-episode-indices contains indices outside [0, {n_episodes}): "
            f"{invalid_excluded_ids}"
        )
    selected_episode_ids = [
        episode_id for episode_id in range(n_episodes) if episode_id not in excluded_episode_ids
    ]
    if n_episodes > 0 and not selected_episode_ids:
        raise ValueError("All selected episodes were excluded")
    _log(
        f"episodes: selected={len(selected_episode_ids)} of {n_episodes} "
        f"(dataset total={dataset.num_episodes}, excluded={excluded_episode_ids}); "
        f"batch_size={args.batch_size} num_workers={args.num_workers}"
    )

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
            episode_ids=selected_episode_ids,
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
            trimmed_episode_count = 0
            trimmed_frame_count = 0
            for k, ep_id in enumerate(eps):
                ep_meta = dataset.meta.episodes
                ep_from = int(ep_meta["dataset_from_index"][ep_id])
                ep_to = int(ep_meta["dataset_to_index"][ep_id])
                cache_from = ep_from
                motion_onset = None
                if args.trim_leading_idle:
                    cache_from, motion_onset = _trimmed_episode_start(
                        dataset,
                        episode_id=ep_id,
                        episode_start=ep_from,
                        episode_stop=ep_to,
                        threshold=args.trim_leading_idle_threshold,
                        hold_frames=args.trim_leading_idle_hold_frames,
                        pre_roll_frames=trim_pre_roll_frames,
                        action_dims=args.trim_leading_idle_action_dims,
                    )
                    trimmed_frames = cache_from - ep_from
                    trimmed_episode_count += int(trimmed_frames > 0)
                    trimmed_frame_count += trimmed_frames
                frame_indices = build_overlap_frame_indices(
                    episode_start=cache_from,
                    episode_stop=ep_to,
                    chunk_length=args.chunk_length,
                    stride=args.frame_stride,
                )
                episode_success = _episode_success_from_metadata(
                    dataset,
                    ep_id,
                    args.missing_episode_success,
                )
                trim_details = ""
                if motion_onset is not None:
                    trim_details = (
                        f" raw_frames={ep_to-ep_from} trimmed={cache_from-ep_from} "
                        f"onset_rel={motion_onset-ep_from}"
                    )
                _log(
                    f"  [{split_name}] ep {k+1}/{len(eps)} id={ep_id} "
                    f"frames={ep_to-cache_from}{trim_details} chunks={len(frame_indices)} "
                    f"(total transitions={len(all_tx)}, wall={time.time()-t_start:.0f}s)"
                )
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
                    demo_reference_mode=args.demo_reference_mode,
                    human_reference_mode=args.human_reference_mode,
                    legacy_handoff_policy=args.legacy_handoff_policy,
                    actor_bc_mode=args.actor_bc_mode,
                    exec_action_is_actual_sent=exec_action_is_actual_sent,
                )
                raw_transition_count = sum(
                    1
                    for frame in frame_indices
                    if frame + args.chunk_length <= ep_to - 1
                )
                dropped = raw_transition_count - len(ep_tx)
                if dropped:
                    _log(
                        f"    ep{ep_id}: filtered {dropped}/{raw_transition_count} "
                        "handoff-boundary transitions"
                    )
                all_tx.extend(ep_tx)
                if (k + 1) % 5 == 0 or (k + 1) == len(eps):
                    _save_partial(out_dir, split_name, all_tx, f"{split_name} ep {k+1}/{len(eps)}")
            _log(
                f"  [{split_name}] final "
                f"{_transition_summary(all_tx, args.residual_delta_scale)}"
            )
            if args.trim_leading_idle:
                _log(
                    f"  [{split_name}] leading-idle trim summary: "
                    f"episodes_trimmed={trimmed_episode_count}/{len(eps)} "
                    f"frames_trimmed={trimmed_frame_count}"
                )
    finally:
        capture.detach()

    _log(f"done, total wall {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
