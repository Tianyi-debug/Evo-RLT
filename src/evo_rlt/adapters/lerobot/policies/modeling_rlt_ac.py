from __future__ import annotations

import copy
import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import torch
from torch import Tensor
from safetensors.torch import load_file as load_safetensors_file
from typing_extensions import Unpack

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from evo_rlt.adapters.lerobot.policies.action_modifier import PrefixOutputCapture, RLTActionModifier
from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
from evo_rlt.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy
from evo_rlt.adapters.lerobot.policies.vla_backbone import (
    configure_vla_rtc,
    infer_num_image_tokens,
)
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic
from evo_rlt.core.corrective_risk import load_corrective_risk_checkpoint
from evo_rlt.core.losses import (
    actor_loss_with_diagnostics,
    critic_loss_with_diagnostics,
)
from evo_rlt.core.interfaces import (
    TRANSITION_SOURCE_DEMO,
    TRANSITION_SOURCE_HUMAN_OVERRIDE,
    TRANSITION_SOURCE_RL_AUTONOMOUS,
    TRANSITION_SOURCE_WARMUP_VLA,
)
from evo_rlt.core.phase_controller import PhaseController
from evo_rlt.core.utils import soft_update

log = logging.getLogger(__name__)


class ChunkACPolicy(PreTrainedPolicy):
    """Chunk-level TD3+BC offline RL policy with VLA reference.

    Holds two frozen backbones in __dict__ (so they don't land in safetensors):
      * `_rl_token_policy`: an RLTokenPolicy loaded from
        config.rl_token_pretrained_path. It internally holds the frozen VLA
        backbone, so we only need one stash for the whole VLA chain.

    Trainable modules (registered as nn submodules → in safetensors + optimizer):
      * `actor`: ChunkActor — refines or replaces the VLA reference chunk.
      * `critic`: TwinCritic — twin Q networks.
      * `target_critic`: TwinCritic — Polyak-averaged copy of critic.

    forward(batch) returns one scalar loss. Each forward == one critic update.
    Actor is updated every `actor_update_interval` calls. Target critic is
    soft-updated with `tau` after every critic step. UTD=k is achieved by
    setting lerobot-train `--steps` to outer*k.
    """

    config_class = ChunkACPolicyConfig
    name = "rlt_ac"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: ChunkACPolicyConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> ChunkACPolicy:
        # PreTrainedPolicy does not copy pretrained_name_or_path into the parsed
        # config.  Preserve it here so __init__ can migrate unversioned local
        # checkpoints to legacy v1 semantics before constructing the modules.
        if config is None:
            config = ChunkACPolicyConfig.from_pretrained(
                pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        config.pretrained_path = str(pretrained_name_or_path)
        policy = super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )
        policy._reset_actor_uncertainty_ema_from_config()
        return policy

    def __init__(
        self,
        config: ChunkACPolicyConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: ChunkACPolicyConfig = config
        self._apply_checkpoint_semantics_compat()

        rl_token_policy = self._load_rl_token_policy()
        object.__setattr__(self, "_rl_token_policy", rl_token_policy)
        self._validate_rl_token_arch(rl_token_policy)

        state_dim = config.rl_token_dim + config.proprio_dim
        chunk_dim = config.chunk_length * config.action_dim

        self.actor = ChunkActor(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            fixed_std=config.actor_fixed_std,
            ref_dropout_p=config.actor_ref_dropout_p,
            activation=config.actor_activation,
            layer_norm=config.actor_layer_norm,
            residual=config.actor_residual,
            proprio_dim=config.proprio_dim,
            state_normalization=config.state_normalization,
            action_residual=config.actor_action_residual,
            delta_scale=config.actor_delta_scale,
            delta_scale_per_action_dim=config.actor_delta_scale_per_action_dim,
        )
        self.critic = TwinCritic(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            activation=config.critic_activation,
            layer_norm=config.critic_layer_norm,
            residual=config.critic_residual,
            proprio_dim=config.proprio_dim,
            state_normalization=config.state_normalization,
        )
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.target_critic.eval()
        if config.training_stage in {"human_bc", "teacher_bc", "actor_refine"}:
            for p in self.critic.parameters():
                p.requires_grad = False
        elif config.training_stage == "critic_only":
            for p in self.actor.parameters():
                p.requires_grad = False

        # Persistent step counter — survives ckpt save/load.
        self.register_buffer("_critic_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("_human_bc_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("_teacher_bc_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("_actor_refine_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer(
            "_actor_refine_batch_fingerprint",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_actor_bc_tau_low_ema",
            torch.tensor(float(config.actor_bc_uncertainty_tau_low)),
            persistent=True,
        )
        self.register_buffer(
            "_actor_bc_tau_high_ema",
            torch.tensor(float(config.actor_bc_uncertainty_tau_high)),
            persistent=True,
        )
        self._diagnostics_jsonl_initialized = False
        self._legacy_transition_schema_warned = False
        # Kept outside nn.Module registration so it is neither optimized nor
        # serialized into the student checkpoint. It is created lazily only by
        # the teacher_bc training path.
        object.__setattr__(self, "_teacher_actor", None)
        # Like the frozen teacher, risk is an independently trained artifact.
        # Keeping it unregistered prevents optimizer/checkpoint contamination.
        object.__setattr__(self, "_corrective_risk_head", None)

        # Deploy-only: lazy build at .reset() time.
        self.modifier: RLTActionModifier | None = None
        # Deploy toggle (set by lerobot_rlt_record). False => the actor sees a
        # zeroed VLA reference chunk in RL phase (mirrors training ref-dropout).
        self.vla_ref: bool = True
        self._rtc_config: RTCConfig | None = None
        self._vla_rtc_config: RTCConfig | None = None
        self._active_pi05_rtc_config: RTCConfig | None = None
        self._rtc_action_queue: ActionQueue | None = None
        self._rtc_latency_tracker: LatencyTracker | None = None
        self._rtc_fps: float = 30.0
        self._rtc_action_queue_size_to_get_new_actions: int = 0
        self._rtc_worker: Thread | None = None
        self._rtc_worker_error: Exception | None = None
        self._rtc_generation: int = 0
        self._rtc_lock = Lock()
        self._rtc_inference_lock = Lock()
        self._rtc_step_metadata: deque[Any] = deque()
        self._rtc_selected_step_metadata: deque[Any] = deque()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _apply_checkpoint_semantics_compat(self) -> None:
        """Keep checkpoints created before AC v2 on their original semantics.

        The v2 transformation is stateless and does not change tensor shapes,
        which makes old weights technically loadable but would silently change
        their behavior.  Local LeRobot deployment records ``pretrained_path``;
        inspect its config and pin unversioned checkpoints to v1.
        """
        path_value = self.config.pretrained_path
        if path_value:
            config_path = Path(path_value).expanduser() / "config.json"
            if config_path.is_file():
                with config_path.open() as f:
                    saved_config = json.load(f)
                if "ac_semantics_version" not in saved_config:
                    self.config.ac_semantics_version = 1

        if self.config.ac_semantics_version == 1:
            self.config.state_normalization = "none"
            self.config.actor_action_residual = False
            log.warning(
                "Loading legacy AC v1 semantics: no RL-token state normalization "
                "and absolute actor actions. Retrain with AC v2 before always_rl deployment."
            )

    def _load_rl_token_policy(self) -> RLTokenPolicy:
        if not self.config.rl_token_pretrained_path:
            raise ValueError(
                "ChunkACPolicy requires config.rl_token_pretrained_path "
                "pointing at a saved RLTokenPolicy checkpoint dir."
            )
        policy = RLTokenPolicy.from_pretrained(self.config.rl_token_pretrained_path)
        for p in policy.parameters():
            p.requires_grad = False
        policy.eval()
        if self.config.vla_type != "auto":
            policy.config.vla_type = self.config.vla_type
        return policy

    def _validate_rl_token_arch(self, rl_token_policy: RLTokenPolicy) -> None:
        rtp_cfg = rl_token_policy.config
        if not self.config.rl_token_dim:
            self.config.rl_token_dim = rtp_cfg.rl_token_dim
        if rtp_cfg.rl_token_dim != self.config.rl_token_dim:
            raise ValueError(
                f"rl_token_dim mismatch: ChunkACPolicyConfig={self.config.rl_token_dim} vs "
                f"RLTokenPolicy ckpt={rtp_cfg.rl_token_dim}"
            )
        if rtp_cfg.rl_token_num_rl_tokens != self.config.rl_token_num_rl_tokens:
            raise ValueError(
                f"num_rl_tokens mismatch: ChunkACPolicyConfig={self.config.rl_token_num_rl_tokens} vs "
                f"RLTokenPolicy ckpt={rtp_cfg.rl_token_num_rl_tokens}"
            )

    # ------------------------------------------------------------------
    # Training: forward
    # ------------------------------------------------------------------

    def _coerce_batch(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Convert ChunkTransition-style batch into the dict expected by losses.

        New caches separate ``proposal_chunk`` (actor input/residual base) from
        ``bc_target_chunk`` (human action on HIL chunks). Legacy ref-only caches
        remain loadable by treating ref as both proposal and target.
        """
        out: dict[str, Tensor] = {}
        if (
            ("proposal_chunk" not in batch or "bc_target_chunk" not in batch)
            and not getattr(self, "_legacy_transition_schema_warned", False)
        ):
            log.warning(
                "Training from a legacy ref-only transition cache: ref_chunk is "
                "being used as both proposal and BC target. Rebuild the raw dataset "
                "cache to learn VLA-to-human corrections."
            )
            self._legacy_transition_schema_warned = True
        for k in (
            "state_vec",
            "exec_chunk",
            "reward_seq",
            "next_state_vec",
            "done",
            "actual_steps",
        ):
            if k not in batch:
                raise KeyError(f"ChunkACPolicy.forward missing batch key: {k!r}")
            v = batch[k]
            if not isinstance(v, Tensor):
                v = torch.as_tensor(v)
            out[k] = v

        proposal = batch.get("proposal_chunk", batch.get("ref_chunk"))
        if proposal is None:
            raise KeyError(
                "ChunkACPolicy.forward requires 'proposal_chunk' "
                "(or legacy 'ref_chunk')"
            )
        bc_target = batch.get("bc_target_chunk", batch.get("ref_chunk", proposal))
        next_proposal = batch.get(
            "next_proposal_chunk",
            batch.get("next_ref_chunk"),
        )
        if next_proposal is None:
            raise KeyError(
                "ChunkACPolicy.forward requires 'next_proposal_chunk' "
                "(or legacy 'next_ref_chunk')"
            )
        for key, value in (
            ("proposal_chunk", proposal),
            ("bc_target_chunk", bc_target),
            ("next_proposal_chunk", next_proposal),
        ):
            if not isinstance(value, Tensor):
                value = torch.as_tensor(value)
            out[key] = value

        out["exec_chunk_flat"] = out["exec_chunk"].flatten(start_dim=-2)
        out["proposal_chunk_flat"] = out["proposal_chunk"].flatten(start_dim=-2)
        out["bc_target_chunk_flat"] = out["bc_target_chunk"].flatten(start_dim=-2)
        out["next_proposal_flat"] = out["next_proposal_chunk"].flatten(start_dim=-2)
        # Compatibility aliases now consistently point to VLA proposals.
        out["ref_chunk_flat"] = out["proposal_chunk_flat"]
        out["next_ref_flat"] = out["next_proposal_flat"]
        for key in (
            "source",
            "intervention",
            "cache_index",
            "critic_mask",
            "actor_q_mask",
            "actor_bc_mask",
            "intervention_reason",
            "bootstrap_mask",
            "cache_semantics_version",
        ):
            if key in batch:
                value = batch[key]
                if not isinstance(value, Tensor):
                    value = torch.as_tensor(value)
                out[key] = value
        return out

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        tx = self._coerce_batch(batch)

        training_stage = getattr(self.config, "training_stage", "mixed_ac")
        if training_stage == "human_bc":
            return self._forward_human_bc(tx)
        if training_stage == "teacher_bc":
            return self._forward_teacher_bc(tx)
        if training_stage == "actor_refine":
            return self._forward_actor_refine(tx)

        c_loss, critic_info = critic_loss_with_diagnostics(
            self.critic,
            self.target_critic,
            self.actor,
            tx,
            gamma=self.config.gamma,
            C=self.config.chunk_length,
            target_q_clip=self.config.target_q_clip,
            bootstrap_mode=getattr(self.config, "critic_bootstrap_mode", "none"),
            bootstrap_keep_prob=getattr(
                self.config,
                "critic_bootstrap_keep_prob",
                0.8,
            ),
            bootstrap_seed=getattr(self.config, "critic_bootstrap_seed", 1000),
        )
        soft_update(self.target_critic, self.critic, self.config.tau)
        self._critic_step += 1

        source_info = self._source_fraction_diagnostics(tx)
        if training_stage == "critic_only":
            raw_info = {
                "loss_total_step": c_loss.detach(),
                "loss_critic": c_loss.detach(),
                "critic_step": self._critic_step.detach().clone(),
                "actor_update": False,
                "critic_only_stage": True,
                **critic_info,
                **source_info,
            }
            info = self._finalize_diagnostics(raw_info)
            return c_loss, info

        do_actor = (int(self._critic_step.item()) % self.config.actor_update_interval) == 0
        if do_actor:
            a_loss, actor_info = self._actor_loss_without_critic_grads(tx)
            total = c_loss + a_loss
            raw_info = {
                "loss_total_step": total.detach(),
                "loss_critic": c_loss.detach(),
                "loss_actor": a_loss.detach(),
                "critic_step": self._critic_step.detach().clone(),
                "actor_update": True,
                **critic_info,
                **source_info,
                **actor_info,
            }
            info = self._finalize_diagnostics(raw_info)
            return total, info

        raw_info = {
            "loss_total_step": c_loss.detach(),
            "loss_critic": c_loss.detach(),
            "critic_step": self._critic_step.detach().clone(),
            "actor_update": False,
            **critic_info,
            **source_info,
        }
        info = self._finalize_diagnostics(raw_info)
        return c_loss, info

    def _forward_human_bc(self, tx: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        source = tx.get("source")
        if source is None:
            raise KeyError("human_bc requires the transition cache to contain source")
        source = source.reshape(-1)
        if not bool(torch.all(source == 3)):
            observed = sorted(int(value) for value in source.detach().unique().cpu().tolist())
            raise ValueError(f"human_bc accepts only source=3 batches, got sources={observed}")

        proposal = tx["proposal_chunk_flat"]
        raw_target = tx["bc_target_chunk_flat"]
        mu, _ = self.actor.forward(tx["state_vec"], proposal, training=True)
        if getattr(self.actor, "action_residual", False):
            raw_target = raw_target.clamp(-1.0, 1.0)
        feasible_target = self.actor.project_to_residual_support(proposal, raw_target)
        bc_target = (
            feasible_target
            if getattr(self.config, "human_bc_target_mode", "raw")
            == "residual_feasible"
            else raw_target
        )
        bc_per_sample = (mu - bc_target).square().sum(dim=-1)
        loss = bc_per_sample.mean()
        self._human_bc_step += 1

        executable_proposal = proposal.detach().clamp(-1.0, 1.0)
        human_delta = raw_target.detach() - executable_proposal
        projection_info = self._human_projection_diagnostics(
            mu=mu,
            raw_target=raw_target,
            feasible_target=feasible_target,
            human_mask=torch.ones(mu.shape[0], dtype=torch.bool, device=mu.device),
        )
        raw_info = {
            "loss_total_step": loss.detach(),
            "loss_actor": loss.detach(),
            "loss_actor_bc_raw": loss.detach(),
            "actor_bc_target_rmse": (mu.detach() - bc_target.detach()).square().mean().sqrt(),
            "actor_ref_rmse": (mu.detach() - executable_proposal).square().mean().sqrt(),
            "human_vla_action_rmse": human_delta.square().mean().sqrt(),
            "human_target_outside_residual_bound_frac": projection_info[
                "human_target_outside_chunk_frac"
            ],
            "human_sample_frac": loss.detach().new_ones(()),
            "source_0_frac": loss.detach().new_zeros(()),
            "source_1_frac": loss.detach().new_zeros(()),
            "source_2_frac": loss.detach().new_zeros(()),
            "source_3_frac": loss.detach().new_ones(()),
            "human_bc_step": self._human_bc_step.detach().clone(),
            "actor_update": True,
            "human_bc_stage": True,
            **projection_info,
        }
        return loss, self._finalize_diagnostics(raw_info)

    def _load_frozen_teacher_actor(self, like: Tensor) -> ChunkActor:
        teacher = getattr(self, "_teacher_actor", None)
        if teacher is not None:
            return teacher

        path_value = getattr(self.config, "actor_teacher_pretrained_path", "")
        if not path_value:
            raise ValueError(
                "teacher_bc requires config.actor_teacher_pretrained_path"
            )
        checkpoint_path = Path(path_value).expanduser()
        weights_path = (
            checkpoint_path
            if checkpoint_path.is_file()
            else checkpoint_path / "model.safetensors"
        )
        if not weights_path.is_file():
            raise FileNotFoundError(
                "Frozen teacher actor weights not found: "
                f"{weights_path}. Pass the AC pretrained_model directory."
            )

        full_state = load_safetensors_file(str(weights_path), device="cpu")
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in full_state.items()
            if key.startswith("actor.")
        }
        if not actor_state:
            raise ValueError(
                f"Frozen teacher checkpoint has no actor.* tensors: {weights_path}"
            )

        teacher = copy.deepcopy(self.actor).cpu()
        try:
            teacher.load_state_dict(actor_state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Frozen teacher actor architecture does not match the student: "
                f"{weights_path}"
            ) from error
        teacher.to(device=like.device, dtype=next(self.actor.parameters()).dtype)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
        object.__setattr__(self, "_teacher_actor", teacher)
        log.info("Loaded frozen warmup actor teacher from %s", weights_path)
        return teacher

    def _load_frozen_corrective_risk(self, like: Tensor):
        risk_head = getattr(self, "_corrective_risk_head", None)
        if risk_head is not None:
            return risk_head
        checkpoint = getattr(self.config, "corrective_risk_checkpoint", "")
        if not checkpoint:
            raise ValueError("corrective_risk trust requires corrective_risk_checkpoint")
        risk_head, metadata = load_corrective_risk_checkpoint(checkpoint, freeze=True)
        expected_state_dim = int(self.config.rl_token_dim + self.config.proprio_dim)
        expected_action_dim = int(self.config.chunk_length * self.config.action_dim)
        if risk_head.state_dim != expected_state_dim or risk_head.action_dim != expected_action_dim:
            raise ValueError(
                "corrective risk dimensions do not match actor_refine: "
                f"risk=({risk_head.state_dim}, {risk_head.action_dim}), "
                f"actor=({expected_state_dim}, {expected_action_dim})"
            )
        checkpoint_horizon = metadata.get("primary_future_k")
        if checkpoint_horizon != int(self.config.corrective_risk_horizon_chunks):
            raise ValueError(
                "corrective risk horizon mismatch: "
                f"checkpoint={checkpoint_horizon}, "
                f"config={self.config.corrective_risk_horizon_chunks}"
            )
        risk_head.to(device=like.device, dtype=like.dtype)
        for parameter in risk_head.parameters():
            parameter.requires_grad_(False)
        risk_head.eval()
        object.__setattr__(self, "_corrective_risk_head", risk_head)
        return risk_head

    @staticmethod
    def _teacher_supervision_masks(tx: dict[str, Tensor]) -> dict[str, Tensor]:
        source = tx.get("source")
        actor_bc_mask = tx.get("actor_bc_mask")
        intervention_reason = tx.get("intervention_reason")
        if source is None or actor_bc_mask is None or intervention_reason is None:
            raise KeyError(
                "teacher_bc requires source, actor_bc_mask, and intervention_reason "
                "from a typed outcome-aware cache"
            )

        source = source.reshape(-1).to(dtype=torch.long)
        actor_bc_valid = actor_bc_mask.reshape(-1) > 0.5
        reason = intervention_reason.reshape(-1).to(dtype=torch.long)
        allowed = (
            (source == TRANSITION_SOURCE_DEMO)
            | (source == TRANSITION_SOURCE_WARMUP_VLA)
            | (source == TRANSITION_SOURCE_RL_AUTONOMOUS)
            | (source == TRANSITION_SOURCE_HUMAN_OVERRIDE)
        )
        if not bool(torch.all(allowed)):
            observed = sorted(int(value) for value in source.detach().unique().cpu().tolist())
            raise ValueError(f"teacher_bc received unsupported source ids: {observed}")

        autonomous = (
            (source == TRANSITION_SOURCE_WARMUP_VLA)
            | (source == TRANSITION_SOURCE_RL_AUTONOMOUS)
        )
        autonomous_success = autonomous & actor_bc_valid
        corrective_prefix = autonomous & (~actor_bc_valid) & (reason == 1)
        proactive_prefix = autonomous & (~actor_bc_valid) & (reason == 2)
        autonomous_failure = autonomous & (~actor_bc_valid) & (reason == 0)
        malformed_success = autonomous_success & (reason != 0)
        if bool(malformed_success.any()):
            raise ValueError(
                "teacher_bc found autonomous BC-valid samples carrying a non-zero "
                "intervention_reason"
            )

        demo = source == TRANSITION_SOURCE_DEMO
        human = source == TRANSITION_SOURCE_HUMAN_OVERRIDE
        teacher = demo | autonomous_success | proactive_prefix
        if bool((human & teacher).any()):
            raise RuntimeError("human and teacher supervision masks must be disjoint")
        ignored = ~(human | teacher)
        return {
            "demo": demo,
            "autonomous_success": autonomous_success,
            "autonomous_failure": autonomous_failure,
            "corrective_prefix": corrective_prefix,
            "proactive_prefix": proactive_prefix,
            "human": human,
            "teacher": teacher,
            "ignored": ignored,
        }

    @staticmethod
    def _masked_scalar_mean(values: Tensor, mask: Tensor) -> Tensor:
        values = values.reshape(-1)
        weights = mask.reshape(-1).to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def _human_projection_diagnostics(
        self,
        *,
        mu: Tensor,
        raw_target: Tensor,
        feasible_target: Tensor,
        human_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Report raw/feasible fit and residual-support projection severity."""
        human_mask = human_mask.reshape(-1).to(device=mu.device, dtype=torch.bool)
        raw_mse = (mu.detach() - raw_target.detach()).square().mean(dim=-1)
        feasible_mse = (
            mu.detach() - feasible_target.detach()
        ).square().mean(dim=-1)
        projection = (raw_target.detach() - feasible_target.detach()).abs()
        projection_per_sample = projection.square().sum(dim=-1).sqrt()
        changed = projection > 1e-6

        action_dim = int(self.config.action_dim)
        if projection.shape[-1] % action_dim != 0:
            raise ValueError(
                f"flattened action width={projection.shape[-1]} is not divisible "
                f"by action_dim={action_dim}"
            )
        changed_steps = changed.reshape(changed.shape[0], -1, action_dim)
        selected_projection = projection_per_sample[human_mask]
        zero = mu.detach().new_zeros(())
        if selected_projection.numel():
            projection_mean = selected_projection.mean()
            projection_p50 = torch.quantile(selected_projection, 0.50)
            projection_p95 = torch.quantile(selected_projection, 0.95)
        else:
            projection_mean = projection_p50 = projection_p95 = zero

        info = {
            "human_raw_target_rmse": self._masked_scalar_mean(
                raw_mse, human_mask
            ).sqrt(),
            "human_feasible_target_rmse": self._masked_scalar_mean(
                feasible_mse, human_mask
            ).sqrt(),
            "human_projection_error_mean": projection_mean,
            "human_projection_error_p50": projection_p50,
            "human_projection_error_p95": projection_p95,
            "human_projection_fraction": self._masked_scalar_mean(
                changed_steps.any(dim=(1, 2)).float(), human_mask
            ),
            "human_target_outside_step_frac": self._masked_scalar_mean(
                changed_steps.any(dim=-1).float().mean(dim=-1), human_mask
            ),
            "human_target_outside_chunk_frac": self._masked_scalar_mean(
                changed_steps.any(dim=(1, 2)).float(), human_mask
            ),
        }
        for dim in range(action_dim):
            info[f"human_projection_dim_{dim}_frac"] = self._masked_scalar_mean(
                changed_steps[:, :, dim].float().mean(dim=-1), human_mask
            )
        if action_dim > 0:
            info["human_projection_gripper_frac"] = info[
                f"human_projection_dim_{action_dim - 1}_frac"
            ]
        return info

    def _forward_teacher_bc(self, tx: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """Learn human corrections while preserving the frozen warmup actor.

        Human BC and teacher distillation use disjoint semantic masks and are
        normalized independently. Demo and fully autonomous successful states
        preserve the warmup function. Corrective prefixes and autonomous
        failures are deliberately ignored; proactive prefixes are preserved.
        """
        masks = self._teacher_supervision_masks(tx)
        proposal = tx["proposal_chunk_flat"]
        student_mu, _ = self.actor.forward(tx["state_vec"], proposal, training=False)
        teacher_actor = self._load_frozen_teacher_actor(student_mu)
        with torch.no_grad():
            teacher_mu, _ = teacher_actor.forward(
                tx["state_vec"],
                proposal,
                training=False,
            )

        raw_human_target = tx["bc_target_chunk_flat"]
        if getattr(self.actor, "action_residual", False):
            raw_human_target = raw_human_target.clamp(-1.0, 1.0)
        feasible_human_target = self.actor.project_to_residual_support(
            proposal,
            raw_human_target,
        )
        human_target = (
            feasible_human_target
            if getattr(self.config, "human_bc_target_mode", "raw")
            == "residual_feasible"
            else raw_human_target
        )

        human_per_sample = (student_mu - human_target).square().sum(dim=-1)
        teacher_per_sample = (student_mu - teacher_mu).square().sum(dim=-1)
        human_raw = self._masked_scalar_mean(human_per_sample, masks["human"])
        teacher_raw = self._masked_scalar_mean(teacher_per_sample, masks["teacher"])
        human_weighted = float(self.config.human_bc_weight) * human_raw
        teacher_weighted = (
            float(self.config.teacher_distillation_weight) * teacher_raw
        )
        loss = human_weighted + teacher_weighted
        self._teacher_bc_step += 1

        student_teacher_mse = (student_mu.detach() - teacher_mu).square().mean(dim=-1)
        student_human_mse = (
            student_mu.detach() - human_target.detach()
        ).square().mean(dim=-1)
        teacher_human_mse = (
            teacher_mu.detach() - raw_human_target.detach()
        ).square().mean(dim=-1)
        projection_info = self._human_projection_diagnostics(
            mu=student_mu,
            raw_target=raw_human_target,
            feasible_target=feasible_human_target,
            human_mask=masks["human"],
        )

        raw_info: dict[str, Any] = {
            "loss_total_step": loss.detach(),
            "loss_actor": loss.detach(),
            "loss_actor_human_bc_raw": human_raw.detach(),
            "loss_actor_human_bc_weighted": human_weighted.detach(),
            "loss_actor_teacher_raw": teacher_raw.detach(),
            "loss_actor_teacher_weighted": teacher_weighted.detach(),
            "teacher_student_rmse": self._masked_scalar_mean(
                student_teacher_mse, masks["teacher"]
            ).sqrt(),
            "human_student_rmse": self._masked_scalar_mean(
                student_human_mse, masks["human"]
            ).sqrt(),
            "human_teacher_rmse": self._masked_scalar_mean(
                teacher_human_mse, masks["human"]
            ).sqrt(),
            "teacher_sample_frac": masks["teacher"].float().mean(),
            "human_sample_frac": masks["human"].float().mean(),
            "teacher_ignored_sample_frac": masks["ignored"].float().mean(),
            "teacher_bc_step": self._teacher_bc_step.detach().clone(),
            "actor_update": True,
            "teacher_bc_stage": True,
            **projection_info,
            **self._source_fraction_diagnostics(tx),
        }
        for category in (
            "demo",
            "autonomous_success",
            "autonomous_failure",
            "corrective_prefix",
            "proactive_prefix",
            "human",
        ):
            mask = masks[category]
            raw_info[f"teacher_{category}_sample_frac"] = mask.float().mean()
            raw_info[f"teacher_{category}_student_drift_rmse"] = (
                self._masked_scalar_mean(student_teacher_mse, mask).sqrt()
            )
        return loss, self._finalize_diagnostics(raw_info)

    def _forward_actor_refine(self, tx: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """Unified frozen-teacher + human BC + optional trusted Q objective."""
        masks = self._teacher_supervision_masks(tx)
        cache_index = tx.get("cache_index")
        if cache_index is None:
            raise KeyError("actor_refine requires stable cache_index for matched-run auditing")
        q_mask_value = tx.get("actor_q_mask")
        if q_mask_value is None:
            raise KeyError("actor_refine requires actor_q_mask from a typed cache")
        q_mask = q_mask_value.reshape(-1) > 0.5
        proposal = tx["proposal_chunk_flat"]
        student_mu, _ = self.actor.forward(tx["state_vec"], proposal, training=False)
        teacher_actor = self._load_frozen_teacher_actor(student_mu)
        with torch.no_grad():
            teacher_mu, _ = teacher_actor.forward(tx["state_vec"], proposal, training=False)

        raw_human_target = tx["bc_target_chunk_flat"]
        if getattr(self.actor, "action_residual", False):
            raw_human_target = raw_human_target.clamp(-1.0, 1.0)
        feasible_human_target = self.actor.project_to_residual_support(
            proposal,
            raw_human_target,
        )
        human_target = (
            feasible_human_target
            if getattr(self.config, "human_bc_target_mode", "raw") == "residual_feasible"
            else raw_human_target
        )
        human_per_sample = (student_mu - human_target).square().sum(dim=-1)
        teacher_per_sample = (student_mu - teacher_mu).square().sum(dim=-1)
        human_raw = self._masked_scalar_mean(human_per_sample, masks["human"])
        teacher_raw = self._masked_scalar_mean(teacher_per_sample, masks["teacher"])
        human_weighted = float(self.config.actor_human_weight) * human_raw
        teacher_weighted = float(self.config.actor_teacher_weight) * teacher_raw

        trust_mode = self.config.actor_q_trust_mode
        if trust_mode == "fixed":
            trust = torch.ones(student_mu.shape[0], device=student_mu.device, dtype=student_mu.dtype)
        elif trust_mode == "corrective_risk":
            risk_head = self._load_frozen_corrective_risk(student_mu)
            with torch.no_grad():
                risk = risk_head(tx["state_vec"].detach(), student_mu.detach()).sigmoid()
                trust = 1.0 - risk
        else:
            raise ValueError(f"unsupported actor_q_trust_mode {trust_mode!r}")
        trust = trust.detach()

        critic_parameters = list(self.critic.parameters())
        original_requires_grad = [parameter.requires_grad for parameter in critic_parameters]
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            q1, q2 = self.critic(tx["state_vec"], student_mu)
            q = torch.minimum(q1, q2).reshape(-1)
        finally:
            for parameter, requires_grad in zip(
                critic_parameters, original_requires_grad, strict=True
            ):
                parameter.requires_grad_(requires_grad)

        q_raw = -self._masked_scalar_mean(q, q_mask)
        q_weight = float(self.config.actor_q_weight_max)
        # The denominator is sum(actor_q_mask), never sum(trust).  This is the
        # invariant that makes risk actually suppress aggregate Q pressure.
        q_weighted = -self._masked_scalar_mean(q_weight * trust * q, q_mask)
        loss = human_weighted + teacher_weighted + q_weighted
        fingerprint = int(self._actor_refine_batch_fingerprint.item())
        modulus = 9_223_372_036_854_775_783
        for index in cache_index.detach().reshape(-1).cpu().tolist():
            fingerprint = (fingerprint * 1_000_003 + int(index) + 1) % modulus
        self._actor_refine_batch_fingerprint.fill_(fingerprint)
        self._actor_refine_step += 1

        selected_trust = trust[q_mask]
        if selected_trust.numel():
            trust_p10, trust_p50, trust_p90 = torch.quantile(
                selected_trust,
                torch.tensor([0.1, 0.5, 0.9], device=selected_trust.device),
            ).unbind()
            trust_mean = selected_trust.mean()
        else:
            trust_mean = trust_p10 = trust_p50 = trust_p90 = trust.new_zeros(())
        student_teacher_mse = (student_mu.detach() - teacher_mu).square().mean(dim=-1)
        student_human_mse = (
            student_mu.detach() - human_target.detach()
        ).square().mean(dim=-1)
        raw_info: dict[str, Any] = {
            "loss_total_step": loss.detach(),
            "loss_actor_total": loss.detach(),
            "loss_actor": loss.detach(),
            "loss_actor_human": human_weighted.detach(),
            "loss_actor_human_raw": human_raw.detach(),
            "loss_actor_teacher": teacher_weighted.detach(),
            "loss_actor_teacher_raw": teacher_raw.detach(),
            "loss_actor_q_raw": q_raw.detach(),
            "loss_actor_q_weighted": q_weighted.detach(),
            "actor_q_weight_max": q_weight,
            "actor_q_trust_mean": trust_mean.detach(),
            "actor_q_trust_p10": trust_p10.detach(),
            "actor_q_trust_p50": trust_p50.detach(),
            "actor_q_trust_p90": trust_p90.detach(),
            "actor_q_valid_frac": q_mask.float().mean(),
            "human_valid_frac": masks["human"].float().mean(),
            "teacher_valid_frac": masks["teacher"].float().mean(),
            "actor_teacher_rmse": self._masked_scalar_mean(
                student_teacher_mse, masks["teacher"]
            ).sqrt(),
            "actor_human_rmse": self._masked_scalar_mean(
                student_human_mse, masks["human"]
            ).sqrt(),
            "actor_refine_step": self._actor_refine_step.detach().clone(),
            "actor_refine_batch_fingerprint": (
                self._actor_refine_batch_fingerprint.detach().clone()
            ),
            "actor_update": True,
            "actor_refine_stage": True,
            **self._human_projection_diagnostics(
                mu=student_mu,
                raw_target=raw_human_target,
                feasible_target=feasible_human_target,
                human_mask=masks["human"],
            ),
            **self._source_fraction_diagnostics(tx),
        }
        return loss, self._finalize_diagnostics(raw_info)

    @staticmethod
    def _scalarize_diagnostics(info: dict[str, Any]) -> dict[str, float | int | bool]:
        scalarized: dict[str, float | int | bool] = {}
        for key, value in info.items():
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        f"Diagnostic {key!r} must be scalar, got shape {tuple(value.shape)}"
                    )
                value = value.detach().cpu().item()
            if isinstance(value, bool):
                scalarized[key] = value
            elif isinstance(value, int):
                scalarized[key] = value
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(f"Diagnostic {key!r} is not finite: {value}")
                scalarized[key] = value
            else:
                raise TypeError(
                    f"Diagnostic {key!r} has unsupported type {type(value).__name__}"
                )
        return scalarized

    @staticmethod
    def _source_fraction_diagnostics(batch: dict[str, Tensor]) -> dict[str, Tensor]:
        source = batch.get("source")
        if source is None:
            return {}
        source = source.detach().reshape(-1)
        return {
            f"source_{source_id}_frac": (source == source_id).float().mean()
            for source_id in range(4)
        }

    def _finalize_diagnostics(self, info: dict[str, Any]) -> dict[str, float | int | bool]:
        scalarized = self._scalarize_diagnostics(info)
        path_value = getattr(self.config, "diagnostics_jsonl_path", None)
        if path_value:
            path = Path(path_value).expanduser()
            if not getattr(self, "_diagnostics_jsonl_initialized", False):
                path.parent.mkdir(parents=True, exist_ok=True)
                self._diagnostics_jsonl_initialized = True
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(scalarized, sort_keys=True) + "\n")
        return scalarized

    def _actor_uncertainty_thresholds(self) -> tuple[float, float]:
        threshold_mode = getattr(
            self.config,
            "actor_bc_uncertainty_threshold_mode",
            "fixed",
        )
        if threshold_mode == "ema_quantile":
            low_buffer = getattr(self, "_actor_bc_tau_low_ema", None)
            high_buffer = getattr(self, "_actor_bc_tau_high_ema", None)
            if low_buffer is not None and high_buffer is not None:
                low = float(low_buffer.item())
                high = float(high_buffer.item())
            else:
                low = float(self.config.actor_bc_uncertainty_tau_low)
                high = float(self.config.actor_bc_uncertainty_tau_high)
        else:
            low = float(getattr(self.config, "actor_bc_uncertainty_tau_low", 0.0))
            high = float(getattr(self.config, "actor_bc_uncertainty_tau_high", 1.0))
        min_gap = float(getattr(self.config, "actor_bc_uncertainty_min_gap", 1e-6))
        return low, max(high, low + min_gap)

    def _reset_actor_uncertainty_ema_from_config(self) -> bool:
        if not getattr(
            self.config,
            "actor_bc_uncertainty_reset_ema_on_load",
            False,
        ):
            return False

        low = float(self.config.actor_bc_uncertainty_tau_low)
        high = float(self.config.actor_bc_uncertainty_tau_high)
        old_low = float(self._actor_bc_tau_low_ema.item())
        old_high = float(self._actor_bc_tau_high_ema.item())
        with torch.no_grad():
            self._actor_bc_tau_low_ema.fill_(low)
            self._actor_bc_tau_high_ema.fill_(high)

        # This is a one-shot load override. Saved checkpoints must resume from
        # their learned EMA buffers instead of resetting on every load.
        self.config.actor_bc_uncertainty_reset_ema_on_load = False
        log.info(
            "Reset actor disagreement EMA thresholds after checkpoint load: "
            "[%g, %g] -> [%g, %g]",
            old_low,
            old_high,
            low,
            high,
        )
        return True

    def _update_actor_uncertainty_thresholds(
        self,
        actor_info: dict[str, Tensor],
    ) -> tuple[float, float]:
        threshold_mode = getattr(
            self.config,
            "actor_bc_uncertainty_threshold_mode",
            "fixed",
        )
        weight_mode = getattr(self.config, "actor_bc_weight_mode", "fixed")
        if threshold_mode != "ema_quantile" or weight_mode != "disagreement":
            return self._actor_uncertainty_thresholds()

        low_buffer = getattr(self, "_actor_bc_tau_low_ema", None)
        high_buffer = getattr(self, "_actor_bc_tau_high_ema", None)
        if low_buffer is None or high_buffer is None:
            raise RuntimeError("EMA uncertainty mode requires persistent tau buffers")

        decay = float(getattr(self.config, "actor_bc_uncertainty_ema_decay", 0.95))
        min_gap = float(getattr(self.config, "actor_bc_uncertainty_min_gap", 1e-6))
        batch_low = actor_info["actor_disagreement_p50"].detach().to(low_buffer.device)
        batch_high = actor_info["actor_disagreement_p95"].detach().to(high_buffer.device)
        with torch.no_grad():
            updated_low = decay * low_buffer + (1.0 - decay) * batch_low
            updated_high = decay * high_buffer + (1.0 - decay) * batch_high
            updated_high = torch.maximum(updated_high, updated_low + min_gap)
            low_buffer.copy_(updated_low)
            high_buffer.copy_(updated_high)
        return float(low_buffer.item()), float(high_buffer.item())

    def _actor_loss_without_critic_grads(
        self,
        tx: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        for p in critic_params:
            p.requires_grad_(False)
        try:
            tau_low, tau_high = self._actor_uncertainty_thresholds()
            loss, info = actor_loss_with_diagnostics(
                self.actor,
                self.critic,
                tx,
                beta=self.config.beta,
                q_weight=getattr(self.config, "actor_q_weight", 1.0),
                weight_mode=getattr(self.config, "actor_bc_weight_mode", "fixed"),
                uncertainty_tau_low=tau_low,
                uncertainty_tau_high=tau_high,
                uncertainty_kappa=getattr(
                    self.config,
                    "actor_bc_uncertainty_kappa",
                    0.0,
                ),
                behavior_preservation_weight=getattr(
                    self.config,
                    "actor_behavior_preservation_weight",
                    0.0,
                ),
            )
            updated_low, updated_high = self._update_actor_uncertainty_thresholds(info)
            info.update(
                {
                    "actor_tau_low_used": torch.tensor(tau_low),
                    "actor_tau_high_used": torch.tensor(tau_high),
                    "actor_tau_low_updated": torch.tensor(updated_low),
                    "actor_tau_high_updated": torch.tensor(updated_high),
                }
            )
            return loss, info
        finally:
            for p in critic_params:
                p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Inference: predict_action_chunk + select_action
    # ------------------------------------------------------------------

    def _ensure_modifier(self) -> RLTActionModifier:
        if self.modifier is None:
            phase_ctrl = self._build_phase_controller()
            rl_token_module = self._rl_token_policy.rl_token
            self.modifier = RLTActionModifier(
                rl_token=rl_token_module,
                actor=self.actor,
                phase_ctrl=phase_ctrl,
                chunk_length=self.config.chunk_length,
                action_dim=self.config.action_dim,
                proprio_dim=self.config.proprio_dim,
                chunk_exec_steps=self.config.chunk_exec_steps,
                vla_ref=self.vla_ref,
                deterministic=self.config.deterministic,
            )
            self._prefix_capture = PrefixOutputCapture(
                token_pool_size=self.config.token_pool_size,
                image_only=self.config.image_only,
                num_image_tokens=self._compute_num_image_tokens(),
                num_per_camera=self.config.num_per_camera,
                active_camera_indices=self.config.active_camera_indices,
            )
            self._prefix_capture.attach(self._rl_token_policy._pi05)
        return self.modifier

    def _apply_phase_mode(self, ctrl: PhaseController) -> None:
        """Pin the phase controller to the phase encoded by phase_mode.

        always_rl -> critical, always_vla -> vla, manual -> left as-is. Must be
        re-applied after every reset(): modifier.reset() drops the controller
        back to VLA, which would otherwise silently disable always_rl from the
        second episode onward (record_loop resets the policy each episode).
        """
        if self.config.phase_mode == "always_rl":
            ctrl.trigger_critical()
        elif self.config.phase_mode == "always_vla":
            ctrl.trigger_vla()

    def _build_phase_controller(self) -> PhaseController:
        # ChunkACPolicyConfig.phase_mode (always_rl / always_vla / manual) is
        # validated in the config __post_init__. PhaseController only does
        # manual/learned transitions, so we pin the deploy phase explicitly.
        ctrl = PhaseController(mode="manual")
        self._apply_phase_mode(ctrl)
        return ctrl

    def _compute_num_image_tokens(self) -> int:
        return infer_num_image_tokens(
            self._rl_token_policy._pi05,
            image_resolution=self.config.image_resolution,
            camera_keys=self.config.camera_keys,
            num_per_camera=self.config.num_per_camera,
            required=self.config.image_only or bool(self.config.active_camera_indices),
        )

    def configure_rtc(
        self,
        rtc_config: RTCConfig,
        fps: float,
        action_queue_size_to_get_new_actions: int | None = None,
        vla_rtc_config: RTCConfig | None = None,
    ) -> None:
        """Enable RTC chunk replacement for deployment-time ``select_action``.

        ``predict_action_chunk`` already forwards RTC kwargs into the frozen VLA
        backbone. This runtime keeps an overlapping action queue so those kwargs
        contain the previous chunk leftovers and measured inference delay.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        self._rtc_config = rtc_config
        self._vla_rtc_config = vla_rtc_config or rtc_config
        self._active_pi05_rtc_config = None
        self._set_pi05_rtc_config(rtc_config)
        self._rtc_action_queue = ActionQueue(rtc_config)
        self._rtc_latency_tracker = LatencyTracker()
        self._rtc_fps = float(fps)
        if action_queue_size_to_get_new_actions is None:
            action_queue_size_to_get_new_actions = max(1, self.config.chunk_length - 1)
        self._rtc_action_queue_size_to_get_new_actions = action_queue_size_to_get_new_actions
        self._rtc_worker = None
        self._rtc_worker_error = None
        self._rtc_generation = 0
        self._rtc_step_metadata.clear()
        self._rtc_selected_step_metadata.clear()

    @property
    def _rtc_runtime_enabled(self) -> bool:
        return self._rtc_config is not None

    def _rtc_config_for_phase(self, mod: RLTActionModifier) -> RTCConfig:
        if self._rtc_config is None:
            raise RuntimeError("RTC runtime is not configured")
        if not mod.is_rl_phase and self._vla_rtc_config is not None:
            return self._vla_rtc_config
        return self._rtc_config

    def _set_pi05_rtc_config(self, rtc_config: RTCConfig) -> None:
        if self._active_pi05_rtc_config is rtc_config:
            return
        configure_vla_rtc(self._rl_token_policy._pi05, rtc_config)
        self._active_pi05_rtc_config = rtc_config

    def _clone_rtc_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                cloned[key] = value.detach().clone()
            elif isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = copy.deepcopy(value)
        return cloned

    def _raise_rtc_worker_error(self) -> None:
        if self._rtc_worker_error is None:
            return
        error = self._rtc_worker_error
        self._rtc_worker_error = None
        raise error

    def _join_finished_rtc_worker(self) -> None:
        if self._rtc_worker is None or self._rtc_worker.is_alive():
            return
        self._rtc_worker.join()
        self._rtc_worker = None

    def _wait_for_rtc_worker(self) -> None:
        if self._rtc_worker is None:
            return
        self._rtc_worker.join()
        self._rtc_worker = None
        self._raise_rtc_worker_error()

    def _reset_rtc_runtime(self) -> None:
        with self._rtc_lock:
            self._rtc_generation += 1
            if self._rtc_action_queue is not None:
                self._rtc_action_queue = ActionQueue(self._rtc_action_queue.cfg)
            if self._rtc_latency_tracker is not None:
                self._rtc_latency_tracker.reset()
            self._rtc_step_metadata.clear()
            self._rtc_selected_step_metadata.clear()
            self._rtc_worker_error = None
        self._join_finished_rtc_worker()

    def _prepare_rtc_request(
        self, batch: dict[str, Tensor]
    ) -> tuple[dict[str, Any], Tensor | None, int, int, float, int]:
        if self._rtc_action_queue is None or self._rtc_latency_tracker is None:
            raise RuntimeError("RTC runtime is not configured")

        time_per_chunk = 1.0 / self._rtc_fps
        with self._rtc_lock:
            prev_actions = self._rtc_action_queue.get_left_over()
            if prev_actions is not None:
                prev_actions = prev_actions.clone()
            action_index_before_inference = self._rtc_action_queue.get_action_index()
            generation = self._rtc_generation
        inference_delay = math.ceil(self._rtc_latency_tracker.max() / time_per_chunk)
        return (
            self._clone_rtc_batch(batch),
            prev_actions,
            inference_delay,
            action_index_before_inference,
            time.perf_counter(),
            generation,
        )

    def _predict_and_merge_rtc_chunk(
        self,
        batch: dict[str, Tensor],
        prev_actions: Tensor | None,
        inference_delay: int,
        action_index_before_inference: int,
        request_start_time: float,
        generation: int,
    ) -> None:
        if (
            self._rtc_action_queue is None
            or self._rtc_latency_tracker is None
            or self._rtc_config is None
        ):
            raise RuntimeError("RTC runtime is not configured")
        if generation != self._rtc_generation:
            return

        with self._rtc_inference_lock:
            if generation != self._rtc_generation:
                return
            mod = self._ensure_modifier()
            mod._step_metadata.clear()
            chunk = self.predict_action_chunk(
                batch,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_actions,
            )
            active_rtc_config = self._active_pi05_rtc_config or self._rtc_config
            if chunk.shape[0] != 1:
                raise ValueError(f"RTC deployment expects batch size 1, got {chunk.shape[0]}")

            original_actions = chunk.squeeze(0).detach()
            step_metadata = list(mod._step_metadata)
            mod._step_metadata.clear()
        if len(step_metadata) != len(original_actions):
            raise RuntimeError(
                f"RTC metadata/action length mismatch: {len(step_metadata)} != {len(original_actions)}"
            )

        new_latency = time.perf_counter() - request_start_time
        time_per_chunk = 1.0 / self._rtc_fps
        estimated_delay = math.ceil(new_latency / time_per_chunk)
        self._rtc_latency_tracker.add(new_latency)
        with self._rtc_lock:
            if generation != self._rtc_generation:
                return
            real_delay = max(0, self._rtc_action_queue.get_action_index() - action_index_before_inference)
            queue_before_merge = self._rtc_action_queue.qsize()
            self._rtc_action_queue.merge(
                original_actions,
                original_actions,
                real_delay,
                action_index_before_inference,
            )
            self._rtc_step_metadata = deque(step_metadata[real_delay:])

        log.info(
            "[RLT RTC] inference latency=%.1fms (estimated_delay=%d steps, real_delay=%d steps) | "
            "inference_delay used=%d | queue before merge=%d",
            new_latency * 1000.0,
            estimated_delay,
            real_delay,
            inference_delay,
            queue_before_merge,
        )
        min_refill_threshold = active_rtc_config.execution_horizon + estimated_delay
        if self._rtc_action_queue_size_to_get_new_actions < min_refill_threshold:
            log.warning(
                "[RLT RTC] action_queue_size_to_get_new_actions=%d is smaller than "
                "execution_horizon + delay (%d + %d). The queue may run dry under load.",
                self._rtc_action_queue_size_to_get_new_actions,
                active_rtc_config.execution_horizon,
                estimated_delay,
            )

    def _run_rtc_worker(
        self,
        batch: dict[str, Tensor],
        prev_actions: Tensor | None,
        inference_delay: int,
        action_index_before_inference: int,
        request_start_time: float,
        generation: int,
    ) -> None:
        try:
            self._predict_and_merge_rtc_chunk(
                batch,
                prev_actions,
                inference_delay,
                action_index_before_inference,
                request_start_time,
                generation,
            )
        except Exception as error:
            with self._rtc_lock:
                if generation == self._rtc_generation:
                    self._rtc_worker_error = error

    def _maybe_start_rtc_worker(self, batch: dict[str, Tensor]) -> None:
        if self._rtc_action_queue is None:
            raise RuntimeError("RTC runtime is not configured")
        if self._rtc_worker is not None and self._rtc_worker.is_alive():
            return
        self._join_finished_rtc_worker()
        self._raise_rtc_worker_error()
        if self._rtc_action_queue.qsize() > self._rtc_action_queue_size_to_get_new_actions:
            return
        request = self._prepare_rtc_request(batch)
        self._rtc_worker = Thread(target=self._run_rtc_worker, args=request, daemon=True)
        self._rtc_worker.start()

    def _select_action_rtc(self, batch: dict[str, Tensor]) -> Tensor:
        if self._rtc_action_queue is None:
            raise RuntimeError("RTC runtime is not configured")

        self._join_finished_rtc_worker()
        self._raise_rtc_worker_error()
        computed_sync = False
        if self._rtc_action_queue.empty():
            self._wait_for_rtc_worker()
            self._raise_rtc_worker_error()
        if self._rtc_action_queue.empty():
            self._predict_and_merge_rtc_chunk(*self._prepare_rtc_request(batch))
            computed_sync = True

        with self._rtc_lock:
            action = self._rtc_action_queue.get()
            step_metadata = self._rtc_step_metadata.popleft() if self._rtc_step_metadata else None
        if action is None:
            self._predict_and_merge_rtc_chunk(*self._prepare_rtc_request(batch))
            with self._rtc_lock:
                action = self._rtc_action_queue.get()
                step_metadata = self._rtc_step_metadata.popleft() if self._rtc_step_metadata else None
        if action is None:
            raise RuntimeError("RTC action queue is empty after refill")
        if step_metadata is None:
            raise RuntimeError("RTC metadata queue is empty after action pop")
        self._rtc_selected_step_metadata.append(step_metadata)

        if not computed_sync:
            self._maybe_start_rtc_worker(batch)
        return action.unsqueeze(0)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        mod = self._ensure_modifier()
        pi05 = self._rl_token_policy._pi05
        if self._rtc_config is not None:
            self._set_pi05_rtc_config(self._rtc_config_for_phase(mod))
        vla_chunk = pi05.predict_action_chunk(batch, **kwargs)
        vla_chunk = vla_chunk[:, :, : self.config.action_dim]
        prefix_tokens = self._prefix_capture.consume()
        proprio = batch["observation.state"][:, : self.config.proprio_dim]
        return mod.compute_chunk(vla_chunk, proprio, prefix_tokens)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        if self._rtc_runtime_enabled:
            return self._select_action_rtc(batch)

        mod = self._ensure_modifier()
        if mod.needs_new_chunk:
            chunk = self.predict_action_chunk(batch, **kwargs)
            mod.enqueue(chunk)
        return mod.pop_action()

    def reset(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        if self.modifier is not None:
            self.modifier.reset()
            # modifier.reset() resets the phase controller to VLA; re-pin the
            # configured deploy phase so always_rl survives episode resets.
            self._apply_phase_mode(self.modifier.phase_ctrl)

    def set_rl_mode(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        self._ensure_modifier().set_rl_mode()

    def set_vla_mode(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        self._ensure_modifier().set_vla_mode()

    def trigger_critical_phase(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        self._ensure_modifier().trigger_critical_phase()

    def interrupt_chunk(self) -> None:
        if self._rtc_runtime_enabled:
            self._reset_rtc_runtime()
        if self.modifier is not None:
            self.modifier.interrupt_chunk()

    def pop_step_metadata(self):
        if self._rtc_runtime_enabled:
            if len(self._rtc_selected_step_metadata) == 0:
                return None
            return self._rtc_selected_step_metadata.popleft()
        if self.modifier is None:
            return None
        return self.modifier.pop_step_metadata()

    def get_optim_params(self) -> list:
        actor_group = {"params": list(self.actor.parameters())}
        critic_group = {"params": list(self.critic.parameters())}
        actor_lr = getattr(self.config, "actor_lr", None)
        critic_lr = getattr(self.config, "critic_lr", None)
        if actor_lr is not None:
            actor_group["lr"] = float(actor_lr)
        if critic_lr is not None:
            critic_group["lr"] = float(critic_lr)

        training_stage = getattr(self.config, "training_stage", "mixed_ac")
        if training_stage in {"human_bc", "teacher_bc", "actor_refine"}:
            return [actor_group]
        if training_stage == "critic_only":
            return [critic_group]
        return [actor_group, critic_group]

    # ------------------------------------------------------------------
    # Device + train-mode plumbing for the stashed backbone
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._rl_token_policy.to(*args, **kwargs)
        teacher = getattr(self, "_teacher_actor", None)
        if teacher is not None:
            teacher.to(*args, **kwargs)
        risk_head = getattr(self, "_corrective_risk_head", None)
        if risk_head is not None:
            risk_head.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self._rl_token_policy.cuda(device)
        teacher = getattr(self, "_teacher_actor", None)
        if teacher is not None:
            teacher.cuda(device)
        risk_head = getattr(self, "_corrective_risk_head", None)
        if risk_head is not None:
            risk_head.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        self._rl_token_policy.cpu()
        teacher = getattr(self, "_teacher_actor", None)
        if teacher is not None:
            teacher.cpu()
        risk_head = getattr(self, "_corrective_risk_head", None)
        if risk_head is not None:
            risk_head.cpu()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen backbones always in eval.
        self._rl_token_policy.eval()
        self.target_critic.eval()
        teacher = getattr(self, "_teacher_actor", None)
        if teacher is not None:
            teacher.eval()
        risk_head = getattr(self, "_corrective_risk_head", None)
        if risk_head is not None:
            risk_head.eval()
        return self
