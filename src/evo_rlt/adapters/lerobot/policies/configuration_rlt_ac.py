from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

_DEFAULT_CAMERA_KEYS: list[str] = ["left_wrist", "right_wrist", "right_front"]


@dataclass
class ChunkACPolicyConfig(PreTrainedConfig):
    """Config for the chunk-level TD3+BC actor-critic policy on top of RL Token.

    Train via lerobot-train CLI with a ChunkTransitionDataset; deploy via
    make_policy + lerobot-record. forward(batch) returns a single scalar TD3+BC
    loss combining critic MSE + (every actor_update_interval steps) the actor
    Q-maximization + BC reg. Target critic is soft-updated with tau after every
    critic step. UTD ratio is achieved by setting outer --steps=outer*utd
    (each forward() == one critic update; actor update gated by counter).
    """

    # --- VLA + RL Token backbones (frozen, not serialized) ---
    vla_pretrained_path: str = "lerobot/pi05_base"
    vla_type: str = "auto"
    vla_revision: str | None = None
    vla_dtype: str = "bfloat16"
    rl_token_pretrained_path: str = ""

    # --- RL Token arch (must match the loaded RLTokenPolicy ckpt) ---
    rl_token_dim: int = 0
    rl_token_num_rl_tokens: int = 1
    token_pool_size: int = 0
    image_only: bool = False
    active_camera_indices: list[int] | None = None
    num_per_camera: int = 0

    # --- Actor ---
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    actor_fixed_std: float = 0.05
    actor_ref_dropout_p: float = 0.0
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = False
    # AC v2: normalize only z_rl, then predict a bounded delta around VLA ref.
    state_normalization: str = "rl_token_layer_norm"
    actor_action_residual: bool = True
    actor_delta_scale: float = 0.1
    actor_delta_scale_per_action_dim: list[float] | None = None
    ac_semantics_version: int = 2

    # --- Critic + target ---
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    critic_activation: str = "relu"
    critic_layer_norm: bool = False
    critic_residual: bool = False

    # --- TD3+BC hyperparams ---
    gamma: float = 0.99
    beta: float = 0.3
    # Independent coefficient for the actor's Q-maximization term.  Keeping
    # this at 1.0 preserves the historical TD3+BC objective exactly; setting
    # it to 0.0 turns mixed_ac actor updates into supervised BC-only updates
    # while the critic can continue training for diagnostics.
    actor_q_weight: float = 1.0
    actor_bc_weight_mode: str = "fixed"
    actor_bc_uncertainty_tau_low: float = 0.0
    actor_bc_uncertainty_tau_high: float = 1.0
    actor_bc_uncertainty_kappa: float = 0.0
    actor_bc_uncertainty_threshold_mode: str = "fixed"
    actor_bc_uncertainty_ema_decay: float = 0.95
    actor_bc_uncertainty_min_gap: float = 1e-6
    actor_bc_uncertainty_reset_ema_on_load: bool = False
    critic_bootstrap_mode: str = "none"
    critic_bootstrap_keep_prob: float = 0.8
    critic_bootstrap_seed: int = 1000
    diagnostics_jsonl_path: str | None = None
    tau: float = 0.005
    utd_ratio: int = 5
    actor_update_interval: int = 2
    target_q_clip: float = 100.0

    # --- Offline training stage + source balancing ---
    # ``human_bc`` performs actor-only residual behavior cloning on source=3
    # transitions. ``teacher_bc`` performs actor-only human correction BC plus
    # frozen warmup-actor distillation on semantically safe old-policy states.
    # ``critic_only`` updates only the critic/target critic while keeping the
    # loaded actor bit-identical. ``mixed_ac`` keeps the regular TD3+BC update.
    training_stage: str = "mixed_ac"
    # Replay-level source weights ordered as [demo, VLA, RL autonomous, human].
    # The transition dataset realizes these weights with a deterministic virtual
    # index map while keeping one epoch equal to the original cache length.
    source_sampling_weights: list[float] | None = None
    source_sampling_seed: int = 1000
    # Backward-compatible optimizer default. Explicit component learning rates
    # take precedence in the optimizer parameter groups when provided.
    training_lr: float = 3e-4
    actor_lr: float | None = None
    critic_lr: float | None = None
    # Conservative online refinement: on source=2 autonomous rollout chunks,
    # penalize drift from the action that the behavior policy actually
    # executed. Assisted prefixes are excluded when intervention_reason is
    # available. Zero preserves the original TD3+BC objective exactly.
    actor_behavior_preservation_weight: float = 0.0
    # Continual-adaptation control. Only the actor tensors are read from this
    # AC checkpoint, and the frozen teacher is built lazily on the first
    # teacher_bc batch. Deployment therefore does not load a second actor.
    actor_teacher_pretrained_path: str = ""
    teacher_distillation_weight: float = 1.0
    human_bc_weight: float = 1.0
    # Unified actor-only refinement objective.  These names are deliberately
    # separate from the legacy teacher_bc knobs so old checkpoints retain
    # their exact objective and serialization semantics.
    actor_human_weight: float = 1.0
    actor_teacher_weight: float = 1.0
    actor_q_weight_max: float = 0.0
    actor_q_trust_mode: str = "fixed"
    corrective_risk_checkpoint: str = ""
    corrective_risk_horizon_chunks: int = 3
    # Explicit opt-in. ``raw`` preserves every historical checkpoint/config;
    # ``residual_feasible`` projects source=3 targets only at loss time while
    # retaining the raw cache action for diagnostics and critic semantics.
    human_bc_target_mode: str = "raw"

    # --- Shapes ---
    chunk_length: int = 10
    action_dim: int = 12
    proprio_dim: int = 12

    # --- Deploy ---
    chunk_exec_steps: int = 25
    phase_mode: str = "always_rl"
    deterministic: bool = True

    # --- Observation mapping (for deploy preprocessor) ---
    camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_CAMERA_KEYS))

    # --- Normalization MATCHES THE SFT VLA (deploy parity) ---
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # --- VLA proxy fields ---
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    tokenizer_path: str | None = None

    @classmethod
    def ensure_registered(cls) -> None:
        import lerobot.policies  # noqa: F401

        PreTrainedConfig._choice_registry["rlt_ac"] = cls

    def __post_init__(self) -> None:
        self.ensure_registered()
        super().__post_init__()
        from evo_rlt.adapters.lerobot.policies.vla_backbone import normalize_vla_type

        self.vla_type = normalize_vla_type(self.vla_type)
        if self.phase_mode not in ("always_rl", "always_vla", "manual"):
            raise ValueError(
                f"phase_mode must be 'always_rl', 'always_vla', or 'manual', got {self.phase_mode!r}"
            )
        if self.state_normalization not in ("none", "rl_token_layer_norm"):
            raise ValueError(
                "state_normalization must be 'none' or 'rl_token_layer_norm', "
                f"got {self.state_normalization!r}"
            )
        if self.actor_delta_scale <= 0:
            raise ValueError(
                f"actor_delta_scale must be positive, got {self.actor_delta_scale}"
            )
        if self.actor_delta_scale_per_action_dim is not None:
            if len(self.actor_delta_scale_per_action_dim) != self.action_dim:
                raise ValueError(
                    "actor_delta_scale_per_action_dim must have one value per action "
                    f"dimension ({self.action_dim}), got "
                    f"{len(self.actor_delta_scale_per_action_dim)}"
                )
            if any(scale <= 0 for scale in self.actor_delta_scale_per_action_dim):
                raise ValueError(
                    "all actor_delta_scale_per_action_dim values must be positive"
                )
        if self.ac_semantics_version not in (1, 2):
            raise ValueError(
                f"ac_semantics_version must be 1 or 2, got {self.ac_semantics_version}"
            )
        if self.actor_bc_weight_mode not in ("fixed", "disagreement"):
            raise ValueError(
                "actor_bc_weight_mode must be 'fixed' or 'disagreement', "
                f"got {self.actor_bc_weight_mode!r}"
            )
        if self.actor_bc_uncertainty_kappa < 0:
            raise ValueError(
                "actor_bc_uncertainty_kappa must be non-negative, "
                f"got {self.actor_bc_uncertainty_kappa}"
            )
        if self.actor_bc_uncertainty_threshold_mode not in ("fixed", "ema_quantile"):
            raise ValueError(
                "actor_bc_uncertainty_threshold_mode must be 'fixed' or "
                f"'ema_quantile', got {self.actor_bc_uncertainty_threshold_mode!r}"
            )
        if not 0.0 <= self.actor_bc_uncertainty_ema_decay < 1.0:
            raise ValueError(
                "actor_bc_uncertainty_ema_decay must be within [0, 1), "
                f"got {self.actor_bc_uncertainty_ema_decay}"
            )
        if self.actor_bc_uncertainty_min_gap <= 0:
            raise ValueError(
                "actor_bc_uncertainty_min_gap must be positive, "
                f"got {self.actor_bc_uncertainty_min_gap}"
            )
        if self.actor_bc_uncertainty_reset_ema_on_load and (
            self.actor_bc_weight_mode != "disagreement"
            or self.actor_bc_uncertainty_threshold_mode != "ema_quantile"
        ):
            raise ValueError(
                "actor_bc_uncertainty_reset_ema_on_load requires "
                "actor_bc_weight_mode='disagreement' and "
                "actor_bc_uncertainty_threshold_mode='ema_quantile'"
            )
        if self.critic_bootstrap_mode not in ("none", "fixed_bernoulli"):
            raise ValueError(
                "critic_bootstrap_mode must be 'none' or 'fixed_bernoulli', "
                f"got {self.critic_bootstrap_mode!r}"
            )
        if not 0.0 < self.critic_bootstrap_keep_prob <= 1.0:
            raise ValueError(
                "critic_bootstrap_keep_prob must be within (0, 1], "
                f"got {self.critic_bootstrap_keep_prob}"
            )
        if self.critic_bootstrap_seed < 0:
            raise ValueError(
                f"critic_bootstrap_seed must be non-negative, got {self.critic_bootstrap_seed}"
            )
        if self.training_stage not in (
            "mixed_ac",
            "human_bc",
            "teacher_bc",
            "actor_refine",
            "critic_only",
        ):
            raise ValueError(
                "training_stage must be 'mixed_ac', 'human_bc', 'teacher_bc', "
                "'actor_refine', or 'critic_only', "
                f"got {self.training_stage!r}"
            )
        if self.source_sampling_weights is not None:
            if len(self.source_sampling_weights) != 4:
                raise ValueError(
                    "source_sampling_weights must contain four values ordered as "
                    "[demo, VLA, RL autonomous, human]"
                )
            if any(weight < 0 for weight in self.source_sampling_weights):
                raise ValueError("source_sampling_weights must be non-negative")
            if sum(self.source_sampling_weights) <= 0:
                raise ValueError("source_sampling_weights must have a positive sum")
        if self.source_sampling_seed < 0:
            raise ValueError(
                f"source_sampling_seed must be non-negative, got {self.source_sampling_seed}"
            )
        if self.training_lr <= 0:
            raise ValueError(f"training_lr must be positive, got {self.training_lr}")
        if self.actor_lr is not None and self.actor_lr <= 0:
            raise ValueError(f"actor_lr must be positive when set, got {self.actor_lr}")
        if self.critic_lr is not None and self.critic_lr <= 0:
            raise ValueError(f"critic_lr must be positive when set, got {self.critic_lr}")
        if self.actor_behavior_preservation_weight < 0:
            raise ValueError(
                "actor_behavior_preservation_weight must be non-negative, "
                f"got {self.actor_behavior_preservation_weight}"
            )
        if self.actor_q_weight < 0:
            raise ValueError(
                f"actor_q_weight must be non-negative, got {self.actor_q_weight}"
            )
        if self.teacher_distillation_weight < 0:
            raise ValueError(
                "teacher_distillation_weight must be non-negative, "
                f"got {self.teacher_distillation_weight}"
            )
        if self.human_bc_weight < 0:
            raise ValueError(
                f"human_bc_weight must be non-negative, got {self.human_bc_weight}"
            )
        if self.actor_human_weight < 0 or self.actor_teacher_weight < 0:
            raise ValueError("actor_human_weight and actor_teacher_weight must be non-negative")
        if self.actor_q_weight_max < 0:
            raise ValueError("actor_q_weight_max must be non-negative")
        if self.actor_q_trust_mode not in ("fixed", "corrective_risk"):
            raise ValueError(
                "actor_q_trust_mode must be 'fixed' or 'corrective_risk', "
                f"got {self.actor_q_trust_mode!r}"
            )
        if self.corrective_risk_horizon_chunks <= 0:
            raise ValueError("corrective_risk_horizon_chunks must be positive")
        if self.human_bc_target_mode not in ("raw", "residual_feasible"):
            raise ValueError(
                "human_bc_target_mode must be 'raw' or 'residual_feasible', "
                f"got {self.human_bc_target_mode!r}"
            )
        if self.training_stage == "teacher_bc":
            if not self.actor_teacher_pretrained_path:
                raise ValueError(
                    "teacher_bc requires actor_teacher_pretrained_path pointing "
                    "to the frozen warmup AC pretrained_model directory"
                )
            if self.teacher_distillation_weight + self.human_bc_weight <= 0:
                raise ValueError(
                    "teacher_bc requires a positive teacher_distillation_weight "
                    "or human_bc_weight"
                )
            if self.actor_q_weight != 0:
                raise ValueError(
                    "teacher_bc is an actor-only Q=0 diagnostic stage; set "
                    "actor_q_weight=0"
                )
        if self.training_stage == "actor_refine":
            if not self.actor_teacher_pretrained_path:
                raise ValueError(
                    "actor_refine requires actor_teacher_pretrained_path pointing "
                    "to the frozen warmup AC pretrained_model directory"
                )
            if self.actor_human_weight + self.actor_teacher_weight <= 0:
                raise ValueError(
                    "actor_refine requires a positive actor_human_weight or "
                    "actor_teacher_weight"
                )
            if self.actor_bc_weight_mode != "fixed":
                raise ValueError(
                    "actor_refine requires actor_bc_weight_mode='fixed' so old "
                    "critic-disagreement BC weighting cannot contaminate Q trust"
                )
            if self.actor_q_trust_mode == "corrective_risk" and not self.corrective_risk_checkpoint:
                raise ValueError(
                    "corrective_risk trust requires corrective_risk_checkpoint"
                )
        if (
            self.actor_bc_weight_mode == "disagreement"
            and self.actor_bc_uncertainty_tau_high
            <= self.actor_bc_uncertainty_tau_low
        ):
            raise ValueError(
                "actor_bc_uncertainty_tau_high must be greater than "
                "actor_bc_uncertainty_tau_low in disagreement mode"
            )

    @property
    def type(self) -> str:
        return "rlt_ac"

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {}
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.proprio_dim,)
            )
        for cam_key in self.camera_keys:
            img_key = f"{OBS_IMAGES}.{cam_key}"
            if img_key not in self.input_features:
                self.input_features[img_key] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, *self.image_resolution)
                )
        if not self.output_features:
            self.output_features = {}
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.action_dim,)
            )

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=self.training_lr, weight_decay=0.0, grad_clip_norm=1.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_length))

    @property
    def reward_delta_indices(self) -> None:
        return None
