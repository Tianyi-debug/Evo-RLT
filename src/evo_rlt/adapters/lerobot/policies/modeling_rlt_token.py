from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import packaging
import safetensors
import torch
from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor
from typing_extensions import Unpack

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from evo_rlt.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
from evo_rlt.adapters.lerobot.policies.vla_backbone import (
    extract_prefix_hidden,
    infer_num_image_tokens,
    infer_vla_token_dim,
    load_vla_policy,
    get_vla_prefix_target,
)
from lerobot.policies.utils import log_model_loading_keys
from evo_rlt.core.rl_token import RLTokenModule
from evo_rlt.core.utils import postprocess_prefix_tokens

log = logging.getLogger(__name__)

VLA_SAFETENSORS_FILE = "vla.safetensors"


def _load_norm_stats(path: str | None) -> Tensor | None:
    """Load per-dim std for weighted reconstruction loss.

    Expected file format: torch.save({"std": Tensor[token_dim]}, path).
    Returns None when path is None.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"norm_stats_path does not exist: {path}")
    blob = torch.load(p, map_location="cpu")
    if isinstance(blob, dict) and "std" in blob:
        return blob["std"].to(dtype=torch.float32)
    if isinstance(blob, Tensor):
        return blob.to(dtype=torch.float32)
    raise ValueError(f"unrecognized norm_stats payload at {path}: keys={list(blob) if isinstance(blob, dict) else type(blob)}")


class RLTokenPolicy(PreTrainedPolicy):
    """Training-only policy that fits an RLTokenModule on top of a frozen VLA.

    forward(batch) returns the reconstruction loss (+ optional VLA supervised
    loss when vla_ft_weight > 0). The frozen VLA backbone is loaded in __init__
    and stashed in self.__dict__ to bypass
    nn.Module's submodule registration — so it is NOT serialized into the
    saved safetensors and does NOT appear in get_optim_params() either.

    deploy is handled by ChunkACPolicy; this policy raises on inference paths.
    """

    config_class = RLTokenPolicyConfig
    name = "rlt_token"

    def __init__(
        self,
        config: RLTokenPolicyConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self.config: RLTokenPolicyConfig = config

        vla = self._load_vla_backbone()
        token_dim = self._resolve_rl_token_dim(vla)

        self.rl_token = RLTokenModule(
            token_dim=token_dim,
            nhead=config.rl_token_nhead,
            num_enc_layers=config.rl_token_enc_layers,
            num_dec_layers=config.rl_token_dec_layers,
            ff_dim=config.rl_token_ff_dim,
            num_rl_tokens=config.rl_token_num_rl_tokens,
            inference_only=False,
        )

        # Stash VLA OUTSIDE nn.Module submodule tracking. nn.Module.__setattr__
        # registers nn.Module values into self._modules; object.__setattr__ stores
        # in self.__dict__ instead — so state_dict() / get_optim_params() skip it.
        object.__setattr__(self, "_pi05", vla)
        object.__setattr__(self, "_vla", vla)

        std = _load_norm_stats(config.norm_stats_path)
        if std is not None:
            self.register_buffer("_dim_std", std, persistent=False)
        else:
            self._dim_std = None  # type: ignore[assignment]

        self._num_image_tokens: int = self._compute_num_image_tokens(vla)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_vla_backbone(self):
        return self._load_pi05_backbone()

    def _load_pi05_backbone(self):
        vla = load_vla_policy(
            self.config.vla_pretrained_path,
            vla_type=self.config.vla_type,
            revision=self.config.vla_revision,
            dtype=self.config.vla_dtype,
            device=self.config.device,
        )
        if self.config.vla_ft_weight == 0:
            for p in vla.parameters():
                p.requires_grad = False
            vla.eval()
        return vla

    def _resolve_rl_token_dim(self, vla) -> int:
        inferred = infer_vla_token_dim(vla)
        configured = int(self.config.rl_token_dim or 0)
        if configured <= 0:
            if inferred is None:
                raise ValueError(
                    "policy.rl_token_dim=0 requests auto-inference, but Evo-RLT could "
                    "not infer the VLA prefix hidden size. Set policy.rl_token_dim explicitly."
                )
            self.config.rl_token_dim = inferred
            return inferred
        if inferred is not None and inferred != configured:
            raise ValueError(
                f"policy.rl_token_dim={configured} does not match VLA prefix hidden size {inferred}. "
                "Set policy.rl_token_dim=0 to auto-infer it."
            )
        return configured

    def _compute_num_image_tokens(self, vla) -> int:
        return infer_num_image_tokens(
            vla,
            image_resolution=self.config.image_resolution,
            camera_keys=self.config.camera_keys,
            num_per_camera=self.config.num_per_camera,
            required=self.config.image_only or bool(self.config.active_camera_indices),
        )

    # ------------------------------------------------------------------
    # Persistence (cotrained VLA lives outside nn.Module._modules)
    # ------------------------------------------------------------------

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save RLT state via super, and also dump the VLA when it was fine-tuned.

        Because `_pi05` is stashed via `object.__setattr__`, it is invisible to
        `state_dict()` and therefore to the standard safetensors save path. When
        `vla_ft_weight > 0` the backbone is being updated and must be persisted
        alongside the RLT weights; otherwise the cotrained gains are silently
        dropped on the next reload.
        """
        super()._save_pretrained(save_directory)
        if self.config.vla_ft_weight > 0:
            vla_path = save_directory / VLA_SAFETENSORS_FILE
            save_model_as_safetensor(self._pi05, str(vla_path))
            n_params = sum(p.numel() for p in self._pi05.parameters())
            log.info("saved cotrained VLA to %s (%d params)", vla_path, n_params)

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """Load RLT state via super, and also VLA if a `vla.safetensors` sits beside it.

        Old checkpoints (frozen-VLA runs or pre-patch saves) have no auxiliary
        file; in that case the fresh VLA loaded by ``__init__`` is kept as-is.
        We deliberately avoid try/except here — file existence is the contract.

        ``strict=False`` mirrors the SFT-baseline load path. Any
        missing/unexpected keys are still surfaced via
        ``log_model_loading_keys`` — same diagnostic channel the base
        ``_load_as_safetensor`` uses for the RLT half.
        """
        model = super()._load_as_safetensor(model, model_file, map_location, strict)
        vla_path = os.path.join(os.path.dirname(model_file), VLA_SAFETENSORS_FILE)
        if os.path.exists(vla_path):
            load_kwargs: dict[str, Any] = {"strict": False}
            if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
                load_kwargs["device"] = map_location
            missing_keys, unexpected_keys = load_model_as_safetensor(model._pi05, vla_path, **load_kwargs)
            log_model_loading_keys(missing_keys, unexpected_keys)
            log.info("loaded cotrained VLA from %s", vla_path)
        return model

    # ------------------------------------------------------------------
    # PreTrainedPolicy abstract methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        pass

    def get_optim_params(self) -> list:
        groups = [
            {"params": list(self.rl_token.parameters()), "lr": self.config.rl_token_lr},
        ]
        if self.config.vla_ft_weight > 0:
            vla_params = [p for p in self._pi05.parameters() if p.requires_grad]
            if vla_params:
                groups.append({"params": vla_params, "lr": self.config.vla_lr})
        return groups

    def _forward_pi05_with_prefix(self, batch: dict[str, Tensor], reduction: str) -> tuple[Tensor, dict, Tensor]:
        target = get_vla_prefix_target(self._pi05)
        original_forward = target.forward
        prefix_hidden: Tensor | None = None

        def patched_forward(*args, **kwargs):
            nonlocal prefix_hidden
            result = original_forward(*args, **kwargs)
            prefix_output = extract_prefix_hidden(result)
            if prefix_output is not None:
                prefix_hidden = prefix_output
            return result

        target.forward = patched_forward
        try:
            loss, info = self._pi05.forward(batch, reduction=reduction)
        finally:
            target.forward = original_forward

        if prefix_hidden is None:
            raise RuntimeError("VLA forward did not produce prefix hidden states")
        return loss, info, prefix_hidden

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Reconstruction loss + optional VLA supervised loss."""
        if self.config.vla_ft_weight > 0:
            loss_vla, info, prefix_hidden = self._forward_pi05_with_prefix(batch, reduction="mean")
        else:
            with torch.no_grad():
                _, info, prefix_hidden = self._forward_pi05_with_prefix(batch, reduction="mean")
            loss_vla = torch.zeros((), device=prefix_hidden.device, dtype=torch.float32)

        prefix_for_rl = postprocess_prefix_tokens(
            prefix_hidden.to(dtype=torch.float32),
            image_only=self.config.image_only,
            num_image_tokens=self._num_image_tokens,
            pool_size=self.config.token_pool_size,
            num_per_camera=self.config.num_per_camera,
            active_camera_indices=self.config.active_camera_indices,
        )

        loss_recon = self.rl_token.reconstruction_loss(
            prefix_for_rl,
            dim_std=self._dim_std,
            gamma=self.config.norm_gamma,
        )
        total = self.config.recon_weight * loss_recon + self.config.vla_ft_weight * loss_vla
        self._fwd_count = getattr(self, "_fwd_count", 0) + 1
        if self._fwd_count % 50 == 0:
            log.info(
                "rlt fwd~%d loss_recon=%.6f loss_vla=%.6f",
                self._fwd_count,
                loss_recon.detach().item(),
                loss_vla.detach().item(),
            )
        return total, {
            "loss": total.detach().item(),
            "loss_recon": loss_recon.detach().item(),
            "loss_vla": loss_vla.detach().item(),
        }

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        raise NotImplementedError(
            "RLTokenPolicy is training-only; use ChunkACPolicy for inference."
        )

    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        raise NotImplementedError(
            "RLTokenPolicy is training-only; use ChunkACPolicy for inference."
        )

    # ------------------------------------------------------------------
    # Device + train-mode plumbing
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._pi05.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self._pi05.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        self._pi05.cpu()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.vla_ft_weight > 0:
            self._pi05.train(mode)
        else:
            self._pi05.eval()
        return self
