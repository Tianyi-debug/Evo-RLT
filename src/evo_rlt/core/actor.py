from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from evo_rlt.core.utils import build_mlp, _get_activation


def normalize_state_vec(
    state_vec: torch.Tensor,
    *,
    proprio_dim: int,
    mode: str,
) -> torch.Tensor:
    """Normalize the high-magnitude RL-token portion of an AC state vector.

    ``state_vec`` is ``[z_rl, proprio]``.  The RL token and proprioception have
    very different semantics and scales, so only the RL-token slice is
    normalized.  This is deliberately stateless: checkpoints do not need extra
    affine parameters or running statistics, and training/deployment use the
    exact same transformation.
    """
    if mode == "none":
        return state_vec
    if mode != "rl_token_layer_norm":
        raise ValueError(
            "state_normalization must be 'none' or 'rl_token_layer_norm', "
            f"got {mode!r}"
        )
    if proprio_dim < 0 or proprio_dim >= state_vec.shape[-1]:
        raise ValueError(
            f"proprio_dim must be in [0, state_dim), got {proprio_dim} "
            f"for state_dim={state_vec.shape[-1]}"
        )

    if proprio_dim == 0:
        return F.layer_norm(state_vec, (state_vec.shape[-1],))

    z_rl = state_vec[..., :-proprio_dim]
    proprio = state_vec[..., -proprio_dim:]
    z_rl = F.layer_norm(z_rl, (z_rl.shape[-1],))
    return torch.cat([z_rl, proprio], dim=-1)


class ResidualMLP(nn.Module):
    """MLP with residual connections between hidden layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: str = "relu",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            block_layers: list[nn.Module] = [nn.Linear(hidden_dim, hidden_dim)]
            if layer_norm:
                block_layers.append(nn.LayerNorm(hidden_dim))
            block_layers.append(_get_activation(activation))
            blocks.append(nn.Sequential(*block_layers))
        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)


class ChunkActor(nn.Module):
    """Actor that predicts an action chunk conditioned on RL state and VLA reference.

    Uses binary reference dropout (per batch element) during training to avoid
    over-reliance on the VLA reference chunk.
    """

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.05,
        ref_dropout_p: float = 0.5,
        activation: str = "relu",
        layer_norm: bool = False,
        residual: bool = False,
        proprio_dim: int = 0,
        state_normalization: str = "none",
        action_residual: bool = False,
        delta_scale: float = 0.1,
    ):
        super().__init__()
        if delta_scale <= 0:
            raise ValueError(f"delta_scale must be positive, got {delta_scale}")
        if residual:
            self.net = ResidualMLP(
                state_dim + chunk_dim, hidden_dim, chunk_dim, num_layers,
                activation=activation, layer_norm=layer_norm,
            )
        else:
            self.net = build_mlp(
                state_dim + chunk_dim, hidden_dim, chunk_dim, num_layers,
                activation=activation, layer_norm=layer_norm,
            )
        if action_residual:
            # Start from the proven VLA policy exactly.  The first optimization
            # step learns a bounded delta head instead of replacing the VLA
            # chunk with a random absolute action.
            output_layer = self.net.output_proj if residual else self.net[-1]
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        self.fixed_std = fixed_std
        self.ref_dropout_p = ref_dropout_p
        self.proprio_dim = proprio_dim
        self.state_normalization = state_normalization
        self.action_residual = action_residual
        self.delta_scale = delta_scale

    def forward(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (mu, std)."""
        # Match the action that the deployment path can actually execute.
        # Clamping before adding the delta also lets the actor correct inward
        # when a quantile-normalized VLA reference slightly exceeds [-1, 1].
        base_ref = (
            ref_chunk_flat.clamp(-1.0, 1.0)
            if self.action_residual
            else ref_chunk_flat
        )
        condition_ref = ref_chunk_flat
        if training:
            mask = (
                torch.rand(state_vec.shape[0], 1, device=state_vec.device) > self.ref_dropout_p
            ).float()
            condition_ref = condition_ref * mask
        state_vec = normalize_state_vec(
            state_vec,
            proprio_dim=self.proprio_dim,
            mode=self.state_normalization,
        )
        x = torch.cat([state_vec, condition_ref], dim=-1)
        network_out = self.net(x)
        if self.action_residual:
            delta = self.delta_scale * torch.tanh(network_out)
            mu = (base_ref + delta).clamp(-1.0, 1.0)
        else:
            mu = network_out
        std = torch.full_like(mu, self.fixed_std)
        return mu, std

    def sample(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action with Gaussian noise. Returns (action, mu)."""
        mu, std = self.forward(state_vec, ref_chunk_flat, training)
        return mu + std * torch.randn_like(std), mu
