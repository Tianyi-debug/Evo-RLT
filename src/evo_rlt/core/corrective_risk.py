"""Independent corrective-takeover risk classifier and checkpoint helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


RISK_CHECKPOINT_SCHEMA_VERSION = 1


class CorrectiveTakeoverRiskMLP(nn.Module):
    """Predict future corrective takeover risk from state and optional action.

    ``action_dim=0`` is reserved for the matched state-only ablation.  The
    production corrective-risk head always has a positive action dimension.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int] = (256, 128),
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim < 0:
            raise ValueError("state_dim must be positive and action_dim non-negative")
        if len(hidden_dims) != 2 or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain two positive widths")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = (int(hidden_dims[0]), int(hidden_dims[1]))
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[1], 1),
        )

    def forward(self, state: Tensor, action: Tensor | None = None) -> Tensor:
        state = state.flatten(start_dim=1)
        if state.shape[1] != self.state_dim:
            raise ValueError(f"expected state_dim={self.state_dim}, got {state.shape[1]}")
        if self.action_dim == 0:
            if action is not None and action.flatten(start_dim=1).shape[1] != 0:
                raise ValueError("state-only risk head does not accept action features")
            features = state
        else:
            if action is None:
                raise ValueError("action-conditioned risk head requires action features")
            action = action.flatten(start_dim=1)
            if state.shape[0] != action.shape[0]:
                raise ValueError("state and action batch sizes differ")
            if action.shape[1] != self.action_dim:
                raise ValueError(f"expected action_dim={self.action_dim}, got {action.shape[1]}")
            features = torch.cat((state, action), dim=-1)
        return self.net(features).squeeze(-1)


def masked_risk_bce_with_logits(
    logits: Tensor,
    labels: Tensor,
    label_mask: Tensor,
    *,
    pos_weight: float | Tensor,
) -> Tensor:
    """BCE over eligible rows only; masked/censored rows have exactly zero credit."""
    logits = logits.reshape(-1)
    labels = labels.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    mask = label_mask.reshape(-1).to(device=logits.device) > 0.5
    if not (logits.numel() == labels.numel() == mask.numel()):
        raise ValueError("logits, labels, and label_mask must have the same size")
    if not bool(mask.any()):
        raise ValueError("risk batch contains no label-eligible samples")
    weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    if weight.numel() != 1 or not bool(torch.isfinite(weight)) or weight.item() <= 0:
        raise ValueError("pos_weight must be one finite positive scalar")
    return F.binary_cross_entropy_with_logits(
        logits[mask],
        labels[mask],
        pos_weight=weight,
    )


def save_corrective_risk_checkpoint(
    path: str | Path,
    model: CorrectiveTakeoverRiskMLP,
    metadata: dict[str, Any],
) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RISK_CHECKPOINT_SCHEMA_VERSION,
        "model_config": {
            "state_dim": model.state_dim,
            "action_dim": model.action_dim,
            "hidden_dims": list(model.hidden_dims),
        },
        "model_state": model.state_dict(),
        "metadata": metadata,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_corrective_risk_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    freeze: bool = True,
) -> tuple[CorrectiveTakeoverRiskMLP, dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"corrective risk checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != RISK_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported corrective risk checkpoint schema: {source}")
    config = payload.get("model_config")
    state = payload.get("model_state")
    metadata = payload.get("metadata")
    if not isinstance(config, dict) or not isinstance(state, dict) or not isinstance(metadata, dict):
        raise ValueError(f"malformed corrective risk checkpoint: {source}")
    model = CorrectiveTakeoverRiskMLP(
        state_dim=int(config["state_dim"]),
        action_dim=int(config["action_dim"]),
        hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
    )
    model.load_state_dict(state, strict=True)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
    return model, metadata
