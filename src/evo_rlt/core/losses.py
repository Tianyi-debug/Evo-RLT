from __future__ import annotations

import torch
import torch.nn.functional as F

from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import TwinCritic
from evo_rlt.core.utils import compute_discount_vector


def discounted_chunk_return(
    reward_seq: torch.Tensor, gamma: float, actual_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute discounted return over a chunk of rewards.

    Args:
        reward_seq: (B, C) rewards for each timestep (padded with 0 beyond actual_steps)
        gamma: discount factor
        actual_steps: (B,) number of valid steps per chunk (if None, assume all C are valid)

    Returns:
        (B, 1) discounted return
    """
    C = reward_seq.shape[1]
    discounts = compute_discount_vector(gamma, C, device=reward_seq.device)
    return (reward_seq * discounts.unsqueeze(0)).sum(dim=1, keepdim=True)


def critic_loss(
    critic: TwinCritic,
    target_critic: TwinCritic,
    actor: ChunkActor,
    batch: dict[str, torch.Tensor],
    gamma: float,
    C: int,
    target_q_clip: float | None = 100.0,
    bootstrap_mode: str = "none",
    bootstrap_keep_prob: float = 0.8,
    bootstrap_seed: int = 1000,
) -> torch.Tensor:
    loss, _ = critic_loss_with_diagnostics(
        critic=critic,
        target_critic=target_critic,
        actor=actor,
        batch=batch,
        gamma=gamma,
        C=C,
        target_q_clip=target_q_clip,
        bootstrap_mode=bootstrap_mode,
        bootstrap_keep_prob=bootstrap_keep_prob,
        bootstrap_seed=bootstrap_seed,
    )
    return loss


def fixed_bootstrap_mask(
    cache_index: torch.Tensor,
    *,
    head_id: int,
    keep_prob: float,
    seed: int,
) -> torch.Tensor:
    """Return a deterministic Bernoulli-like mask for one critic head."""
    if head_id not in (0, 1):
        raise ValueError(f"head_id must be 0 or 1, got {head_id}")
    if not 0.0 < keep_prob <= 1.0:
        raise ValueError(f"keep_prob must be within (0, 1], got {keep_prob}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    index = cache_index.detach().reshape(-1).to(dtype=torch.int64)
    modulus = 2_147_483_647
    if head_id == 0:
        multiplier, increment = 1_103_515_245, 12_345
    else:
        multiplier, increment = 1_664_525, 1_013_904_223
    seed_offset = (seed * 2_654_435_761) % modulus
    hashed = torch.remainder((index + 1 + seed_offset) * multiplier + increment, modulus)
    mask = hashed.to(dtype=torch.float64) < keep_prob * modulus
    if mask.numel() > 0 and not bool(mask.any()):
        mask = mask.clone()
        mask[int(hashed.argmin().item())] = True
    return mask


def _masked_mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    per_sample = (prediction - target).square().reshape(prediction.shape[0], -1).mean(dim=1)
    weights = mask.to(device=prediction.device, dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum()


def critic_loss_with_diagnostics(
    critic: TwinCritic,
    target_critic: TwinCritic,
    actor: ChunkActor,
    batch: dict[str, torch.Tensor],
    gamma: float,
    C: int,
    target_q_clip: float | None = 100.0,
    bootstrap_mode: str = "none",
    bootstrap_keep_prob: float = 0.8,
    bootstrap_seed: int = 1000,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """TD3-style chunk-level TD loss with correct truncated-chunk handling.

    Uses actual_steps to compute the correct bootstrap exponent gamma^k
    instead of always using gamma^C.
    """
    x = batch["state_vec"]
    a = batch["exec_chunk_flat"]
    x_next = batch["next_state_vec"]
    ref_next = batch["next_ref_flat"]
    reward_seq = batch["reward_seq"]
    done = batch["done"]
    actual_steps = batch.get("actual_steps")

    with torch.no_grad():
        # Use deterministic mean for target action (TD3-style), clamped to [-1,1]
        mu_next, _ = actor.forward(x_next, ref_next)
        mu_next = mu_next.clamp(-1.0, 1.0)
        q_next = target_critic.min_q(x_next, mu_next)
        if target_q_clip is not None and target_q_clip > 0:
            q_next = q_next.clamp(-target_q_clip, target_q_clip)
        r = discounted_chunk_return(reward_seq, gamma, actual_steps)

        # Bootstrap with gamma^k where k = actual steps executed
        if actual_steps is not None:
            bootstrap_exp = actual_steps.unsqueeze(-1).float()
        else:
            bootstrap_exp = torch.full_like(done.unsqueeze(-1), C, dtype=torch.float32)
        bootstrap = (gamma ** bootstrap_exp) * (1.0 - done.unsqueeze(-1)) * q_next
        target = r + bootstrap

    q1, q2 = critic(x, a)
    if bootstrap_mode == "none":
        q1_mask = torch.ones(q1.shape[0], dtype=torch.bool, device=q1.device)
        q2_mask = torch.ones(q2.shape[0], dtype=torch.bool, device=q2.device)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    elif bootstrap_mode == "fixed_bernoulli":
        if "cache_index" not in batch:
            raise KeyError("fixed_bernoulli critic bootstrap requires batch['cache_index']")
        cache_index = batch["cache_index"].to(q1.device)
        q1_mask = fixed_bootstrap_mask(
            cache_index,
            head_id=0,
            keep_prob=bootstrap_keep_prob,
            seed=bootstrap_seed,
        )
        q2_mask = fixed_bootstrap_mask(
            cache_index,
            head_id=1,
            keep_prob=bootstrap_keep_prob,
            seed=bootstrap_seed,
        )
        loss = _masked_mean_squared_error(q1, target, q1_mask) + _masked_mean_squared_error(
            q2,
            target,
            q2_mask,
        )
    else:
        raise ValueError(
            "bootstrap_mode must be 'none' or 'fixed_bernoulli', "
            f"got {bootstrap_mode!r}"
        )
    diagnostics = {
        "critic_q1_exec_mean": q1.detach().mean(),
        "critic_q2_exec_mean": q2.detach().mean(),
        "critic_q_min_exec_mean": torch.minimum(q1, q2).detach().mean(),
        "critic_target_mean": target.detach().mean(),
        "critic_td_abs_mean": (
            ((q1.detach() - target.detach()).abs() + (q2.detach() - target.detach()).abs())
            * 0.5
        ).mean(),
        "critic_exec_disagreement_mean": ((q1.detach() - q2.detach()).abs() * 0.5).mean(),
        "critic_bootstrap_q1_frac": q1_mask.float().mean(),
        "critic_bootstrap_q2_frac": q2_mask.float().mean(),
        "critic_bootstrap_overlap_frac": (q1_mask & q2_mask).float().mean(),
        "critic_bootstrap_union_frac": (q1_mask | q2_mask).float().mean(),
    }
    return loss, diagnostics


def actor_loss(
    actor: ChunkActor,
    critic: TwinCritic,
    batch: dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    loss, _ = actor_loss_with_diagnostics(
        actor=actor,
        critic=critic,
        batch=batch,
        beta=beta,
    )
    return loss


def actor_loss_with_diagnostics(
    actor: ChunkActor,
    critic: TwinCritic,
    batch: dict[str, torch.Tensor],
    beta: float,
    weight_mode: str = "fixed",
    uncertainty_tau_low: float = 0.0,
    uncertainty_tau_high: float = 1.0,
    uncertainty_kappa: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Q-maximization + BC regularization toward VLA reference.

    Uses deterministic mean (not noisy samples) for stable optimization.
    BC term is the per-sample squared distance summed across action dims, then
    averaged over the batch — matching the paper's β-scaling convention. This
    differs from mean-MSE by a factor of C*D_flat.
    """
    x = batch["state_vec"]
    ref = batch["ref_chunk_flat"]
    mu, _ = actor.forward(x, ref, training=True)
    if callable(critic):
        q1, q2 = critic(x, mu)
    else:
        # Preserve compatibility with simple min_q-only critic adapters used
        # by existing callers and tests. Such adapters have no ensemble signal.
        q1 = critic.min_q(x, mu)
        q2 = q1
    q = torch.minimum(q1, q2).squeeze(-1)
    disagreement = ((q1.detach() - q2.detach()).abs() * 0.5).squeeze(-1)

    if weight_mode == "fixed":
        rho = torch.zeros_like(disagreement)
        beta_per_sample = torch.full_like(disagreement, float(beta))
    elif weight_mode == "disagreement":
        if uncertainty_tau_high <= uncertainty_tau_low:
            raise ValueError(
                "uncertainty_tau_high must be greater than uncertainty_tau_low, "
                f"got {uncertainty_tau_high} <= {uncertainty_tau_low}"
            )
        if uncertainty_kappa < 0:
            raise ValueError(
                f"uncertainty_kappa must be non-negative, got {uncertainty_kappa}"
            )
        rho = (
            (disagreement - uncertainty_tau_low)
            / (uncertainty_tau_high - uncertainty_tau_low)
        ).clamp(0.0, 1.0)
        beta_per_sample = float(beta) * (1.0 + float(uncertainty_kappa) * rho)
    else:
        raise ValueError(
            "weight_mode must be 'fixed' or 'disagreement', "
            f"got {weight_mode!r}"
        )

    rho = rho.detach()
    beta_per_sample = beta_per_sample.detach()
    bc_target = (
        ref.clamp(-1.0, 1.0)
        if getattr(actor, "action_residual", False)
        else ref
    )
    bc_per_sample = ((mu - bc_target) ** 2).sum(dim=-1)
    q_loss = -q.mean()
    bc_raw = bc_per_sample.mean()
    bc_weighted = (beta_per_sample * bc_per_sample).mean()
    loss = q_loss + bc_weighted

    diagnostics = {
        "loss_actor_q": q_loss.detach(),
        "loss_actor_bc_raw": bc_raw.detach(),
        "loss_actor_bc_weighted": bc_weighted.detach(),
        "actor_q_min_mean": q.detach().mean(),
        "actor_ref_rmse": ((mu.detach() - bc_target.detach()) ** 2).mean().sqrt(),
        "actor_disagreement_mean": disagreement.mean(),
        "actor_disagreement_p50": torch.quantile(disagreement, 0.50),
        "actor_disagreement_p90": torch.quantile(disagreement, 0.90),
        "actor_disagreement_p95": torch.quantile(disagreement, 0.95),
        "actor_disagreement_max": disagreement.max(),
        "actor_rho_mean": rho.mean(),
        "actor_rho_zero_frac": (rho <= 0.0).float().mean(),
        "actor_rho_one_frac": (rho >= 1.0).float().mean(),
        "actor_beta_mean": beta_per_sample.mean(),
        "actor_beta_min": beta_per_sample.min(),
        "actor_beta_max": beta_per_sample.max(),
    }

    source = batch.get("source")
    if source is not None:
        source = source.detach().reshape(-1).to(disagreement.device)
        for source_id in range(4):
            mask = source == source_id
            diagnostics[f"source_{source_id}_frac"] = mask.float().mean()
            if bool(mask.any()):
                diagnostics[f"source_{source_id}_disagreement_mean"] = disagreement[mask].mean()
                diagnostics[f"source_{source_id}_beta_mean"] = beta_per_sample[mask].mean()

    return loss, diagnostics
