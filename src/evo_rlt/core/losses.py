from __future__ import annotations

import torch

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
    return _masked_mean(per_sample, mask)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid samples, with a differentiable zero for an empty mask."""
    per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
    weights = mask.reshape(-1).to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)


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
    proposal_next = batch.get("next_proposal_flat", batch.get("next_ref_flat"))
    if proposal_next is None:
        raise KeyError("critic loss requires next_proposal_flat (or legacy next_ref_flat)")
    reward_seq = batch["reward_seq"]
    done = batch["done"]
    actual_steps = batch.get("actual_steps")

    with torch.no_grad():
        # Use deterministic mean for target action (TD3-style), clamped to [-1,1]
        mu_next, _ = actor.forward(x_next, proposal_next)
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
    critic_mask = batch.get("critic_mask")
    if critic_mask is None:
        critic_valid = torch.ones(q1.shape[0], dtype=torch.bool, device=q1.device)
    else:
        critic_valid = critic_mask.reshape(-1).to(q1.device) > 0.5
    if bootstrap_mode == "none":
        q1_mask = critic_valid
        q2_mask = critic_valid
        loss = _masked_mean_squared_error(q1, target, q1_mask) + _masked_mean_squared_error(
            q2, target, q2_mask
        )
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
        q1_mask = q1_mask & critic_valid
        q2_mask = q2_mask & critic_valid
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
    td_abs = (
        (q1.detach() - target.detach()).abs() + (q2.detach() - target.detach()).abs()
    ) * 0.5
    diagnostics = {
        "critic_q1_exec_mean": _masked_mean(q1.detach(), critic_valid),
        "critic_q2_exec_mean": _masked_mean(q2.detach(), critic_valid),
        "critic_q_min_exec_mean": _masked_mean(torch.minimum(q1, q2).detach(), critic_valid),
        "critic_target_mean": _masked_mean(target.detach(), critic_valid),
        "critic_td_abs_mean": _masked_mean(td_abs, critic_valid),
        "critic_exec_disagreement_mean": _masked_mean(
            (q1.detach() - q2.detach()).abs() * 0.5,
            critic_valid,
        ),
        "critic_valid_frac": critic_valid.float().mean(),
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
    behavior_preservation_weight: float = 0.0,
    q_weight: float = 1.0,
) -> torch.Tensor:
    loss, _ = actor_loss_with_diagnostics(
        actor=actor,
        critic=critic,
        batch=batch,
        beta=beta,
        q_weight=q_weight,
        behavior_preservation_weight=behavior_preservation_weight,
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
    behavior_preservation_weight: float = 0.0,
    q_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Q-maximization plus BC toward an independently selected target.

    The actor always receives the VLA proposal. The cache independently selects
    the BC target and can use ``actor_bc_mask`` to exclude transitions whose
    executed action must not be cloned (for example failed autonomous rollouts
    or policy prefixes preceding a corrective takeover). Uses deterministic
    mean (not noisy samples) for stability.
    BC term is the per-sample squared distance summed across action dims, then
    averaged over the batch — matching the paper's β-scaling convention. This
    differs from mean-MSE by a factor of C*D_flat. An optional conservative
    source=2 term anchors the actor to the action executed by the behavior
    policy. When intervention_reason is present, assisted prefixes are excluded
    from that anchor so failed/censored takeover context is not preserved.
    ``q_weight`` independently scales direct Q-gradient trust; zero leaves the
    supervised BC objectives active while preventing Q from moving the actor.
    """
    if q_weight < 0:
        raise ValueError(f"q_weight must be non-negative, got {q_weight}")
    if behavior_preservation_weight < 0:
        raise ValueError(
            "behavior_preservation_weight must be non-negative, "
            f"got {behavior_preservation_weight}"
        )
    x = batch["state_vec"]
    proposal = batch.get("proposal_chunk_flat", batch.get("ref_chunk_flat"))
    if proposal is None:
        raise KeyError("actor loss requires proposal_chunk_flat (or legacy ref_chunk_flat)")
    bc_target_raw = batch.get("bc_target_chunk_flat", proposal)
    mu, _ = actor.forward(x, proposal, training=True)
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
        bc_target_raw.clamp(-1.0, 1.0)
        if getattr(actor, "action_residual", False)
        else bc_target_raw
    )
    bc_per_sample = ((mu - bc_target) ** 2).sum(dim=-1)
    actor_bc_mask = batch.get("actor_bc_mask")
    if actor_bc_mask is None:
        actor_bc_valid = torch.ones_like(bc_per_sample, dtype=torch.bool)
    else:
        actor_bc_valid = actor_bc_mask.reshape(-1).to(bc_per_sample.device) > 0.5
    actor_q_mask = batch.get("actor_q_mask")
    if actor_q_mask is None:
        actor_q_valid = torch.ones_like(q, dtype=torch.bool)
    else:
        actor_q_valid = actor_q_mask.reshape(-1).to(q.device) > 0.5
    q_loss = -_masked_mean(q, actor_q_valid)
    bc_raw = _masked_mean(bc_per_sample, actor_bc_valid)
    bc_weighted = _masked_mean(beta_per_sample * bc_per_sample, actor_bc_valid)

    source = batch.get("source")
    if source is not None:
        source = source.detach().reshape(-1).to(disagreement.device)
    behavior_mask = torch.zeros_like(q, dtype=torch.bool)
    behavior_bc_raw = q.new_zeros(())
    behavior_target_rmse = q.new_zeros(())
    if behavior_preservation_weight > 0:
        if source is None:
            raise KeyError(
                "behavior preservation requires batch['source'] to select source=2"
            )
        behavior_target_raw = batch.get("exec_chunk_flat")
        if behavior_target_raw is None:
            raise KeyError(
                "behavior preservation requires batch['exec_chunk_flat']"
            )
        behavior_mask = source == 2
        # Outcome-aware caches use the same mask to prevent the optional
        # behavior-preservation term from re-introducing BC on failed policy
        # actions that the primary BC objective deliberately excludes.
        behavior_mask = behavior_mask & actor_bc_valid
        intervention_reason = batch.get("intervention_reason")
        if intervention_reason is not None:
            reason = intervention_reason.detach().reshape(-1).to(source.device)
            behavior_mask = behavior_mask & (reason == 0)
        behavior_target = (
            behavior_target_raw.clamp(-1.0, 1.0)
            if getattr(actor, "action_residual", False)
            else behavior_target_raw
        )
        behavior_per_sample = (mu - behavior_target).square().sum(dim=-1)
        behavior_bc_raw = _masked_mean(behavior_per_sample, behavior_mask)
        behavior_target_rmse = _masked_mean(
            (mu.detach() - behavior_target.detach()).square().mean(dim=-1),
            behavior_mask,
        ).sqrt()
    behavior_bc_weighted = float(behavior_preservation_weight) * behavior_bc_raw
    q_weighted = float(q_weight) * q_loss
    loss = q_weighted + bc_weighted + behavior_bc_weighted

    diagnostics = {
        # Keep loss_actor_q as the term that actually contributes to the total
        # loss.  At the backward-compatible default q_weight=1 it is identical
        # to the historical value.  The raw value remains available when the
        # Q term is disabled or down-weighted.
        "loss_actor_q": q_weighted.detach(),
        "loss_actor_q_raw": q_loss.detach(),
        "loss_actor_q_weighted": q_weighted.detach(),
        "actor_q_weight": q.new_tensor(float(q_weight)),
        "loss_actor_bc_raw": bc_raw.detach(),
        "loss_actor_bc_weighted": bc_weighted.detach(),
        "loss_actor_behavior_bc_raw": behavior_bc_raw.detach(),
        "loss_actor_behavior_bc_weighted": behavior_bc_weighted.detach(),
        "actor_behavior_sample_frac": behavior_mask.float().mean(),
        "actor_behavior_target_rmse": behavior_target_rmse.detach(),
        "actor_q_min_mean": _masked_mean(q.detach(), actor_q_valid),
        "actor_ref_rmse": (
            (mu.detach() - proposal.detach().clamp(-1.0, 1.0)) ** 2
        ).mean().sqrt(),
        "actor_bc_target_rmse": _masked_mean(
            (mu.detach() - bc_target.detach()).square().mean(dim=-1),
            actor_bc_valid,
        ).sqrt(),
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
        "actor_q_valid_frac": actor_q_valid.float().mean(),
        "actor_bc_valid_frac": actor_bc_valid.float().mean(),
    }

    if source is not None:
        for source_id in range(4):
            mask = source == source_id
            diagnostics[f"source_{source_id}_frac"] = mask.float().mean()
            if bool(mask.any()):
                diagnostics[f"source_{source_id}_disagreement_mean"] = disagreement[mask].mean()
                diagnostics[f"source_{source_id}_beta_mean"] = beta_per_sample[mask].mean()
                diagnostics[f"source_{source_id}_actor_bc_valid_frac"] = (
                    actor_bc_valid[mask].float().mean()
                )

    intervention = batch.get("intervention")
    if source is not None:
        human_mask = source == 3
    elif intervention is not None:
        human_mask = intervention.detach().reshape(-1).to(disagreement.device) > 0.5
    else:
        human_mask = torch.zeros_like(disagreement, dtype=torch.bool)

    zero = disagreement.new_zeros(())
    diagnostics["human_sample_frac"] = human_mask.float().mean()
    if bool(human_mask.any()):
        executable_proposal = proposal.detach().clamp(-1.0, 1.0)
        executable_target = bc_target.detach()
        human_delta = executable_target[human_mask] - executable_proposal[human_mask]
        diagnostics["human_vla_action_rmse"] = human_delta.square().mean().sqrt()
        if getattr(actor, "action_residual", False):
            if hasattr(actor, "residual_delta_bound"):
                residual_bound = actor.residual_delta_bound(human_delta)
            else:
                residual_bound = torch.as_tensor(
                    float(actor.delta_scale),
                    device=human_delta.device,
                    dtype=human_delta.dtype,
                )
            outside = (human_delta.abs() > residual_bound + 1e-6).any(dim=-1)
            diagnostics["human_target_outside_residual_bound_frac"] = outside.float().mean()
            diagnostics["actor_delta_scale_min"] = residual_bound.min()
            diagnostics["actor_delta_scale_max"] = residual_bound.max()
        else:
            diagnostics["human_target_outside_residual_bound_frac"] = zero
        diagnostics["human_bc_target_rmse"] = (
            (mu.detach()[human_mask] - executable_target[human_mask]).square().mean().sqrt()
        )
    else:
        diagnostics["human_vla_action_rmse"] = zero
        diagnostics["human_target_outside_residual_bound_frac"] = zero
        diagnostics["human_bc_target_rmse"] = zero

    return loss, diagnostics
