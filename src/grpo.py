"""GRPO advantage and policy-gradient loss.

The loss is a deliberate re-implementation of TRL's token-level GRPO
(`trl/trainer/grpo_trainer.py`, lines ~3095-3180) so that `tests/test_grpo_align.py`
can assert numerical agreement with the library version. If the two disagree,
the audit's measurements are not trustworthy, so this is tested rather than
assumed.

Why not just use TRL? Because the audit needs per-mu-step *observations* that
the trainer does not expose: `kl_per_rollout_policy` at every mu, the clipped
gradient-norm share, and the micro-batch gradient spread sigma that the drift
clock is built from. That is what this loop exists for.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# advantages
# ---------------------------------------------------------------------------


def group_advantages(rewards: torch.Tensor, G: int) -> tuple[torch.Tensor, dict]:
    """GRPO group-normalised advantages.

    rewards: (P*G,) with layout [p0_g0..p0_g{G-1}, p1_g0, ...]
    returns: (P*G,) advantages, stats dict

    Zero-variance groups (all G rollouts identical in reward) carry no learning
    signal; their advantage is exactly 0 and they are counted, because the
    fraction of such groups is itself a difficulty diagnostic (the "dead prompt"
    rate that quietly changes the effective batch size across arms).
    """
    if rewards.numel() % G:
        raise ValueError(f"rewards ({rewards.numel()}) not divisible by G ({G})")
    r = rewards.view(-1, G)
    mean = r.mean(dim=1, keepdim=True)
    # unbiased std to match torch/TRL defaults; +1e-4 keeps degenerate groups finite
    std = r.std(dim=1, keepdim=True, unbiased=True)
    adv = (r - mean) / (std + 1e-4)
    zero_var = (std.squeeze(1) < 1e-8)
    adv = torch.where(zero_var.view(-1, 1), torch.zeros_like(adv), adv)

    stats = {
        "zero_var_group_frac": float(zero_var.float().mean().item()),
        "adv_abs_mean": float(adv.abs().mean().item()),
        "reward_mean": float(rewards.mean().item()),
        "group_std_mean": float(std.mean().item()),
    }
    return adv.reshape(-1), stats


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    clip_delta: float | None = 10.0,
    norm: str = "token_mean",
) -> tuple[torch.Tensor, dict]:
    """Token-level GRPO with asymmetric (dual-segment) clipping + optional dual clip.

    logp / old_logp / mask : (B, T)  completion-token log-probs and 0/1 mask
    adv                    : (B, 1) or (B,)
    """
    if logp.shape != old_logp.shape or logp.shape != mask.shape:
        raise ValueError(f"shape mismatch: {logp.shape} {old_logp.shape} {mask.shape}")
    if adv.dim() == 1:
        adv = adv.unsqueeze(1)

    log_ratio = (logp - old_logp) * mask          # padded positions -> ratio 1
    coef_1 = torch.exp(log_ratio)
    coef_2 = torch.clamp(coef_1, 1.0 - eps_low, 1.0 + eps_high)
    if clip_delta is not None:
        # bounded pessimistic branch (INTELLECT-2 / TRL `delta`): only the upper
        # side of coef_1 is capped, exactly as TRL does it.
        coef_1 = torch.clamp(coef_1, max=clip_delta)

    per_token = -torch.min(coef_1 * adv, coef_2 * adv)

    n_valid = mask.sum().clamp(min=1.0)
    if norm == "token_mean":
        loss = (per_token * mask).sum() / n_valid
    elif norm == "rollout_token":
        loss = (per_token * mask).sum() / (per_token.size(0) * per_token.size(1))
    else:
        raise ValueError(f"unknown loss norm: {norm}")

    with torch.no_grad():
        # ratio diagnostics use the *unclamped* ratio
        ratio = torch.exp((logp - old_logp).detach())
        valid = mask > 0
        n = valid.sum().clamp(min=1.0)
        is_low = (ratio < 1.0 - eps_low) & (adv < 0) & valid
        is_high = (ratio > 1.0 + eps_high) & (adv > 0) & valid
        kl = ((old_logp - logp).detach() * mask).sum() / n
        vals = ratio[valid]
        p95 = torch.quantile(vals, 0.95) if vals.numel() > 1 else vals.new_tensor(1.0)
        # C6, second half. Clipping zeroes the gradient of the tokens it clips, so
        # "how much gradient norm do clipped tokens carry" has to be asked of the
        # unclipped surrogate: the per-token dL/dlogp magnitude there is
        # |ratio * adv| exactly, and this share is how much of that mass the clip
        # dropped. NOTE: this is the logprob-space mass, not the parameter-space
        # norm share (which would need a per-token backward and is NOT measured).
        mass = (ratio * adv).abs() * mask
        mass_total = mass.sum().clamp(min=1e-12)
        mass_clipped = (mass * (is_low | is_high)).sum()
        stats = {
            "loss": float(loss.detach().item()),
            "ratio_mean": float(vals.mean().item()) if vals.numel() else 1.0,
            "ratio_p95": float(p95.item()),
            # SAC-mu tail diagnostics (v2 doc section 4): a p95 bounds nothing
            # about the remaining 5% tail, so the tail is recorded directly.
            # Same tensor, zero extra forward.
            "ratio_max": float(vals.max().item()) if vals.numel() else 1.0,
            "ratio_p99": (float(torch.quantile(vals, 0.99).item())
                          if vals.numel() > 1 else 1.0),
            "ratio_over_high_frac": (float((vals > 1.0 + eps_high).float().mean().item())
                                     if vals.numel() else 0.0),
            "kl_per_rollout_policy": float(kl.item()),
            "clip_frac_low": float((is_low.sum() / n).item()),
            "clip_frac_high": float((is_high.sum() / n).item()),
            "clip_frac_any": float(((is_low | is_high).sum() / n).item()),
            "clip_grad_mass_share": float((mass_clipped / mass_total).item()),
        }
    return loss, stats


# ---------------------------------------------------------------------------
# gradient statistics (the drift clock's inputs)
# ---------------------------------------------------------------------------


class GradStats:
    """Accumulates the two numbers the drift clock needs.

    For M micro-batches with gradients g_1..g_M and g_bar = mean(g_i):

        ||g_bar||^2  is measured directly from the averaged gradient
        sigma^2      = ( sum_i ||g_i||^2 - M * ||g_bar||^2 ) / (M - 1)

    Both are estimated from the same M samples, so sigma is the *within-step*
    gradient noise scale -- the quantity in Eq. (3) of arXiv:2610.18853.
    Only scalars are retained, so this costs no extra GPU memory.
    """

    def __init__(self) -> None:
        self.sum_sq = 0.0
        self.n = 0

    def add(self, grad_norm: float) -> None:
        self.sum_sq += float(grad_norm) ** 2
        self.n += 1

    def sigma(self, g_bar_norm: float) -> float | None:
        if self.n < 2:
            return None
        var = (self.sum_sq - self.n * float(g_bar_norm) ** 2) / (self.n - 1)
        return var ** 0.5 if var > 0 else 0.0


def drift_clock(g_bar_norm: float, sigma: float | None, kappa: float = 1.0) -> float | None:
    """d = kappa * ||g_bar|| / sigma  (Eq. 3 of arXiv:2610.18853).

    kappa is a unit convention. The audit only compares d across arms and
    assumes a tau crossing; kappa cancels in every ratio, so its absolute value
    does not affect any claim that is made.
    """
    if sigma is None or sigma <= 0:
        return None
    return kappa * float(g_bar_norm) / float(sigma)
