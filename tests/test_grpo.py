"""Correctness tests for the one thing the audit cannot get wrong: the loss.

Three independent checks, because "it matches the library" is not verifiable
without instantiating a trainer:

1.  reference loop -- a scalar, per-token implementation written with plain
    Python math, compared against the vectorised version to 1e-6;
2.  finite differences -- autograd gradient vs a numerical gradient on a tiny
    problem, which catches sign errors and missing masks;
3.  invariants -- ratio == 1 gives the plain policy gradient, clipping engages
    only on the documented side, and padded tokens contribute nothing.

The operational definition being reproduced is TRL's token-level GRPO
(`trl/trainer/grpo_trainer.py:3095-3180`): asymmetric clip (1-eps_low, 1+eps_high),
optional `delta` cap on the pessimistic branch, `-min(coef1*A, coef2*A)`.

    python -m pytest tests -q        (or: python tests/test_grpo.py)
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import grpo  # noqa: E402


def _reference_loss(logp, old, adv, mask, eps_low, eps_high, delta):
    """Token mean of -min(c1*A, c2*A) with c1 possibly capped by delta, scalar loop."""
    B, T = len(logp), len(logp[0])
    total, n = 0.0, 0
    for b in range(B):
        for t in range(T):
            if mask[b][t] <= 0:
                continue
            r = math.exp(logp[b][t] - old[b][t])
            c1 = r if delta is None else min(r, delta)
            c2 = min(max(r, 1 - eps_low), 1 + eps_high)
            total += -min(c1 * adv[b], c2 * adv[b])
            n += 1
    return total / max(1, n)


def test_matches_reference_loop() -> None:
    torch.manual_seed(0)
    B, T = 4, 7
    logp = torch.randn(B, T, dtype=torch.float64)
    old = logp + 0.3 * torch.randn(B, T, dtype=torch.float64)
    adv = torch.randn(B, 1, dtype=torch.float64)
    mask = torch.ones(B, T)
    mask[1, 5:] = 0.0
    mask[3, :] = 0.0

    loss, _ = grpo.grpo_loss(logp, old, adv, mask, eps_low=0.2, eps_high=0.28,
                             clip_delta=10.0, norm="token_mean")
    ref = _reference_loss(logp.tolist(), old.tolist(), adv.squeeze(1).tolist(),
                          mask.tolist(), 0.2, 0.28, 10.0)
    assert abs(float(loss) - ref) < 1e-9, (float(loss), ref)


def test_gradient_matches_finite_differences() -> None:
    torch.manual_seed(1)
    B, T = 3, 5
    base = torch.randn(B, T, dtype=torch.float64)
    old = base + 0.2 * torch.randn(B, T, dtype=torch.float64)
    adv = torch.tensor([[-1.5], [0.7], [2.3]], dtype=torch.float64)
    mask = torch.ones(B, T)
    mask[2, 4] = 0.0

    x = base.clone().requires_grad_(True)
    loss, _ = grpo.grpo_loss(x, old, adv, mask, eps_low=0.2, eps_high=0.28,
                             clip_delta=10.0)
    loss.backward()
    g_auto = x.grad.clone()

    eps = 1e-6
    g_num = torch.zeros_like(base)
    for b in range(B):
        for t in range(T):
            xp = base.clone(); xp[b, t] += eps
            xm = base.clone(); xm[b, t] -= eps
            lp, _ = grpo.grpo_loss(xp, old, adv, mask, eps_low=0.2, eps_high=0.28, clip_delta=10.0)
            lm, _ = grpo.grpo_loss(xm, old, adv, mask, eps_low=0.2, eps_high=0.28, clip_delta=10.0)
            g_num[b, t] = (lp - lm) / (2 * eps)
    assert torch.allclose(g_auto, g_num, atol=1e-6), (g_auto - g_num).abs().max()


def test_invariants() -> None:
    # ratio == 1 -> loss is the plain policy gradient -mean(A)
    logp = torch.zeros(2, 3, dtype=torch.float64)
    old = logp.clone()
    adv = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    mask = torch.ones(2, 3)
    loss, stats = grpo.grpo_loss(logp, old, adv, mask, eps_low=0.2, eps_high=0.28, clip_delta=10.0)
    assert abs(float(loss) - (-(1.0 - 1.0) / 2)) < 1e-9
    assert stats["kl_per_rollout_policy"] == 0.0
    assert stats["clip_frac_any"] == 0.0

    # high side only clips for A > 0: with A<0 and r>1+eps nothing is clipped
    old2 = torch.full((1, 1), -1.0, dtype=torch.float64)   # r = e^1 = 2.72
    logp2 = torch.zeros(1, 1, dtype=torch.float64)
    _, s_pos = grpo.grpo_loss(logp2, old2, torch.tensor([[1.0]], dtype=torch.float64),
                              torch.ones(1, 1, dtype=torch.float64), eps_low=0.2, eps_high=0.28)
    _, s_neg = grpo.grpo_loss(logp2, old2, torch.tensor([[-1.0]], dtype=torch.float64),
                              torch.ones(1, 1, dtype=torch.float64), eps_low=0.2, eps_high=0.28)
    assert s_pos["clip_frac_high"] == 1.0 and s_pos["clip_frac_low"] == 0.0
    assert s_neg["clip_frac_high"] == 0.0 and s_neg["clip_frac_low"] == 0.0


def test_masked_tokens_are_ignored() -> None:
    torch.manual_seed(2)
    logp = torch.randn(2, 6, dtype=torch.float64)
    old = logp + 0.1
    adv = torch.tensor([[0.5], [-0.5]], dtype=torch.float64)
    mask_a = torch.ones(2, 6)
    mask_b = mask_a.clone()
    mask_b[0, 3:] = 0.0
    logp_b = logp.clone()
    logp_b[0, 3:] = 99.0                       # garbage in padded positions
    la, _ = grpo.grpo_loss(logp, old, adv, mask_a, norm="token_mean")
    lb, _ = grpo.grpo_loss(logp_b, old, adv, mask_b, norm="token_mean")
    assert abs(float(la) - float(lb)) > 1e-6   # different token counts -> different mean
    # ... but the padded positions must not add loss: compare with them zeroed out
    logp_c = logp.clone()
    old_c = old.clone()
    logp_c[0, 3:] = 0.0
    old_c[0, 3:] = 0.0                         # r = 1 -> contributes -min(A, A) = -A
    lc, _ = grpo.grpo_loss(logp_c, old_c, adv, mask_b, norm="token_mean")
    assert abs(float(lb) - float(lc)) < 1e-9


def test_group_advantages() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    adv, stats = grpo.group_advantages(rewards, G=4)
    assert adv[:4].abs().sum() == 0.0, "zero-variance group must give exactly zero advantage"
    assert stats["zero_var_group_frac"] == 0.5
    v = adv[4:]
    assert abs(float(v.mean())) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")
