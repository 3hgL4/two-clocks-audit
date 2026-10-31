"""Group filtering + resampling for the format-signal-audit arms B and C.

Three arms share one harness and differ ONLY in what happens to a generated
group of G=8 responses before it can enter a gradient step:

  A ("none")      keep every group; the batch is the first
                  prompts_per_rollout prompts of the rollout permutation.
  B ("accuracy")  reject groups whose accuracy is degenerate -- every response
                  correct (acc==1) or every response wrong (acc==0). This is
                  the DAPO dynamic-sampling predicate (Eq. 11 there, with
                  `is_equivalent` == our `verify.is_correct`), applied to the
                  composed reward harness (correctness + format bonus).
  C ("variance")  reject groups whose composed-reward standard deviation
                  (ddof=1) is < 1e-8 -- the exact predicate `grpo.group_advantages`
                  uses to zero an advantage. Rejected groups are RESAMPLED, not
                  masked: GRPO already zeroes their advantage, so masking would
                  make arm C gradient-identical to arm A (v1.0 correction).

Refill rule (frozen text, 方案 §三): rejected groups are replaced by fresh
prompts drawn from LATER positions of the SAME per-rollout permutation, one
round at a time, each round sized to exactly the number of groups still
missing, until either n_target_groups valid groups are assembled or the
resample cap (`train.resample.max_extra_prompts`, counting REFILL prompts
only) or the permutation is exhausted. Every generated response -- kept or
discarded -- counts against the arm's generation-token budget.

Determinism contract: the whole procedure is a pure function of
(seed, rollout_idx, pool, mode, cap). Prompt order comes from
`data.rollout_permutation` (same stream as `data.prompt_batch`, so round 1 of
every arm sees the byte-identical prompt batch of arm A). Sampler seeds are
derived per request as in the backends, with an `attempt` term that is 0 for
round 1 -- so round 1 of arms B/C is token-identical to arm A -- and strictly
increasing afterwards. vLLM sampling reproducibility itself is the already
proven L1 property (fixed batch shape + per-request seeds, 0/512 rows differ),
not something this module re-proves.
"""

from __future__ import annotations

from typing import Sequence

import data as data_mod
import verify
from rollout import Rollout

VALID_MODES = ("none", "accuracy", "variance")
# Same threshold as grpo.group_advantages' zero-variance zeroing: the filter and
# the advantage code must agree on what "zero variance" means, bit for bit.
VAR_EPS = 1e-8


def group_verdicts(
    group_rewards: Sequence[Sequence[float]],
    mode: str,
    group_correct: Sequence[Sequence[bool]] | None = None,
    var_eps: float = VAR_EPS,
) -> list[bool]:
    """Per-group keep/reject decision. Order-aligned with group_rewards.

    mode="accuracy" needs group_correct (the `is_equivalent` counter); the
    composed reward alone cannot recover it because the format bonus makes
    "all wrong" rewards nonzero (-0.1 each).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown resample mode: {mode!r}")
    if mode == "none":
        return [True] * len(group_rewards)
    out: list[bool] = []
    for gi, r in enumerate(group_rewards):
        if len(r) == 0:
            out.append(False)
            continue
        if mode == "accuracy":
            if group_correct is None:
                raise ValueError("mode='accuracy' requires group_correct")
            corr = group_correct[gi]
            acc = sum(bool(c) for c in corr) / len(corr)
            out.append(0.0 < acc < 1.0)
        else:  # variance, on the composed reward, ddof=1 like grpo.group_advantages
            if len(r) < 2:
                out.append(False)
                continue
            mean = sum(r) / len(r)
            var = sum((x - mean) ** 2 for x in r) / (len(r) - 1)
            out.append(var**0.5 >= var_eps)
    return out


def rollout_with_resample(
    pool: list[dict],
    *,
    seed: int,
    rollout_idx: int,
    G: int,
    n_target_groups: int,
    backend,
    mode: str,
    max_extra_prompts: int,
    var_eps: float = VAR_EPS,
    log=print,
) -> tuple[Rollout, dict]:
    """Generate with group filtering + refill; return the merged rollout + stats.

    The returned Rollout contains ONLY the kept groups, prompt-major, in
    generation order -- so the downstream train.py code (rewards, advantages,
    partition) runs unchanged. `stats` carries what the frozen budget text
    requires the run to report: every attempted completion token, kept vs
    discarded split, rounds, rejection counts.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown resample mode: {mode!r}")
    if mode == "none":
        raise ValueError("mode='none' must not enter the resample path")
    perm = data_mod.rollout_permutation(len(pool), seed, rollout_idx)

    kept_items: list[dict] = []   # expanded rows, index-aligned with kept responses
    kept_pids: list[list[int]] = []
    kept_cids: list[list[int]] = []
    kept_texts: list[str] = []
    kept_groups = 0  # NB: rb.items is G-expanded, so the group count is its own counter
    # Discarded responses are archived by the caller (they are the audit's
    # object of study: the discarded groups' own format/accuracy statistics),
    # so they are returned here in full, not just counted.
    dis_pids: list[list[int]] = []
    dis_cids: list[list[int]] = []
    dis_texts: list[str] = []
    n_responses_seen = 0  # global response counter across rounds (drives vLLM seeds)
    attempt = 0
    cursor = 0
    refill_used = 0
    stats = {
        "resample_mode": mode,
        "resample_rounds": 0,
        "resample_attempted_prompts": 0,
        "resample_attempted_responses": 0,
        "resample_kept_groups": 0,
        "resample_discarded_groups": 0,
        "resample_discarded_responses": 0,
        "resample_discarded_correct": 0,
        "resample_capped": False,
        "resample_target_groups": n_target_groups,
        # raw rows for the caller's discarded-rollout archive
        "discarded_prompt_ids": dis_pids,
        "discarded_completion_ids": dis_cids,
        "discarded_completion_texts": dis_texts,
    }

    while kept_groups < n_target_groups:
        need = n_target_groups - kept_groups
        draw = need
        if attempt > 0:
            # rounds 2+ are refill: bounded by the cap and by what is left of
            # the permutation after the initial batch.
            draw = min(need, max_extra_prompts - refill_used, len(perm) - cursor)
        else:
            draw = min(need, len(perm) - cursor)
        if draw <= 0:
            stats["resample_capped"] = True
            break

        items = [pool[perm[cursor + k]] for k in range(draw)]
        cursor += draw
        if attempt > 0:
            refill_used += draw
        rb = backend.generate(items, G, rollout_idx, seed=seed, attempt=attempt,
                              response_offset=n_responses_seen)
        attempt += 1

        rewards = [
            verify.reward(int(it["answer"]), txt)
            for it, txt in zip(rb.items, rb.completion_texts)
        ]
        correct = [
            verify.is_correct(int(it["answer"]), txt)
            for it, txt in zip(rb.items, rb.completion_texts)
        ]
        group_rewards = [rewards[i * G : (i + 1) * G] for i in range(rb.n_prompts)]
        group_correct = [correct[i * G : (i + 1) * G] for i in range(rb.n_prompts)]
        keep = group_verdicts(group_rewards, mode, group_correct, var_eps)

        stats["resample_rounds"] += 1
        stats["resample_attempted_prompts"] += draw
        stats["resample_attempted_responses"] += rb.n_responses
        for gi, k in enumerate(keep):
            lo, hi = gi * G, (gi + 1) * G
            if k:
                kept_items.extend(rb.items[lo:hi])
                kept_pids.extend(rb.prompt_ids[lo:hi])
                kept_cids.extend(rb.completion_ids[lo:hi])
                kept_texts.extend(rb.completion_texts[lo:hi])
                kept_groups += 1
            else:
                stats["resample_discarded_groups"] += 1
                stats["resample_discarded_responses"] += G
                stats["resample_discarded_correct"] += sum(bool(c) for c in correct[lo:hi])
                dis_pids.extend(rb.prompt_ids[lo:hi])
                dis_cids.extend(rb.completion_ids[lo:hi])
                dis_texts.extend(rb.completion_texts[lo:hi])
        n_responses_seen += rb.n_responses
        log(
            f"[resample r{rollout_idx} round {attempt}] drew {draw} prompts, "
            f"kept {sum(keep)}/{len(keep)} groups, total {kept_groups}/{n_target_groups}"
        )

    stats["resample_kept_groups"] = kept_groups
    if kept_groups == 0:
        raise RuntimeError(
            f"resample exhausted with 0 kept groups "
            f"(mode={mode}, cap={max_extra_prompts}, pool={len(pool)})"
        )
    merged = Rollout(
        items=kept_items,
        prompt_ids=kept_pids,
        completion_ids=kept_cids,
        completion_texts=kept_texts,
        group_size=G,
        wall_ms=0.0,  # the caller accumulates wall time across rounds itself
        backend=getattr(backend, "name", "unknown"),
    )
    return merged, stats
