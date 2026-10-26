"""Run train.py with the chunked-logsumexp scorer, recording per-micro-batch Lmax.

Two questions this answers at once (2026-10-25, 48G session):

  1. Headroom. The shipped scorer materialises a fp32 (B,T,V) copy of the logits,
     so peak grows ~21% when completions run to the cap (Lmax 584 -> 712, measured
     in tools/probe_memory.py: 18.19 -> 21.99 GiB). The first 512-response run left
     only 3.1 GiB of the card free, so that growth is the whole question. The
     chunked scorer reduces logsumexp one 64-position slice at a time in
     checkpointed form: 18.04 GiB at Lmax 712, i.e. cheaper at the worst length
     than the shipped scorer at the average length.

  2. How long the sequences actually get. metrics.jsonl only stores resp_len_mean.
     Lmax is a max over the micro-batch, so the mean says nothing about the peak
     that decides an OOM. This wrapper records every call's Lmax, which is exactly
     the quantity the memory goes as.

Numerics: max |Δlogp| vs the fp32 path is 1.9e-06 (probe_memory numeric check),
because each 64-position slice is still upcast to fp32 before the reduction.
score(fp32=False) is NOT an option: it drifts 3e-02 nats, and grpo_loss rejects
bf16 outright (quantile() on a bf16 tensor).

Usage: python tools/run_with_chunk.py --config configs/cloud48g_throughput.yaml
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import model as M  # noqa: E402
import train as T  # noqa: E402
from probe_memory import score_chunk  # noqa: E402

calls: list[tuple[int, int, bool]] = []
_orig = M.Policy.score


def patched(self, prompt_ids, completion_ids, *, requires_grad=False, fp32=True):
    if not calls:
        print(f"[patch] first call in pid={os.getpid()} grad={requires_grad}", flush=True)
    lmax = max(len(p) + len(c) for p, c in zip(prompt_ids, completion_ids))
    calls.append((len(prompt_ids), lmax, requires_grad))
    if requires_grad:
        return score_chunk(self, prompt_ids, completion_ids)
    with torch.no_grad():
        return score_chunk(self, prompt_ids, completion_ids)


def report() -> None:
    grad = [(b, l) for b, l, g in calls if g]
    if not grad:
        print("[lmax] no grad calls recorded", flush=True)
        return
    ls = sorted(l for _, l in grad)
    n = len(ls)
    q = lambda p: ls[min(n - 1, int(p * n))]  # noqa: E731
    print(f"[lmax] grad calls={n} B={grad[0][0]} "
          f"min={ls[0]} p50={q(.5)} p90={q(.9)} p99={q(.99)} max={ls[-1]}", flush=True)
    hist = collections.Counter((l // 50) * 50 for l in ls)
    print("[lmax] histogram (bucket=50): "
          + " ".join(f"{k}:{v}" for k, v in sorted(hist.items())), flush=True)
    print(f"[lmax] cap+prompt budget = 512 + {ls[0] - 512 if ls[0] >= 512 else '?'}", flush=True)


if __name__ == "__main__":
    M.Policy.score = patched
    print(f"[patch] installed in pid={os.getpid()} bound={M.Policy.score is patched}", flush=True)
    rc = 1
    try:
        rc = T.main(sys.argv[1:])
    finally:
        report()
    sys.exit(rc)
