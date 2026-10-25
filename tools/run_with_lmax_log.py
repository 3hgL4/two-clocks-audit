"""Run train.py unchanged, recording the L_max of every score() call.

Why: `metrics.jsonl` stores `resp_len_mean` (322 tokens on the counting pool), but
peak memory is driven by the *maximum* sequence in a micro-batch, and the mean
says nothing about that maximum. L_max = prompt + completion, capped at
prompt_max + max_completion_length, so this recorder is what turns "memory might
grow if completions get long" into a measured bound.

Usage: python tools/run_with_lmax_log.py --config configs/cloud48g_throughput.yaml

Since D-15 the chunked scorer is shipped in Policy.score, so this wrapper only
observes; it does not change the numerics. (Before that it monkeypatched a
chunked scorer in, which is why the file used to be called run_with_chunk.py.)

Measured with it on 2026-10-25, 512 responses, cap 512, micro=4:
    min=210 p50=586 p90=610 p99=614 max=614   (128 grad calls)
and with micro=8: p50=591 p90=607 max=607    (64 grad calls)
"""

from __future__ import annotations

import collections
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import model as M  # noqa: E402
import train as T  # noqa: E402

calls: list[tuple[int, int, bool]] = []
_orig = M.Policy.score


def patched(self, prompt_ids, completion_ids, **kw):
    if not calls:
        print(f"[lmax] first call in pid={os.getpid()} grad={kw.get('requires_grad')}", flush=True)
    lmax = max(len(p) + len(c) for p, c in zip(prompt_ids, completion_ids))
    calls.append((len(prompt_ids), lmax, bool(kw.get("requires_grad"))))
    return _orig(self, prompt_ids, completion_ids, **kw)


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


if __name__ == "__main__":
    M.Policy.score = patched
    rc = 1
    try:
        rc = T.main(sys.argv[1:])
    finally:
        report()
    sys.exit(rc)
