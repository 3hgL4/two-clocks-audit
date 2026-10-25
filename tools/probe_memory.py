"""Peak-memory probe for one GRPO micro-step, replicating train.py's inner loop.

Runs the whole micro-step (score -> grpo_loss -> backward -> _accumulate ->
_write_back -> opt.step) so the number is comparable to a real run, not just to
`score()` alone. Reports peak and reserved VRAM for a grid of (variant, B, Lmax).

Variants
  chunk   the shipped path: log-sum-exp in 64-position slices, recomputed in
          backward (D-15). This is what production runs.
  full    `score(chunk=0)`: the pre-D-15 single-shot fp32 upcast of (B, T, V).
          Kept only for A/B; it is what used to OOM on the 48G box.

Measured 2026-10-25 on one 48G 4090 (see notes/2026-10-25_cloud48g/HEADROOM.md):

  chunk  B=4 Lmax=712 -> 18.04 GiB peak   full B=4 Lmax=712 -> 21.99
  chunk  B=8 Lmax=712 -> 31.98 GiB peak   full B=8 Lmax=712 -> 39.89
  chunk  B=8 Lmax=584 -> 27.04 GiB peak   full B=8 Lmax=584 -> 28.73 reserved

Run:  /opt/conda/bin/python tools/probe_memory.py [--lp 200] [--lc 512]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
import grpo  # noqa: E402
import train as T  # noqa: E402
from model import Policy  # noqa: E402

GiB = 2**30


def one(policy, params, opt, variant: str, B: int, lp: int, lc: int) -> None:
    pids = [[100 + i for i in range(lp)] for _ in range(B)]
    cids = [[2000 + i for i in range(lc)] for _ in range(B)]
    mask = policy.mask_for(cids)
    adv = torch.linspace(-1.0, 1.0, B, device=policy.device)

    policy.model.zero_grad(set_to_none=True)
    for prm in params:
        prm.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tag = f"[{variant:5s} B={B} Lp={lp} Lc={lc} Lmax={lp + lc}]"
    try:
        logp = policy.score(pids, cids, requires_grad=True, chunk=0 if variant == "full" else None)
        fwd = torch.cuda.max_memory_allocated() / GiB
        loss, _ = grpo.grpo_loss(logp, logp.detach(), adv, mask)
        loss.backward()
        buf = T._grad_buffer(policy)
        T._accumulate(buf, policy, 1.0)
        T._write_back(buf, policy)
        opt.step()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / GiB
        res = torch.cuda.max_memory_reserved() / GiB
        print(f"{tag} fwd={fwd:6.2f} peak={peak:6.2f} reserved={res:6.2f} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        del logp, loss
    except torch.cuda.OutOfMemoryError:
        print(f"{tag} OOM at {torch.cuda.memory_allocated() / GiB:.2f} GiB allocated", flush=True)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def numeric_check(policy, B: int = 2, lp: int = 200, lc: int = 384) -> None:
    """The chunked reduction must agree with the single-shot fp32 one."""
    pids = [[100 + i for i in range(lp)] for _ in range(B)]
    cids = [[2000 + i for i in range(lc)] for _ in range(B)]
    with torch.no_grad():
        a = policy.score(pids, cids, chunk=0)
        b = policy.score(pids, cids)
    m = policy.mask_for(cids).bool()
    d = (a - b)[m].abs()
    print(f"[numeric] full-vs-chunk max={d.max():.3e} mean={d.mean():.3e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp", type=int, default=200)
    ap.add_argument("--lc", type=int, default=0, help="override completion length")
    ap.add_argument("--cfg", default="configs/cloud48g.yaml")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-numeric", action="store_true")
    args = ap.parse_args()

    cfg = C.load_config(str(ROOT / args.cfg))
    policy = Policy(cfg, device="cuda")
    params = T._trainable(policy)
    opt = torch.optim.AdamW(params, lr=1e-4)
    n = sum(p.numel() for p in params)
    print(f"[info] vocab={len(policy.tokenizer)} trainable={n / 1e6:.2f}M "
          f"tensors={len(params)} dtype={next(policy.model.parameters()).dtype} "
          f"logit_chunk={policy.logit_chunk}", flush=True)
    print(f"[info] card={torch.cuda.get_device_properties(0).total_memory / GiB:.1f} GiB", flush=True)

    if not args.no_numeric:
        numeric_check(policy)

    plan = [
        ("chunk", 4, 384),   # calibration point: matched probe2's 18.20 pre-D-15
        ("chunk", 4, 512),   # worst case: completions at the cap
        ("full", 4, 512),
        ("chunk", 8, 512),
        ("full", 8, 512),
    ]
    for variant, B, lc in plan:
        if args.only and variant != args.only:
            continue
        one(policy, params, opt, variant, B, args.lp, args.lc or lc)


if __name__ == "__main__":
    main()
