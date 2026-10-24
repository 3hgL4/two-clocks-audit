"""Measure the base model's pass rate per item, then build a usable-band pool.

Why this exists (and why it runs *before* the paid experiments):

Two Clocks' failure clock `N_fail = C√b/η` is a statement about failure rate
rising with training. That requires a starting point where the policy sometimes
succeeds and sometimes fails. A 1.7B *base* model on depth-5..7 counting items
can sit at a pass rate of 0, in which case every GRPO group has identical reward,
every advantage is zero, and the "failure window" is not short -- it is
unmeasurable. Discovering that after paying for 9 runs would waste the budget.

So: spend a tenth of a run here, get the pass-rate distribution, and only then
decide whether to train on the full mixed pool, on the [lo, hi] band, or to
re-generate easier items.

    python tools/calibrate_pool.py --limit 400 --samples 8 \
        --pool data/prompts_mixed2000.jsonl \
        --out data/calibration_qwen3-1.7b.json \
        --band-pool data/prompts_band.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import data as data_mod  # noqa: E402
import rollout as rollout_mod  # noqa: E402
import verify  # noqa: E402
from config import load_config  # noqa: E402
from model import Policy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--pool", default=None, help="defaults to configs data.pool")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--backend", default="hf", choices=["hf", "fake"])
    ap.add_argument("--lo", type=float, default=0.1, help="band lower bound on pass rate")
    ap.add_argument("--hi", type=float, default=0.9, help="band upper bound on pass rate")
    ap.add_argument("--out", default="data/calibration.json")
    ap.add_argument("--band-pool", default="data/prompts_band.jsonl")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    pool_path = a.pool or cfg["data.pool"]
    pool = data_mod.load_pool(pool_path, cfg["data.sha256"] or None)
    if a.limit < len(pool):
        # spread the subset across the pool instead of taking a prefix,
        # otherwise the depth histogram of the calibration set is not the pool's
        step = len(pool) / a.limit
        pool = [pool[int(i * step)] for i in range(a.limit)]

    policy = Policy(cfg, device=a.device)
    backend = rollout_mod.build_backend(a.backend, cfg, policy)

    t0 = __import__("time").perf_counter()
    rb = backend.generate(pool, a.samples, rollout_idx=0, seed=12345)
    elapsed = __import__("time").perf_counter() - t0

    G = a.samples
    per_item = []
    for i, it in enumerate(pool):
        hits = sum(
            verify.is_correct(int(it["answer"]), rb.completion_texts[i * G + g])
            for g in range(G)
        )
        per_item.append({
            "idx": i,
            "depth": int(it.get("depth", -1)),
            "prompt": it["prompt"],
            "answer": it["answer"],
            "n_correct": hits,
            "pass_rate": hits / G,
        })

    rates = np.array([p["pass_rate"] for p in per_item])
    by_depth: dict[int, list[float]] = {}
    for p in per_item:
        by_depth.setdefault(p["depth"], []).append(p["pass_rate"])

    band = [p for p in per_item if a.lo <= p["pass_rate"] <= a.hi]
    report = {
        "pool": pool_path,
        "pool_sha256": data_mod.sha256_file(pool_path),
        "n_items": len(per_item),
        "samples_per_item": G,
        "backend": a.backend,
        "model": cfg["model.name"],
        "elapsed_s": round(elapsed, 2),
        "responses": len(per_item) * G,
        "overall_pass_rate": round(float(rates.mean()), 4),
        "zero_pass_frac": round(float((rates == 0).mean()), 4),
        "one_pass_frac": round(float((rates == 1).mean()), 4),
        "band": {"lo": a.lo, "hi": a.hi, "n_items": len(band),
                 "frac_of_pool": round(len(band) / max(1, len(per_item)), 4)},
        "pass_rate_by_depth": {
            str(d): {"n": len(v), "mean": round(float(np.mean(v)), 4)}
            for d, v in sorted(by_depth.items())
        },
        "histogram": dict(sorted(Counter(np.round(rates, 3).tolist()).items())),
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"report": report, "per_item": per_item}, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    # band pool keeps the original item schema so training code is unchanged
    if band:
        keep_prompts = {p["prompt"] for p in band}
        kept = [it for it in data_mod.load_pool(pool_path) if it["prompt"] in keep_prompts]
        import gen_counting
        gen_counting.save_jsonl(
            [gen_counting.CountingItem(**{k: v for k, v in it.items() if k != "_idx"})
             for it in kept],
            a.band_pool,
        )
        mf = data_mod.write_manifest([a.band_pool], str(Path(a.band_pool).with_suffix(".MANIFEST.json")),
                                     extra={"derived_from": a.out, "band": report["band"]})
        print(f"\nband pool -> {a.band_pool} ({len(kept)} items)")
        print(f'sha256: "{list(mf["files"].values())[0]["sha256"]}"')
    else:
        print("\n[warn] no item fell in the band: lower --lo or regenerate easier items "
              "before spending budget on training")

    if report["overall_pass_rate"] < 0.02:
        print("[warn] overall pass rate < 2%: GRPO groups will be mostly zero-variance; "
              "training would be uninformative regardless of mu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
