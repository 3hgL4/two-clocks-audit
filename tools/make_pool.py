"""Generate the frozen prompt pool, then hash it.

Run once. The pool is never regenerated during training, and `src/data.py`
refuses to run if the hash does not match `configs/base.yaml`.

    python tools/make_pool.py --size 2000 --seed 20261024

Why 2000 when a rollout only needs 512 prompts:
  * the largest arm (Br = 4096, G = 8) needs 512 distinct prompts per rollout;
  * the S1 pool held 500, i.e. 12 fewer than required, which would have forced
    silent prompt reuse inside single rollouts;
  * 2000 gives ~4 rollouts of headroom, so no prompt repeats within a run of 8
    rollouts, and different seeds still see different prompt subsets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import data as data_mod  # noqa: E402
import gen_counting  # noqa: E402


def dedupe(items, keep: int):
    """Drop repeated prompt texts, preserving generator order.

    Duplicates are not harmless here: if the same prompt appears twice inside one
    rollout, the 8 generations of "the same question" are split across two groups,
    which inflates within-group variance and quietly changes the effective batch.
    """
    seen: set[str] = set()
    out = []
    for it in items:
        if it.prompt in seen:
            continue
        seen.add(it.prompt)
        out.append(it)
        if len(out) >= keep:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--kind", default="mixed", choices=["mixed", "easy"])
    ap.add_argument("--out", default="data/prompts_mixed2000.jsonl")
    ap.add_argument("--manifest", default="data/MANIFEST.json")
    a = ap.parse_args()

    # over-generate, then dedupe down to exactly --size unique prompts
    raw = gen_counting.build_dataset(a.kind, int(a.size * 1.06), seed=a.seed)
    items = dedupe(raw, a.size)
    if len(items) < a.size:
        raise SystemExit(
            f"only {len(items)} unique prompts from {len(raw)} generated; raise the over-generation factor"
        )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gen_counting.save_jsonl(items, str(out))

    pool = data_mod.load_pool(out)
    summary = data_mod.pool_summary(pool)
    manifest = data_mod.write_manifest(
        [out],
        a.manifest,
        extra={
            "generator": "src/gen_counting.py:build_dataset",
            "kind": a.kind,
            "size": len(items),
            "seed": a.seed,
            "deduped_from": len(raw),
            "pool_summary": summary,
        },
    )
    sha = list(manifest["files"].values())[0]["sha256"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out} ({out.stat().st_size/1024:.1f} KiB)")
    print(f"sha256 {sha}")
    print("\nput this in configs/base.yaml -> data.sha256:")
    print(f'  sha256: "{sha}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
