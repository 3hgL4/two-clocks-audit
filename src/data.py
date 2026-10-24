"""Frozen prompt pool: loading, sha256 provenance, deterministic sampling.

Two properties matter for the audit:

1.  **Freeze + hash.** The pool is generated once, hashed, and never edited.
    `load_pool` refuses to run against a pool whose hash does not match the
    config, so every run in the comparison provably saw the same items.

2.  **Deterministic per-rollout sampling without carried state.** The prompt
    set for rollout `k` is a pure function of (seed, k). That gives us
    (a) exact resume across a machine death, and (b) *prompt-paired* arms: the
    mu=1 and mu=8 runs see byte-identical prompt sets at the same k, so the
    comparison is paired rather than unpaired.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# hashing / manifests
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(paths: list[str | Path], out: str | Path, extra: dict | None = None) -> dict:
    entries = {}
    for p in paths:
        p = Path(p)
        entries[p.as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    doc: dict[str, Any] = {"files": entries}
    if extra:
        doc.update(extra)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return doc


# ---------------------------------------------------------------------------
# pool loading
# ---------------------------------------------------------------------------


def load_pool(path: str | Path, expect_sha256: str | None = None) -> list[dict]:
    """Load a frozen jsonl pool of counting items (prompt/answer/depth/...)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prompt pool not found: {p} (run tools/make_pool.py)")
    if expect_sha256:
        got = sha256_file(p)
        if got != expect_sha256:
            raise ValueError(
                f"prompt pool hash mismatch for {p}\n  expected {expect_sha256}\n  got      {got}\n"
                "The pool must be frozen: regenerate it or update data.sha256, never edit in place."
            )
    items: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for key in ("prompt", "answer"):
                if key not in d:
                    raise ValueError(f"{p}:{i + 1} missing required field {key!r}")
            d["_idx"] = i
            items.append(d)
    if not items:
        raise ValueError(f"prompt pool is empty: {p}")
    return items


# ---------------------------------------------------------------------------
# deterministic sampling
# ---------------------------------------------------------------------------


def _rollout_rng(seed: int, rollout_idx: int) -> random.Random:
    # Independent stream per rollout; stable across processes and Python versions
    # because random.Random(int) is defined in terms of the Mersenne Twister seed.
    return random.Random((seed * 1_000_003) ^ (rollout_idx * 9_176 + 1))


def prompt_batch(pool: list[dict], seed: int, rollout_idx: int, n: int) -> list[dict]:
    """The n prompts used by `rollout_idx`. Pure function of (pool, seed, idx, n).

    Without replacement when the pool is large enough; otherwise a fresh
    permutation per rollout is cycled, which is documented as prompt reuse
    (see REPORT: 'pool reuse' row) rather than silently duplicated.
    """
    rng = _rollout_rng(seed, rollout_idx)
    ids = list(range(len(pool)))
    rng.shuffle(ids)
    if n <= len(ids):
        return [pool[i] for i in ids[:n]]
    out = [pool[i] for i in ids]
    while len(out) < n:                       # cycle a fresh permutation
        rng.shuffle(ids)
        out.extend(pool[i] for i in ids)
    return out[:n]


def pool_summary(pool: list[dict]) -> dict:
    depths = [int(d.get("depth", -1)) for d in pool]
    by_depth: dict[int, int] = {}
    for d in depths:
        by_depth[d] = by_depth.get(d, 0) + 1
    answers = [int(d["answer"]) for d in pool]
    return {
        "n_items": len(pool),
        "depth_hist": dict(sorted(by_depth.items())),
        "answer_mean": round(sum(answers) / len(answers), 4),
        "answer_min": min(answers),
        "answer_max": max(answers),
        "n_distinct_prompts": len({d["prompt"] for d in pool}),
    }
