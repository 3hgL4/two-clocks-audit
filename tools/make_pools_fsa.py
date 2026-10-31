"""Generate and freeze the format-signal-audit prompt pools (2026-10-31).

Outputs (under data/):
  heldout_counting_512.jsonl   held-out frozen test set, counting family
  prompts_mod2000.jsonl        training pool, modular-arithmetic family
  heldout_mod_512.jsonl        held-out frozen test set, mod family
  MANIFEST_fsa.json            sha256 + generation parameters for all four
                               pools (the legacy counting pool included, for
                               one-stop provenance)

Frozen-seed discipline: each pool's seed is a fixed literal in this script;
regenerating must reproduce byte-identical files (newline="\\n", no timestamps
in the items). Zero overlap between any training pool and any held-out pool,
enforced by prompt-text exclusion AND asserted at the end.

  python tools/make_pools_fsa.py           # writes the three pools + manifest
  python tools/make_pools_fsa.py --check   # verify existing files instead
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gen_counting as gen  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"

# Frozen generation parameters. Changing ANY of these invalidates the pools;
# the manifest records them so the change is visible, not silent.
SEED_HELDOUT_COUNTING = 20261101
SEED_MOD_TRAIN = 20261102
SEED_HELDOUT_MOD = 20261103
N_HELDOUT = 512
N_MOD_TRAIN = 2000
DRAW_MARGIN = 768  # draw this many candidates before cutting to N_HELDOUT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_pool_texts(path: Path) -> set[str]:
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.add(json.loads(line)["prompt"])
    return out


def _cut_to(items: list, n: int) -> list:
    assert len(items) >= n, f"candidate pool {len(items)} < {n}; raise DRAW_MARGIN"
    return items[:n]


def _sanity(items, name: str) -> None:
    for it in items:
        assert isinstance(it.answer, int), (name, it.prompt)
        assert "\\n" not in it.prompt.replace("\\n", "\n"), it.prompt
    assert len({it.prompt for it in items}) == len(items), f"{name}: duplicate prompts"


def build() -> None:
    legacy_texts = _load_pool_texts(DATA / "prompts_mixed2000.jsonl")

    # 1. held-out counting test set: same generator + tier mix as the training
    #    pool, different seed, prompt-text-disjoint from the training pool.
    cands = gen.build_dataset("mixed", DRAW_MARGIN, seed=SEED_HELDOUT_COUNTING)
    fresh = [it for it in cands if it.prompt not in legacy_texts]
    heldout_c = _cut_to(fresh, N_HELDOUT)
    _sanity(heldout_c, "heldout_counting")

    # 2. mod training pool
    mod_train = gen.build_mod_dataset("mixed", N_MOD_TRAIN, seed=SEED_MOD_TRAIN)
    _sanity(mod_train, "mod_train")

    # 3. mod held-out test set, disjoint from BOTH training pools
    exclude = legacy_texts | {it.prompt for it in mod_train}
    cands = gen.build_mod_dataset(
        "mixed", DRAW_MARGIN, seed=SEED_HELDOUT_MOD, exclude_prompts=exclude
    )
    heldout_m = _cut_to(cands, N_HELDOUT)
    _sanity(heldout_m, "heldout_mod")

    files = {
        "heldout_counting_512.jsonl": heldout_c,
        "prompts_mod2000.jsonl": mod_train,
        "heldout_mod_512.jsonl": heldout_m,
    }
    for name, items in files.items():
        gen.save_jsonl(items, str(DATA / name))
        print(f"wrote {name}: {len(items)} items, "
              f"depth hist {_depth_hist(items)}")

    manifest = {
        "generated": "2026-10-31",
        "generator": "src/gen_counting.py @ fsa-resample branch (see COMMIT file)",
        "note": "held-out test sets are evaluation-only; training pools are "
                "training-only. Disjointness is by prompt text, asserted below.",
        "pools": {
            "data/prompts_mixed2000.jsonl": {
                "role": "train/counting",
                "seed": "legacy (S1 rebuild pool, sha256 d781f955...)",
                "n_items": len(legacy_texts),
            },
            "data/heldout_counting_512.jsonl": {
                "role": "heldout/counting",
                "seed": SEED_HELDOUT_COUNTING,
                "n_items": len(heldout_c),
                "draw_margin": DRAW_MARGIN,
                "tier_mix": "uniform easy/medium/hard, ranges 30/60/100",
            },
            "data/prompts_mod2000.jsonl": {
                "role": "train/mod",
                "seed": SEED_MOD_TRAIN,
                "n_items": len(mod_train),
                "ops": "filters=mod3/5/7 residues>=1; transforms include mod_9/mod_10; "
                       "aggregations include sum_mod_9/sum_mod_10; every item has >=1 mod op",
            },
            "data/heldout_mod_512.jsonl": {
                "role": "heldout/mod",
                "seed": SEED_HELDOUT_MOD,
                "n_items": len(heldout_m),
                "draw_margin": DRAW_MARGIN,
                "disjoint_from": ["data/prompts_mixed2000.jsonl", "data/prompts_mod2000.jsonl"],
            },
        },
        "sha256": {},
        "disjointness_assertions": {},
    }
    all_files = [
        "prompts_mixed2000.jsonl",
        "heldout_counting_512.jsonl",
        "prompts_mod2000.jsonl",
        "heldout_mod_512.jsonl",
    ]
    for name in all_files:
        manifest["sha256"][f"data/{name}"] = sha256_file(DATA / name)

    texts = {n: _load_pool_texts(DATA / n) for n in all_files}
    pairs = [
        ("prompts_mixed2000.jsonl", "heldout_counting_512.jsonl"),
        ("prompts_mixed2000.jsonl", "heldout_mod_512.jsonl"),
        ("prompts_mod2000.jsonl", "heldout_mod_512.jsonl"),
        ("prompts_mod2000.jsonl", "heldout_counting_512.jsonl"),
        ("heldout_counting_512.jsonl", "heldout_mod_512.jsonl"),
    ]
    for a, b in pairs:
        inter = texts[a] & texts[b]
        manifest["disjointness_assertions"][f"{a} x {b}"] = len(inter)
        assert not inter, f"OVERLAP between {a} and {b}: {len(inter)} prompts"

    with open(DATA / "MANIFEST_fsa.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("wrote MANIFEST_fsa.json; all disjointness assertions passed")


def _depth_hist(items) -> dict:
    h: dict[int, int] = {}
    for it in items:
        h[it.depth] = h.get(it.depth, 0) + 1
    return dict(sorted(h.items()))


def check() -> int:
    ok = True
    for name in [
        "heldout_counting_512.jsonl",
        "prompts_mod2000.jsonl",
        "heldout_mod_512.jsonl",
        "MANIFEST_fsa.json",
    ]:
        p = DATA / name
        if not p.exists():
            print(f"MISSING {name}")
            ok = False
    if not ok:
        return 1
    manifest = json.loads((DATA / "MANIFEST_fsa.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        got = sha256_file(DATA / Path(name).name)
        status = "ok" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"{status}  {name}  {got[:16]}")
    for a, b in [
        ("prompts_mixed2000.jsonl", "heldout_counting_512.jsonl"),
        ("prompts_mixed2000.jsonl", "heldout_mod_512.jsonl"),
        ("prompts_mod2000.jsonl", "heldout_mod_512.jsonl"),
    ]:
        inter = _load_pool_texts(DATA / a) & _load_pool_texts(DATA / b)
        if inter:
            print(f"OVERLAP {a} x {b}: {len(inter)}")
            ok = False
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    raise SystemExit(check() if a.check else build() or 0)
