"""Procedural counting task generator.

Rebuilt from the specification in Bauer et al. (2026), arXiv:2604.18381
"Learning from Less: Measuring the Effectiveness of RLVR in Low Data and
Compute Regimes" (Section: Counting Problems).

Task shape (per paper):
  - A prompt defines an integer range, a chain of conditional filters
    (1-4) and optional transformations (0-3), then asks for a count or
    an arithmetic aggregate.
  - The ground truth is computed programmatically by executing the same
    chain, so the verifier is exact.

Difficulty definition (per paper): difficulty is NOT a parameter of the
generator but an empirical property - the solve rate of 10 base models.
  Easy: 67-100% of models correct, Medium: 34-66%, Hard: 0-33%.
We implement `depth` as a generation knob and provide a calibration
script (calibrate_difficulty.py) that buckets generated items into the
three tiers by measured solve rate, reproducing the paper's methodology.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from typing import Callable

# ---------------------------------------------------------------------------
# Operator taxonomy (subset faithful to the paper's counting family)
# ---------------------------------------------------------------------------

# Conditional filters: keep only numbers satisfying a predicate.
FILTERS: dict[str, tuple[str, Callable[[int], bool]]] = {
    "even": ("keep only the numbers that are even", lambda x: x % 2 == 0),
    "odd": ("keep only the numbers that are odd", lambda x: x % 2 != 0),
    "positive": ("keep only the numbers that are positive", lambda x: x > 0),
    "negative": ("keep only the numbers that are negative", lambda x: x < 0),
    "divisible_by_3": ("keep only the numbers that are divisible by 3", lambda x: x % 3 == 0),
    "divisible_by_5": ("keep only the numbers that are divisible by 5", lambda x: x % 5 == 0),
    "divisible_by_7": ("keep only the numbers that are divisible by 7", lambda x: x % 7 == 0),
}

# Unary transformations applied to the surviving numbers before aggregation.
TRANSFORMS: dict[str, tuple[str, Callable[[int], int]]] = {
    "add_1": ("add 1 to each remaining number", lambda x: x + 1),
    "double": ("double each remaining number", lambda x: 2 * x),
    "square": ("square each remaining number", lambda x: x * x),
    "negate": ("negate each remaining number", lambda x: -x),
}

# Final aggregation over the resulting multiset.
AGGREGATIONS: dict[str, tuple[str, Callable[[list[int]], float | int]]] = {
    "count": ("count how many values remain", lambda xs: len(xs)),
    "sum": ("compute the sum of the remaining values", lambda xs: sum(xs)),
    "max": ("report the largest remaining value", lambda xs: max(xs)),
    "min": ("report the smallest remaining value", lambda xs: min(xs)),
    "range": ("report the difference between the largest and smallest remaining values", lambda xs: max(xs) - min(xs)),
}


def _unique_count(xs: list[int]) -> int:
    return len(set(xs))


AGGREGATIONS["unique_count"] = ("count how many distinct values remain", _unique_count)


# ---------------------------------------------------------------------------
# Item dataclass
# ---------------------------------------------------------------------------


@dataclass
class CountingItem:
    prompt: str
    answer: int
    # provenance (needed later for difficulty calibration and analysis)
    lo: int
    hi: int
    filters: list[str]
    transforms: list[str]
    aggregation: str
    depth: int  # 1-7 in the paper: n_filters + n_transforms + 1 aggregation step

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _make_prompt(lo: int, hi: int, filters: list[str], transforms: list[str], agg: str) -> str:
    parts = [f"Consider the integers from {lo} to {hi}, inclusive."]
    for f in filters:
        parts.append("First, " + FILTERS[f][0] + "." if f == filters[0] else "Then, " + FILTERS[f][0] + ".")
    for t in transforms:
        parts.append("Then, " + TRANSFORMS[t][0] + ".")
    parts.append("Of these numbers, " + AGGREGATIONS[agg][0] + ".")
    parts.append("Think step by step, then give your final answer on a new line in the exact format 'Answer: <number>'.")
    return " ".join(parts)


def _execute(lo: int, hi: int, filters: list[str], transforms: list[str], agg: str) -> int:
    xs = list(range(lo, hi + 1))
    for f in filters:
        pred = FILTERS[f][1]
        xs = [x for x in xs if pred(x)]
    for t in transforms:
        fn = TRANSFORMS[t][1]
        xs = [fn(x) for x in xs]
    if not xs:
        # aggregations on an empty set: count/unique_count -> 0; others undefined
        if agg in ("count", "unique_count"):
            return 0
        return 0  # generator avoids this combination for non-count aggregations
    value = AGGREGATIONS[agg][1](xs)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return int(value)


def generate_item(
    rng: random.Random,
    depth: int | None = None,
    max_range_scale: int = 100,
) -> CountingItem:
    """Generate one counting item.

    depth: compositional depth, 1-7 per the paper (n_filters + n_transforms + 1).
           If None, sampled uniformly from 1-4 (light tail).
    max_range_scale: upper bound for the interval width; the paper's example
           uses [1, 100]. Larger scales raise arithmetic burden.
    """
    if depth is None:
        depth = rng.randint(1, 4)
    if not 1 <= depth <= 7:
        raise ValueError(f"depth must be in 1-7, got {depth}")

    n_agg = 1
    n_filters = rng.randint(0, min(4, depth - n_agg))
    n_transforms = min(3, depth - n_agg - n_filters)

    # interval: width bounded by max_range_scale; allow negative ranges
    width = rng.randint(10, max_range_scale)
    lo = rng.randint(-max(20, width // 2), 100)
    hi = lo + width - 1

    filters = rng.sample(list(FILTERS), n_filters) if n_filters else []
    # avoid degenerate chains: negative-only after negation etc. is fine -
    # ground truth is computed, so any chain is well-posed.
    transforms = rng.sample(list(TRANSFORMS), n_transforms) if n_transforms else []

    # aggregations that need a non-empty survivor set
    safe_aggs = list(AGGREGATIONS)
    if n_filters >= 3 and width <= 30:
        # high risk of empty survivor set; prefer counting aggregations
        safe_aggs = ["count", "unique_count"]
    agg = rng.choice(safe_aggs)

    answer = _execute(lo, hi, filters, transforms, agg)
    prompt = _make_prompt(lo, hi, filters, transforms, agg)
    return CountingItem(
        prompt=prompt,
        answer=answer,
        lo=lo,
        hi=hi,
        filters=filters,
        transforms=transforms,
        aggregation=agg,
        depth=1 + n_filters + n_transforms,
    )


# ---------------------------------------------------------------------------
# Dataset builders (paper: Easy-100/200/500, Mixed-100/200/500, test-200)
# ---------------------------------------------------------------------------


def build_dataset(kind: str, size: int, seed: int = 0, max_range_scale: int = 100) -> list[CountingItem]:
    """kind: 'easy' | 'mixed'. Tier definitions driven by the 2026-10-21
    calibration run (data/calibration.json, Qwen3-4B-Base greedy on 500 items):
    depth 1 solved at 0.62 with ranges up to ~110 -> nothing reached the paper's
    easy band (>=0.67). Fix: narrow ranges per tier, keeping depth as the main
    knob (solve rate decreases monotonically-ish with depth).
      easy:   depth 1-2, range <= 30
      medium: depth 3-4, range <= 60
      hard:   depth 5-7, range <= 100
      mixed:  uniform across the three tiers (paper: ~1/3 each)
    Re-verify with a spot check after regeneration.
    """
    rng = random.Random(seed)
    items: list[CountingItem] = []
    tier_range = {"easy": 30, "medium": 60, "hard": 100}
    if kind == "easy":
        depths = [("easy", rng.randint(1, 2)) for _ in range(size)]
    elif kind == "mixed":
        third = size // 3
        depths = (
            [("easy", rng.randint(1, 2)) for _ in range(third)]
            + [("medium", rng.randint(3, 4)) for _ in range(third)]
            + [("hard", rng.randint(5, 7)) for _ in range(size - 2 * third)]
        )
        rng.shuffle(depths)
    else:
        raise ValueError(f"unknown kind: {kind}")
    for tier, d in depths:
        items.append(generate_item(rng, depth=d, max_range_scale=tier_range[tier]))
    return items


def save_jsonl(items: list[CountingItem], path: str) -> None:
    # newline="\n" pins the byte layout on every OS: without it Windows text mode
    # turns \n into \r\n, and the sha256 recorded in data/MANIFEST.json stops
    # matching as soon as the file is checked out on Linux (see DEVLOG D-09).
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(it.to_json() + "\n")


def load_jsonl(path: str) -> list[CountingItem]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(CountingItem(**d))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["easy", "mixed"], required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = build_dataset(a.kind, a.size, seed=a.seed)
    save_jsonl(items, a.out)
    print(f"wrote {len(items)} items to {a.out}")
    for it in items[:3]:
        print("---", it.depth, it.answer)
        print(it.prompt)
