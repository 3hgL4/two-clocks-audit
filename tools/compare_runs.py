"""Print the drift-clock numbers of several runs side by side.

Used for the D-15 ablation (configs/ab_g8m8.yaml vs ab_g4m4.yaml vs ab_g8m4.yaml),
and reusable for the 9-run campaign headline table.

Usage: python tools/compare_runs.py runs/ab_g8m8 runs/ab_g4m4 runs/ab_g8m4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLS = ["n_groups", "n_forward_chunks", "grad_norm_raw", "grad_noise_sigma", "drift_d"]
SHORT = {
    "n_groups": "M",
    "n_forward_chunks": "nchunk",
    "grad_norm_raw": "||g_bar||",
    "grad_noise_sigma": "sigma",
    "drift_d": "d",
}


def load(run: Path) -> list[dict]:
    rows = []
    with open(run / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def fmt(v) -> str:
    if v is None:
        return "  None"
    if isinstance(v, int):
        return f"{v:>6d}"
    return f"{v:8.5f}"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    runs = [Path(a) for a in argv]
    data = {}
    for r in runs:
        try:
            data[r.name] = load(r)
        except FileNotFoundError:
            print(f"[skip] {r} has no metrics.jsonl")
    if not data:
        return 1

    print("step " + " | ".join(f"{k:>26s}" for k in data))
    print("     " + " | ".join(
        " ".join(f"{SHORT[c]:>8s}" for c in COLS) for _ in data))
    n = max(len(v) for v in data.values())
    for i in range(n):
        cells = []
        for k in data:
            rows = data[k]
            if i < len(rows):
                cells.append(" ".join(fmt(rows[i].get(c)) for c in COLS))
            else:
                cells.append(" " * (9 * len(COLS)))
        print(f"{i:4d} " + " | ".join(cells))

    # pairwise ratios on sigma, which is the quantity the ablation is about
    names = list(data)
    print()
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            sa = [r.get("grad_noise_sigma") for r in data[names[a]]]
            sb = [r.get("grad_noise_sigma") for r in data[names[b]]]
            pairs = [(x, y) for x, y in zip(sa, sb) if x and y and x > 0 and y > 0]
            if not pairs:
                continue
            ratios = [y / x for x, y in pairs]
            print(f"sigma({names[b]}) / sigma({names[a]}) = "
                  f"{sum(ratios) / len(ratios):.4f} over {len(pairs)} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
