"""Aggregate runs/*/metrics.jsonl into the table the report quotes.

Design rules that come from the protocol:

*   Every number is a function of recorded rows -- nothing is retyped.
*   Uncertainty is bootstrapped over *rollouts* (the independent unit), not over
    rows, and the number of resamples is fixed so the output is reproducible.
*   Crossings of a drift threshold are reported with right-censoring: "6/9 runs
    reached d/d0 >= 2 within 8 rollouts" is a fact; "the time constant is 6" is
    not, and the table does not print it.
*   Comparing arms with fewer than 3 seeds prints per-seed differences instead of
    a confidence interval, because a bootstrap over 2 points is theatre.

    python analysis/make_table.py --runs runs --out analysis/out
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import metrics as metrics_mod  # noqa: E402

BOOT = 2000
RNG = np.random.default_rng(12345)


# ---------------------------------------------------------------------------


def load_runs(root: Path, pattern: str = "mu*_s*") -> dict[tuple[str, int], list[dict]]:
    """Read runs/*/metrics.jsonl, keyed by (arm, seed).

    The directory-name filter is not cosmetic: runs/ also collects probes and
    pilots (det_*, ab_*, reg_*, dryrun, hf48g ...), and a pilot carrying
    arm=mu1 seed=0 silently overwrites the real mu1_s0 row because the key comes
    from the row, not from the path. Measured 2026-10-26: the unfiltered table
    replaced mu1/s0 (8 rollouts, 32768 responses) with a 1-rollout probe.
    """
    runs: dict[tuple[str, int], list[dict]] = {}
    for mpath in sorted(root.glob(f"{pattern}/metrics.jsonl")):
        rows = metrics_mod.read_metrics(mpath)
        if not rows:
            continue
        mu = int(rows[-1]["mu"])
        seed = int(rows[-1]["seed"])
        key = (f"mu{mu}", seed)
        if key in runs:
            raise SystemExit(
                f"duplicate {key}: {mpath} and another run share arm+seed. "
                f"Use --pattern to select one family.")
        runs[key] = rows
    return runs


def rollout_means(rows: list[dict], field: str, step: int | None = None) -> list[float]:
    """Mean of `field` per rollout (optionally restricted to one mu_step)."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if step is not None and int(r["mu_step"]) != step:
            continue
        v = r.get(field)
        if v is None:
            continue
        buckets[int(r["rollout_idx"])].append(float(v))
    return [float(np.mean(v)) for _, v in sorted(buckets.items())]


def boot_ci(xs: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    if not xs:
        return (float("nan"),) * 3
    arr = np.asarray(xs, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 3:
        return mean, float("nan"), float("nan")
    idx = RNG.integers(0, len(arr), size=(BOOT, len(arr)))
    means = arr[idx].mean(axis=1)
    return mean, float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def first_crossing(series: list[float], ratio: float) -> tuple[int | None, bool]:
    """(index, reached). index is 1-based in rollouts; None if right-censored."""
    if not series or series[0] <= 0:
        return None, False
    thr = series[0] * ratio
    for i, v in enumerate(series):
        if v >= thr:
            return i, True
    return None, False


def summarise_run(rows: list[dict]) -> dict:
    mu = int(rows[-1]["mu"])
    last_idx = max(int(r["rollout_idx"]) for r in rows)
    # "last half" must be half the ROLLOUTS, not half the rows. For mu>1 the row
    # count is mu times the rollout count, so len(rows)//2 overshoots and keeps
    # every rollout -- which silently turns "pass in the last half" into "pass
    # over the whole run" and halves the measured mu effect (caught 2026-10-26).
    half = max(1, (last_idx + 1) // 2)
    keep_from = last_idx - half + 1

    pass_last = rollout_means([r for r in rows if int(r["rollout_idx"]) >= keep_from], "pass_rate")
    d_series = rollout_means(rows, "drift_d")
    d_series = [v for v in d_series if not math.isnan(v)]
    kls = {st: rollout_means(rows, "kl_per_rollout_policy", st) for st in {int(r["mu_step"]) for r in rows}}
    mu_last = max(kls)
    kl_last = boot_ci(kls[mu_last]) if kls[mu_last] else (float("nan"),) * 3
    clip = rollout_means(rows, "clip_frac_any")

    cx = {r: first_crossing(d_series, r) for r in (1.5, 2.0)}
    out = {
        "arm": f"mu{mu}",
        "seed": int(rows[-1]["seed"]),
        "mu": mu,
        "rollouts": last_idx + 1,
        "b": int(rows[-1]["b_responses"]),
        "br": int(rows[-1]["br_responses"]),
        "pass_last_mean": round(pass_last_mean(pass_last), 4),
        "pass_last_lo": round(boot_ci(pass_last)[1], 4),
        "pass_last_hi": round(boot_ci(pass_last)[2], 4),
        "kl_last_mean": kl_last[0],
        "kl_last_lo": kl_last[1],
        "kl_last_hi": kl_last[2],
        "kl_first": float(np.mean(kls[0])) if kls.get(0) else float("nan"),
        "clip_any_mean": round(float(np.mean(clip)), 5) if clip else float("nan"),
        "d_mean": round(float(np.mean(d_series)), 4) if d_series else float("nan"),
        "d0": round(d_series[0], 4) if d_series else float("nan"),
        "d_last": round(d_series[-1], 4) if d_series else float("nan"),
        "reached_d15": cx[1.5][1],
        "n_d15": cx[1.5][0],
        "reached_d20": cx[2.0][1],
        "n_d20": cx[2.0][0],
        "resp_len_mean": round(float(np.mean(rollout_means(rows, "resp_len_mean"))), 2),
        "zero_var_frac": round(float(np.mean(rollout_means(rows, "zero_var_group_frac"))), 4),
        "cost_cny": float(rows[-1]["cost_cny_cum"]),
        "flops": float(rows[-1]["flops_cum"]),
        "responses": int(rows[-1]["responses_cum"]),
        "n_rows": len(rows),
    }
    return out


def pass_last_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


# ---------------------------------------------------------------------------


def to_markdown(rows: list[dict]) -> str:
    cols = [
        ("arm", "arm"), ("seed", "seed"), ("mu", "mu"), ("b", "b"),
        ("rollouts", "rollouts"), ("pass_last_mean", "pass (last half)"),
        ("pass_last_lo", "boot 2.5%"), ("pass_last_hi", "boot 97.5%"),
        ("kl_last_mean", "KL@last mu"), ("kl_first", "KL@mu_step0"),
        ("clip_any_mean", "clip frac"), ("d0", "d (first)"), ("d_last", "d (last)"),
        ("n_d15", "cross d/d0>=1.5"), ("n_d20", "cross d/d0>=2"),
        ("zero_var_frac", "zero-var groups"), ("cost_cny", "¥"),
    ]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r[k]
            if isinstance(v, float):
                if math.isnan(v):
                    cells.append("--")
                elif abs(v) < 1e-3 and v != 0:
                    cells.append(f"{v:+.2e}")
                else:
                    cells.append(f"{v:.4g}")
            elif v is None:
                cells.append("cens.")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_table(rows: list[dict]) -> str:
    """Per-seed differences against the mu=1 arm, when seeds are shared."""
    by_seed: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["arm"]] = r
    base = "mu1"
    out = ["| seed | " + " | ".join(f"{a} - {base}" for a in ("mu8", "mu16")) + " |",
           "|---|---|---|"]
    for seed, arms in sorted(by_seed.items()):
        if base not in arms:
            continue
        cells = []
        for a in ("mu8", "mu16"):
            if a in arms:
                d = arms[a]["pass_last_mean"] - arms[base]["pass_last_mean"]
                cells.append(f"{d:+.4f}")
            else:
                cells.append("--")
        out.append(f"| {seed} | " + " | ".join(cells) + " |")
    if len(out) == 2:
        return "_no seed is shared by all arms yet_"
    out.append("\n_Differences are per seed, not a CI: with <3 seeds a bootstrap "
               "interval would be misleading._")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--pattern", default="mu*_s*",
                    help="directory glob under --runs; the default keeps the 9 "
                         "arm_seed runs and drops probes/pilots")
    a = ap.parse_args()

    runs = load_runs(Path(a.runs), a.pattern)
    if not runs:
        print(f"no runs found under {a.runs}/*/metrics.jsonl")
        return 1

    rows = [summarise_run(r) for r in runs.values()]
    rows.sort(key=lambda r: (r["mu"], r["seed"]))

    md = ["# Two Clocks audit -- run summary", "",
          f"runs: {len(rows)}  |  boot resamples: {BOOT} (seed 12345)", "",
          to_markdown(rows), "", "## Paired differences", "", paired_table(rows), ""]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    import csv
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
