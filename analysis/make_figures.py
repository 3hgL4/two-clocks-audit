"""Figures for the audit, with a CSV/ASCII fallback when matplotlib is absent.

The local dev box has no matplotlib on purpose: plotting is a cloud-side or
post-hoc job, and a deploy that needs a plotting library to *read* its own
results is a deploy that fails at the worst moment. So every figure is also
emitted as a flat CSV, and a text summary is always printed.

    python analysis/make_figures.py --runs runs --out analysis/out/fig
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import metrics as metrics_mod  # noqa: E402


def collect(rows: list[dict]) -> dict:
    """Group into the three curves the report shows, averaged over seeds."""
    kl: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))   # mu -> step -> vals
    drift: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))  # mu -> rollout -> vals
    acc: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    clip: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        mu = int(r["mu"])
        kl[mu][int(r["mu_step"])].append(float(r["kl_per_rollout_policy"]))
        clip[mu][int(r["mu_step"])].append(float(r["clip_frac_any"]))
        drift[mu][int(r["rollout_idx"])].append(float(r["drift_d"] or 0.0))
        acc[mu][int(r["rollout_idx"])].append(float(r["pass_rate"]))
    return {"kl": kl, "drift": drift, "acc": acc, "clip": clip}


def to_csv(path: Path, series: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["curve", "mu", "x", "mean", "n"])
        for name, by_mu in series.items():
            for mu, xs in sorted(by_mu.items()):
                for x, vals in sorted(xs.items()):
                    w.writerow([name, mu, x, round(float(np.mean(vals)), 8), len(vals)])


def ascii_plot(series: dict) -> str:
    blocks = ["█▉▊▋▌▍▎▏"]
    out = []
    for name, by_mu in sorted(series.items()):
        out.append(f"\n{name}")
        for mu, xs in sorted(by_mu.items()):
            ys = [float(np.mean(v)) for _, v in sorted(xs.items())]
            if not ys:
                continue
            lo, hi = min(ys), max(ys)
            span = (hi - lo) or 1.0
            bars = "".join(blocks[0][min(7, int((y - lo) / span * 7))] for y in ys)
            out.append(f"  mu={mu:<3} [{lo:+.4g} .. {hi:+.4g}]  {bars}")
    return "\n".join(out)


def plot(series: dict, out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    (ax_kl, ax_d), (ax_acc, ax_clip) = axes

    for name, ax, ylabel, xlabel in (
        ("kl", ax_kl, "KL(policy || rollout policy)", "mu_step"),
        ("clip", ax_clip, "clipped token fraction", "mu_step"),
    ):
        for mu, xs in sorted(series[name].items()):
            x = sorted(xs)
            ax.plot(x, [float(np.mean(xs[i])) for i in x], marker="o", label=f"mu={mu}")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=.3)

    for name, ax, ylabel in (("drift", ax_d, "drift clock d"), ("acc", ax_acc, "pass rate")):
        for mu, xs in sorted(series[name].items()):
            x = sorted(xs)
            ax.plot(x, [float(np.mean(xs[i])) for i in x], marker="o", label=f"mu={mu}")
        ax.set_xlabel("rollout"); ax.set_ylabel(ylabel); ax.set_title(ylabel, fontsize=9)
        ax.grid(alpha=.3); ax.legend()

    fig.suptitle("Two Clocks audit -- measured curves (mean over seeds)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=140)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="analysis/out/fig/curves")
    a = ap.parse_args()

    rows: list[dict] = []
    for mpath in sorted(Path(a.runs).glob("*/metrics.jsonl")):
        rows.extend(metrics_mod.read_metrics(mpath))
    if not rows:
        print("no metrics found")
        return 1

    series = collect(rows)
    out = Path(a.out)
    to_csv(out.with_suffix(".csv"), series)
    ok = plot(series, out)
    print(ascii_plot(series))
    print(f"\ncsv  -> {out.with_suffix('.csv')}")
    print(("png  -> " + str(out.with_suffix('.png'))) if ok else
          "matplotlib not installed: png skipped (csv and ascii are complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
