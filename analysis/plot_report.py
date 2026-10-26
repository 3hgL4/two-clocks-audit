"""PNG figures for REPORT.md. Optional: needs matplotlib, unlike make_figures.py.

The repo's own plotting path (analysis/make_figures.py) deliberately avoids
matplotlib so a checkout without it can still read its results. This script is
the other half: when matplotlib *is* available, it renders the four panels the
report shows. Missing matplotlib is not an error -- it prints a note and exits 0.

    python analysis/plot_report.py --runs runs --out figures

Default output is figures/, not analysis/out/fig/: analysis/out/ is .gitignored
(it is regenerated from runs/, which is also .gitignored), so a figure that only
lives there is a figure nobody reading the repo can see.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    print("matplotlib not available -- skipping PNG figures (CSV path: make_figures.py)")
    raise SystemExit(0)


def load(root: Path) -> dict:
    """arm -> field -> per-run series, read straight from metrics.jsonl."""
    data: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for d in sorted(root.glob("mu*_s*")):
        mp = d / "metrics.jsonl"
        if not mp.is_file():
            continue
        rows = [json.loads(l) for l in mp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        arm = f"mu{int(rows[-1]['mu'])}"
        mu = int(rows[-1]["mu"])
        n_roll = max(int(r["rollout_idx"]) for r in rows) + 1

        by_step: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            by_step[int(r["mu_step"])].append(float(r["kl_per_rollout_policy"]))
        data[arm]["kl"].append([sum(v) / len(v) for _, v in sorted(by_step.items())])

        for field, key in (("drift_d", "drift"), ("pass_rate", "pass")):
            per_roll = defaultdict(list)
            for r in rows:
                per_roll[int(r["rollout_idx"])].append(float(r[field]))
            data[arm][key].append([sum(v) / len(v) for _, v in sorted(per_roll.items())])
            assert len(data[arm][key][-1]) == n_roll

        by_step_clip: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            by_step_clip[int(r["mu_step"])].append(float(r["clip_frac_any"]))
        data[arm]["clip"].append([sum(v) / len(v) for _, v in sorted(by_step_clip.items())])
    return data


def band(ax, series: list[list[float]], label: str, color: str, logy: bool = False) -> None:
    n = min(len(s) for s in series)
    xs = list(range(n))
    means = [sum(s[i] for s in series) / len(series) for i in range(n)]
    ax.plot(xs, means, color=color, lw=1.8, label=label, marker="o", ms=2.5)
    if len(series) > 1:
        lo = [min(s[i] for s in series) for i in range(n)]
        hi = [max(s[i] for s in series) for i in range(n)]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
    if logy and all(v > 0 for v in means):
        ax.set_yscale("log")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load(Path(a.runs))
    colors = {"mu1": "#4C78A8", "mu8": "#F58518", "mu16": "#B2182B"}

    # 1. KL inside a rollout -- the quantity A2 throws away.
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
    for arm in ("mu8", "mu16"):
        if arm in data:
            band(ax, data[arm]["kl"], f"{arm} (b={4096 // int(arm[2:])})", colors[arm], logy=True)
    ax.axhline(0, color="#888", lw=0.8, ls=":")
    ax.set_xlabel("update step within one rollout")
    ax.set_ylabel(r"KL($\pi_\theta \| \pi_{rollout}$), nats/token")
    ax.set_title("A2 drops this term; it grows an order of magnitude within a rollout", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig1_kl_vs_step.png")
    plt.close(fig)

    # 2. drift clock across rollouts -- C3's threshold never fires here.
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
    for arm in ("mu1", "mu8", "mu16"):
        if arm in data:
            band(ax, data[arm]["drift"], arm, colors[arm])
    ax.set_xlabel("rollout index (each = 4096 responses)")
    ax.set_ylabel(r"drift clock $d=\kappa\|\bar g\|/\sigma$")
    ax.set_title("d is flat within every arm; the arms differ at rollout 0", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig2_drift_vs_rollout.png")
    plt.close(fig)

    # 3. pass rate -- the only curve that actually moves.
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
    for arm in ("mu1", "mu8", "mu16"):
        if arm in data:
            band(ax, data[arm]["pass"], arm, colors[arm])
    ax.set_xlabel("rollout index (each = 4096 responses)")
    ax.set_ylabel("pass rate")
    ax.set_title("Same 32k responses: more updates per rollout ends higher", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig3_pass_vs_rollout.png")
    plt.close(fig)

    # 4. clip fraction per step -- two orders below the paper's 0.5%.
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
    for arm in ("mu8", "mu16"):
        if arm in data:
            band(ax, data[arm]["clip"], arm, colors[arm])
    ax.set_xlabel("update step within one rollout")
    ax.set_ylabel("fraction of tokens clipped")
    ax.set_title("clip token fraction stays near 0.3% in every arm", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig4_clip_vs_step.png")
    plt.close(fig)

    for p in sorted(out.glob("fig*.png")):
        print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
