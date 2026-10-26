"""Derived statistics for REPORT.md -- every number the report quotes comes from here.

`make_table.py` answers "how did each arm do". This script answers the questions
the report asks *around* that table, and prints them as markdown so no figure in
the prose is hand-copied:

  - per-arm aggregation across seeds (mean and min-max range, never a CI: n=3)
  - the d0 decomposition (||g_bar||, sigma, sigma/||g_bar||) that C4 turns on
  - C2's monotonicity diagnostics (rises, negative-KL rows, monotonish count)
  - C6's two shares and their ratio
  - the drift-clock crossing window (why C3 is censored and not merely "large")
  - cost and wall clock in both currencies the accounting keeps

Deliberately independent of make_table.py: it re-reads metrics.jsonl itself, so
the two agreeing is a check rather than a tautology.

    python analysis/report_stats.py --runs runs --out analysis/out/report_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import metrics as metrics_mod  # noqa: E402


def by_rollout(rows: list[dict], field: str, step: int | None = None) -> dict[int, float]:
    """Mean of `field` per rollout_idx (a rollout-level field repeats across its mu rows)."""
    bucket: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if step is not None and int(r["mu_step"]) != step:
            continue
        v = r.get(field)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        bucket[int(r["rollout_idx"])].append(float(v))
    return {k: sum(v) / len(v) for k, v in sorted(bucket.items())}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def rng(xs) -> tuple[float, float]:
    xs = list(xs)
    return (min(xs), max(xs)) if xs else (float("nan"), float("nan"))


def c2_diagnostics(rows: list[dict]) -> dict:
    """Is KL(policy||rollout) monotone inside each rollout? Count it, don't eyeball it."""
    per_rollout: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        per_rollout[int(r["rollout_idx"])].append((int(r["mu_step"]), float(r["kl_per_rollout_policy"])))
    rises = pairs = monotone_rollouts = neg_rows = 0
    firsts, lasts = [], []
    for _, seq in sorted(per_rollout.items()):
        seq.sort()
        vals = [v for _, v in seq]
        neg_rows += sum(1 for v in vals if v < 0)
        r_ = sum(1 for a, b in zip(vals, vals[1:]) if b > a)
        p_ = max(0, len(vals) - 1)
        rises += r_
        pairs += p_
        if p_ and r_ / p_ >= 0.8 and vals[-1] > vals[0]:
            monotone_rollouts += 1
        if vals:
            firsts.append(vals[1] if len(vals) > 1 else vals[0])
            lasts.append(vals[-1])
    return {
        "rollouts": len(per_rollout),
        "rise_frac": round(rises / pairs, 4) if pairs else None,
        "rises": rises,
        "pairs": pairs,
        "monotonish_rollouts": monotone_rollouts,
        "neg_kl_rows": neg_rows,
        "kl_step1_mean": mean(firsts),
        "kl_last_mean": mean(lasts),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--pattern", default="mu*_s*")
    ap.add_argument("--out", default="analysis/out/report_stats.json")
    a = ap.parse_args()

    root = Path(a.runs)
    runs: dict[tuple[str, int], dict] = {}
    for d in sorted(root.glob(a.pattern)):
        mp = d / "metrics.jsonl"
        if not mp.is_file():
            continue
        rows = metrics_mod.read_metrics(mp)
        if not rows:
            continue
        tag = f"mu{int(rows[-1]['mu'])}"
        seed = int(rows[-1]["seed"])
        runs[(tag, seed)] = {"rows": rows, "dir": d}

    per_run = []
    for (arm, seed), rec in sorted(runs.items(), key=lambda kv: (int(kv[0][0][2:]), kv[0][1])):
        rows = rec["rows"]
        mu = int(rows[-1]["mu"])
        rollouts = max(int(r["rollout_idx"]) for r in rows) + 1
        half_from = rollouts - max(1, rollouts // 2)
        pass_last = [v for i, v in by_rollout(rows, "pass_rate").items() if i >= half_from]
        d_series = list(by_rollout(rows, "drift_d").values())
        d0 = d_series[0]
        ratio_path = [v / d0 for v in d_series] if d0 else []
        r0_sigma = by_rollout([r for r in rows if int(r["rollout_idx"]) == 0], "grad_noise_sigma")[0]
        r0_gn = by_rollout([r for r in rows if int(r["rollout_idx"]) == 0], "grad_norm_raw")[0]
        summ = rec["dir"] / "summary.json"
        tot = json.loads(summ.read_text(encoding="utf-8"))["totals"] if summ.is_file() else {}
        per_run.append({
            "arm": arm, "seed": seed, "mu": mu,
            "b": int(rows[-1]["b_responses"]),
            "rollouts": rollouts, "rows": len(rows),
            "pass_last": mean(pass_last),
            "kl_last": mean(by_rollout(rows, "kl_per_rollout_policy", mu - 1).values()),
            "clip_any": mean(by_rollout(rows, "clip_frac_any").values()),
            "clip_mass": mean(by_rollout(rows, "clip_grad_mass_share").values()),
            "d0": d0, "d_last": d_series[-1], "d_mean": mean(d_series),
            "d_path_min": min(ratio_path), "d_path_max": max(ratio_path),
            "sigma_r0": r0_sigma, "gn_r0": r0_gn,
            "resp_len": mean(by_rollout(rows, "resp_len_mean").values()),
            "zero_var": mean(by_rollout(rows, "zero_var_group_frac").values()),
            "cost": tot.get("cost_cny"), "wall_h": tot.get("wall_hours_measured"),
            "responses": tot.get("responses", 0),
            "flops": tot.get("flops_total"),
            "c2": c2_diagnostics(rows),
        })

    arms = {}
    for arm in sorted({r["arm"] for r in per_run}, key=lambda s: int(s[2:])):
        rs = [r for r in per_run if r["arm"] == arm]
        arms[arm] = {
            "mu": rs[0]["mu"], "b": rs[0]["b"], "seeds": sorted(r["seed"] for r in rs),
            **{
                k: {"mean": mean(r[k] for r in rs), "min": rng(r[k] for r in rs)[0], "max": rng(r[k] for r in rs)[1]}
                for k in ("pass_last", "kl_last", "clip_any", "clip_mass", "d0", "d_last",
                          "sigma_r0", "gn_r0", "resp_len", "zero_var", "cost", "wall_h")
            },
        }

    base = {r["seed"]: r for r in per_run if r["arm"] == "mu1"}
    paired = {}
    for arm in ("mu8", "mu16"):
        if arm not in {r["arm"] for r in per_run}:
            continue
        rows_ = [r for r in per_run if r["arm"] == arm]
        paired[arm] = {
            "pass_minus_mu1": {r["seed"]: r["pass_last"] - base[r["seed"]]["pass_last"] for r in rows_ if r["seed"] in base},
            "d0_ratio_to_mu1": {r["seed"]: r["d0"] / base[r["seed"]]["d0"] for r in rows_ if r["seed"] in base},
            "d0_ratio_arm_mean": mean(r["d0"] for r in rows_) / mean(base[s]["d0"] for s in base),
        }

    total = {
        "runs": len(per_run),
        "responses": sum(r["responses"] for r in per_run),
        "wall_h": sum(r["wall_h"] or 0 for r in per_run),
        "cost": sum(r["cost"] or 0 for r in per_run),
        "flops": sum(r["flops"] or 0 for r in per_run),
    }

    out = {"per_run": per_run, "arms": arms, "paired": paired, "total": total}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    # ---- markdown, so the report can be assembled without retyping numbers ----
    print("### A. per arm (mean over 3 seeds, range in brackets)\n")
    print("| arm | mu | b | pass(last 4) | KL@last step | clip token frac | d0 | d_last | ||g||@r0 | sigma@r0 | resp_len | zero-var | GPU-h | CNY |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, s in arms.items():
        f = lambda k, n=4: f"{s[k]['mean']:.{n}f} [{s[k]['min']:.{n}f}-{s[k]['max']:.{n}f}]"
        print(f"| {arm} | {s['mu']} | {s['b']} | {f('pass_last')} | {s['kl_last']['mean']:.2e} | "
              f"{s['clip_any']['mean']:.5f} | {f('d0')} | {s['d_last']['mean']:.4f} | "
              f"{s['gn_r0']['mean']:.5f} | {s['sigma_r0']['mean']:.5f} | {s['resp_len']['mean']:.1f} | "
              f"{s['zero_var']['mean']:.4f} | {s['wall_h']['mean']:.3f} | {s['cost']['mean']:.3f} |")

    print("\n### B. paired (same seed)\n")
    print("| seed | mu8-mu1 pass | mu16-mu1 pass | mu8 d0 ratio | mu16 d0 ratio |")
    print("|---|---|---|---|---|")
    seeds = sorted({s for v in paired.values() for s in v["pass_minus_mu1"]})
    for s in seeds:
        print(f"| {s} | {paired['mu8']['pass_minus_mu1'][s]:+.4f} | {paired['mu16']['pass_minus_mu1'][s]:+.4f} | "
              f"{paired['mu8']['d0_ratio_to_mu1'][s]:.3f} | {paired['mu16']['d0_ratio_to_mu1'][s]:.3f} |")
    print(f"\narm-mean d0 ratio mu8/mu1={paired['mu8']['d0_ratio_arm_mean']:.3f}  "
          f"mu16/mu1={paired['mu16']['d0_ratio_arm_mean']:.3f}")

    print("\n### C. C2 monotonicity\n")
    print("| run | rise frac | rises/pairs | monotonish | neg-KL rows | KL step1 | KL last |")
    print("|---|---|---|---|---|---|---|")
    for r in per_run:
        if r["mu"] < 2:
            continue
        c = r["c2"]
        print(f"| {r['arm']}_s{r['seed']} | {c['rise_frac']:.2%} | {c['rises']}/{c['pairs']} | "
              f"{c['monotonish_rollouts']}/{c['rollouts']} | {c['neg_kl_rows']} | "
              f"{c['kl_step1_mean']:+.2e} | {c['kl_last_mean']:+.2e} |")

    print("\n### D. C6 shares\n")
    print("| run | clip token frac | clip grad mass share | share ratio |")
    print("|---|---|---|---|")
    for r in per_run:
        if r["mu"] < 2:
            continue
        ratio = r["clip_mass"] / r["clip_any"] if r["clip_any"] else float("nan")
        print(f"| {r['arm']}_s{r['seed']} | {r['clip_any']:.5f} | {r['clip_mass']:.5f} | {ratio:.2f}x |")

    print("\n### E. drift path (d/d0 within one run)\n")
    for r in per_run:
        print(f"  {r['arm']}_s{r['seed']}: d/d0 in [{r['d_path_min']:.3f}, {r['d_path_max']:.3f}]  "
              f"d0={r['d0']:.4f} d_last={r['d_last']:.4f}")

    print(f"\n### F. totals\n  runs={total['runs']}  GPU-h={total['wall_h']:.3f}  "
          f"CNY={total['cost']:.3f}  FLOPs={total['flops']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
