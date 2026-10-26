"""Read-only review of the nine archived runs: what the numbers can and cannot say.

Why a separate script from `report_stats.py`: that one answers "did each
pre-registered claim pass", this one answers "what does the data actually
constrain" -- noise floors, schedule confounds, token accounting, cost per
unit of progress. No claim in `analysis/claims.yaml` is read here on purpose,
so nothing in this file can be tuned to a verdict.

    python analysis/data_review.py [--runs runs] [--json analysis/out/data_review.json]

stdlib only, no torch, no GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys


ARMS = ("mu1", "mu8", "mu16")


# --- the LR schedule is re-implemented here, deliberately --------------------
# It must match src/train.py:_lr_at(). If it drifts, every "slope vs lr" number
# below is wrong in a way no test would catch, so the formula is copied with the
# source line instead of imported (importing trainer pulls in torch).
def lr_at(step: int, total: int, base: float = 5e-5, warmup_ratio: float = 0.1) -> float:
    """src/train.py:266-273, verbatim."""
    warm = max(1, int(total * warmup_ratio))
    if step < warm:
        return base * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def load(runs_dir: str) -> dict:
    """arm -> seed -> list of metric rows (one per mu-step, in file order)."""
    out = {a: {} for a in ARMS}
    for d in sorted(glob.glob(os.path.join(runs_dir, "mu*_s*"))):
        name = os.path.basename(d)
        arm, seed_s = name.split("_")
        if arm not in out:
            continue
        p = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
        out[arm][int(seed_s[1:])] = rows
    return out


def by_rollout(rows: list[dict]) -> dict:
    """rollout_idx -> list of rows (len == mu for a finished run)."""
    g: dict[int, list[dict]] = {}
    for r in rows:
        g.setdefault(int(r["rollout_idx"]), []).append(r)
    return {k: g[k] for k in sorted(g)}


def series(data: dict, field: str, agg="mean") -> dict:
    """arm -> [per-rollout value, averaged over seeds]."""
    out = {}
    for arm in ARMS:
        per = {}
        for seed, rows in data[arm].items():
            g = by_rollout(rows)
            # reward/length are per-rollout constants repeated on every mu-step;
            # pass_rate and friends are per-mu-step, so mean over the rollout.
            per[seed] = [st.mean(float(x[field]) for x in g[k]) for k in sorted(g)]
        n = min(len(v) for v in per.values())
        out[arm] = [st.mean(per[s][i] for s in per) for i in range(n)] if per else []
        out[arm + "_per_seed"] = {s: per[s] for s in per}
    return out


def slope(vals: list[float]) -> float:
    """Least-squares slope per rollout."""
    n = len(vals)
    if n < 2:
        return float("nan")
    mx = (n - 1) / 2
    my = sum(vals) / n
    num = sum((i - mx) * (y - my) for i, y in enumerate(vals))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--json", default="analysis/out/data_review.json")
    a = ap.parse_args(argv)

    data = load(a.runs)
    n_runs = sum(len(v) for v in data.values())
    n_rows = sum(len(r) for arms in data.values() for r in arms.values())
    print(f"loaded {n_runs} runs, {n_rows} metric rows")

    out: dict = {"runs": n_runs, "rows": n_rows}

    # ---------------------------------------------------------------- 1. curves
    section("1. per-rollout curves (seed mean; range in brackets)")
    fields = ["pass_rate", "reward_mean", "resp_len_mean", "completion_tokens",
              "grad_noise_sigma", "drift_d", "kl_per_rollout_policy",
              "clip_frac_any", "zero_var_group_frac", "lr", "cost_cny_cum",
              "reward_mean_minus_pass"]
    # format bonus = reward - pass, both are rollout means
    for arm in ARMS:
        for seed, rows in data[arm].items():
            g = by_rollout(rows)
            for k in g:
                for x in g[k]:
                    x["reward_mean_minus_pass"] = x["reward_mean"] - x["pass_rate"]

    curves = {}
    for f in fields:
        curves[f] = series(data, f)
    out["curves"] = {f: {arm: curves[f][arm] for arm in ARMS} for f in fields}

    hdr = f"{'arm':5s} " + " ".join(f"{'r'+str(i):>7s}" for i in range(8))
    for f in ("pass_rate", "reward_mean_minus_pass", "resp_len_mean",
              "grad_noise_sigma", "drift_d", "zero_var_group_frac", "lr"):
        print(f"\n-- {f}")
        print(hdr)
        for arm in ARMS:
            v = curves[f][arm]
            print(f"{arm:5s} " + " ".join((f"{x:7.4f}" if abs(x) < 1 else f"{x:7.2e}") for x in v))

    # ------------------------------------------------------------ 2. lr confound
    section("2. schedule confound: slope of pass vs learning rate")
    sched = [lr_at(i, 8) for i in range(8)]
    print("H=8 lr: " + " ".join(f"{x:.3g}" for x in sched))
    print(f"{'arm':5s} {'slope r0-r4':>12s} {'slope r4-r7':>12s} {'ratio':>7s} "
          f"{'mean lr r0-r4':>13s} {'mean lr r4-r7':>13s} {'slope/lr early':>14s} {'slope/lr late':>13s}")
    sched_out = {}
    for arm in ARMS:
        p = curves["pass_rate"][arm]
        s_early = slope(p[0:5])          # r0..r4, lr still 3e-5..5e-5
        s_late = slope(p[4:8])           # r4..r7, lr falls 12x
        lr_early = st.mean(sched[0:5])
        lr_late = st.mean(sched[4:8])
        print(f"{arm:5s} {s_early:12.5f} {s_late:12.5f} {s_early/max(s_late,1e-9):7.2f} "
              f"{lr_early:13.3g} {lr_late:13.3g} "
              f"{s_early/lr_early*1e6:14.3f} {s_late/lr_late*1e6:13.3f}")
        sched_out[arm] = {"slope_r0_r4": s_early, "slope_r4_r7": s_late,
                          "lr_r0_r4": lr_early, "lr_r4_r7": lr_late,
                          "slope_per_lr_early": s_early / lr_early,
                          "slope_per_lr_late": s_late / lr_late}
    out["schedule"] = sched_out

    # ------------------------------------------------------------- 3. noise floor
    section("3. noise floor: is one rollout enough to place a crossing?")
    print(f"{'arm':5s} {'p@r7':>7s} {'binomial SE':>12s} {'seed spread':>12s} "
          f"{'SE(mean of 3)':>14s}")
    noise = {}
    for arm in ARMS:
        p7 = [curves["pass_rate"][arm + "_per_seed"][s][-1] for s in data[arm]]
        pbar = st.mean(p7)
        se_bin = math.sqrt(pbar * (1 - pbar) / 4096)
        spread = max(p7) - min(p7)
        noise[arm] = {"p_r7": pbar, "se_binomial": se_bin, "seed_spread": spread,
                      "se_of_mean": se_bin / math.sqrt(len(p7)),
                      "per_seed": p7}
        print(f"{arm:5s} {pbar:7.4f} {se_bin:12.5f} {spread:12.5f} "
              f"{se_bin/math.sqrt(len(p7)):14.5f}")
    print("\npairwise gap at r7 in units of SE(mean of 3):")
    out["noise"] = noise
    for i, x in enumerate(ARMS):
        for y in ARMS[i + 1:]:
            d = noise[y]["p_r7"] - noise[x]["p_r7"]
            se = math.hypot(noise[x]["se_of_mean"], noise[y]["se_of_mean"])
            print(f"  {y} - {x} = {d:+.4f}  ({d/se:.1f} SE)")

    # ------------------------------------------------------------- 4. crossing
    section("4. crossing points, with right-censoring made explicit")
    thresholds = (0.28, 0.30, 0.32, 0.35, 0.40)
    cross = {}
    for arm in ARMS:
        p = curves["pass_rate"][arm]
        cross[arm] = {}
        for t in thresholds:
            hit = next((i for i, v in enumerate(p) if v >= t), None)
            cross[arm][t] = hit
        print(f"{arm:5s} " + "  ".join(
            f"{t:.2f}->{'r%d' % cross[arm][t] if cross[arm][t] is not None else 'censored'}"
            for t in thresholds))
    out["crossing"] = {a: {str(k): v for k, v in cross[a].items()} for a in ARMS}

    # per-seed crossing at the two thresholds that matter
    print("\nper-seed crossing (None = censored at 8 rollouts):")
    for arm in ARMS:
        for s in sorted(data[arm]):
            p = curves["pass_rate"][arm + "_per_seed"][s]
            c30 = next((i for i, v in enumerate(p) if v >= 0.30), None)
            c35 = next((i for i, v in enumerate(p) if v >= 0.35), None)
            print(f"  {arm}_s{s}: r7={p[-1]:.3f}  0.30->{c30}  0.35->{c35}")

    # --------------------------------------------------------------- 5. tokens
    section("5. token accounting: is 'same responses' also 'same tokens'?")
    for f in ("completion_tokens", "resp_len_mean"):
        print(f"-- {f}")
        for arm in ARMS:
            v = curves[f][arm]
            print(f"{arm:5s} " + " ".join(f"{x:9.1f}" for x in v))
    tok = {}
    for arm in ARMS:
        tot = sum(curves["completion_tokens"][arm])
        tok[arm] = {"total_completion_tokens": tot,
                    "tokens_per_response_r7": curves["resp_len_mean"][arm][-1],
                    "pass_per_1k_tokens_r7": curves["pass_rate"][arm][-1] / (tot / 1000)}
        print(f"{arm:5s} total completion tokens {tot:12.0f}  "
              f"len@r7 {tok[arm]['tokens_per_response_r7']:7.1f}")
    out["tokens"] = tok

    # ------------------------------------------------------- 6. cost per point
    section("6. cost and FLOPs per unit of progress (accounting meter, not the bill)")
    cost = {}
    for arm in ARMS:
        p = curves["pass_rate"][arm]
        c = curves["cost_cny_cum"][arm][-1]
        gain = p[-1] - p[0]
        cost[arm] = {"cny_cum_r7": c, "pass_gain": gain,
                     "cny_per_pp": c / (gain * 100) if gain else float("inf"),
                     "pass_per_cny": gain / c if c else float("inf")}
        print(f"{arm:5s} cum ¥{c:5.2f}  gain {gain:+.4f}  "
              f"¥{cost[arm]['cny_per_pp']:.3f} per +1pp  "
              f"{cost[arm]['pass_per_cny']:.4f} pass per ¥")
    out["cost"] = cost

    # ----------------------------------------------------------- 7. wall split
    section("7. where the wall time goes (sum over all rows, per arm)")
    wall = {}
    for arm in ARMS:
        roll = upd = old = 0.0
        for rows in data[arm].values():
            for r in rows:
                roll += r["wall_ms_rollout"]
                upd += r["wall_ms_update"]
        wall[arm] = {"rollout_h": roll / 3.6e6, "update_h": upd / 3.6e6,
                     "update_share": upd / (roll + upd)}
        print(f"{arm:5s} rollout {wall[arm]['rollout_h']:5.2f} h  "
              f"update {wall[arm]['update_h']:5.2f} h  "
              f"update share {wall[arm]['update_share']*100:4.1f}%")
    out["wall"] = wall

    # --------------------------------------------------- 8. within-rollout mu
    section("8. within-rollout behaviour: sigma and KL across mu-steps")
    for arm in ARMS:
        g0 = by_rollout(data[arm][0])
        sig = [float(x["grad_noise_sigma"]) for x in g0[sorted(g0)[0]]]
        kl = [float(x["kl_per_rollout_policy"]) for x in g0[sorted(g0)[-1]]]
        print(f"{arm:5s} sigma over mu-steps (s0,r0): " + " ".join(f"{v:.4f}" for v in sig[:16]))
        print(f"{arm:5s} KL over mu-steps (s0,r7)   : " + " ".join(f"{v:.2e}" for v in kl[:16]))

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
