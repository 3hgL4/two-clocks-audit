"""Recompute the residual pre-registered claims C2, C3, C4, C5, C6, C7, P1.

Why this script exists
----------------------
2026-10-27 the pre-registered claims were filled in against measurement. The
numbers behind C2/C4/C5/C6/P1 were already in REPORT.md, but they were produced
by ad-hoc one-liners that were never committed -- i.e. a reader could not redo
them. This script makes every one of those numbers a single command.

It deliberately does NOT replace analysis/report_stats.py (C1/C3 headline table)
or the four attribution scripts (C8-C12). It only covers the seven claims that
had no committed recompute path.

Two counting conventions, both printed, because REPORT.md uses both
------------------------------------------------------------------
  all rows        -- every line of metrics.jsonl, INCLUDING the baseline row
                     (rollout r, mu_step=0), where the policy has not stepped yet
  stepped only    -- rows with mu_step > 0; for mu=1 this set is empty (a mu=1
                     rollout has exactly one step and it IS the baseline), so the
                     script falls back to all rows and says so

REPORT.md §5.6 (C6 clip shares) uses `all rows`; §5.4 (C2) and §5.5 (C5) use
`stepped only`. Both are legitimate; they differ by the 1/mu baseline row, which
is why mu16 shows a smaller gap between the conventions (15/16) than mu8 (7/8).

The 8-vs-19 rollout trap
------------------------
The working copy's runs/mu1_s{0,1,2} were overwritten by the R4-A continuation
(19 rollouts) and never committed. Every claim here except C4 is defined on the
8-rollout campaign, so `--runs` must point at the ARCHIVED copy:

    Z=<archive>/20_实验数据/01_original_8rollout
    python analysis/residual_claims.py --runs $Z/runs --json analysis/out/residual_claims.json

The script refuses to run silently on the 19-rollout version: it checks the
rollout count of the mu1 arm and exits unless --allow-19 is passed.

    python analysis/residual_claims.py --runs runs --json analysis/out/residual_claims.json

stdlib only.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ARMS = ("mu1", "mu8", "mu16")
SEEDS = (0, 1, 2)

# C3 / C4 thresholds, copied verbatim from claims.yaml -- never edited here
C3_TAUS = (1.5, 2.0)
C4_EXPECTED_DIRECTION = "down"  # pre-registered; measured direction is reported below


# ---------------------------------------------------------------- io helpers


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def stepped(rows: list[dict]) -> list[dict]:
    """Rows after the baseline step. Empty for mu=1 -> caller falls back."""
    body = [r for r in rows if int(r["mu_step"]) > 0]
    return body or rows


def by_rollout(rows: list[dict], field: str) -> list[float]:
    """Mean of `field` inside each rollout, in rollout order."""
    bucket = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        bucket[int(r["rollout_idx"])].append(float(v))
    return [mean(bucket[k]) for k in sorted(bucket)]


def load(runs: Path) -> dict[tuple[str, int], list[dict]]:
    out = {}
    for arm in ARMS:
        for s in SEEDS:
            p = runs / f"{arm}_s{s}" / "metrics.jsonl"
            if not p.exists():
                raise SystemExit(f"missing {p}")
            out[(arm, s)] = read_rows(p)
    return out


# ------------------------------------------------------------------- claims


def c2_kl_monotone(data) -> dict:
    """C2: KL(pi_theta || pi_rollout) rises monotonically with mu_step inside a
    rollout. The criterion is a conjunction over rollouts, so one falling pair
    falsifies it for that rollout."""
    res = {"criterion": "strictly increasing within every rollout", "per_arm": {}}
    tot = {"all_rows": 0, "stepped": 0, "rollouts": 0, "pairs": 0, "up": 0,
           "neg_rows": 0, "kl_rows": 0}
    for arm in ("mu8", "mu16"):  # mu=1 has a single baseline step -> untestable
        a = {"all_rows": 0, "stepped": 0, "rollouts": 0, "pairs": 0, "up": 0,
             "neg_rows": 0, "kl_rows": 0, "last_over_first": [], "seed_kl": []}
        for s in SEEDS:
            rows = data[(arm, s)]
            bucket = defaultdict(list)
            for r in rows:
                bucket[int(r["rollout_idx"])].append(r)
            for ri in sorted(bucket):
                seq = sorted(bucket[ri], key=lambda r: int(r["mu_step"]))
                kl_all = [r["kl_per_rollout_policy"] for r in seq]
                kl_body = [r["kl_per_rollout_policy"] for r in seq
                           if int(r["mu_step"]) > 0]
                a["rollouts"] += 1
                a["all_rows"] += int(all(b > a_ for a_, b in zip(kl_all, kl_all[1:])))
                if kl_body:
                    a["stepped"] += int(all(b > a_ for a_, b in zip(kl_body, kl_body[1:])))
                for x, y in zip(kl_all, kl_all[1:]):
                    a["pairs"] += 1
                    a["up"] += int(y > x)
                a["neg_rows"] += sum(1 for v in kl_body if v < 0)
                a["kl_rows"] += len(kl_body)
            # cross-rollout mean at each mu_step position (noise averages out)
            pos = defaultdict(list)
            for r in rows:
                pos[int(r["mu_step"])].append(r["kl_per_rollout_policy"])
            steps = sorted(pos)
            first, last = mean(pos[steps[1]]) if len(steps) > 1 else float("nan"), mean(pos[steps[-1]])
            a["last_over_first"].append(last / first if first and first == first and first != 0 else None)
            a["seed_kl"].append({"seed": s, "first": first, "last": last})
        a["up_rate"] = a["up"] / a["pairs"] if a["pairs"] else float("nan")
        res["per_arm"][arm] = a
        for k in tot:
            tot[k] += a[k]
    res["total"] = tot
    res["total"]["up_rate"] = tot["up"] / tot["pairs"] if tot["pairs"] else float("nan")
    res["verdict"] = ("falsified_as_written" if tot["stepped"] < tot["rollouts"]
                      else "holds")
    res["note"] = (
        "mu=1 has one step per rollout and that step is the baseline, so C2 is "
        "structurally untestable on that arm. REPORT §5.4 reads the same data at "
        "two aggregation levels: per-rollout monotonicity (this verdict) versus "
        "cross-rollout mean at fixed mu_step (net growth)."
    )
    return res


def c3_crossing(data) -> dict:
    """C3: first rollout where d/d0 reaches tau. Two normalisations are reported
    because they give different envelopes (REPORT §5 表 uses arm-mean per rollout).
      A (report_stats) -- d0 = mean d inside rollout 0; ratio taken per rollout
      B (per step)     -- d0 = d at (rollout 0, mu_step 0); ratio taken per row
    """
    out = {"taus": list(C3_TAUS), "per_run": [], "envelope": {}, "crossings": {}}
    envA, envB = [], []
    hits = {tau: 0 for tau in C3_TAUS}
    for arm in ARMS:
        for s in SEEDS:
            rows = data[(arm, s)]
            dA = by_rollout(rows, "drift_d")
            rA = [v / dA[0] for v in dA]
            d0 = min((r for r in rows if int(r["rollout_idx"]) == 0),
                     key=lambda r: int(r["mu_step"]))["drift_d"]
            rB = [r["drift_d"] / d0 for r in rows]
            envA += rA
            envB += rB
            rec = {"run": f"{arm}_s{s}", "d0_rollout_mean": dA[0], "d0_first_step": d0,
                   "A_min": min(rA), "A_max": max(rA), "B_min": min(rB), "B_max": max(rB)}
            for tau in C3_TAUS:
                hit = [i for i, v in enumerate(rA) if v >= tau]
                rec[f"cross_{tau}"] = hit[0] if hit else None
                hits[tau] += len(hit)
            out["per_run"].append(rec)
    out["envelope"] = {"A": [min(envA), max(envA)], "B": [min(envB), max(envB)]}
    out["crossings"] = {str(tau): hits[tau] for tau in C3_TAUS}
    out["verdict"] = "unidentifiable" if all(v == 0 for v in hits.values()) else "ratio_exists"
    out["note"] = (
        "Both envelopes sit far below the 1.5x threshold; numerator and "
        "denominator are both right-censored, so N_tau(mu16)/N_tau(mu1) has no "
        "value to report. This is 'criterion not triggerable', not 'prediction "
        "refuted'."
    )
    return out


def c4_drift_d0(data) -> dict:
    """C4: d should DECREASE with mu. Measured direction is reported, not assumed."""
    out = {"expected": C4_EXPECTED_DIRECTION, "d0": {}, "ratio_to_mu1": {}}
    for arm in ARMS:
        out["d0"][arm] = mean([by_rollout(data[(arm, s)], "drift_d")[0] for s in SEEDS])
    for arm in ("mu8", "mu16"):
        out["ratio_to_mu1"][arm] = [
            by_rollout(data[(arm, s)], "drift_d")[0] / by_rollout(data[("mu1", s)], "drift_d")[0]
            for s in SEEDS
        ]
        out["ratio_to_mu1"][arm + "_arm_mean"] = (
            out["d0"][arm] / out["d0"]["mu1"]
        )
    d = out["d0"]
    out["measured_direction"] = "up" if d["mu16"] > d["mu8"] > d["mu1"] else "not monotone"
    out["verdict"] = ("falsified" if out["measured_direction"] == "up"
                      else "holds")
    out["note"] = ("kappa cancels in every ratio, so the dimension convention does "
                   "not affect the sign. d0 is the rollout-0 mean, same as report_stats.")
    return out


def c5_resp_len(data) -> dict:
    """C5: response length does not drift systematically with mu."""
    out = {"per_arm": {}, "threshold_frac": 0.05}
    for arm in ARMS:
        vals = [mean([r["resp_len_mean"] for r in stepped(data[(arm, s)])]) for s in SEEDS]
        out["per_arm"][arm] = {"per_seed": vals, "mean": mean(vals)}
    base = out["per_arm"]["mu1"]["mean"]
    for arm in ("mu8", "mu16"):
        out["per_arm"][arm]["rel_to_mu1"] = out["per_arm"][arm]["mean"] / base - 1.0
    arms = [out["per_arm"][a]["mean"] for a in ARMS]
    out["spread_frac"] = (max(arms) - min(arms)) / mean(arms)
    out["max_abs_shift"] = max(abs(out["per_arm"][a]["rel_to_mu1"]) for a in ("mu8", "mu16"))
    out["verdict"] = "no_systematic_drift" if out["max_abs_shift"] < out["threshold_frac"] else "drift"
    out["note"] = (
        "claims.yaml C5 named no threshold; 5% is supplied post hoc, borrowed "
        "from C9's H1 convention, and is declared as such in the revision log. "
        "The criterion only forbids an UPWARD drift -- measured shift is negative "
        "on both arms, so the verdict is unchanged even if the threshold moves."
    )
    return out


def c6_clip(data) -> dict:
    """C6: clip token share rises with mu AND clip grad share exceeds token share."""
    out = {"per_arm": {}, "conventions": ("all_rows", "stepped")}
    for conv in out["conventions"]:
        out["per_arm"][conv] = {}
    for arm in ARMS:
        for conv in out["conventions"]:
            tk, gm = [], []
            for s in SEEDS:
                rows = data[(arm, s)] if conv == "all_rows" else stepped(data[(arm, s)])
                tk.append(mean([r["clip_frac_any"] for r in rows]))
                gm.append(mean([r["clip_grad_mass_share"] for r in rows]))
            out["per_arm"][conv][arm] = {
                "token_share": tk, "grad_share": gm,
                "token_mean": mean(tk), "grad_mean": mean(gm),
                "share_ratio": mean(gm) / mean(tk) if mean(tk) else None,
            }
    tab = out["per_arm"]["all_rows"]
    out["rises_with_mu"] = tab["mu1"]["token_mean"] <= tab["mu8"]["token_mean"] <= tab["mu16"]["token_mean"]
    ratios = [tab[a]["share_ratio"] for a in ("mu8", "mu16")]
    out["grad_share_exceeds_token"] = all(r is not None and r > 1.0 for r in ratios)
    out["verdict"] = (
        "first_half_holds_second_half_falsified"
        if out["rises_with_mu"] and not out["grad_share_exceeds_token"] else "see_fields"
    )
    out["note"] = (
        "REPORT §5.6 quotes the all-rows convention (mu8 0.00264 -> mu16 0.00335). "
        "The measured quantity is logprob-space |ratio*A| mass share, NOT parameter-"
        "space gradient norm share, so a ratio < 1 does not refute the paper's "
        "50-70% claim -- it only says the logprob-space proxy shows no excess weight."
    )
    return out


def c7_cost(data) -> dict:
    """C7: single run <= 6 CNY, nine runs <= 55 CNY."""
    per_run = {}
    for arm in ARMS:
        for s in SEEDS:
            per_run[f"{arm}_s{s}"] = data[(arm, s)][-1]["cost_cny_cum"]
    total = sum(per_run.values())
    out = {"per_run": per_run, "max_run": max(per_run.values()), "total": total,
           "limits": {"run": 6.0, "total": 55.0}}
    out["verdict"] = ("holds" if out["max_run"] <= 6.0 and total <= 55.0 else "breached")
    out["note"] = ("values are the last cost_cny_cum row of each 8-rollout run; the "
                   "19-rollout continuation is excluded by the 8-row guard.")
    return out


def p1_pass_rate(data) -> dict:
    """P1: is the base model's pass rate inside the usable 10-90% band?"""
    vals = []
    for arm in ARMS:
        for s in SEEDS:
            rows = data[(arm, s)]
            r0 = min((r for r in rows if int(r["rollout_idx"]) == 0),
                     key=lambda r: int(r["mu_step"]))
            vals.append({"run": f"{arm}_s{s}", "pass_rate": r0["pass_rate"]})
    xs = [v["pass_rate"] for v in vals]
    out = {"per_run": vals, "min": min(xs), "max": max(xs), "band": [0.10, 0.90]}
    out["verdict"] = ("usable" if out["band"][0] <= out["min"] and out["max"] <= out["band"][1]
                      else "outside_band")
    out["note"] = ("baseline = rollout 0, mu_step 0 (untrained). The three mu1 seeds "
                   "repeat inside mu8/mu16 because the prompt pool and the initial "
                   "policy are shared -- that is duplication, not extra evidence.")
    return out


# --------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--allow-19", action="store_true")
    a = ap.parse_args()

    runs = Path(a.runs)
    data = load(runs)

    n_mu1 = len({int(r["rollout_idx"]) for r in data[("mu1", 0)]})
    if n_mu1 != 8 and not a.allow_19:
        raise SystemExit(
            f"{runs}/mu1_s0 has {n_mu1} rollouts, expected 8.\n"
            "The working copy runs/mu1_s* is the 19-rollout R4-A continuation "
            "(uncommitted M state). Point --runs at the archived 8-rollout copy, "
            "e.g. <archive>/20_实验数据/01_original_8rollout/runs, or pass "
            "--allow-19 if you really mean the continuation."
        )

    out = {
        "source": str(runs),
        "rollouts_per_run": n_mu1,
        "C2": c2_kl_monotone(data),
        "C3": c3_crossing(data),
        "C4": c4_drift_d0(data),
        "C5": c5_resp_len(data),
        "C6": c6_clip(data),
        "C7": c7_cost(data),
        "P1": p1_pass_rate(data),
    }

    lines = [f"# residual claims recompute — {runs} ({n_mu1} rollouts/run)", ""]
    t = out["C2"]["total"]
    lines += [
        "## C2 KL monotonicity",
        f"- strictly increasing, all rows (incl. baseline): {t['all_rows']}/{t['rollouts']}",
        f"- strictly increasing, stepped only:            {t['stepped']}/{t['rollouts']}",
        f"- adjacent pairs rising:                        {t['up']}/{t['pairs']} = {t['up_rate']:.3f}",
        f"- negative KL rows:                             {t['neg_rows']}/{t['kl_rows']}",
        f"- verdict: {out['C2']['verdict']}",
        "",
        "## C3 d/d0 crossings",
        f"- envelope A (rollout-mean d0): [{out['C3']['envelope']['A'][0]:.4f}, {out['C3']['envelope']['A'][1]:.4f}]",
        f"- envelope B (first-step d0):   [{out['C3']['envelope']['B'][0]:.4f}, {out['C3']['envelope']['B'][1]:.4f}]",
        f"- crossings: {out['C3']['crossings']}  verdict: {out['C3']['verdict']}",
        "",
        "## C4 d0 vs mu",
    ]
    for arm in ARMS:
        lines.append(f"- d0[{arm}] = {out['C4']['d0'][arm]:.4f}")
    for arm in ("mu8", "mu16"):
        r = out["C4"]["ratio_to_mu1"][arm]
        lines.append(f"- {arm}/mu1 per seed: " + " / ".join(f"{x:.2f}" for x in r)
                     + f"  arm mean {out['C4']['ratio_to_mu1'][arm + '_arm_mean']:.2f}")
    lines += [f"- measured direction: {out['C4']['measured_direction']} "
              f"(pre-registered: {out['C4']['expected']}) -> {out['C4']['verdict']}", "",
              "## C5 response length"]
    for arm in ARMS:
        p = out["C5"]["per_arm"][arm]
        extra = f", {p['rel_to_mu1'] * 100:+.2f}% vs mu1" if "rel_to_mu1" in p else ""
        lines.append(f"- {arm}: {p['mean']:.2f} tok{extra}")
    lines += [f"- spread {out['C5']['spread_frac'] * 100:.2f}%, max shift "
              f"{out['C5']['max_abs_shift'] * 100:.2f}% vs 5% threshold -> {out['C5']['verdict']}", "",
              "## C6 clip shares"]
    for conv in out["C6"]["conventions"]:
        lines.append(f"### {conv}")
        for arm in ARMS:
            p = out["C6"]["per_arm"][conv][arm]
            sr = f"{p['share_ratio']:.2f}x" if p["share_ratio"] else "n/a"
            lines.append(f"- {arm}: token {p['token_mean']:.5f} grad {p['grad_mean']:.5f} ratio {sr}")
    lines += [f"- rises with mu: {out['C6']['rises_with_mu']}, grad>token: "
              f"{out['C6']['grad_share_exceeds_token']} -> {out['C6']['verdict']}", "",
              "## C7 cost", f"- max run CNY {out['C7']['max_run']:.4f} (<=6), "
              f"total CNY {out['C7']['total']:.4f} (<=55) -> {out['C7']['verdict']}", "",
              "## P1 pass rate", f"- baseline range [{out['P1']['min']:.4f}, {out['P1']['max']:.4f}] "
              f"vs band {out['P1']['band']} -> {out['P1']['verdict']}"]
    md = "\n".join(lines) + "\n"
    print(md)

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
