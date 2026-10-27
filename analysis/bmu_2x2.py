"""C10: b x mu 2x2 -- is per-step sample count b, or update count mu, the driver?

C10's criterion (analysis/claims.yaml):
  response = r7 pass rate (endpoint, not the second-half mean the main table uses)
  estimate two main effects across three seeds
    if |b main effect| > |mu main effect| AND the paired CIs do not overlap
      -> b, not mu, is the primary cause of the pass gap
    if the interaction is positive (b helps more at mu=16)
      -> mu16's penalty can be bought back with more samples per step

Design, as it actually landed (verified against metrics.jsonl, not configs):
  cell              mu   b     responses/rollout   source
  mu8_b512           8  512              4096       01_original_8rollout
  mu16_b256         16  256              4096       01_original_8rollout
  mu8_b256           8  256              2048       02_bxmu
  mu16_b512         16  512              8192       02_bxmu

Br = b * mu is an identity, so a true 2x2 cannot hold Br fixed: two cells are
half-budget and two are double. That is a second-order confound and it pushes in
OPPOSITE directions for the two main effects (C10 note), so the honest reading
needs both numbers on the table:

  with budget (default) -- the four cells as run, budget varies
  budget-matched         -- only the two 4096-response cells are comparable, so
                            mu's effect is measured there alone and b's effect is
                            measured as the change from the 4096 cell to the
                            adjacent half/double cell

Neither is "the" answer; both are printed.

    python analysis/bmu_2x2.py --runs runs --ab runs_ab2x2 --out analysis/out/bmu_2x2.json

stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

# t(0.975, df=2) -- n=3 per cell, same convention the rest of the audit uses
T_DF2 = 4.302653


def read_endpoint(path):
    """Last row of metrics.jsonl = (last rollout, last mu_step) = r7 endpoint."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("empty metrics: %s" % path)
    last = rows[-1]
    return {
        "mu": int(last["mu"]),
        "b": int(last["b_responses"]),
        "br": int(last["br_responses"]),
        "seed": int(last["seed"]),
        "rollout": int(last["rollout_idx"]),
        "mu_step": int(last["mu_step"]),
        "pass_r7": float(last["pass_rate"]),
        "prompts": int(last["prompts_per_rollout"]),
        "responses_cum": int(last["responses_cum"]),
        "cost": float(last.get("cost_cny_cum", 0.0)),
        "dir": os.path.basename(os.path.dirname(path)),
    }


def load_cells(dirs):
    cells = defaultdict(dict)  # (mu, b) -> seed -> record
    for d in dirs:
        for name in sorted(os.listdir(d)):
            mp = os.path.join(d, name, "metrics.jsonl")
            if not os.path.isfile(mp):
                continue
            rec = read_endpoint(mp)
            cells[(rec["mu"], rec["b"])][rec["seed"]] = rec
    return cells


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def ci95(xs):
    m = mean(xs)
    if len(xs) < 2:
        return m, m, 0.0
    half = T_DF2 * sd(xs) / (len(xs) ** 0.5)
    return m, m - half, m + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="dir holding the two 4096-response cells")
    ap.add_argument("--ab", default="runs_ab2x2", help="dir holding the half/double budget cells")
    ap.add_argument("--json", default="analysis/out/bmu_2x2.json")
    ap.add_argument("--md", default="")
    a = ap.parse_args()

    cells = load_cells([a.runs, a.ab])
    need = [(8, 512), (16, 256), (8, 256), (16, 512)]
    missing = [k for k in need if k not in cells]
    if missing:
        raise SystemExit("missing cells: %s (found: %s)" % (missing, sorted(cells)))

    seeds = sorted(set.intersection(*[set(cells[k]) for k in need]))
    if len(seeds) < 3:
        raise SystemExit("need 3 shared seeds, have %s" % seeds)

    lines = []
    w = lines.append
    w("# C10  b x mu 2x2   (response = r7 endpoint pass rate)")
    w("")
    w("shared seeds: %s" % seeds)
    w("")
    w("| cell | mu | b | responses/rollout | prompts | r7 pass per seed | mean | cost/run |")
    w("|---|---|---|---|---|---|---|---|")
    for (mu, b) in need:
        recs = [cells[(mu, b)][s] for s in seeds]
        vals = [r["pass_r7"] for r in recs]
        w("| mu%d_b%d | %d | %d | %d | %d | %s | %.4f | %.2f |" % (
            mu, b, mu, b, recs[0]["br"], recs[0]["prompts"],
            " / ".join("%.4f" % v for v in vals), mean(vals), mean([r["cost"] for r in recs])))
    w("")

    # per-seed paired differences
    d_mu_at_b256 = [cells[(16, 256)][s]["pass_r7"] - cells[(8, 256)][s]["pass_r7"] for s in seeds]
    d_mu_at_b512 = [cells[(16, 512)][s]["pass_r7"] - cells[(8, 512)][s]["pass_r7"] for s in seeds]
    d_b_at_mu8 = [cells[(8, 512)][s]["pass_r7"] - cells[(8, 256)][s]["pass_r7"] for s in seeds]
    d_b_at_mu16 = [cells[(16, 512)][s]["pass_r7"] - cells[(16, 256)][s]["pass_r7"] for s in seeds]

    mu_main = [(x + y) / 2 for x, y in zip(d_mu_at_b256, d_mu_at_b512)]
    b_main = [(x + y) / 2 for x, y in zip(d_b_at_mu8, d_b_at_mu16)]
    inter = [x - y for x, y in zip(d_mu_at_b512, d_mu_at_b256)]
    # interaction from the b side (identical by construction, printed as a check)
    inter_b = [x - y for x, y in zip(d_b_at_mu16, d_b_at_mu8)]

    w("## per-seed paired differences (pp)")
    w("")
    w("| seed | mu 8->16 at b=256 | mu 8->16 at b=512 | b 256->512 at mu=8 | b 256->512 at mu=16 |")
    w("|---|---|---|---|---|")
    for i, s in enumerate(seeds):
        w("| %d | %+.2f | %+.2f | %+.2f | %+.2f |" % (
            s, 100 * d_mu_at_b256[i], 100 * d_mu_at_b512[i],
            100 * d_b_at_mu8[i], 100 * d_b_at_mu16[i]))
    w("")

    w("## main effects (pp)  n=3, t(0.975,df=2)=%.3f" % T_DF2)
    w("")
    w("| effect | mean | 95% CI | sd |")
    w("|---|---|---|---|")
    for label, xs in (("mu main (8->16)", mu_main), ("b main (256->512)", b_main),
                      ("interaction", inter)):
        m, lo, hi = ci95(xs)
        w("| %s | %+.2f | [%+.2f, %+.2f] | %.2f |" % (label, 100 * m, 100 * lo, 100 * hi, 100 * sd(xs)))
    w("")
    m, lo, hi = ci95(inter_b)
    w("interaction recomputed from the b side: %+.2f [%+.2f, %+.2f]  (must match the row above)"
      % (100 * m, 100 * lo, 100 * hi))
    w("")

    # the decision C10 asks for
    mu_m, mu_lo, mu_hi = ci95(mu_main)
    b_m, b_lo, b_hi = ci95(b_main)
    overlap = not (b_lo > mu_hi or b_hi < mu_lo)
    gap = [x - y for x, y in zip(mu_main, b_main)]
    g_m, g_lo, g_hi = ci95(gap)

    w("## C10 verdict")
    w("")
    w("- |b main| = %.2f pp, |mu main| = %.2f pp -> %s" % (
        100 * abs(b_m), 100 * abs(mu_m),
        "b larger" if abs(b_m) > abs(mu_m) else "mu larger"))
    w("- CIs %s" % ("overlap" if overlap else "are disjoint"))
    w("- paired gap (mu effect - b effect) per seed: %s" % " / ".join("%+.2f" % (100 * g) for g in gap))
    w("- that gap: %+.2f pp [%+.2f, %+.2f]" % (100 * g_m, 100 * g_lo, 100 * g_hi))
    if abs(b_m) > abs(mu_m) and not overlap:
        head = "b_is_primary"
    elif g_lo > 0:
        head = "mu_is_primary"
    elif g_hi < 0:
        head = "b_is_primary"
    else:
        head = "unresolved"
    w("- **HEADLINE: %s**" % head)
    w("")

    i_m, i_lo, i_hi = ci95(inter)
    w("interaction: %+.2f pp [%+.2f, %+.2f] -> %s" % (
        100 * i_m, 100 * i_lo, 100 * i_hi,
        "positive: b helps more at mu=16" if i_lo > 0 else
        ("negative: b helps less at mu=16" if i_hi < 0 else
         "not distinguishable from zero: the two effects are additive on this data")))
    w("")

    w("## budget confound, stated explicitly")
    w("")
    w("| cell | Br = b*mu | vs standard 4096 | direction of bias |")
    w("|---|---|---|---|")
    for (mu, b) in need:
        br = cells[(mu, b)][seeds[0]]["br"]
        ratio = br / 4096.0
        tag = "standard" if abs(ratio - 1) < 1e-9 else ("optimistic" if ratio > 1 else "conservative")
        w("| mu%d_b%d | %d | %.2fx | %s |" % (mu, b, br, ratio, tag))
    w("")
    w("The mu effect is measured inside a fixed b, so it is NOT budget-matched either:")
    w("b=256 column compares 2048 vs 4096 responses, b=512 column compares 4096 vs 8192.")
    w("What IS budget-matched is the 4096 column of the main table (mu1/mu8/mu16 all at 4096).")
    w("")

    out = "\n".join(lines)
    print(out)
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        payload = {
            "claim": "C10",
            "response": "r7 endpoint pass rate",
            "seeds": seeds,
            "cells": {
                "mu%d_b%d" % (mu, b): {
                    "mu": mu, "b": b, "br": cells[(mu, b)][seeds[0]]["br"],
                    "prompts": cells[(mu, b)][seeds[0]]["prompts"],
                    "pass_r7": [cells[(mu, b)][s]["pass_r7"] for s in seeds],
                    "dir": cells[(mu, b)][seeds[0]]["dir"],
                } for (mu, b) in need
            },
            "paired": {
                "d_mu_at_b256": d_mu_at_b256, "d_mu_at_b512": d_mu_at_b512,
                "d_b_at_mu8": d_b_at_mu8, "d_b_at_mu16": d_b_at_mu16,
                "mu_main": mu_main, "b_main": b_main, "interaction": inter,
            },
            "verdict": {
                "mu_main_pp": 100 * mu_m, "mu_ci": [100 * mu_lo, 100 * mu_hi],
                "b_main_pp": 100 * b_m, "b_ci": [100 * b_lo, 100 * b_hi],
                "cis_overlap": overlap,
                "gap_pp": 100 * g_m, "gap_ci": [100 * g_lo, 100 * g_hi],
                "headline": head,
                "interaction_pp": 100 * i_m,
                "interaction_ci": [100 * i_lo, 100 * i_hi],
            },
        }
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        print("\nwrote %s" % a.json)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        print("wrote %s" % a.md)


if __name__ == "__main__":
    main()
