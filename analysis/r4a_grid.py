"""C12: R4-A grid verdict -- did mu=1 reach pass >= 0.30 by rollout 19?

C12's criterion, locked before the run (analysis/claims.yaml):
  grid only -- check pass >= 0.30 at each rollout boundary, never interpolate
  between boundaries (interpolating is what C8 already assumes; using it here
  would make the check circular).
    3/3 seeds hit  -> report the first hitting rollout index and cumulative responses
    3/3 seeds miss -> N_0.30(mu1) > 19*4096 = 77,824 responses, and the
                      two ratios become lower bounds rather than point estimates

Why the grid matters: a crossing that falls between two boundaries can only be
resolved to "somewhere in this rollout", so the grid estimator is quantised and
it inflates fast arms more than slow ones (mu8 gets 8 samples per unit of
training, mu16 gets 16, mu1 gets 1). The script prints the grid ratios next to
C8's interpolated ones so the two 口径 are never mixed by accident.

    python analysis/r4a_grid.py --runs runs_r4a --base runs --json analysis/out/r4a_grid.json

stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os

TAU = 0.30  # fixed by C12, not a knob


def read_series(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("empty metrics: %s" % path)
    # one row per rollout boundary: keep the LAST mu_step row of each rollout_idx
    by_rollout = {}
    for r in rows:
        idx = int(r["rollout_idx"])
        if idx not in by_rollout or int(r["mu_step"]) > int(by_rollout[idx]["mu_step"]):
            by_rollout[idx] = r
    return [by_rollout[k] for k in sorted(by_rollout)]


def first_hit(series, tau=TAU):
    for r in series:
        if float(r["pass_rate"]) >= tau:
            return {
                "rollout_idx": int(r["rollout_idx"]),
                "pass_rate": float(r["pass_rate"]),
                "responses_cum": int(r["responses_cum"]),
                "cost_cny_cum": float(r.get("cost_cny_cum", 0.0)),
            }
    return None


def load(dirs):
    out = {}
    for d in dirs:
        for name in sorted(os.listdir(d)):
            mp = os.path.join(d, name, "metrics.jsonl")
            if not os.path.isfile(mp):
                continue
            series = read_series(mp)
            out[name] = {
                "mu": int(series[0]["mu"]),
                "seed": int(series[0]["seed"]),
                "n_rollouts": len(series),
                "series": series,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_r4a", help="dir with the 19-rollout mu1 runs")
    ap.add_argument("--base", default="runs", help="dir with the 8-rollout mu8/mu16 runs")
    ap.add_argument("--json", default="analysis/out/r4a_grid.json")
    ap.add_argument("--md", default="")
    a = ap.parse_args()

    r4a = load([a.runs])
    base = load([a.base])

    mu1 = {k: v for k, v in r4a.items() if v["mu"] == 1}
    if not mu1:
        raise SystemExit("no mu=1 runs under %s" % a.runs)
    seeds = sorted(v["seed"] for v in mu1.values())

    lines = []
    w = lines.append
    w("# C12  R4-A grid verdict   (tau = %.2f, grid only, no interpolation)" % TAU)
    w("")
    w("## mu=1 over 19 rollouts")
    w("")
    w("| run | seed | rollouts | first hit idx | pass there | responses_cum | pass at r18 |")
    w("|---|---|---|---|---|---|---|")
    hits = {}
    for k in sorted(mu1):
        v = mu1[k]
        h = first_hit(v["series"])
        hits[v["seed"]] = h
        last = v["series"][-1]
        w("| %s | %d | %d | %s | %s | %s | %.4f |" % (
            k, v["seed"], v["n_rollouts"],
            "--" if h is None else str(h["rollout_idx"]),
            "--" if h is None else "%.4f" % h["pass_rate"],
            "--" if h is None else "%d" % h["responses_cum"],
            float(last["pass_rate"])))
    w("")

    n_hit = sum(1 for h in hits.values() if h is not None)
    w("hit %d/%d seeds" % (n_hit, len(seeds)))
    w("")

    # the whole trajectory, so a near-miss is visible rather than inferred
    w("## pass trajectory per seed (mu=1)")
    w("")
    hdr = "| rollout | " + " | ".join("s%d" % s for s in seeds) + " |"
    w(hdr)
    w("|---" * (len(seeds) + 1) + "|")
    for i in range(max(v["n_rollouts"] for v in mu1.values())):
        cells = []
        for s in seeds:
            v = [x for x in mu1.values() if x["seed"] == s][0]
            cells.append("%.4f" % float(v["series"][i]["pass_rate"]) if i < v["n_rollouts"] else "--")
        w("| r%d | %s |" % (i, " | ".join(cells)))
    w("")

    # grid vs interpolation: the two 口径 side by side
    w("## grid ratios vs C8's interpolated ones")
    w("")
    w("| arm | first hit idx (per seed) | responses_cum | grid N_tau(mu1)/N_tau(arm) |")
    w("|---|---|---|---|")
    mu1_N = {s: (hits[s]["responses_cum"] if hits[s] else None) for s in seeds}
    grid_ratio = {}
    for k in sorted(base):
        v = base[k]
        if v["mu"] == 1:
            continue
        h = first_hit(v["series"])
        if h is None:
            w("| %s (mu=%d) | never crossed in 8 rollouts | -- | -- |" % (k, v["mu"]))
            continue
        s = v["seed"]
        if mu1_N.get(s) is None:
            continue
        r = mu1_N[s] / h["responses_cum"]
        grid_ratio.setdefault(v["mu"], {})[s] = r
        w("| %s (mu=%d) | %d | %d | %.2f |" % (k, v["mu"], h["rollout_idx"], h["responses_cum"], r))
    w("")
    for mu, per_seed in sorted(grid_ratio.items()):
        vals = [per_seed[s] for s in sorted(per_seed)]
        w("- mu1/mu%d grid: mean %.2f  (per seed %s)" % (
            mu, sum(vals) / len(vals), " / ".join("%.2f" % x for x in vals)))
    w("")
    w("C8 interpolated (r0-r7 window): mu1/mu8 = 4.98, mu1/mu16 = 6.14")
    w("")

    if n_hit == len(seeds):
        head = "hit_%d_of_%d" % (n_hit, len(seeds))
        w("**HEADLINE: %s** -- report the first hitting index; the point is that mu=1"
          " does move, just very slowly. Do NOT read the grid ratio as a correction to C8." % head)
    elif n_hit == 0:
        head = "no_hit"
        bound = max(v["n_rollouts"] for v in mu1.values()) * 4096
        w("**HEADLINE: no_hit** -- N_0.30(mu1) > %d responses (lower bound, no assumption used)." % bound)
    else:
        head = "partial_%d_of_%d" % (n_hit, len(seeds))
        w("**HEADLINE: %s** -- seeds disagree; neither the hit nor the bound is available"
          " as a single number. Report per seed." % head)
    w("")

    out = "\n".join(lines)
    print(out)
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        payload = {
            "claim": "C12",
            "tau": TAU,
            "method": "grid only, no interpolation",
            "seeds": seeds,
            "mu1_hits": {str(s): hits[s] for s in seeds},
            "n_hit": n_hit,
            "grid_ratios": {("mu%d" % mu): grid_ratio[mu] for mu in grid_ratio},
            "c8_interpolated": {"mu8": 4.98, "mu16": 6.14},
            "verdict": head,
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
