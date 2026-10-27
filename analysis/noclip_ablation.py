"""C11: does dual clip matter at mu=16? Evidence file for claims.yaml.

The ablation is one switch -- `train.clip_delta` 10.0 -> null in grpo.py:83-86 --
with everything else fixed. Three seeds per side. The pre-registered criterion is

    |pass(noclip) - pass(clip=10)| < 3.0pp  ->  indistinguishable
    <= -3.0pp                              -> dual clip is necessary at mu=16
    >= +3.0pp                              -> dual clip is harmful

Two things this script refuses to do quietly:

1. It does not pick a pass definition for you. C11's text does not say which one
   to use, and the archive supports at least three (endpoint r7, second half
   r4-r7, all eight rollouts). They disagree by a factor of 1.6 on this data
   (+0.97pp endpoint vs +0.60pp second half), so all three are computed and
   labelled. The headline uses the second-half mean because that is the口径
   REPORT.md's main table uses; the other two are printed next to it.

2. It reports a mechanism statistic alongside the p-value-sized one. `ratio_p95`
   is the 95th percentile of the importance ratio; if it never approaches
   `clip_delta`, then the switch is inert and "indistinguishable" means "this
   configuration never exercised the clamp", which is a strictly weaker claim
   than "dual clip is harmless". claims.yaml records the inertness as primary.

    python analysis/noclip_ablation.py
    python analysis/noclip_ablation.py --json analysis/out/noclip_mu16.json

stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
THRESHOLD_PP = 3.0      # n=3 MDE, fixed by C11 before the runs
T_CRIT_DF2 = 4.303      # two-sided 95%, df = n-1 = 2


def read_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_rollout(rows: list[dict]) -> dict[int, list[dict]]:
    """metrics.jsonl is one row per mu_step, not per rollout."""
    groups: dict[int, list[dict]] = {}
    for r in rows:
        groups.setdefault(int(r["rollout_idx"]), []).append(r)
    for k in groups:
        groups[k].sort(key=lambda x: x["mu_step"] if x["mu_step"] is not None else -1)
    return dict(sorted(groups.items()))


def pass_series(rows: list[dict]) -> list[float]:
    """Rollout-level pass_rate: it is constant within a rollout, read it once."""
    out = []
    for _, grp in by_rollout(rows).items():
        zero = [x for x in grp if x["mu_step"] == 0]
        out.append(float((zero or grp)[0]["pass_rate"]))
    return out


def max_field(rows: list[dict], field: str) -> float | None:
    vals = [float(r[field]) for r in rows if r.get(field) is not None]
    return max(vals) if vals else None


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def stdev(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def arm(tag: str) -> dict:
    path = os.path.join(ROOT, "runs", tag, "metrics.jsonl")
    rows = read_rows(path)
    s = pass_series(rows)
    n = len(s)
    return {
        "tag": tag,
        "n_rollouts": n,
        "rows": len(rows),
        "pass_series": [round(v, 6) for v in s],
        "pass_endpoint": s[-1],
        "pass_second_half": mean(s[n // 2:]),
        "pass_all": mean(s),
        "ratio_p95_max": max_field(rows, "ratio_p95"),
        "clip_frac_high_max": max_field(rows, "clip_frac_high"),
        "clip_frac_any_max": max_field(rows, "clip_frac_any"),
    }


def paired(clip: list[dict], noclip: list[dict], key: str) -> dict:
    diffs = [(n[key] - c[key]) * 100.0 for c, n in zip(clip, noclip)]
    m = mean(diffs)
    sd = stdev(diffs)
    se = sd / math.sqrt(len(diffs))
    t = m / se if se else float("inf")
    return {
        "definition": key,
        "per_seed_pp": [round(d, 4) for d in diffs],
        "mean_pp": round(m, 4),
        "sd_pp": round(sd, 4),
        "se_pp": round(se, 4),
        "t_df2": round(t, 3),
        "ci95_pp": [round(m - T_CRIT_DF2 * se, 4), round(m + T_CRIT_DF2 * se, 4)],
        "all_seeds_same_sign": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
        "criterion": verdict(m),
    }


def verdict(diff_pp: float) -> str:
    if diff_pp <= -THRESHOLD_PP:
        return "dual_clip_necessary"
    if diff_pp >= THRESHOLD_PP:
        return "dual_clip_harmful"
    return "indistinguishable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    clip = [arm("mu16_s%d" % s) for s in args.seeds]
    noclip = [arm("noclip_mu16_s%d" % s) for s in args.seeds]

    for a in clip + noclip:
        if a["n_rollouts"] != 8:
            print("[warn] %s has %d rollouts, expected 8" % (a["tag"], a["n_rollouts"]))

    comparisons = [paired(clip, noclip, k)
                   for k in ("pass_endpoint", "pass_second_half", "pass_all")]
    primary = next(c for c in comparisons if c["definition"] == "pass_second_half")

    # Mechanism: is the clamp ever approached? Upper bound is clip_delta = 10.
    r95 = max(a["ratio_p95_max"] or 0.0 for a in noclip)
    r95_clip = max(a["ratio_p95_max"] or 0.0 for a in clip)
    CLAMP = 10.0
    inert = r95 < CLAMP / 2.0

    if inert:
        headline = "clip_is_inert"
    else:
        headline = primary["criterion"]

    out = {
        "claim": "C11",
        "threshold_pp": THRESHOLD_PP,
        "t_crit_df2": T_CRIT_DF2,
        "arms": {"clip_delta_10": clip, "clip_delta_null": noclip},
        "comparisons": comparisons,
        "primary_definition": "pass_second_half",
        "primary": primary,
        "mechanism": {
            "clip_delta_upper_bound": CLAMP,
            "ratio_p95_max_noclip": round(r95, 6),
            "ratio_p95_max_clip": round(r95_clip, 6),
            "clamp_headroom_x": round(CLAMP / r95, 2) if r95 else None,
            "verdict": ("clamp never approached -- ablation is inert, not harmless"
                        if inert else "clamp approached -- ablation is informative"),
        },
        "headline": headline,
        "note": (
            "C11's text does not fix a pass definition. All three are reported; "
            "primary uses the second-half mean (r4-r7), which is REPORT.md's main-table "
            "convention. Endpoint gives a larger difference (+%.2fpp vs +%.2fpp), "
            "so the choice is not neutral." % (
                next(c["mean_pp"] for c in comparisons if c["definition"] == "pass_endpoint"),
                primary["mean_pp"],
            )
        ),
    }

    if args.json:
        os.makedirs(os.path.dirname(os.path.join(ROOT, args.json)), exist_ok=True)
        with open(os.path.join(ROOT, args.json), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print("[json] %s" % args.json)

    print("=" * 78)
    print("C11  dual-clip ablation at mu=16   (threshold %.1fpp, n=3)" % THRESHOLD_PP)
    print("=" * 78)
    print("%-12s %10s %12s %12s" % ("seed", "clip=10", "noclip", "diff pp"))
    for i, s in enumerate(args.seeds):
        print("%-12s %10.4f %12.4f %+12.2f" % (
            "s%d" % s, clip[i]["pass_second_half"], noclip[i]["pass_second_half"],
            (noclip[i]["pass_second_half"] - clip[i]["pass_second_half"]) * 100))
    print()
    for c in comparisons:
        print("%-18s %+7.2f pp  SD %.2f  t %.2f  CI95 [%+.2f, %+.2f]  %s" % (
            c["definition"], c["mean_pp"], c["sd_pp"], c["t_df2"],
            c["ci95_pp"][0], c["ci95_pp"][1], c["criterion"]))
    print()
    print("mechanism: ratio_p95 max noclip %.4f vs clamp %.1f  (headroom %.1fx) -> %s" % (
        r95, CLAMP, CLAMP / r95 if r95 else 0, out["mechanism"]["verdict"]))
    print()
    print("HEADLINE: %s   (statistics says '%s', mechanism says '%s')" % (
        headline, primary["criterion"], out["mechanism"]["verdict"]))
    print(out["note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
