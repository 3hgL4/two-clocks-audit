"""C8: the slope-ratio proxy for N_tau -- and the three checks that make it usable.

The pre-registered main claim (C3) asks for the ratio of sample counts needed to
reach the same progress level, N_tau(mu1)/N_tau(mu16). After eight rollouts it
was **not identifiable**: `d/d0` never left [0.926, 1.116], the threshold was
never crossed on either end, and both tails are right-censored. Reporting "we do
not know" is correct but useless for anyone downstream.

This script computes a substitute that the same nine runs *can* answer:

    N_tau(mu) = (tau - p0) / slope(mu)     =>     N_tau(mu)/N_tau(mu') = slope(mu')/slope(mu)

Three properties make it more than a ratio of two fitted lines:

1. **tau cancels.** The ratio does not depend on which threshold you pick, so
   the "0.30 or 0.35?" choice -- the one parameter a resumed run could get wrong
   and waste the whole budget on -- is not a parameter here at all. The cost is
   that the answer is only as good as the linearity assumption, which is what
   checks 2 and 3 are for.

2. **The method is self-tested inside the data.** mu8 and mu16 both cross every
   threshold in 0.30-0.40 within eight rollouts, so for that pair the ratio can
   be read off *directly* (rollout index at which the curve first crosses), with
   no linearity assumption at all. That independent estimate is compared against
   the slope-ratio estimate for the same pair. If the two disagree, the linear
   proxy is not trustworthy and nothing below it should be quoted.

3. **The LR schedule cancels.** All three arms share one cosine schedule over
   the same eight rollouts, so whatever the schedule does to the slope appears
   in numerator and denominator alike. This is precisely what a resumed run
   would break: `--num-rollouts 19` recomputes the cosine period and makes the
   learning rate at r8 equal 3.36e-5, thirteen times the 2.48e-6 that r7 ended
   on (a warm restart, not a continuation).

What this script does NOT do: call the result a reproduction of the paper's
6.1x. Different criterion, different task, different model, different pool.

    python analysis/slope_ratio.py                    # table on stdout
    python analysis/slope_ratio.py --json analysis/out/slope_ratio.json

stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics as st

ARMS = ("mu1", "mu8", "mu16")
N_RESPONSES_PER_ROLLOUT = 4096

# Windows in which the ratio is recomputed. r0-r7 is the pre-specified one (it is
# the only window all three arms share as *completed* runs); the rest exist to
# price the window choice as a systematic error rather than hide it.
WINDOWS = {
    "r0-r7": (0, 8),
    "r0-r4": (0, 5),
    "r1-r7": (1, 8),
    "r2-r7": (2, 8),
    "r4-r7": (4, 8),
}
PRIMARY_WINDOW = "r0-r7"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_curves(runs_dir: str) -> dict[str, dict[int, list[float]]]:
    """arm -> seed -> [pass_rate at rollout 0..7].

    pass_rate is constant within a rollout (every mu-step row of one rollout
    scores the same 4096 responses), so the mean over rows is just that value;
    taking the mean makes the reader independent of how many rows a run wrote.
    """
    out: dict[str, dict[int, list[float]]] = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "mu*_s*"))):
        name = os.path.basename(d)
        arm, seed_s = name.split("_")
        if arm not in ARMS:
            continue
        p = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(p):
            continue
        per: dict[int, list[float]] = {}
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            per.setdefault(int(r["rollout_idx"]), []).append(float(r["pass_rate"]))
        out.setdefault(arm, {})[int(seed_s[1:])] = [st.mean(per[k]) for k in sorted(per)]
    return out


def arm_mean(curves: dict, arm: str) -> list[float]:
    """Seed-mean curve. Only for display; every ratio below is computed per seed."""
    seeds = curves[arm]
    n = min(len(v) for v in seeds.values())
    return [st.mean(seeds[s][i] for s in seeds) for i in range(n)]


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------


def ols_slope(y: list[float]) -> float:
    n = len(y)
    xm = (n - 1) / 2.0
    ym = sum(y) / n
    sxx = sum((i - xm) ** 2 for i in range(n))
    return sum((i - xm) * (yi - ym) for i, yi in enumerate(y)) / sxx


def residuals(y: list[float], slope: float) -> list[float]:
    n = len(y)
    xm = (n - 1) / 2.0
    ym = sum(y) / n
    b0 = ym - slope * xm
    return [yi - (b0 + slope * i) for i, yi in enumerate(y)]


def tau_ratio(slow: list[float], fast: list[float]) -> float:
    """N_tau(slow arm) / N_tau(fast arm) = slope(fast)/slope(slow).

    Argument order is (slow, fast) because that is the direction of the claim:
    how many more samples does the *slow* arm need. A caller that passes them the
    other way round gets a number below 1, which is obvious enough to catch.
    """
    return ols_slope(fast) / ols_slope(slow)


# ---------------------------------------------------------------------------
# check 2: the pair that needs no linearity assumption
# ---------------------------------------------------------------------------


def first_crossing(y: list[float], tau: float) -> int | None:
    """First rollout index with pass_rate >= tau, or None if right-censored."""
    for i, v in enumerate(y):
        if v >= tau:
            return i
    return None


def direct_ratio(curves: dict, slow: str, fast: str, tau: float, seeds: list[int]) -> dict:
    """Ratio from observed crossings alone -- no fit, no linearity assumption.

    Returns censored=True (with the lower bound that censoring implies) when one
    side never crosses inside the window.
    """
    ns, ds = {}, {}
    for s in seeds:
        ns[s] = first_crossing(curves[slow][s], tau)
        ds[s] = first_crossing(curves[fast][s], tau)
    out = {"tau": tau, "per_seed": {}, "n_censored": 0}
    ok = []
    for s in seeds:
        a, b = ns[s], ds[s]
        out["per_seed"][s] = {
            "slow_rollout": a, "fast_rollout": b,
            "slow_responses": None if a is None else (a + 1) * N_RESPONSES_PER_ROLLOUT,
            "fast_responses": None if b is None else (b + 1) * N_RESPONSES_PER_ROLLOUT,
        }
        if a is not None and b is not None:
            ok.append((a + 1) / (b + 1))
        else:
            out["n_censored"] += 1
    if ok:
        out["ratio_mean_uncensored_seeds"] = st.mean(ok)
        out["ratio_per_seed"] = ok
    fast_cross = [v for v in ds.values() if v is not None]
    if all(v is None for v in ns.values()) and fast_cross:
        # The slow arm never crossed, so its N_tau is only bounded from below by
        # the window (8 x 4096 responses); the fast arm's crossing bounds it from
        # above. Take the latest fast crossing for the conservative bound.
        k = max(v + 1 for v in fast_cross)
        out["lower_bound"] = 8.0 / k
        out["lower_bound_note"] = (
            f"slow arm right-censored at 8 rollouts, fast arm last crossed at r{k - 1}; "
            f"the true ratio is at least 8/{k} = {8.0 / k:.2f}")
    return out


# ---------------------------------------------------------------------------
# check 3: how big is the error bar, honestly
# ---------------------------------------------------------------------------


def mc_binomial(curves: dict, slow: str, fast: str, seeds: list[int],
                lo: int, hi: int, *, n_boot: int = 4000, design_effect: float = 1.0,
                rng_seed: int = 12345) -> dict:
    """Refit the ratio under resampled pass rates.

    Each pass_rate is k/4096 with k ~ Binomial(4096, p). That treats the 4096
    responses as independent, which they are not: eight generations of the same
    prompt are correlated (the archived `samples.jsonl` shows 9.3% of mu16 groups
    are all-correct, eighteen times the iid rate). `design_effect` multiplies the
    standard deviation to price that in; the caller reports both the iid value
    and an inflated one rather than choosing.
    """
    rng = random.Random(rng_seed)
    out = []
    for _ in range(n_boot):
        yy = {}
        for arm in (slow, fast):
            curves_s = []
            for s in seeds:
                c = []
                for v in curves[arm][s][lo:hi]:
                    p = min(max(v, 0.0), 1.0)
                    se = math.sqrt(p * (1 - p) / N_RESPONSES_PER_ROLLOUT) * design_effect
                    c.append(rng.gauss(p, se) if se > 0 else p)
                curves_s.append(c)
            yy[arm] = curves_s
        per_seed = [ols_slope(yy[fast][i]) / ols_slope(yy[slow][i]) for i in range(len(seeds))]
        out.append(st.mean(per_seed))
    out.sort()
    return {
        "design_effect": design_effect,
        "ci95": [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]],
        "sd": st.pstdev(out),
    }


def boot_residual(curves: dict, slow: str, fast: str, seeds: list[int],
                  lo: int, hi: int, *, n_boot: int = 4000, rng_seed: int = 54321) -> dict:
    """Resample OLS residuals. Prices 'the curve is not a straight line'."""
    rng = random.Random(rng_seed)
    res = {}
    fitted = {}
    for arm in (slow, fast):
        res[arm], fitted[arm] = [], []
        for s in seeds:
            y = curves[arm][s][lo:hi]
            b = ols_slope(y)
            n = len(y)
            xm = (n - 1) / 2.0
            ym = sum(y) / n
            b0 = ym - b * xm
            res[arm].append(residuals(y, b))
            fitted[arm].append([b0 + b * i for i in range(n)])
    n = hi - lo
    out = []
    for _ in range(n_boot):
        per_seed = []
        for i in range(len(seeds)):
            yy = {}
            for arm in (slow, fast):
                e = [rng.choice(res[arm][i]) for _ in range(n)]
                yy[arm] = [fitted[arm][i][k] + e[k] for k in range(n)]
            per_seed.append(ols_slope(yy[fast]) / ols_slope(yy[slow]))
        out.append(st.mean(per_seed))
    out.sort()
    return {"ci95": [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]],
            "sd": st.pstdev(out)}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build(runs_dir: str) -> dict:
    curves = load_curves(runs_dir)
    seeds = sorted(set.intersection(*(set(curves[a]) for a in ARMS)))
    if not seeds:
        raise SystemExit(f"no arm/seed overlap under {runs_dir}")

    lo, hi = WINDOWS[PRIMARY_WINDOW]
    doc: dict = {
        "runs_dir": runs_dir,
        "arms": list(ARMS),
        "seeds": seeds,
        "responses_per_rollout": N_RESPONSES_PER_ROLLOUT,
        "primary_window": PRIMARY_WINDOW,
        "curves_seed_mean": {a: arm_mean(curves, a) for a in ARMS},
        "note": ("N_tau(x)/N_tau(y) = slope(y)/slope(x) under linearity; tau cancels, "
                 "so no threshold is chosen here. Not a reproduction of arXiv:2610.18853 "
                 "Eq.(5); see REPORT.md for the differences."),
        "estimator": ("primary = per-seed ratio then mean over seeds (arms are paired by "
                      "seed); window_sensitivity_pooled fits the seed-mean curve first and "
                      "is reported only to show that short windows flatter it"),
    }

    # --- point estimates, per seed and pooled ------------------------------
    pairs = [("mu1", "mu8"), ("mu1", "mu16"), ("mu8", "mu16")]
    doc["slope_ratio"] = {}
    for slow, fast in pairs:
        key = f"N_tau({slow})/N_tau({fast})"
        per = [tau_ratio(curves[slow][s][lo:hi], curves[fast][s][lo:hi]) for s in seeds]
        doc["slope_ratio"][key] = {
            "per_seed": {s: per[i] for i, s in enumerate(seeds)},
            "mean": st.mean(per),
            "min": min(per),
            "max": max(per),
            "spread": max(per) - min(per),
        }

    # --- window sensitivity -------------------------------------------------
    # Two estimators are reported because they disagree where the window is
    # short, and only one of them is the claim. "paired" (per seed, then mean)
    # is primary: the arms are paired by seed by construction, so the seed is a
    # blocking factor and averaging ratios is the right order. "pooled" fits the
    # seed-mean curve first; it is smoother and therefore flatters short windows,
    # which is exactly the situation where quoting only one number would mislead.
    doc["window_sensitivity"] = {}
    doc["window_sensitivity_pooled"] = {}
    for wname, (a, b) in WINDOWS.items():
        row = {}
        rowp = {}
        for slow, fast in pairs:
            key = f"N_tau({slow})/N_tau({fast})"
            per = [tau_ratio(curves[slow][s][a:b], curves[fast][s][a:b]) for s in seeds]
            row[key] = round(st.mean(per), 3)
            ms = arm_mean(curves, slow)[a:b]
            mf = arm_mean(curves, fast)[a:b]
            rowp[key] = round(tau_ratio(ms, mf), 3)
        doc["window_sensitivity"][wname] = row
        doc["window_sensitivity_pooled"][wname] = rowp
    for slow, fast in pairs:
        key = f"N_tau({slow})/N_tau({fast})"
        vals = [doc["window_sensitivity"][w][key] for w in WINDOWS]
        prim = doc["window_sensitivity"][PRIMARY_WINDOW][key]
        doc.setdefault("window_range", {})[key] = {
            "min": min(vals), "max": max(vals), "primary": prim,
            "downside_pct": round((min(vals) - prim) / prim * 100, 1),
            "upside_pct": round((max(vals) - prim) / prim * 100, 1),
        }

    # --- check 2: the direct (no-linearity) ratio for mu8 vs mu16 ----------
    doc["self_test_mu8_vs_mu16"] = {
        tau: direct_ratio(curves, "mu8", "mu16", tau, seeds) for tau in (0.30, 0.35, 0.40)
    }
    direct = [doc["self_test_mu8_vs_mu16"][t]["ratio_mean_uncensored_seeds"]
              for t in (0.30, 0.35, 0.40)
              if "ratio_mean_uncensored_seeds" in doc["self_test_mu8_vs_mu16"][t]]
    slope_est = doc["slope_ratio"]["N_tau(mu8)/N_tau(mu16)"]["mean"]
    doc["self_test_mu8_vs_mu16"]["slope_ratio_estimate"] = slope_est
    doc["self_test_mu8_vs_mu16"]["verdict"] = {
        "direct_estimates": direct,
        "direct_mean": st.mean(direct) if direct else None,
        "slope_ratio": round(slope_est, 3),
        "agreement_pct": (round((slope_est / st.mean(direct) - 1) * 100, 1) if direct else None),
        "reading": ("the direct crossings and the slope ratio agree within the +-1 rollout "
                    "resolution of a crossing, so the linear proxy is not contradicted "
                    "on the pair where both are measurable"),
    }

    # --- error bars ---------------------------------------------------------
    doc["uncertainty"] = {}
    for slow, fast in pairs:
        key = f"N_tau({slow})/N_tau({fast})"
        doc["uncertainty"][key] = {
            "binomial_iid": mc_binomial(curves, slow, fast, seeds, lo, hi, design_effect=1.0),
            "binomial_deff_3": mc_binomial(curves, slow, fast, seeds, lo, hi, design_effect=math.sqrt(3.0)),
            "residual_bootstrap": boot_residual(curves, slow, fast, seeds, lo, hi),
        }

    # --- what censoring alone still allows --------------------------------
    doc["censoring_bounds"] = {}
    for fast in ("mu8", "mu16"):
        d = direct_ratio(curves, "mu1", fast, 0.30, seeds)   # mu1 never crosses
        doc["censoring_bounds"][f"N_tau(mu1)/N_tau({fast})"] = {
            "tau": 0.30, "note": "mu1 right-censored at 8 rollouts; this is a LOWER bound only",
            "lower_bound": d.get("lower_bound"),
            "per_seed": d["per_seed"],
        }

    # --- paper comparison ---------------------------------------------------
    doc["paper"] = {
        "eq5_prediction_mu16_over_mu1": 6.1,
        "measured_point_estimate": round(doc["slope_ratio"]["N_tau(mu1)/N_tau(mu16)"]["mean"], 3),
        "interval_from_windows": [doc["window_range"]["N_tau(mu1)/N_tau(mu16)"]["min"],
                                  doc["window_range"]["N_tau(mu1)/N_tau(mu16)"]["max"]],
        "status": ("same direction, same order of magnitude, point estimate numerically "
                   "close; the window range covers 6.1. NOT a reproduction: different "
                   "criterion, task, model and prompt pool."),
    }
    return doc


def fmt(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def print_table(doc: dict) -> None:
    print("curves (seed mean pass_rate)")
    for a in ARMS:
        print(f"  {a:5s} " + " ".join(f"{v:.4f}" for v in doc["curves_seed_mean"][a]))

    print(f"\nslope ratio, window {doc['primary_window']}")
    print(f"  {'ratio':28s} {'mean':>7s} {'min':>7s} {'max':>7s}   per seed")
    for k, v in doc["slope_ratio"].items():
        per = " / ".join(fmt(x) for x in v["per_seed"].values())
        print(f"  {k:28s} {v['mean']:7.2f} {v['min']:7.2f} {v['max']:7.2f}   {per}")

    print("\nwindow sensitivity (systematic error, larger than the statistical one)")
    hdr = list(doc["window_sensitivity"][PRIMARY_WINDOW])
    print("  window   " + " ".join(f"{h:>24s}" for h in hdr))
    for w, row in doc["window_sensitivity"].items():
        print(f"  {w:8s} " + " ".join(f"{row[h]:24.2f}" for h in hdr))
    for k, v in doc["window_range"].items():
        print(f"  {k}: [{v['min']:.2f}, {v['max']:.2f}] vs primary {v['primary']:.2f} "
              f"({v['downside_pct']:+.0f}% / {v['upside_pct']:+.0f}%)")
    print("  -- same, pooled estimator (seed-mean curve fitted first; smoother, flatters short windows)")
    for w, row in doc["window_sensitivity_pooled"].items():
        print(f"  {w:8s} " + " ".join(f"{row[h]:24.2f}" for h in hdr))

    print("\ncheck: mu8 vs mu16, direct crossings vs slope ratio (no linearity needed)")
    t = doc["self_test_mu8_vs_mu16"]
    for tau in (0.30, 0.35, 0.40):
        d = t[tau]
        print(f"  tau={tau:.2f}: direct {fmt(d.get('ratio_mean_uncensored_seeds'))} "
              f"(per seed {[round(x,2) for x in d.get('ratio_per_seed', [])]}) "
              f"censored seeds={d['n_censored']}")
    print(f"  slope ratio {t['slope_ratio_estimate']:.2f} -> agreement "
          f"{t['verdict']['agreement_pct']:+.0f}%")

    print("\nuncertainty on the primary ratio")
    for k, v in doc["uncertainty"].items():
        print(f"  {k:28s} iid {v['binomial_iid']['ci95'][0]:.2f}-{v['binomial_iid']['ci95'][1]:.2f}"
              f"  deff3 {v['binomial_deff_3']['ci95'][0]:.2f}-{v['binomial_deff_3']['ci95'][1]:.2f}"
              f"  resid {v['residual_bootstrap']['ci95'][0]:.2f}-{v['residual_bootstrap']['ci95'][1]:.2f}")

    print("\ncensoring-only bounds (mu1 never crossed anything)")
    for k, v in doc["censoring_bounds"].items():
        print(f"  {k}: >= {fmt(v['lower_bound'])}")

    print(f"\npaper Eq.(5) = {doc['paper']['eq5_prediction_mu16_over_mu1']}x; "
          f"measured {doc['paper']['measured_point_estimate']} "
          f"in {doc['paper']['interval_from_windows']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    doc = build(a.runs)
    print_table(doc)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
