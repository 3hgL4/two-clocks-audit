"""SAC-mu main analysis: P1 / P2 / P3 / P5' per v2 frozen doc section 4.

Data sources
    - SAC arms (sac_s0/1/2, sacS_*, sacR_*), fixed arms (mu32_*, mu64_*):
      live-logged runs/<run>/prompts.jsonl (512 prompts x 8 responses, per cycle).
    - mu16 baseline: tools/rescore_mu16.py output (bit-exact validated; the
      campaign archive kept only 1 group/cycle, per-prompt bits were rebuilt
      from npz via pool->prompt_batch->verify).

Caliber (frozen)
    - Last segment = cycles r6+r7, response grain (equal group sizes 8).
    - P1: mean over 3 seeds of (SAC - mu16) >= +5pp AND per-seed paired-diff
      cluster bootstrap 95% CI excludes 0 (resample 1024 prompts with
      replacement per arm, 2000 draws, 2.5/97.5 percentile; c4 caliber).
    - P2: every SAC seed last-segment pass >= 0.517 - 0.01 = 0.507.
    - P3: sacR vs mu16 per-seed CIs contain 0 or point <= 0 -> attribution holds.
    - P5': trajectory diagnostics (ladder, trigger stages, cap/floor, walls).
      Descriptive only.

    python analysis/sac_p1p3.py            # prints tables, writes JSON

stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis", "out")
BASELINE_JSON = os.path.join(OUT, "mu16_rescored_r6r7.json")

SAC_RUNS = ["sac_s0", "sac_s1", "sac_s2"]          # runs_sac/
SACS_RUNS = ["sacS_s0", "sacS_s1", "sacS_s2"]      # runs_sac/
SACR_RUNS = ["sacR_s0", "sacR_s1", "sacR_s2"]      # runs_sac/
FIX_RUNS = ["mu32_s0", "mu32_s1", "mu32_s2", "mu64_s0", "mu64_s1", "mu64_s2"]  # runs_fixed/
LAST_CYCLES = (6, 7)
N_DRAWS = 2000
P2_THRESHOLD = 0.507  # 0.517 - 1pp tolerance
P1_THRESHOLD = 0.05


def run_dir(run: str) -> str:
    for sub in ("runs_sac", "runs_fixed", "runs"):
        p = os.path.join(REPO, sub, run)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(run)


def load_last_cycle_bits(run: str) -> list[str]:
    """512x8 bit-strings for cycles r6+r7 from live prompts.jsonl."""
    path = os.path.join(run_dir(run), "prompts.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    sel = [r for r in rows if r["rollout_idx"] in LAST_CYCLES]
    assert len(sel) == 2, f"{run}: expected 2 last-cycle rows, got {len(sel)}"
    bits = []
    for r in sorted(sel, key=lambda r: r["rollout_idx"]):
        assert r["group_size"] == 8 and r["n_prompts"] == 512
        bits.extend(r["correct"])
    return bits


def pass_of(bits: list[str]) -> float:
    return sum(b.count("1") for b in bits) / (len(bits) * len(bits[0]))


def load_baseline() -> dict[str, list[str]]:
    """seed -> bits for mu16, from the validated rescore."""
    d = json.load(open(BASELINE_JSON, encoding="utf-8"))
    by_seed: dict[int, list[str]] = collections.defaultdict(list)
    for rec in d["runs"]:
        if rec["rollout_idx"] in LAST_CYCLES:
            by_seed[rec["seed"]].extend(rec["correct_bits"])
    return {f"mu16_s{s}": b for s, b in sorted(by_seed.items())}


def paired_ci(bits_a: list[str], bits_b: list[str], rng: random.Random) -> tuple[float, float, float]:
    """Point diff (a-b) and cluster bootstrap CI over prompts (independent resample)."""
    na, nb = len(bits_a), len(bits_b)
    pa = [b.count("1") / len(b) for b in bits_a]
    pb = [b.count("1") / len(b) for b in bits_b]
    point = sum(pa) / na - sum(pb) / nb
    diffs = []
    for _ in range(N_DRAWS):
        ma = sum(pa[rng.randrange(na)] for _ in range(na)) / na
        mb = sum(pb[rng.randrange(nb)] for _ in range(nb)) / nb
        diffs.append(ma - mb)
    diffs.sort()
    lo = diffs[int(0.025 * N_DRAWS)]
    hi = diffs[int(0.975 * N_DRAWS) - 1]
    return point, lo, hi


def seed_of(run: str) -> int:
    return int(run.rsplit("_s", 1)[1])


def match_seeds(runs: list[str]) -> list[str]:
    return sorted(runs, key=seed_of)


def metrics_rows(run: str) -> list[dict]:
    return [json.loads(l) for l in open(os.path.join(run_dir(run), "metrics.jsonl"), encoding="utf-8")]


def cycle_table(run: str) -> list[dict]:
    by = collections.defaultdict(list)
    for r in metrics_rows(run):
        by[r["rollout_idx"]].append(r)
    out = []
    for k in sorted(by):
        v = sorted(by[k], key=lambda r: r["mu_step"])
        out.append({
            "cycle": k,
            "plan": v[0]["mu_plan"],
            "executed": v[0]["mu_executed"],
            "trigger": any(r["sac_trigger"] for r in v),
            "rows": len(v),
        })
    return out


def wall_hours(run: str) -> float:
    tot_ms = sum(r["wall_ms_rollout"] + r["wall_ms_update"] for r in metrics_rows(run))
    return tot_ms / 3.6e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(OUT, "sac_p1p3.json"))
    args = ap.parse_args()
    rng = random.Random(20261031)

    base = load_baseline()
    sac = {r: load_last_cycle_bits(r) for r in SAC_RUNS}
    sacs = {r: load_last_cycle_bits(r) for r in SACS_RUNS}
    sacrs = {r: load_last_cycle_bits(r) for r in SACR_RUNS}
    fixed = {r: load_last_cycle_bits(r) for r in FIX_RUNS}

    passes = {}
    for group in (sac, sacs, sacrs, fixed):
        for r, b in group.items():
            passes[r] = pass_of(b)
    for r, b in base.items():
        passes[r] = pass_of(b)

    print("=== last-segment (r6+r7) pass ===")
    for r in sorted(passes):
        print(f"  {r:10s} {passes[r]:.4f}")
    base_mean = sum(passes[r] for r in sorted(base)) / 3
    print(f"  mu16 3-seed mean = {base_mean:.4f}  (frozen F4 ref 0.517)")

    # ---- P1 ----
    print("\n=== P1  SAC vs mu16 (paired by seed) ===")
    p1 = {"per_seed": {}, "threshold": P1_THRESHOLD}
    for r in match_seeds(SAC_RUNS):
        s = seed_of(r)
        b_run = sac[r]
        b_base = base[f"mu16_s{s}"]
        pt, lo, hi = paired_ci(b_run, b_base, rng)
        excl0 = not (lo <= 0 <= hi)
        p1["per_seed"][r] = {"diff": pt, "ci": [lo, hi], "ci_excludes_0": excl0}
        print(f"  {r} - mu16_s{s}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] excl0={excl0}")
    mean_diff = sum(v["diff"] for v in p1["per_seed"].values()) / 3
    all_excl0 = all(v["ci_excludes_0"] for v in p1["per_seed"].values())
    p1["mean_diff"] = mean_diff
    p1["all_ci_exclude_0"] = all_excl0
    p1["verdict"] = bool(mean_diff >= P1_THRESHOLD and all_excl0)
    print(f"  3-seed mean diff = {mean_diff:+.4f} (>= +5pp: {mean_diff >= P1_THRESHOLD}); "
          f"all CIs excl 0: {all_excl0}  => P1 {'PASS' if p1['verdict'] else 'FAIL'}")

    # ---- P2 ----
    print("\n=== P2  guardrail SAC >= 0.507 ===")
    p2 = {"threshold": P2_THRESHOLD, "per_seed": {}}
    for r in match_seeds(SAC_RUNS):
        ok = passes[r] >= P2_THRESHOLD
        p2["per_seed"][r] = {"pass": passes[r], "ok": ok}
        print(f"  {r}: {passes[r]:.4f} {'OK' if ok else 'FAIL'}")
    p2["verdict"] = all(v["ok"] for v in p2["per_seed"].values())
    print(f"  => P2 {'PASS' if p2['verdict'] else 'FAIL'}")

    # ---- P3 ----
    print("\n=== P3  sacR vs mu16 (attribution) ===")
    p3 = {"per_seed": {}}
    for r in match_seeds(SACR_RUNS):
        s = seed_of(r)
        pt, lo, hi = paired_ci(sacrs[r], base[f"mu16_s{s}"], rng)
        excl0 = not (lo <= 0 <= hi)
        p3["per_seed"][r] = {"diff": pt, "ci": [lo, hi], "ci_excludes_0": excl0}
        print(f"  {r} - mu16_s{s}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] excl0={excl0}")
    mean_r = sum(v["diff"] for v in p3["per_seed"].values()) / 3
    p3["mean_diff"] = mean_r
    # attribution holds if R is NOT significantly better than mu16
    p3["verdict"] = bool(mean_r <= 0 or not all(v["ci_excludes_0"] for v in p3["per_seed"].values()))
    print(f"  mean diff = {mean_r:+.4f} => P3 {'OK (R not better than mu16)' if p3['verdict'] else 'VIOLATED (R also better -> P1 downgraded)'}")

    # descriptive: sacS vs SAC
    print("\n=== descriptive  sacS vs SAC, fixed arms vs mu16 ===")
    desc = {}
    for r in match_seeds(SACS_RUNS):
        s = seed_of(r)
        pt, lo, hi = paired_ci(sacs[r], sac[f"sac_s{s}"], rng)
        desc[f"sacS_s{s}-sac_s{s}"] = {"diff": pt, "ci": [lo, hi]}
        print(f"  sacS_s{s} - sac_s{s}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for r in match_seeds(FIX_RUNS):
        s = seed_of(r)
        pt, lo, hi = paired_ci(fixed[r], base[f"mu16_s{s}"], rng)
        desc[f"{r}-mu16_s{s}"] = {"diff": pt, "ci": [lo, hi]}
        print(f"  {r} - mu16_s{s}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # ---- P5' ----
    print("\n=== P5'  trajectory diagnostics ===")
    p5 = {}
    for r in SAC_RUNS:
        tab = cycle_table(r)
        post = [t for t in tab if t["cycle"] >= 2]
        stages = {
            "early_r2_3": [t["trigger"] for t in post if t["cycle"] in (2, 3)],
            "mid_r4_5": [t["trigger"] for t in post if t["cycle"] in (4, 5)],
            "late_r6_7": [t["trigger"] for t in post if t["cycle"] in (6, 7)],
        }
        rates = {k: sum(v) / len(v) if v else None for k, v in stages.items()}
        p5[r] = {"cycles": tab, "trigger_rate_by_stage": rates,
                 "cap_hits": sum(1 for t in post if t["plan"] == 32),
                 "floor_hits": sum(1 for t in post if t["plan"] == 4),
                 "wall_hours_2comp": round(wall_hours(r), 3),
                 "planned_updates_post_burnin": sum(t["plan"] for t in post),
                 "executed_updates_post_burnin": sum(t["executed"] for t in post)}
        t_ = p5[r]
        print(f"  {r}: plans={[t['plan'] for t in tab]} wall={t_['wall_hours_2comp']}h "
              f"stages={ {k: round(v,2) for k,v in rates.items() if v is not None} } "
              f"exec/planned={t_['executed_updates_post_burnin']}/{t_['planned_updates_post_burnin']}")
    for r in SAC_RUNS + SACS_RUNS + SACR_RUNS + FIX_RUNS:
        p5.setdefault("walls", {})[r] = round(wall_hours(r), 3)

    result = {"p1": p1, "p2": p2, "p3": p3, "descriptive": desc, "p5": p5,
              "passes_last_segment": passes, "baseline_mean": base_mean}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("\nwrote", args.json)


if __name__ == "__main__":
    main()
