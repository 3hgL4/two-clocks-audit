"""Group-grain analysis on what the archive actually kept: 72 groups, not 36,864.

`samples.jsonl` stored `log_rollout_samples: 8` rows per rollout, and responses
are laid out prompt-major, so those eight rows are *one prompt's whole group*.
Nine runs x 8 rollouts = 72 groups. That is tiny, and it is the only group-grain
evidence the completed runs have -- which is the argument for `src/prompt_log.py`
existing before the next run rather than after it.

What 72 groups still licenses:

  1. the all-correct / all-wrong rate by arm, against the iid counterfactual at
     the arm's own pass rate (the counterfactual is a computation, not a claim
     about the data);
  2. the intraclass correlation, which is what says whether "4096 responses"
     bought 4096 independent observations or many fewer;
  3. a paired comparison across arms: the three arms of one seed see the same
     prompt in the same rollout, so the same 24 prompts are scored under mu=1,
     mu=8 and mu=16.

What it cannot do: one prompt per rollout means no difficulty stratification, no
first-sight vs repeat contrast, and an ICC whose standard error is roughly 1/sqrt(24).

    python analysis/group_grain.py
    python analysis/group_grain.py --json analysis/out/group_grain.json

stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import prompt_log  # noqa: E402

ARMS = ("mu1", "mu8", "mu16")


def load(runs_dir: str) -> dict[str, dict[int, dict[int, list[dict]]]]:
    """arm -> seed -> rollout_idx -> the 8 rows of that rollout's first group."""
    out: dict[str, dict[int, dict[int, list[dict]]]] = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "mu*_s*"))):
        arm, seed_s = os.path.basename(d).split("_")
        if arm not in ARMS:
            continue
        p = os.path.join(d, "samples.jsonl")
        if not os.path.exists(p):
            continue
        per: dict[int, list[dict]] = {}
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            per.setdefault(int(r["rollout_idx"]), []).append(r)
        out.setdefault(arm, {})[int(seed_s[1:])] = per
    return out


def arm_global_pass(runs_dir: str, arm: str) -> float | None:
    """The arm's pass rate over all 4096-response rollouts, for the iid contrast.

    The 24 prompts in this file are a sample: their own pass rate (0.208 / 0.401 /
    0.401) differs from the arm's (0.269 / 0.459 / 0.517) by up to 6pp, and
    p**G is very sensitive to p, so both baselines are reported.
    """
    vals = []
    for d in sorted(glob.glob(os.path.join(runs_dir, f"{arm}_s*"))):
        p = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(p):
            continue
        last = {}
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                last[int(r["rollout_idx"])] = float(r["pass_rate"])
        if last:
            vals.append(last[max(last)])
    return sum(vals) / len(vals) if vals else None


def iid_expected(G: int, p: float, kind: str) -> float:
    """P(all G identical) if the G generations were independent draws at rate p."""
    return p ** G if kind == "correct" else (1.0 - p) ** G


def icc_ci_halfwidth(n_groups: int) -> float:
    """Crude 95% half-width for an ICC estimated from n_groups groups.

    Fisher-z on the ICC is not exact here, but the order of magnitude is what
    matters: with 24 groups the interval is so wide that an ICC of 0.2 and 0.5
    are not distinguishable, and the report must say so instead of quoting a point.
    """
    return 1.96 / math.sqrt(max(1, n_groups - 3))


def build(runs_dir: str) -> dict:
    data = load(runs_dir)
    G = 8
    doc: dict = {
        "runs_dir": runs_dir,
        "group_size": G,
        "caveat": ("one prompt per rollout (samples.jsonl keeps only the first 8 rows, "
                   "which is one group); 72 groups total across nine runs"),
        "by_arm": {},
        "paired_same_prompt": {},
    }

    for arm in ARMS:
        seeds = sorted(data.get(arm, {}))
        bits: list[bool] = []
        rewards: list[float] = []
        all_right = all_wrong = 0
        n_groups = 0
        for s in seeds:
            for r in sorted(data[arm][s]):
                rows = data[arm][s][r][:G]
                if len(rows) < G:
                    continue
                c = [bool(x["correct"]) for x in rows]
                w = [float(x["reward"]) for x in rows]
                bits.extend(c)
                rewards.extend(w)
                k = sum(c)
                all_right += (k == G)
                all_wrong += (k == 0)
                n_groups += 1
        if not n_groups:
            continue
        gs = prompt_log.group_summary(bits, G)
        p = sum(bits) / len(bits)
        pg = arm_global_pass(runs_dir, arm)
        doc["by_arm"][arm] = {
            "n_groups": n_groups,
            "pass_rate": round(p, 4),
            "pass_rate_global": None if pg is None else round(pg, 4),
            "all_correct": all_right,
            "all_wrong": all_wrong,
            "all_correct_frac": round(all_right / n_groups, 4),
            "all_wrong_frac": round(all_wrong / n_groups, 4),
            "iid_all_correct_expected": round(iid_expected(G, p, "correct"), 4),
            "iid_all_wrong_expected": round(iid_expected(G, p, "wrong"), 4),
            "iid_all_correct_expected_global": (
                None if pg is None else round(iid_expected(G, pg, "correct"), 4)),
            "iid_all_wrong_expected_global": (
                None if pg is None else round(iid_expected(G, pg, "wrong"), 4)),
            "icc": None if gs["group_icc"] is None else round(gs["group_icc"], 3),
            "icc_ci95_halfwidth": round(icc_ci_halfwidth(n_groups), 3),
            # the field the metrics row already carries, restricted to these groups
            "reward_zero_var_frac": round(
                sum(1 for i in range(n_groups)
                    if len(set(rewards[i * G:(i + 1) * G])) == 1) / n_groups, 4),
            "correct_zero_var_frac": round((all_right + all_wrong) / n_groups, 4),
            "group_pass_hist": {
                str(k): sum(1 for i in range(n_groups)
                            if sum(bits[i * G:(i + 1) * G]) == k)
                for k in range(G + 1)
            },
        }

    # paired: one seed, one rollout -> the same prompt under all three arms
    common = set(data.get("mu1", {}))
    for a in ("mu8", "mu16"):
        common &= set(data.get(a, {}))
    pairs = []
    for s in sorted(common):
        rollouts = sorted(set(data["mu1"][s]) & set(data["mu8"][s]) & set(data["mu16"][s]))
        for r in rollouts:
            row = {}
            for arm in ARMS:
                rows = data[arm][s][r][:G]
                row[arm] = sum(1 for x in rows if x["correct"])
            row["prompt_head"] = data["mu1"][s][r][0]["item"][:60]
            row["seed"] = s
            row["rollout_idx"] = r
            pairs.append(row)
    doc["paired_same_prompt"] = {
        "n_prompts": len(pairs),
        "note": ("the three arms of one seed see the same prompt in the same rollout, "
                 "so this is a paired comparison of group-level successes out of 8"),
        "group_success_mean": {
            arm: round(sum(p[arm] for p in pairs) / max(1, len(pairs)), 3) for arm in ARMS
        },
        "rows": pairs,
    }
    return doc


def print_table(doc: dict) -> None:
    print(f"{doc['caveat']}\n")
    hdr = (f"{'arm':5s} {'groups':>6s} {'pass':>6s} {'glob':>6s} {'allR':>5s} {'iidR':>6s} "
           f"{'allW':>5s} {'iidW':>6s} {'ICC':>7s} {'zvR':>6s} {'zvC':>6s}")
    print(hdr)
    for arm, v in doc["by_arm"].items():
        icc_s = "    n/a" if v["icc"] is None else f"{v['icc']:7.3f}"
        gp = "   n/a" if v["pass_rate_global"] is None else f"{v['pass_rate_global']:6.3f}"
        print(f"{arm:5s} {v['n_groups']:6d} {v['pass_rate']:6.3f} {gp} "
              f"{v['all_correct']:5d} {v['iid_all_correct_expected']:6.3f} "
              f"{v['all_wrong']:5d} {v['iid_all_wrong_expected']:6.3f} "
              f"{icc_s} "
              f"{v['reward_zero_var_frac']:6.3f} {v['correct_zero_var_frac']:6.3f}")
    print("\n  iidR/iidW use this sample's own pass rate; the same expectation at the")
    print("  arm's global pass rate (glob) is 4-5x smaller because p**8 is steep.")
    for arm, v in doc["by_arm"].items():
        if v["iid_all_wrong_expected_global"] is None:
            continue
        print(f"  {arm:5s} at p={v['pass_rate_global']:.3f}: "
              f"iid all-right {v['iid_all_correct_expected_global']:.4f}, "
              f"iid all-wrong {v['iid_all_wrong_expected_global']:.4f}")
    print("\n(iidR/iidW = expected fractions if the 8 generations were independent draws "
          "at that arm's own pass rate; zvR = reward-defined zero-variance share, "
          "zvC = correctness-defined, which is what C8-era rows will carry)")
    print("\ngroup pass histogram (successes out of 8)")
    for arm, v in doc["by_arm"].items():
        print(f"  {arm:5s} " + " ".join(f"{v['group_pass_hist'][str(k)]:3d}" for k in range(9)))
    print("         " + " ".join(f"{k:3d}" for k in range(9)))

    pp = doc["paired_same_prompt"]
    print(f"\npaired by prompt ({pp['n_prompts']} prompts seen by all three arms)")
    print("  mean successes out of 8: " +
          ", ".join(f"{a}={pp['group_success_mean'][a]}" for a in ARMS))


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
