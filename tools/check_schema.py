"""Validate a run's metrics file and print the numbers a claim would rest on.

This is the gate between "I ran something" and "I have a record": it refuses a
file whose rows are missing fields, whose (rollout, mu_step) keys repeat (the
signature of a botched resume), or whose cumulative counters go backwards.

    python tools/check_schema.py runs/mu8
    python tools/check_schema.py runs/mu8 --strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import metrics as metrics_mod  # noqa: E402


def check(run_dir: Path, *, strict: bool = False) -> tuple[bool, list[str]]:
    problems: list[str] = []
    mpath = run_dir / "metrics.jsonl"
    if not mpath.exists():
        return False, [f"missing {mpath}"]

    rows = metrics_mod.read_metrics(mpath)
    if not rows:
        return False, [f"{mpath} is empty"]

    for i, rec in enumerate(rows):
        try:
            metrics_mod.check_record(rec, allow_extra=True)
        except metrics_mod.SchemaError as e:
            problems.append(f"row {i + 1}: {e}")

    keys = [(r["rollout_idx"], r["mu_step"]) for r in rows]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        problems.append(f"duplicate (rollout_idx, mu_step) keys: {dup[:5]}")

    r_idx = [r["rollout_idx"] for r in rows]
    if r_idx != sorted(r_idx):
        problems.append("rollout_idx is not non-decreasing (rows out of order)")

    cum = [r["responses_cum"] for r in rows]
    if any(b < a for a, b in zip(cum, cum[1:])):
        problems.append("responses_cum decreases (cost accounting broken by a resume)")

    mu_set = {r["mu"] for r in rows}
    if len(mu_set) > 1:
        problems.append(f"file mixes arms: mu values {sorted(mu_set)}")

    # informational: on-policy rows must show zero staleness
    first = [r for r in rows if r["mu_step"] == 0]
    if first:
        max_kl0 = max(abs(r["kl_per_rollout_policy"]) for r in first)
        if max_kl0 > 1e-4:
            msg = (f"mu_step=0 rows show |kl| up to {max_kl0:.2e}: the 'rollout "
                   "policy' log-probs were not captured before the first update")
            if strict:
                problems.append(msg)
            else:
                print(f"[warn] {msg}")

    return (not problems), problems


def summarise(run_dir: Path) -> dict:
    rows = metrics_mod.read_metrics(run_dir / "metrics.jsonl")
    if not rows:
        return {}
    last = rows[-1]
    wall_h = sum(r["wall_ms_rollout"] + r["wall_ms_update"] for r in rows) / 3_600_000.0
    return {
        "rows": len(rows),
        "arm": last["arm"],
        "seed": last["seed"],
        "rollouts": last["rollout_idx"] + 1,
        "mu": last["mu"],
        "pass_rate_last": last["pass_rate"],
        "kl_last": last["kl_per_rollout_policy"],
        "drift_d_last": last["drift_d"],
        "responses_cum": last["responses_cum"],
        "wall_hours": round(wall_h, 4),
        "cost_cny_cum": last["cost_cny_cum"],
        "flops_cum": last["flops_cum"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="+")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ok_all = True
    out = {}
    for d in a.run_dir:
        p = Path(d)
        ok, problems = check(p, strict=a.strict)
        ok_all &= ok
        out[p.name] = {"ok": ok, "problems": problems, "summary": summarise(p)}
        if not a.json:
            tag = "OK  " if ok else "FAIL"
            print(f"[{tag}] {p}")
            for pr in problems:
                print(f"        - {pr}")
            s = out[p.name]["summary"]
            if s:
                print(f"        rows={s['rows']} arm={s['arm']} seed={s['seed']} "
                      f"pass={s['pass_rate_last']:.3f} kl={s['kl_last']:+.2e} "
                      f"d={s['drift_d_last']} cost=¥{s['cost_cny_cum']:.2f}")
    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
