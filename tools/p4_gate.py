#!/usr/bin/env python
"""P4 cost-gate evaluator (v2 freeze doc section 5, two-branch pre-registration).

Judges the mu=32 / mu=64 cost probes against the frozen band: per-cycle wall
time within 890 s +/- 25% (i.e. [667.5, 1112.5] s), based on F2 (update cost
is flat in mu because the optimizer always sees Br=4096 samples per cycle).

PASS  -> all arms keep R=8 cycles (sampling/wall-clock double alignment).
FAIL  -> stop before the smoke gate; re-plan the upper rungs (no silent
         reduction of cycle counts).

Pure stdlib; run with any python >= 3.10. Exits 0 on PASS, 3 on FAIL, 2 when
the inputs are missing/incomplete (gate cannot be judged yet).

Usage: python tools/p4_gate.py runs/p4_mu32 runs/p4_mu64 [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BAND_LOW_S = 667.5   # 890 * 0.75
BAND_HIGH_S = 1112.5  # 890 * 1.25


def cycle_walls(run_dir: Path) -> list[float]:
    """Per-cycle wall seconds: sum of booked rollout+update ms per rollout."""
    rows = []
    with open(run_dir / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    cycles: dict[int, float] = {}
    for r in rows:
        ri = int(r["rollout_idx"])
        ms = float(r.get("wall_ms_rollout") or 0.0) + float(r.get("wall_ms_update") or 0.0)
        cycles[ri] = cycles.get(ri, 0.0) + ms
    return [cycles[k] / 1000.0 for k in sorted(cycles)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {"band_s": [BAND_LOW_S, BAND_HIGH_S], "runs": {}}
    ok = True
    for rd in args.run_dirs:
        m = rd / "metrics.jsonl"
        if not m.exists():
            report["runs"][str(rd)] = {"status": "missing"}
            ok = False
            continue
        walls = cycle_walls(rd)
        in_band = [BAND_LOW_S <= w <= BAND_HIGH_S for w in walls]
        report["runs"][str(rd)] = {
            "status": "ok",
            "n_cycles": len(walls),
            "cycle_wall_s": [round(w, 1) for w in walls],
            "in_band": in_band,
            "mean_s": round(sum(walls) / len(walls), 1) if walls else None,
        }
        if len(walls) < 2 or not all(in_band):
            ok = False

    report["verdict"] = "PASS" if ok else "FAIL"
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
