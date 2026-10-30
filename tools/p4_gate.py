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

Provenance: the JSON report embeds utc_time, host, gate_commit (from the
GATE_COMMIT env var; set it at deploy time to the script's git commit),
a sha256 of this script itself, and a sha256 of every input metrics.jsonl.
The frozen band is never derived from data at run time.

Accounting conventions this gate relies on (verified against the 2026-10-30
P4 probes; if src/train.py's wall bookkeeping ever changes, the per-row sum
caliber below MUST be re-checked in sync, otherwise this gate will silently
double-count or drop time):

1. wall_ms_rollout and wall_ms_old_logp are booked only on the mu_step==0
   row of each cycle (one sampling pass + one old-logp pass per cycle);
   every other row carries 0 in these two fields.
2. wall_ms_update is booked on every row with the same per-step value
   (p4_mu32: 15155.293 ms x 32 rows; p4_mu64: 7579.954 ms x 64 rows).
3. The per-row sum in cycle_walls() therefore equals the true cycle wall
   time only through this mixed convention: a once-per-cycle booking plus
   a per-row-constant booking. It is NOT a plain "sum of per-row timings".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

BAND_LOW_S = 667.5   # 890 * 0.75
BAND_HIGH_S = 1112.5  # 890 * 1.25
BAND_SOURCE = "历史 mu16 三 seed 三分量校准 840.4–939.2 s（runs/mu16_s0/s1/s2）"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cycle_walls(run_dir: Path) -> list[float]:
    """Per-cycle wall seconds: sum of booked rollout+update ms per rollout.

    See module docstring: this sum is valid only under the mixed booking
    convention documented there (rollout/old_logp once per cycle on the
    mu_step==0 row, update as a per-row constant).
    """
    rows = []
    with open(run_dir / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    cycles: dict[int, float] = {}
    for r in rows:
        ri = int(r["rollout_idx"])
        ms = (float(r.get("wall_ms_rollout") or 0.0)
              + float(r.get("wall_ms_old_logp") or 0.0)
              + float(r.get("wall_ms_update") or 0.0))
        cycles[ri] = cycles.get(ri, 0.0) + ms
    return [cycles[k] / 1000.0 for k in sorted(cycles)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    report: dict = {
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate_commit": os.environ.get("GATE_COMMIT", "unknown"),
        "gate_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "host": socket.gethostname(),
        "band_s": [BAND_LOW_S, BAND_HIGH_S],
        "band_source": BAND_SOURCE,
        "runs": {},
    }
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
            "metrics_sha256": sha256_file(m),
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
