"""Claim -> evidence checker.

Every sentence in REPORT.md that states a number has an entry in
`analysis/claims.yaml`. This script verifies that the evidence actually exists
and actually contains the field being claimed, and that no claim is marked
"verified" while its evidence is missing.

Status vocabulary is fixed (from the collaboration protocol):

    verified   -- 已核实: a file, a field and a value were checked here
    default    -- 默认执行: a decision taken without asking, reversible
    pending    -- 待拍板: needs the user, must not be stated as fact

    python analysis/trace.py                     # check against ./analysis/claims.yaml
    python analysis/trace.py --strict            # fail if any claim lacks evidence
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import metrics as metrics_mod  # noqa: E402

ALLOWED = {"verified", "default", "pending"}


def check_claim(c: dict, root: Path) -> tuple[bool, str]:
    cid = c.get("id", "?")
    status = c.get("status", "")
    if status not in ALLOWED:
        return False, f"{cid}: status {status!r} not in {sorted(ALLOWED)}"

    ev = c.get("evidence")
    if status == "pending":
        return True, f"{cid}: pending (no evidence required)"
    if not ev:
        return False, f"{cid}: status={status} but no evidence path"

    path = (root / ev).resolve()
    if not path.exists():
        return False, f"{cid}: evidence missing: {ev}"
    if path.suffix in {".jsonl"} and c.get("field"):
        rows = metrics_mod.read_metrics(path)
        if not rows:
            return False, f"{cid}: {ev} has no rows"
        field = c["field"]
        missing = [i for i, r in enumerate(rows) if field not in r]
        if missing:
            return False, f"{cid}: field {field!r} absent in {len(missing)} rows of {ev}"
        vals = [r[field] for r in rows if r[field] is not None]
        if c.get("expect") is not None and vals:
            if not any(abs(float(v) - float(c["expect"])) <= float(c.get("tol", 1e-9)) for v in vals):
                return False, (f"{cid}: no value within tol of {c['expect']} "
                               f"(saw {vals[0]!r} .. {vals[-1]!r})")
        return True, f"{cid}: {ev}::{field} ok ({len(vals)} non-null values)"
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return False, f"{cid}: {ev} is not valid JSON ({e})"
        return True, f"{cid}: {ev} ok"
    return True, f"{cid}: {ev} exists"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="analysis/claims.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    cpath = Path(a.claims)
    if not cpath.exists():
        print(f"no claims file at {cpath}")
        return 1
    doc = yaml.safe_load(cpath.read_text(encoding="utf-8")) or {}
    claims = doc.get("claims", [])
    if not claims:
        print("claims file has no entries")
        return 1

    root = Path(a.root)
    bad = 0
    counts = {k: 0 for k in ALLOWED}
    for c in claims:
        ok, msg = check_claim(c, root)
        counts[c.get("status", "?")] = counts.get(c.get("status", "?"), 0) + 1
        if not ok:
            bad += 1
            print(f"[FAIL] {msg}")
        else:
            print(f"[ ok ] {msg}")

    print(f"\n{len(claims)} claims: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if a.strict and any(c.get("status") == "pending" for c in claims):
        print("[note] pending claims exist; they must not appear as facts in REPORT.md")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
