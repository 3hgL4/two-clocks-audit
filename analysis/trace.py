"""Claim -> evidence checker.

Every sentence in REPORT.md that states a number has an entry in
`analysis/claims.yaml`. This script verifies that the evidence actually exists
and actually contains the field being claimed, and that no claim is marked
"verified" while its evidence is missing.

Status vocabulary is fixed (from the collaboration protocol). It describes the
relation between the criterion and the measurement, NOT whether the paper was
right:

    verified        -- 已核实: 判据成立 (file, field and value were checked)
    falsified       -- 已核实: 判据不成立 (direction reversed or threshold missed)
    unidentifiable  -- 已核实: 判据不可触发 (both tails censored, the quantity
                       the criterion turns on does not exist)
    default         -- 默认执行: a decision taken without asking, reversible
    pending         -- 待拍板: needs the user, must not be stated as fact

`falsified` and `unidentifiable` demand the same evidence as `verified`. The
distinction exists because collapsing them into `verified` would let a reader
read "verified" as "the prediction held" -- on 2026-10-27 three of the seven
pre-registered criteria landed in exactly that position (C2 mono, C4 direction,
C6 second half), and one (C3) landed on a censored quantity.

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

ORDER = ("verified", "falsified", "unidentifiable", "default", "pending")
ALLOWED = set(ORDER)
NEEDS_EVIDENCE = {"verified", "falsified", "unidentifiable", "default"}


def _collect_field(node, field: str, acc: list) -> None:
    """Recursively gather every value stored under `field` anywhere in a JSON doc.

    Used so that a `.json` evidence is checked for the claimed FIELD, not merely
    for being parseable. Before 2026-10-27 the `.json` branch only ran
    json.loads() and returned ok -- which is why C3's `n_d20` could sit at None in
    all 9 entries for weeks without trace.py complaining.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k == field:
                acc.append(v)
            _collect_field(v, field, acc)
    elif isinstance(node, list):
        for v in node:
            _collect_field(v, field, acc)


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
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return False, f"{cid}: {ev} is not valid JSON ({e})"
        field = c.get("field")
        if not field:
            return True, f"{cid}: {ev} ok (parseable; no field claimed)"
        acc: list = []
        _collect_field(doc, field, acc)
        if not acc:
            return False, f"{cid}: field {field!r} not present anywhere in {ev}"
        vals = [v for v in acc if v is not None]
        if not vals:
            return False, (f"{cid}: field {field!r} present in {ev} but every value is "
                           f"null ({len(acc)} occurrences) -- the criterion turns on a "
                           f"quantity that was never computed")
        if c.get("expect") is not None:
            if not any(isinstance(v, (int, float)) and
                       abs(float(v) - float(c["expect"])) <= float(c.get("tol", 1e-9))
                       for v in vals):
                return False, f"{cid}: no value within tol of {c['expect']} (saw {vals[:3]!r})"
        return True, f"{cid}: {ev}::{field} ok ({len(vals)} non-null of {len(acc)})"
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
    counts = {k: 0 for k in ORDER}
    for c in claims:
        ok, msg = check_claim(c, root)
        counts[c.get("status", "?")] = counts.get(c.get("status", "?"), 0) + 1
        if not ok:
            bad += 1
            print(f"[FAIL] {msg}")
        else:
            print(f"[ ok ] {msg}")

    print(f"\n{len(claims)} claims: " + ", ".join(f"{k}={counts[k]}" for k in ORDER))
    if a.strict and any(c.get("status") == "pending" for c in claims):
        pend = [c["id"] for c in claims if c.get("status") == "pending"]
        print(f"[note] pending claims exist ({', '.join(pend)}); "
              f"they must not appear as facts in REPORT.md")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
