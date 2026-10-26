#!/usr/bin/env python3
"""Verify a local ``runs/`` archive against the manifests generated on the box.

The manifests are produced on the machine that ran the campaign, before its data
disk is released::

    cd <box>/two-clocks-audit/runs
    find mu*_s* -type f ! -name state.pt -print0 | sort -z | xargs -0 sha256sum \\
        > <box>/MANIFEST.nostate.sha256
    find mu*_s* -name state.pt -print0 | sort -z | xargs -0 sha256sum \\
        > <box>/MANIFEST.state.sha256
    find mu*_s* -type f -printf '%s\\t%p\\n' | sort -k2 > <box>/MANIFEST.sizes.tsv

They are plain ``sha256sum`` output so that ``sha256sum -c`` also works; this
script exists for the parts ``-c`` cannot do: byte-size screening before hashing
(a truncated transfer then costs no hash time), a per-run completeness table, and
one exit code for the whole archive.

Usage::

    python tools/verify_archive.py                       # everything
    python tools/verify_archive.py --small-only          # skip adapter/state
    python tools/verify_archive.py --skip-state          # adapters, no state.pt
    python tools/verify_archive.py --manifests notes/2026-10-26_pull --runs runs

Exit code 0 only if every manifest entry that was checked matched.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

BIG = ("adapter_model.safetensors", "state.pt")


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_sha(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        out[rel.strip()] = digest.strip()
    return out


def read_sizes(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        size, rel = line.split("\t", 1)
        out[rel.strip()] = int(size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default="notes/2026-10-26_pull")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--small-only", action="store_true",
                    help="skip every file whose name is in BIG")
    ap.add_argument("--skip-state", action="store_true",
                    help="skip state.pt only")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    mdir, root = Path(a.manifests), Path(a.runs)
    man = read_sha(mdir / "MANIFEST.nostate.sha256")
    man.update(read_sha(mdir / "MANIFEST.state.sha256"))
    sizes = read_sizes(mdir / "MANIFEST.sizes.tsv")
    if not man:
        print(f"no manifests under {mdir}", file=sys.stderr)
        return 2

    skip = set()
    if a.small_only:
        skip = set(BIG)
    elif a.skip_state:
        skip = {"state.pt"}

    ok: list[str] = []
    bad: list[tuple[str, str]] = []
    missing: list[str] = []
    skipped: list[str] = []

    for rel, want in sorted(man.items()):
        name = rel.rsplit("/", 1)[-1]
        if name in skip:
            skipped.append(rel)
            continue
        p = root / rel
        if not p.is_file():
            missing.append(rel)
            continue
        expected = sizes.get(rel)
        got_size = p.stat().st_size
        if expected is not None and got_size != expected:
            bad.append((rel, f"size {got_size} != {expected}"))
            continue
        got = sha256(p)
        if got == want:
            ok.append(rel)
        else:
            bad.append((rel, f"sha256 {got[:16]} != {want[:16]}"))

    print(f"manifests   : {len(man)} entries from {mdir}")
    print(f"checked ok  : {len(ok)}")
    print(f"mismatch    : {len(bad)}")
    print(f"missing     : {len(missing)}")
    if skipped:
        print(f"skipped     : {len(skipped)}")

    if bad and not a.quiet:
        print("\n-- mismatch --")
        for rel, why in bad[:40]:
            print(f"  {rel}: {why}")

    # per-run completeness, so "which run is short" needs no second pass
    runs = sorted({rel.split("/", 1)[0] for rel in man})
    if runs and not a.quiet:
        print("\n-- per-run completeness --")
        for r in runs:
            keys = [k for k in man if k.startswith(r + "/")]
            have = sum(1 for k in keys if (root / k).is_file())
            big = sum(1 for k in keys if k.rsplit("/", 1)[-1] in BIG)
            big_have = sum(1 for k in keys
                           if k.rsplit("/", 1)[-1] in BIG and (root / k).is_file())
            print(f"  {r:9s} {have:3d}/{len(keys):3d} files"
                  f"   (incl. {big_have}/{big} big)")

    good = not bad and not missing
    print("\nRESULT:", "consistent" if good else "NOT consistent")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
