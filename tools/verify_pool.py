"""Verify the frozen pool against MANIFEST.json and configs, on any OS.

Why this exists: the sha256 gate in src/data.py compares bytes on disk with a
recorded hash. On Windows the writers used to emit CRLF, so the hash matched
locally and failed on Linux (DEVLOG D-09). A gate is only trustworthy if it is
run in the environment that will actually execute the training, so this script
is meant to be the first command on the cloud box:

    python tools/verify_pool.py            # checks data/MANIFEST.json + configs

Exit code 0 = every hash and byte count agrees, 1 = mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

RE_SHA = re.compile(r'sha256:\s*"([0-9a-f]{64})"')


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--config", default="configs/base.yaml")
    a = ap.parse_args()

    root = Path(a.root)
    manifest_path = root / "data/MANIFEST.json"
    if not manifest_path.exists():
        print(f"[fail] {manifest_path} missing")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cfg_text = (root / a.config).read_text(encoding="utf-8")
    m = RE_SHA.search(cfg_text)
    cfg_sha = m.group(1) if m else None

    ok = True
    for rel, rec in manifest["files"].items():
        p = root / rel
        if not p.exists():
            print(f"[fail] {rel} missing")
            ok = False
            continue
        raw = p.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        crlf = raw.count(b"\r\n")
        good_hash = got == rec["sha256"]
        good_size = len(raw) == rec["bytes"]
        ok &= good_hash and good_size
        print(f"{'[ok]  ' if good_hash and good_size else '[fail]'} {rel}")
        print(f"       manifest {rec['sha256']}  ({rec['bytes']} bytes)")
        print(f"       on disk  {got}  ({len(raw)} bytes)")
        if crlf:
            print(f"       [warn] {crlf} CRLF line endings: a Linux checkout will hash differently")
        if cfg_sha is not None:
            same_cfg = cfg_sha == rec["sha256"]
            ok &= same_cfg
            print(f"       {a.config} {'matches' if same_cfg else 'DIFFERS: ' + cfg_sha}")

    print("\nresult:", "consistent" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
