"""Fetch a ModelScope model with parallel range requests.

Why this exists: from the rented box, `modelscope.snapshot_download` measured
~1.0 MB/s for the 3.44 GB Qwen3-1.7B-Base safetensors (~57 min), while a single
range request on the same CDN endpoint measured 6.3 MB/s and 4 parallel ranges
~22 MB/s. The SDK's downloader, not the network, was the bottleneck.

Files smaller than --chunk-threshold are fetched in one shot; larger ones are
split into --workers byte ranges written to .partN and concatenated, then the
sha256 from the ModelScope API is verified. The result is a plain directory that
`AutoModelForCausalLM.from_pretrained(path)` accepts directly.

    python tools/fetch_model.py --model Qwen/Qwen3-1.7B-Base \
        --out /root/rivermind-data/models/Qwen3-1.7B-Base --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://www.modelscope.cn/api/v1/models/{model}/repo/files"
FILE = "https://www.modelscope.cn/api/v1/models/{model}/repo?Revision=master&FilePath={path}"
UA = {"User-Agent": "two-clocks-audit/fetch_model"}


def list_files(model: str, revision: str = "master") -> list[dict]:
    r = requests.get(API.format(model=model), params={"Revision": revision, "Recursive": "true"},
                     headers=UA, timeout=30)
    r.raise_for_status()
    files = r.json()["Data"]["Files"]
    return [f for f in files if f.get("Type") == "blob"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_range(url: str, start: int, end: int, out: str, retries: int = 4) -> None:
    for attempt in range(retries):
        try:
            with requests.get(url, headers={**UA, "Range": f"bytes={start}-{end}"},
                              stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            got = os.path.getsize(out)
            want = end - start + 1
            if got == want:
                return
            print(f"    [warn] {os.path.basename(out)}: got {got} want {want}, retry {attempt + 1}")
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] {os.path.basename(out)}: {type(e).__name__} {e}, retry {attempt + 1}")
    raise RuntimeError(f"range {start}-{end} failed after {retries} attempts: {out}")


def fetch(model: str, meta: dict, out_dir: str, workers: int, threshold: int) -> None:
    rel = meta["Path"]
    size = int(meta["Size"])
    dst = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(dst) or out_dir, exist_ok=True)
    url = FILE.format(model=model, path=requests.utils.quote(rel))
    print(f"[file] {rel}  {size / 2**20:.1f} MiB")

    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print("       already present, skip")
    elif size <= threshold:
        get_range(url, 0, size - 1, dst)
    else:
        step = (size + workers - 1) // workers
        parts = []
        jobs = []
        for i in range(workers):
            s = i * step
            e = min(size - 1, s + step - 1)
            if s > e:
                break
            parts.append((s, e, dst + f".part{i}"))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for s, e, p in parts:
                jobs.append(ex.submit(get_range, url, s, e, p))
            for j in jobs:
                j.result()
        with open(dst, "wb") as f:
            for _, _, p in parts:
                with open(p, "rb") as pf:
                    while True:
                        block = pf.read(1 << 22)
                        if not block:
                            break
                        f.write(block)
        for _, _, p in parts:
            os.remove(p)

    got = os.path.getsize(dst)
    want_sha = (meta.get("Sha256") or "").lower()
    if got != size:
        raise RuntimeError(f"{rel}: size {got} != {size}")
    if want_sha:
        got_sha = sha256_file(dst)
        if got_sha != want_sha:
            raise RuntimeError(f"{rel}: sha256 mismatch\n  api  {want_sha}\n  disk {got_sha}")
        print(f"       sha256 ok {got_sha[:16]}...")
    else:
        print("       (api gave no sha256)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default="master")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-threshold", type=int, default=32 << 20)
    a = ap.parse_args()

    files = list_files(a.model, a.revision)
    total = sum(int(f["Size"]) for f in files)
    print(f"{len(files)} files, {total / 2**30:.2f} GiB -> {a.out}")
    os.makedirs(a.out, exist_ok=True)
    for meta in files:
        fetch(a.model, meta, a.out, a.workers, a.chunk_threshold)
    print("DONE", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
