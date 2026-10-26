#!/usr/bin/env python3
"""Emit one frame of campaign telemetry as a single JSON line on stdout.

Read-only: opens metrics.jsonl, reads nvidia-smi, `ps`, and the guard log.
Never writes to the repo. Runs on the box under /opt/conda/bin/python; the
local dashboard polls it every 15 s (tools/live_dashboard.sh).

Why a separate feed instead of pointing the dashboard at the files over SSH:
the dashboard needs derived state (which run is live, how far into it, what the
guard is doing) that would otherwise be recomputed client-side from four
sources on every poll. One JSON frame keeps the polling script dumb.

    /opt/conda/bin/python tools/dash_feed.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only the fields the dashboard draws. Shipping all 42 fields would triple the
# frame size for numbers nobody looks at during a campaign.
KEEP = [
    "rollout_idx", "mu", "seed", "resp_len_mean", "pass_rate", "reward_mean",
    "grad_noise_sigma", "drift_d", "loss", "ratio_mean", "clip_frac_any",
    "grad_norm_raw", "grad_norm_clipped", "clip_grad_mass_share",
    "zero_var_group_frac", "wall_ms_rollout", "wall_ms_old_logp",
    "wall_ms_update", "cost_cny_cum", "responses_cum", "ts", "n_groups",
    "n_forward_chunks",
]


def sh(cmd: str, timeout: int = 15) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:                                  # noqa: BLE001
        return ""


def main() -> int:
    now = time.time()
    frame: dict = {
        "epoch": now,
        "utc": datetime.fromtimestamp(now, timezone.utc).strftime("%FT%TZ"),
        "bjt": datetime.fromtimestamp(now, timezone.utc).astimezone().strftime("%F %T"),
    }

    # Container start, from init's own start time: this is the billing clock,
    # not the host's uptime (the host has been up for days).
    try:
        frame["container_start"] = os.stat("/proc/1").st_ctime
    except OSError:
        frame["container_start"] = None

    total, used, free = "", "", ""
    v = sh("nvidia-smi --query-gpu=memory.total,memory.used,memory.free "
           "--format=csv,noheader,nounits").split(",")
    if len(v) >= 3:
        total, used, free = (x.strip() for x in v[:3])
    frame["vram"] = {
        "total": int(total) if total.isdigit() else None,
        "used": int(used) if used.isdigit() else None,
        "free": int(free) if free.isdigit() else None,
    }
    apps = []
    for line in sh("nvidia-smi --query-compute-apps=pid,used_memory "
                   "--format=csv,noheader").splitlines():
        if "," in line:
            pid, mib = (x.strip() for x in line.split(",", 1))
            # query-compute-apps ignores nounits and returns "18772 MiB"
            apps.append({"pid": pid, "mib": int(mib.split()[0])
                         if mib.split() and mib.split()[0].isdigit() else None})
    frame["apps"] = apps

    procs = {}
    for name, pat in (("run_9", "[r]un_9.sh"), ("train", "[s]rc/train.py"),
                      ("engine", "[E]ngineCore"), ("guard", "[v]ram_guard.sh")):
        procs[name] = sh(f"ps -eo pid,etime,args | grep -E '{pat}' | grep -v grep")
    frame["procs"] = procs

    tag = None
    parts = procs["train"].replace("\n", " ").split()
    for i, tok in enumerate(parts):
        if tok == "--tag" and i + 1 < len(parts):
            tag = parts[i + 1].strip()
    frame["current_tag"] = tag

    runs = []
    metrics = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", "mu*_s*"))):
        if not os.path.isdir(d):
            continue
        t = os.path.basename(d)
        state = os.path.join(d, "state.pt")
        mpath = os.path.join(d, "metrics.jsonl")
        rows = 0
        if os.path.isfile(mpath):
            with open(mpath) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rows += 1
                    rec["tag"] = t
                    metrics.append({k: rec[k] for k in KEEP if k in rec} | {"tag": t})
        runs.append({
            "tag": t,
            "summary": os.path.isfile(os.path.join(d, "summary.json")),
            "rows": rows,
            "npz": len(glob.glob(os.path.join(d, "rollouts", "r*.npz"))),
            "adapters": len(glob.glob(os.path.join(d, "adapters", "r*"))),
            "has_state": os.path.isfile(state),
            "state_age_s": round(now - os.stat(state).st_mtime) if os.path.isfile(state) else None,
        })
    frame["runs"] = runs
    frame["metrics"] = metrics

    frame["guard_tail"] = sh(f"tail -8 '{ROOT}/runs/_logs/vram_guard.log'").splitlines()
    frame["halt"] = {
        "flag": os.path.isfile(os.path.join(ROOT, "runs", "_vram_guard_halt.flag")),
        "report": os.path.isfile(os.path.join(ROOT, "runs", "_vram_guard_halt.md")),
    }
    if tag:
        frame["log_tail"] = sh(f"tail -3 '{ROOT}/runs/_logs/{tag}.log'").splitlines()
    else:
        frame["log_tail"] = []

    try:
        st = os.statvfs(ROOT)
        frame["disk_free_gib"] = round(st.f_bavail * st.f_frsize / 2**30, 1)
    except OSError:
        frame["disk_free_gib"] = None

    commit_path = os.path.join(ROOT, "COMMIT")
    frame["commit"] = (open(commit_path).read().strip()
                       if os.path.isfile(commit_path) else None)

    sys.stdout.write(json.dumps(frame, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
