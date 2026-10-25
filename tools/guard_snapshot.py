#!/usr/bin/env python3
"""Snapshot the campaign state so a human can decide, not a script (D-26).

`vram_guard.sh` used to stop the campaign and immediately relaunch it on
`cloud48g_lowmem.yaml`, which meant the 9 runs would finish on a config nobody
had approved, and the only evidence would be a line in a log. This tool runs
before anything is killed and writes down what is actually on disk: which runs
are complete, which one is mid-flight, how far it got, and how the numbers that
drive memory (resp_len) are trending. Then the guard stops and exits.

Read-only. Safe to run while training is going. Writes two files:
  runs/_vram_guard_halt.md    human-readable
  runs/_vram_guard_halt.json  machine-readable (same facts)

    /opt/conda/bin/python tools/guard_snapshot.py
    /opt/conda/bin/python tools/guard_snapshot.py --out-md /tmp/x.md --out-json /tmp/x.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BJT = timezone(timedelta(hours=8))

# The fields a decision actually turns on. resp_len drives the training-side
# peak; zero_var_group_frac says the sampler is still alive (D-18/D-19);
# cost_cny_cum answers "what has this already cost".
METRIC_FIELDS = [
    "rollout_idx", "resp_len_mean", "pass_rate", "zero_var_group_frac",
    "grad_noise_sigma", "clip_grad_mass_share", "grad_norm_raw",
    "wall_ms_rollout", "wall_ms_old_logp", "wall_ms_update", "cost_cny_cum",
    "n_groups", "n_forward_chunks",
]


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception as exc:                      # noqa: BLE001 - best effort
        return f"<failed: {exc}>"


def vram() -> dict:
    out = sh("nvidia-smi --query-gpu=memory.total,memory.used,memory.free,"
             "utilization.gpu --format=csv,noheader,nounits")
    d = {"raw": out}
    parts = [p.strip() for p in out.split(",")]
    if len(parts) >= 4 and parts[0].isdigit():
        d.update(total_mib=int(parts[0]), used_mib=int(parts[1]),
                 free_mib=int(parts[2]), util_pct=int(parts[3]))
    d["compute_apps"] = sh(
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    return d


def procs() -> dict:
    pats = {"run_9": "[r]un_9.sh", "train": "[s]rc/train.py",
            "engine": "[E]ngineCore", "guard": "[v]ram_guard.sh"}
    return {k: sh(f"ps -eo pid,etime,args | grep -E '{p}' | grep -v grep")
            for k, p in pats.items()}


def tag_of(train_ps: str) -> str | None:
    parts = train_ps.split()
    for i, tok in enumerate(parts):
        if tok == "--tag" and i + 1 < len(parts):
            return parts[i + 1].strip()
    return None


def run_dir(tag: str) -> dict:
    d = os.path.join(ROOT, "runs", tag)
    info: dict = {"tag": tag, "dir": f"runs/{tag}", "exists": os.path.isdir(d)}
    if not info["exists"]:
        return info
    info["has_summary"] = os.path.isfile(os.path.join(d, "summary.json"))
    state = os.path.join(d, "state.pt")
    if os.path.isfile(state):
        st = os.stat(state)
        info["state_pt"] = {
            "size_mib": round(st.st_size / 2**20, 1),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc)
            .strftime("%FT%TZ"),
            "age_s": round(datetime.now(timezone.utc).timestamp() - st.st_mtime),
        }
    else:
        info["state_pt"] = None
    info["rollout_npz"] = len(glob.glob(os.path.join(d, "rollouts", "r*.npz")))
    # runs/<tag>/adapters/rNNN, one per rollout (D-22); 266 MiB each.
    info["adapter_snapshots"] = len(glob.glob(os.path.join(d, "adapters", "r*")))

    mpath = os.path.join(d, "metrics.jsonl")
    rows: list[dict] = []
    if os.path.isfile(mpath):
        with open(mpath) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    info["metrics_rows"] = len(rows)
    info["last"] = {k: rows[-1].get(k) for k in METRIC_FIELDS if k in rows[-1]} \
        if rows else {}
    tail = rows[-4:]
    info["trajectory"] = [
        {k: r.get(k) for k in ("rollout_idx", "resp_len_mean", "pass_rate",
                               "zero_var_group_frac", "cost_cny_cum")
         if k in r}
        for r in tail
    ]
    # resp_len slope over the last few rollouts: the single number that says
    # whether the training-side peak is still climbing (HEADROOM.md §8.3).
    lens = [(r.get("resp_len_mean"), r.get("rollout_idx")) for r in tail
            if isinstance(r.get("resp_len_mean"), (int, float))]
    if len(lens) >= 2:
        (y0, x0), (y1, x1) = lens[0], lens[-1]
        span = (x1 - x0) if (x1 is not None and x0 is not None) else len(lens) - 1
        info["resp_len_slope_per_rollout"] = round(
            (y1 - y0) / span, 4) if span else None
    info["log_tail"] = sh(f"tail -12 'runs/_logs/{tag}.log' 2>/dev/null")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default=os.path.join(
        ROOT, "runs", "_vram_guard_halt.md"))
    ap.add_argument("--out-json", default=os.path.join(
        ROOT, "runs", "_vram_guard_halt.json"))
    ap.add_argument("--reason", default="")
    args = ap.parse_args()

    now_utc = datetime.now(timezone.utc)
    p = procs()
    tag = tag_of(p["train"])
    snap = {
        "generated_utc": now_utc.strftime("%FT%TZ"),
        "generated_bjt": now_utc.astimezone(BJT).strftime("%F %T"),
        "reason": args.reason,
        "git_commit": (open(os.path.join(ROOT, "COMMIT")).read().strip()
                       if os.path.isfile(os.path.join(ROOT, "COMMIT")) else None),
        "vram": vram(),
        "procs": p,
        "current_tag": tag,
        "disk_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 1),
    }

    tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "runs",
                                                                      "mu*_s*"))
                  if os.path.isdir(d))
    if tag and tag not in tags:
        tags.append(tag)
    snap["runs"] = [run_dir(t) for t in tags]

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(snap, fh, indent=2)

    v = snap["vram"]
    L: list[str] = []
    L.append("# VRAM guard halt — snapshot before anything was killed\n")
    L.append(f"- generated: **{snap['generated_bjt']} 北京** ({snap['generated_utc']})")
    if snap["reason"]:
        L.append(f"- reason: {snap['reason']}")
    L.append(f"- code: `{snap['git_commit']}`  ·  disk free "
             f"{snap['disk_free_gib']} GiB")
    L.append(f"- VRAM: free **{v.get('free_mib')}** / used {v.get('used_mib')} "
             f"/ total {v.get('total_mib')} MiB, util {v.get('util_pct')}%")
    L.append(f"- compute apps: `{v.get('compute_apps') or 'none'}`\n")

    L.append("## Processes at halt\n")
    for k, val in snap["procs"].items():
        L.append(f"- **{k}**: `{val or 'not running'}`")
    L.append("")

    L.append("## What is on disk\n")
    L.append("| run | summary | metrics rows | rollouts | state.pt | resp_len (last) "
             "| pass_rate | zero_var_frac | cost CNY |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in snap["runs"]:
        if not r["exists"]:
            continue
        st = r["state_pt"]
        st_s = f"{st['size_mib']} MiB, {st['age_s']}s old" if st else "—"
        last = r.get("last", {})
        L.append(
            f"| {r['tag']} | {'yes' if r['has_summary'] else 'no'} "
            f"| {r['metrics_rows']} | {r['rollout_npz']} | {st_s} "
            f"| {last.get('resp_len_mean', '—')} | {last.get('pass_rate', '—')} "
            f"| {last.get('zero_var_group_frac', '—')} "
            f"| {last.get('cost_cny_cum', '—')} |")
    L.append("")

    cur = next((r for r in snap["runs"] if r["tag"] == tag), None)
    if cur:
        L.append(f"## Current run `{tag}` — recent rollouts\n")
        L.append("| rollout_idx | resp_len_mean | pass_rate | zero_var_frac | cost CNY |")
        L.append("|---|---|---|---|---|")
        for t in cur.get("trajectory", []):
            L.append(f"| {t.get('rollout_idx', '—')} | {t.get('resp_len_mean', '—')} "
                     f"| {t.get('pass_rate', '—')} "
                     f"| {t.get('zero_var_group_frac', '—')} "
                     f"| {t.get('cost_cny_cum', '—')} |")
        slope = cur.get("resp_len_slope_per_rollout")
        L.append("")
        if isinstance(slope, (int, float)):
            verdict = "still climbing" if slope > 0 else "flat or falling"
            L.append(f"resp_len slope: **{slope}** tokens/rollout ({verdict})")
        else:
            L.append(f"resp_len slope: not computable — only "
                     f"{cur.get('metrics_rows', 0)} completed rollout(s) in this run, "
                     f"so the trend that forces this decision is not yet observable")
        if cur.get("log_tail"):
            L.append("\n```\n" + cur["log_tail"] + "\n```")
        L.append("")

    L.append("## Decisions this snapshot was taken for (nothing has been run)\n")
    L.append("Resume always picks up from `state.pt`; the rollout in flight at halt "
             "is lost (state.pt is written every rollout, D-21).\n")
    L.append("```bash")
    L.append("# A. same config, same risk — resume the campaign where it stopped")
    L.append("setsid nohup bash scripts/run_9.sh >>/tmp/campaign_main.log 2>&1 &")
    L.append("")
    L.append("# B. buy headroom with the reservation, ~+0.8 h over 9 runs")
    L.append("#    configs/cloud48g_util042.yaml: util 0.42 (frees ~1.4 GiB of the "
             "21.3 GiB reservation)")
    L.append("CFG=configs/cloud48g_util042.yaml setsid nohup bash scripts/run_9.sh "
             ">>/tmp/campaign_util042.log 2>&1 &")
    L.append("")
    L.append("# C. the old fallback: util 0.38 / max_num_seqs 192")
    L.append("CFG=configs/cloud48g_lowmem.yaml setsid nohup bash scripts/run_9.sh "
             ">>/tmp/campaign_lowmem.log 2>&1 &")
    L.append("```")
    L.append("\nDiscount rate for sitting idle on the box: the instance bills "
             "whether or not it trains.")

    with open(args.out_md, "w") as fh:
        fh.write("\n".join(L) + "\n")

    print(f"snapshot -> {args.out_json}")
    print(f"snapshot -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
