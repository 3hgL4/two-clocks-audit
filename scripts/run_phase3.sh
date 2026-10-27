#!/usr/bin/env bash
# Phase 3 = "plan B", in the order approved on 2026-10-26:
#
#   (a) offline sigma   ~0.5 h   ~CNY 1     [C9]
#   (b) b x mu 2x2      ~15 h    ~CNY 31    [C10]   (mu8@b256 3x1.0h + mu16@b512 3x4.0h)
#   R4-A                ~8.7 h   ~CNY 18    [C12]
#   ------------------------------------------------------------------
#   total               ~24 h    ~CNY 50    (balance left was CNY 49.8 -> no slack)
#
# Two failure-isolation decisions that are deliberate, not lazy:
#
#   1. (a) failing does NOT abort the script. (a) only backprops already-archived
#      data; its verdict says nothing about whether the 2x2 is readable. Letting a
#      0.5 h stage kill a 15 h stage would be the wrong trade.
#   2. (b) failing does NOT prevent R4-A. R4-A resumes runs/mu1_s*/state.pt, which
#      is orthogonal to the 2x2 cells. If the expensive cell OOMs, the cheap cell
#      plus R4-A are still worth the remaining budget.
#
# Inside (b), the cheap cell runs first (scripts/run_phase2.sh already orders it
# that way), so a crash costs 3 h rather than 12 h.
#
# Every stage is resumable: run_phase2.sh skips runs whose summary.json exists
# (and, for R4-A, seeds already at 19 rollouts). Re-running this script after a
# crash continues; it does not re-bill finished work.
#
#   setsid nohup bash scripts/run_phase3.sh > runs/_logs/phase3.out 2>&1 < /dev/null &
#
# Do NOT overwrite this file while it is running: bash reads a script by byte
# offset, so replacing it mid-run makes the reader resume at the wrong place.
# (That exact mistake truncated the (c) stage on 2026-10-26.)

set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/opt/conda/bin/python}"
# D-13: peak without this is 45,140 MiB of 47,370 usable and the update dies in
# the allocator with 4.22 GiB reserved-but-unused. Not optional.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p runs/_logs analysis/out

banner() { printf '\n========== %s ==========\n' "$*"; }
stamp()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

banner "phase3 start  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
[[ -f COMMIT ]] && echo "  COMMIT=$(cat COMMIT)"
df -h . | tail -1

# ------------------------------------------------------------------ (a) offline sigma
banner "(a) offline sigma: fixed data D, adapter staleness 0/3/7 + init anchor  [C9]"

# Smoke first: two groups through the whole pipeline. A wiring bug here costs
# ninety seconds; discovered after the full sweep it would cost the GPU time of
# 4096 responses x 4 policies for nothing.
stamp "smoke: 2 groups, one adapter"
"$PY" tools/offline_sigma.py \
    --run runs/mu1_s0 --rollout 0 \
    --adapters 000 --include-init --limit-groups 2 \
    --json analysis/out/_sigma_smoke.json > runs/_logs/phase3_a_smoke.log 2>&1
rc_smoke=$?
if [[ $rc_smoke -ne 0 ]]; then
  tail -25 runs/_logs/phase3_a_smoke.log >&2
  rc_a=1
  stamp "smoke FAILED (rc=$rc_smoke); skipping the full sweep, continuing to (b)"
else
  stamp "smoke ok"
  "$PY" tools/offline_sigma.py \
      --run runs/mu1_s0 --rollout 0 \
      --adapters 000 003 007 --include-init \
      --json analysis/out/offline_sigma.json 2>&1 | tee runs/_logs/phase3_a.log
  rc_a=${PIPESTATUS[0]}
fi

if [[ $rc_a -ne 0 ]]; then
  printf '\n[(a) FAILED rc=%d] continuing to (b) -- (a) is independent of the 2x2\n' "$rc_a" >&2
fi

# ------------------------------------------------------------------ (b) 2x2
banner "(b) b x mu 2x2: two new cells x 3 seeds  [C10]"
stamp "starting (b) at $(date -u +%H:%M:%S) UTC"
bash scripts/run_phase2.sh b 2>&1 | tee runs/_logs/phase3_b.out
rc_b=${PIPESTATUS[0]}
stamp "(b) exited rc=$rc_b"

# ------------------------------------------------------------------ R4-A
banner "R4-A: resume mu=1 from 8 to 19 rollouts, 3 seeds  [C12]"
if [[ $rc_b -ne 0 ]]; then
  printf '\n[(b) FAILED rc=%d] running R4-A anyway -- it needs only runs/mu1_s*/state.pt\n' "$rc_b" >&2
fi
stamp "starting R4-A at $(date -u +%H:%M:%S) UTC"
bash scripts/run_phase2.sh r4a 2>&1 | tee runs/_logs/phase3_r4a.out
rc_r4a=${PIPESTATUS[0]}
stamp "R4-A exited rc=$rc_r4a"

# ------------------------------------------------------------------ summary
banner "phase3 complete  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
printf '  (a)   rc=%s   -> analysis/out/offline_sigma.json\n' "$rc_a"
printf '  (b)   rc=%s   -> runs/ab2x2_*_s{0,1,2}/summary.json\n' "$rc_b"
printf '  R4-A  rc=%s   -> runs/mu1_s{0,1,2}/metrics.jsonl (19 rollouts)\n' "$rc_r4a"

# Cost line: sum the CostMeter of everything that actually has a summary. This is
# the meter's own number, which under-counts old_logp by ~13% (D-25), so the true
# spend is roughly this / 0.885. Printed both ways so nobody has to remember.
#
# Split by container age on purpose: runs/mu1_s{0,1,2} were uploaded with the
# resume package, so their ~CNY 11 was billed on a previous instance. Folding them
# into "this instance" would overstate the spend by 25% -- the dashboard made
# exactly this mistake before it started using PID 1's start time.
"$PY" - <<'PY'
import glob, json, os, time

def container_start():
    """Epoch when PID 1 started = when this container booted. /proc/uptime is the
    HOST's uptime (52 days here), so it cannot be used directly."""
    try:
        stat = open("/proc/1/stat").read()
        start_ticks = float(stat[stat.rindex(")") + 2:].split()[19])
        hz = os.sysconf("SC_CLK_TCK")
        uptime = float(open("/proc/uptime").read().split()[0])
        return time.time() - (uptime - start_ticks / hz)
    except Exception:
        return None

boot = container_start()
tot_mine = tot_old = 0.0
for p in sorted(glob.glob("runs/*/summary.json")):
    tag = os.path.basename(os.path.dirname(p))
    try:
        d = json.load(open(p, encoding="utf-8"))
        c = float((d.get("totals") or {}).get("cost_cny") or 0)
    except Exception:
        continue
    if not c:
        continue
    old = bool(boot) and os.path.getmtime(p) < boot
    if old:
        tot_old += c
    else:
        tot_mine += c
    print("  %-24s %6.2f  %s" % (tag, c, "uploaded (前一个实例)" if old else "this instance"))
print("  %-24s %6.2f  meter, true ~ %.2f" % ("THIS INSTANCE", tot_mine, tot_mine / 0.885))
if tot_old:
    print("  %-24s %6.2f  (not billed to this instance)" % ("uploaded runs", tot_old))
PY

df -h . | tail -1
stamp "done"
