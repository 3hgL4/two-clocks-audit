#!/usr/bin/env bash
# VRAM guard for the 9-run campaign (D-25, behaviour changed in D-26).
#
# Watches free VRAM. If it stays below the floor for two consecutive reads it
# takes a snapshot of the campaign (`tools/guard_snapshot.py`), stops the
# campaign, and EXITS. It does not relaunch.
#
# Why the change: the first version stopped the campaign and immediately
# brought it back up on `configs/cloud48g_lowmem.yaml`, i.e. a script decided,
# at 3 a.m. and with no one watching, that the remaining eight runs should
# finish on a configuration nobody had approved. Two configs across one
# campaign is exactly the kind of thing that makes a 16-hour result
# unfalsifiable: the µ arms would no longer be comparable, and the only trace
# would be one line in a log. A memory guard's job is to protect the data that
# exists, not to pick the next experiment. So now it protects, records, stops,
# and hands the decision back.
#
# What "judge" is expected to look at (all in runs/_vram_guard_halt.md):
#   - resp_len_mean slope: drives the training-side peak, which grows with it
#     (292.9 -> 299.5 over two rollouts took the peak from 40,172 to 40,974 MiB)
#   - which runs are complete, which is mid-flight, how many rollouts survived
#   - free/used split at the moment of the trip, and per-process VRAM
#
# The trip order matters. `run_9.sh` is killed first so the loop cannot start
# the next run the moment `train.py` dies; `state.pt` is written every rollout
# (D-21), so the recovery cost is one rollout, not one run.
#
#   setsid nohup bash tools/vram_guard.sh >/dev/null 2>&1 &
#   THRESH_MIB=3072 INTERVAL_S=15 bash tools/vram_guard.sh    # tighter floor
#   DRY_RUN=1 THRESH_MIB=99999 bash tools/vram_guard.sh       # self-test, kills nothing
#
# Log: runs/_logs/vram_guard.log
set -uo pipefail
cd "$(dirname "$0")/.."

THRESH_MIB="${THRESH_MIB:-2048}"      # trip floor: this is where a decision is forced
WARN_MIB="${WARN_MIB:-4096}"          # log a one-time warning here
INTERVAL_S="${INTERVAL_S:-20}"
STRIKES_NEEDED="${STRIKES_NEEDED:-2}"
PY="${PY:-/opt/conda/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
FLAG="runs/_vram_guard_halt.flag"
LOG="runs/_logs/vram_guard.log"
mkdir -p runs/_logs

if [[ "$DRY_RUN" == "1" ]]; then
  SNAP_MD="/tmp/_guard_dryrun_halt.md"
  SNAP_JSON="/tmp/_guard_dryrun_halt.json"
else
  SNAP_MD="runs/_vram_guard_halt.md"
  SNAP_JSON="runs/_vram_guard_halt.json"
fi

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
now_bjt() { TZ=Asia/Shanghai date '+%F %T'; }

snapshot() {
  # Never let a snapshot failure keep the campaign running into an OOM.
  if ! "$PY" tools/guard_snapshot.py --out-md "$SNAP_MD" --out-json "$SNAP_JSON" \
        --reason "$1" >>"$LOG" 2>&1; then
    log "WARN snapshot failed, continuing to stop anyway"
  fi
}

stop_campaign() {
  log "stop: pkill run_9.sh (so the loop cannot advance to the next run)"
  pkill -f "[r]un_9.sh" 2>/dev/null || true
  sleep 1

  log "stop: SIGTERM src/train.py (state.pt is already on disk, D-21)"
  pkill -TERM -f "[s]rc/train.py" 2>/dev/null || true
  for _ in $(seq 1 25); do
    pgrep -f "[s]rc/train.py" >/dev/null || break
    sleep 1
  done
  if pgrep -f "[s]rc/train.py" >/dev/null; then
    log "stop: train.py ignored SIGTERM for 25s, SIGKILL"
    pkill -KILL -f "[s]rc/train.py" 2>/dev/null || true
  fi
  sleep 2

  log "stop: EngineCore (frees the vLLM reservation)"
  pkill -f "[E]ngineCore" 2>/dev/null || true
  sleep 4

  local left
  left=$(ps -eo args | grep -E "[r]un_9.sh|[s]rc/train.py|[E]ngineCore" | wc -l)
  log "stop: verify -> ${left} campaign/engine process(es) still alive"
  return 0
}

log "guard up: trip below ${THRESH_MIB} MiB free for ${STRIKES_NEEDED} consecutive reads, polling ${INTERVAL_S}s"
log "guard up: on trip -> snapshot + stop + exit. NO auto-relaunch (D-26)."
log "guard up: warn below ${WARN_MIB} MiB, dry_run=${DRY_RUN}"

strikes=0
warned=0
while :; do
  free=$(gpu_free)
  if [[ -z "$free" ]]; then
    sleep "$INTERVAL_S"
    continue
  fi

  if (( free < THRESH_MIB )); then
    strikes=$((strikes + 1))
    log "LOW free=${free}MiB used=$(gpu_used)MiB strike=${strikes}/${STRIKES_NEEDED}"
    if (( strikes >= STRIKES_NEEDED )); then
      if [[ -f "$FLAG" && "$DRY_RUN" != "1" ]]; then
        log "floor hit again but this guard trip is already recorded in $FLAG; exiting without touching anything"
        exit 0
      fi
      reason="free ${free} MiB < ${THRESH_MIB} MiB for ${strikes} consecutive reads at $(now_bjt) 北京"
      log "TRIPPED: ${reason}"
      log "TRIPPED: snapshotting to ${SNAP_MD}"
      snapshot "$reason"
      if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY_RUN: would stop the campaign now; killing nothing. snapshot -> ${SNAP_MD}"
        exit 0
      fi
      touch "$FLAG"
      printf 'tripped_utc=%s\nreason=%s\nsnapshot=%s\n' \
        "$(date -u +%FT%TZ)" "$reason" "$SNAP_MD" >"runs/_vram_guard_halt.meta"
      stop_campaign
      log "HALTED at $(now_bjt) 北京. Campaign is down, nothing restarted."
      log "HALTED: decide from ${SNAP_MD} ; resume with a CFG=... line from that file."
      exit 0
    fi
  else
    (( strikes > 0 )) && log "recovered: free=${free}MiB, strikes reset"
    strikes=0
    if (( warned == 0 && free < WARN_MIB )); then
      log "WARN free=${free}MiB below ${WARN_MIB}MiB (watch resp_len slope); not acting"
      warned=1
    elif (( warned == 1 && free > WARN_MIB + 1024 )); then
      warned=0
    fi
  fi
  sleep "$INTERVAL_S"
done
