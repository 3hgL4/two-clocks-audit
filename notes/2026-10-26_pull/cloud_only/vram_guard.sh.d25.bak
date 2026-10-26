#!/usr/bin/env bash
# VRAM guard for the 9-run campaign (D-25).
#
# Watches free VRAM and, if it drops below a floor for two consecutive reads,
# stops the campaign and brings it back on configs/cloud48g_lowmem.yaml.
# `run_9.sh` skips runs that already have summary.json and resumes runs that
# have state.pt, so the relaunch continues rather than restarting: at most the
# rollout in flight is lost (state.pt is written every rollout, D-21).
#
# Why it downgrades both knobs: at Br=4096 vLLM reserves 0.45 x 47,370 = 21,317
# MiB up front and the rest of the peak (~19.7 GiB) is the training side, which
# grows with resp_len (292.9 -> 299.5 over two rollouts took the peak from
# 40,172 to 40,974 MiB). max_num_seqs alone shrinks the KV working set but not
# the reservation, so util is the knob that actually returns memory; mns comes
# down with it so the scheduler does not oversubscribe the smaller KV budget.
#
# It downgrades ONCE. If the floor is hit again after that, it logs and exits
# instead of looping the campaign through another restart.
#
#   setsid nohup bash tools/vram_guard.sh >/dev/null 2>&1 &
#   THRESH_MIB=3072 INTERVAL_S=15 bash tools/vram_guard.sh   # tighter
#
# Log: runs/_logs/vram_guard.log
set -uo pipefail
cd "$(dirname "$0")/.."

THRESH_MIB="${THRESH_MIB:-2048}"
INTERVAL_S="${INTERVAL_S:-20}"
STRIKES_NEEDED="${STRIKES_NEEDED:-2}"
LOWMEM_CFG="${LOWMEM_CFG:-configs/cloud48g_lowmem.yaml}"
LOWMEM_LOG="${LOWMEM_LOG:-/tmp/campaign_lowmem.log}"
FLAG="runs/_vram_guard_downgraded"
LOG="runs/_logs/vram_guard.log"
mkdir -p runs/_logs

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >>"$LOG"; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

log "guard up: trip below ${THRESH_MIB} MiB free for ${STRIKES_NEEDED} consecutive reads, polling ${INTERVAL_S}s"
log "guard up: fallback config ${LOWMEM_CFG}"

strikes=0
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
      if [[ -f "$FLAG" ]]; then
        log "floor hit again but a downgrade already happened; not looping, guard exits"
        exit 0
      fi
      touch "$FLAG"
      log "TRIPPED: stopping campaign"
      pkill -f "[r]un_9.sh" 2>/dev/null || true
      sleep 1
      pkill -f "[s]rc/train.py" 2>/dev/null || true
      sleep 6
      pkill -f "[E]ngineCore" 2>/dev/null || true
      sleep 3
      log "TRIPPED: freeing ${free}MiB at stop; relaunching with CFG=${LOWMEM_CFG}"
      CFG="$LOWMEM_CFG" setsid nohup bash scripts/run_9.sh >>"$LOWMEM_LOG" 2>&1 </dev/null &
      sleep 5
      log "TRIPPED: campaign relaunched, guard exits (output -> ${LOWMEM_LOG})"
      exit 0
    fi
  else
    (( strikes > 0 )) && log "recovered: free=${free}MiB, strikes reset"
    strikes=0
  fi
  sleep "$INTERVAL_S"
done
