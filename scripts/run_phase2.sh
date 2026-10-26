#!/usr/bin/env bash
# Phase 2: attribution (a)(b)(c) + R4-A.  Claims C9/C10/C11/C12.
#
# Budget line (measured, 2026-10-26, 4090-48G at CNY 2.08/h):
#   (c) noclip_mu16   3 seeds x 1.97 h = 5.9 h   ~CNY 13
#   (b) 2x2           3 seeds x (1.0+4.0) h = 15 h ~CNY 31
#   R4-A             3 seeds x 11 rollouts x 15.3 min = 8.7 h ~CNY 18
#   (a) offline sigma is NOT here: it is tools/offline_sigma.py, runs in ~0.5 h,
#       and needs no rollout backend (it only backprops archived data).
#
#   ./scripts/run_phase2.sh c        # (c) 3 seeds
#   ./scripts/run_phase2.sh b        # (b) 2x2, 6 runs
#   ./scripts/run_phase2.sh r4a      # R4-A: resume mu1 to 19 rollouts
#   ./scripts/run_phase2.sh all      # all of the above, in that order
#
# Every stage is resumable: a run whose summary.json exists is skipped, so re-running
# the same stage after a crash continues instead of redoing (and re-billing).
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/opt/conda/bin/python}"
STAGE="${1:-all}"

# D-13: without expandable_segments the peak is 45,140 MiB of 47,370 usable and the
# update dies in the allocator with 4.22 GiB reserved-but-unused. Not optional.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# D-23: block buffering makes an 8-rollout run look frozen for ten minutes at a time.
export PYTHONUNBUFFERED=1
mkdir -p runs/_logs

fail() { printf '\n[fail] %s\n' "$*" >&2; exit 1; }
banner() { printf '\n========== %s ==========\n' "$*"; }

# Distinct completed rollouts in a metrics.jsonl (0 if absent).
count_rollouts() {
  "$PY" - "$1" <<'PY'
import json, sys
try:
    rows = {json.loads(l)["rollout_idx"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()}
except Exception:
    rows = set()
print(len(rows))
PY
}

# One run. Skips if summary.json exists (a finished run is never re-billed).
run_one() {
  local cfg="$1" tag="$2"; shift 2
  local dir="runs/$tag"
  if [[ -f "$dir/summary.json" ]]; then
    echo "[skip] $tag already has summary.json"
    return 0
  fi
  local log="runs/_logs/${tag}.log"
  printf '[start] %s  cfg=%s  %s\n' "$tag" "$cfg" "$*"
  local t0=$SECONDS
  "$PY" src/train.py --config "$cfg" --tag "$tag" --rollout vllm "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local dt=$(( SECONDS - t0 ))
  if [[ $rc -ne 0 ]]; then
    printf '[fail] %s exited %d after %d min; log %s\n' "$tag" "$rc" "$((dt/60))" "$log" >&2
    return 1
  fi
  printf '[done] %s in %d min\n' "$tag" "$((dt/60))"
}

# ---------------------------------------------------------------- pre-flight
banner "pre-flight"
"$PY" tests/run_all.py || fail "tests failed -- do not spend GPU time"

# Config gate: the arm definitions must resolve to what the claims file says.
"$PY" - <<'EOF' || fail "config gate"
import sys
sys.path.insert(0, "src")
import config as config_mod

expect = {
    # cfg                          mu  Br    b    group micro
    "configs/noclip_mu16.yaml":  (16, 4096, 256, 8, 4),
    "configs/ab2x2_mu8_b256.yaml": (8, 2048, 256, 8, 4),
    "configs/ab2x2_mu16_b512.yaml": (16, 8192, 512, 8, 4),
}
bad = []
for path, (mu, br, b, grp, micro) in expect.items():
    cfg = config_mod.load_config(path)
    got = (cfg["train.mu"], cfg["train.responses_per_rollout"], cfg.responses_per_step,
           cfg.micro_group_responses, cfg["train.micro_batch_responses"])
    ok = got == (mu, br, b, grp, micro)
    print(f"  {'ok ' if ok else 'BAD'} {path}: mu={got[0]} Br={got[1]} b={got[2]} "
          f"group={got[3]} micro={got[4]} prompts/rollout={cfg.prompts_per_rollout}")
    if not ok:
        bad.append((path, (mu, br, b, grp, micro), got))
    # the arm must keep the D-18/D-19 sampler fixes and the local weights
    if not str(cfg["model.name"]).startswith("/root/"):
        bad.append((path, "model.name must be the local path", cfg["model.name"]))
    if cfg.get("rollout.vllm.seed_per_request") is not True:
        bad.append((path, "seed_per_request must be true (D-18)", None))
if bad:
    for x in bad:
        print("  [fail]", x)
    sys.exit(1)
print("  clip_delta(noclip) =", config_mod.load_config("configs/noclip_mu16.yaml")["train.clip_delta"])
EOF

FREE_GB=$(df -Pk . | awk 'NR==2 {printf "%d", $4/1048576}')
echo "  free ${FREE_GB} GiB"
# Phase 2 needs: 1.97h-run x4 snapshots ~2.6 GiB each for (c)/(b) + 4.0h-run x3
# ~5.2 GiB each for the big cell + R4-A 11 rollouts ~3.2 GiB per seed.
[[ "$FREE_GB" -lt 20 ]] && fail "under 20 GiB free: adapter snapshots would start skipping (D-22)"

if [[ -f COMMIT ]]; then echo "  COMMIT=$(cat COMMIT)"; else fail "no COMMIT file"; fi

# ---------------------------------------------------------------- stage (c)
if [[ "$STAGE" == "all" || "$STAGE" == "c" ]]; then
  banner "(c) dual-clip ablation: mu=16, clip_delta=null, 3 seeds  [C11]"
  for s in 0 1 2; do
    run_one configs/noclip_mu16.yaml "noclip_mu16_s$s" --seed "$s" || fail "stage (c) seed $s"
  done
fi

# ---------------------------------------------------------------- stage (b)
if [[ "$STAGE" == "all" || "$STAGE" == "b" ]]; then
  banner "(b) b x mu 2x2: two new cells x 3 seeds  [C10]"
  # Cheap cell first (1.0 h): if the expensive one has to be abandoned, the pair
  # is still incomplete either way, but the cheap one finishing means a crash
  # costs less.
  for s in 0 1 2; do
    run_one configs/ab2x2_mu8_b256.yaml "ab2x2_mu8_b256_s$s" --seed "$s" || fail "stage (b) mu8@b256 seed $s"
  done
  for s in 0 1 2; do
    run_one configs/ab2x2_mu16_b512.yaml "ab2x2_mu16_b512_s$s" --seed "$s" || fail "stage (b) mu16@b512 seed $s"
  done
fi

# ---------------------------------------------------------------- stage R4-A
if [[ "$STAGE" == "all" || "$STAGE" == "r4a" ]]; then
  banner "R4-A: resume mu=1 from 8 to 19 rollouts, 3 seeds  [C12]"
  # --resume needs runs/mu1_s{0,1,2}/state.pt to be present already. The resumed
  # run re-plans total_steps = 19*mu (train.py:430), so mu=1's lr jumps 13.6x at
  # r8: this is a two-segment schedule, and C12 forbids reading a slope across it.
  #
  # This stage cannot use run_one's summary.json skip: the pre-resume run already
  # shipped its own summary.json (8 rollouts), which would make every seed look
  # finished and silently skip the whole stage. Completion is read from the
  # rollout count inside metrics.jsonl instead, so a crash mid-stage resumes.
  r4a_one() {
    local s="$1" tag="mu1_s$s" dir="runs/mu1_s$s"
    [[ -f "$dir/state.pt" ]] || {
      printf '[fail] runs/mu1_s%s/state.pt missing -- upload it before R4-A\n' "$s" >&2
      exit 1
    }
    local n; n=$(count_rollouts "$dir/metrics.jsonl")
    if [[ "$n" -ge 19 ]]; then
      echo "[skip] $tag already at $n rollouts"
      return 0
    fi
    local log="runs/_logs/${tag}_r4a.log"
    printf '[start] %s  from r%s -> 19  (seed %s)\n' "$tag" "$n" "$s"
    local t0=$SECONDS
    "$PY" src/train.py --config configs/cloud48g_main.yaml --tag "$tag" --rollout vllm \
      --mu 1 --seed "$s" --resume --num-rollouts 19 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]} dt=$(( SECONDS - t0 ))
    [[ $rc -ne 0 ]] && { printf '[fail] %s exited %d after %d min; log %s\n' "$tag" "$rc" "$((dt/60))" "$log" >&2; return 1; }
    printf '[done] %s in %d min, now %s rollouts\n' "$tag" "$((dt/60))" "$(count_rollouts "$dir/metrics.jsonl")"
  }
  for s in 0 1 2; do
    r4a_one "$s" || fail "R4-A seed $s"
  done
fi

banner "stage '$STAGE' complete"
df -h . | tail -1
