#!/usr/bin/env bash
# The 9-run campaign, in the order it has to run in.  NOT LAUNCHED YET.
#
#   3 mu arms (1, 8, 16) x 3 seeds (0, 1, 2), one 48G card, option A:
#   micro 4 / group 8 / util 0.45 / max_num_seqs 256 (configs/cloud48g_main.yaml).
#
# Expected 1.84 h and ~CNY 3.8 per run, 16.5 h and ~CNY 34 for the nine
# (notes/2026-10-25_cloud48g/HEADROOM.md §8.3 + the corrected extrapolation in
# DEVLOG D-17). Budget ceiling is CNY 100, so this leaves room for the
# calibration gate and one repeat if a run dies.
#
#   ./scripts/run_9.sh              run everything that is not finished yet
#   ./scripts/run_9.sh --force      redo runs even if summary.json exists
#   MU=1 SEED=0 ./scripts/run_9.sh  a single arm (useful for a pilot)
#
# Two things this script does that a bare `python src/train.py` does not:
#
# 1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Measured on this card:
#    peak 45,140 MiB without it and 40,172 MiB with it, on the same data. Without
#    it the micro=8 arm died in the allocator at 4.22 GiB reserved-but-unused.
# 2. A clean `runs/_adapter` before each run. The adapter directory is a fixed
#    path shared by every run (base.yaml rollout.vllm.adapter_dir); leaving a
#    previous run's export there is exactly the kind of cross-run contamination
#    that makes a 16-hour campaign unfalsifiable.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/opt/conda/bin/python}"
CFG="${CFG:-configs/cloud48g_main.yaml}"
MUS="${MUS:-1 8 16}"
SEEDS="${SEEDS:-0 1 2}"
MU="${MU:-}"
SEED="${SEED:-}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
# Without this every line sits in Python's block buffer until 8 KiB accumulate,
# so an 8-rollout run looks frozen for ten minutes at a time (D-23).
export PYTHONUNBUFFERED=1
mkdir -p runs/_logs

if [[ -n "$MU" ]]; then MUS="$MU"; fi
if [[ -n "$SEED" ]]; then SEEDS="$SEED"; fi

echo "=== pre-flight: tests ==="
"$PY" tests/run_all.py

echo "=== pre-flight: config resolves ==="
"$PY" - "$CFG" <<'EOF'
import sys
sys.path.insert(0, "src")
import config as config_mod
cfg = config_mod.load_config(sys.argv[1])
print(f"  micro={cfg['train.micro_batch_responses']} group={cfg.micro_group_responses} "
      f"Br={cfg['train.responses_per_rollout']} rollouts={cfg['train.num_rollouts']} "
      f"util={cfg['rollout.vllm.gpu_memory_utilization']} "
      f"mns={cfg['rollout.vllm.max_num_seqs']}")
assert cfg["train.micro_batch_responses"] == 4, "campaign runs option A: micro must be 4"
assert cfg.micro_group_responses == 8, "sigma group must stay at the pre-registered 8"
# D-18 / D-19: without these the campaign measures a degenerate sampler
assert cfg.get("rollout.vllm.seed_per_request") is True, \
    "seed_per_request must be on: one shared seed collapses G samples into one (D-18)"
assert cfg.get("rollout.vllm.enable_prefix_caching") is False, "prefix cache off (D-19)"
assert cfg.get("rollout.vllm.enable_chunked_prefill") is False, "chunked prefill off (D-19)"
assert cfg.get("rollout.vllm.max_num_batched_tokens"), "fixed token budget (D-19)"
assert cfg.get("logging.save_rollout_ids") is True, "rollout archive must stay on (D-17)"
assert cfg.get("logging.save_adapter_every_rollout") is True, \
    "per-rollout adapter snapshots must stay on (D-22)"
assert cfg.get("rollout.vllm.seed_per_request") is True, "per-request seeds (D-18)"
EOF

echo "=== pre-flight: disk ==="
# 9 runs need ~7.5 GiB of state.pt + ~19 GiB of adapter snapshots (D-22). The
# snapshot helper skips itself below 4 GiB free, but a campaign that silently
# stops archiving is worth refusing to start.
FREE_GB=$(df -Pk . | awk 'NR==2 {printf "%d", $4/1048576}')
echo "  free ${FREE_GB} GiB"
if [[ "$FREE_GB" -lt 28 ]]; then
  echo "  [warn] under 28 GiB free: adapter snapshots may start skipping mid-campaign" >&2
fi

echo "=== pre-flight: provenance ==="
# D-21/D-23: run_meta.json records the commit that actually ran. The code gets
# here by scp, so the stamped COMMIT file is the authority -- a .git left on the
# box can name a commit that is not what is deployed (it did: 0fe107b, while the
# deployed tree was a73dc91). Never let a stale .git overwrite the stamp.
if [[ -f COMMIT ]]; then
  stamp=$(cat COMMIT)
  if [[ ! "$stamp" =~ ^[0-9a-f]{40}$ || "$stamp" =~ ^0{40}$ ]]; then
    echo "  [error] COMMIT file is '$stamp' -- stamp it: git rev-parse HEAD > COMMIT" >&2
    exit 1
  fi
  echo "  COMMIT=$stamp (from file)"
else
  echo "  [error] no COMMIT file: run_meta.json could not name the code version" >&2
  exit 1
fi
if [[ -d .git ]]; then
  head=$(git rev-parse HEAD 2>/dev/null || echo "?")
  [[ "$head" == "$(cat COMMIT)" ]] || \
    echo "  [warn] .git HEAD is $head, differs from COMMIT: the stamp wins" >&2
fi

for mu in $MUS; do
  for seed in $SEEDS; do
    tag="mu${mu}_s${seed}"
    dir="runs/$tag"
    if [[ -f "$dir/summary.json" && $FORCE -eq 0 ]]; then
      echo "[skip] $tag already finished ($dir/summary.json)"
      continue
    fi
    if [[ -d "$dir" && ! -f "$dir/state.pt" && $FORCE -eq 0 ]]; then
      echo "[stop] $dir exists without state.pt. Metrics would be appended twice." >&2
      echo "       Inspect it, then either remove the directory or pass --force." >&2
      exit 1
    fi
    resume=()
    [[ -f "$dir/state.pt" ]] && resume=(--resume)

    rm -rf runs/_adapter
    echo "=== $tag ${resume[*]:-} $(date -u +%FT%TZ) ==="
    "$PY" src/train.py --config "$CFG" --mu "$mu" --seed "$seed" --tag "$tag" \
        "${resume[@]}" 2>&1 | tee "runs/_logs/$tag.log"
    "$PY" tools/check_schema.py "$dir" --strict | tee -a "runs/_logs/$tag.log"
    "$PY" tools/audit_rollout_ids.py "$dir/rollouts/r000.npz" --group 8 \
        --json "$dir/rollouts/audit_r000.json" | tee -a "runs/_logs/$tag.log"
  done
done

echo "=== analysis ==="
"$PY" analysis/make_table.py --runs runs --out analysis/out
"$PY" analysis/trace.py || true
echo "done. Cost so far:"
"$PY" - <<'EOF'
import json, glob
tot = 0.0
for p in sorted(glob.glob("runs/*/summary.json")):
    t = json.load(open(p))["totals"]
    tot += float(t.get("cost_cny", 0.0))
    print(f"  {p.split('/')[1]:>10}  {t.get('responses', 0):>6} resp  "
          f"{t.get('gpu_hours_billed', 0):.2f} GPU-h  CNY {t.get('cost_cny', 0):.2f}")
print(f"  total CNY {tot:.2f}")
EOF
