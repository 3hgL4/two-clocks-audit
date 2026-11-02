#!/bin/sh
# Watcher for fsa_a_align s2 (R-1 revision, user-approved 2026-11-02 01:20):
# waits for run_align.sh driver to exit; if s2 was skipped by the 06:00 gate,
# automatically trains + evals it. If the gate let s2 through, this is a no-op.
REPO=/root/rivermind-data/format-signal-audit/repo
PY=/opt/conda/bin/python
BASE=/root/rivermind-data/two-clocks-audit/models/Qwen3-1.7B-Base
LOG=$REPO/out/eval_heldout/eval_fsa_a_align_s2/summary_heldout_counting_512.json

echo "watcher start $(date -u +%FT%TZ)"
while pgrep -f "scripts/run_align.sh" > /dev/null; do
  sleep 120
done
echo "driver exited $(date -u +%FT%TZ)"

if [ -f "$LOG" ] || [ -f "$REPO/runs/fsa_a_align_s2/metrics.jsonl" ]; then
  echo "s2 already handled by driver, nothing to do"
else
  echo "s2 was skipped by gate -> running it now"
  cd $REPO || exit 1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "=== $(date -u +%FT%TZ) train fsa_a_align_s2 (watcher)"
  $PY src/train.py --config configs/fsa_a_align.yaml --seed 2 --tag fsa_a_align_s2 \
      > /tmp/fsa_a_align_s2.log 2>&1 || echo "FAILED train fsa_a_align_s2"
  echo "=== $(date -u +%FT%TZ) eval fsa_a_align_s2 (watcher)"
  $PY tools/eval_heldout.py --pool data/heldout_counting_512.jsonl --model $BASE \
      --adapter runs/fsa_a_align_s2/adapters/r010 --n 4 --temperature 1.0 --seed 0 \
      --tag eval_fsa_a_align_s2 --out out/eval_heldout > /tmp/eval_fsa_a_align_s2.log 2>&1 \
      && grep -h '"pass_rate"' out/eval_heldout/eval_fsa_a_align_s2/summary_*.json \
      || echo "FAILED eval fsa_a_align_s2"
fi
echo "S2_WATCHER_DONE $(date -u +%FT%TZ)"
