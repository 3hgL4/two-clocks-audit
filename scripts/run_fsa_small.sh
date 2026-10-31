#!/bin/sh
# Exploration-stage driver: 3 arms x 3 seeds small runs + held-out eval each.
# Run on the cloud box, from the repo root, under nohup:
#   nohup sh scripts/run_fsa_small.sh > /tmp/fsa_small_driver.log 2>&1 &
set -e
cd /root/rivermind-data/format-signal-audit/repo
PY=/opt/conda/bin/python
BASE=/root/rivermind-data/two-clocks-audit/models/Qwen3-1.7B-Base
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for arm in a b c; do
  for s in 0 1 2; do
    tag=fsa_small_${arm}_s${s}
    echo "=== $(date -u +%FT%TZ) train $tag"
    $PY src/train.py --config configs/fsa_small_${arm}.yaml --seed $s --tag $tag \
        > /tmp/${tag}.log 2>&1
    echo "=== $(date -u +%FT%TZ) eval $tag"
    $PY tools/eval_heldout.py \
        --pool data/heldout_counting_512.jsonl \
        --model $BASE \
        --adapter runs/${tag}/adapters/r001 \
        --n 4 --temperature 1.0 --seed 0 \
        --tag eval_${tag} --out out/eval_heldout \
        > /tmp/eval_${tag}.log 2>&1
    grep -h '"pass_rate"' out/eval_heldout/eval_${tag}/summary_*.json || true
  done
done
echo "ALL_DONE $(date -u +%FT%TZ)"
