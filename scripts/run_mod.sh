#!/bin/sh
# Relaunch driver: mod family A/B x3 seeds only (counting B + A evals already
# done in the 2026-11-01 night run). Config sha fix: fsa_mod_a.yaml overrides
# base.yaml's counting-pool sha256.
set -u
PY=/opt/conda/bin/python
BASE=/root/rivermind-data/two-clocks-audit/models/Qwen3-1.7B-Base
step() { echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

step "mod family A/B x3 seeds (8 rollouts x 4096)"
for arm in a b; do
  for s in 0 1 2; do
    tag=fsa_mod_${arm}_s${s}
    step "train $tag"
    $PY src/train.py --config configs/fsa_mod_${arm}.yaml --seed $s --tag $tag \
        > /tmp/${tag}.log 2>&1 || echo "FAILED train $tag"
    step "eval $tag"
    $PY tools/eval_heldout.py --pool data/heldout_mod_512.jsonl --model $BASE \
        --adapter runs/${tag}/adapters/r007 --n 4 --temperature 1.0 --seed 0 \
        --tag eval_${tag} --out out/eval_heldout > /tmp/eval_${tag}.log 2>&1 \
        && grep -h '"pass_rate"' out/eval_heldout/eval_${tag}/summary_*.json \
        || echo "FAILED eval $tag"
  done
done
step "mod rerun ALL_DONE"
