#!/bin/sh
# Overnight driver (2026-11-01): would-be analysis -> A evals -> B counting
# x3 -> mod A/B x3. Continues on per-step failure (marks FAILED, keeps going).
cd /root/rivermind-data/format-signal-audit/repo
PY=/opt/conda/bin/python
BASE=/root/rivermind-data/two-clocks-audit/models/Qwen3-1.7B-Base
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
AD=adapters_mu1   # uploaded campaign A-arm final adapters: adapters_mu1/s{0,1,2}/r007

step() { echo "=== $(date -u +%FT%TZ) $*"; }

step "A-arm held-out evals (campaign r007 adapters, counting pool)"
for s in 0 1 2; do
  $PY tools/eval_heldout.py --pool data/heldout_counting_512.jsonl --model $BASE \
      --adapter ${AD}/s${s}/r007 --n 4 --temperature 1.0 --seed 0 \
      --tag eval_mu1_s${s} --out out/eval_heldout > /tmp/eval_mu1_s${s}.log 2>&1 \
      && grep -h '"pass_rate"' out/eval_heldout/eval_mu1_s${s}/summary_*.json \
      || echo "FAILED eval mu1_s${s}"
done

step "B counting arm x3 seeds (8 rollouts x 4096)"
for s in 0 1 2; do
  tag=fsa_b_full_s${s}
  step "train $tag"
  $PY src/train.py --config configs/fsa_b_full.yaml --seed $s --tag $tag \
      > /tmp/${tag}.log 2>&1 || echo "FAILED train $tag"
  step "eval $tag"
  $PY tools/eval_heldout.py --pool data/heldout_counting_512.jsonl --model $BASE \
      --adapter runs/${tag}/adapters/r007 --n 4 --temperature 1.0 --seed 0 \
      --tag eval_${tag} --out out/eval_heldout > /tmp/eval_${tag}.log 2>&1 \
      && grep -h '"pass_rate"' out/eval_heldout/eval_${tag}/summary_*.json \
      || echo "FAILED eval $tag"
done

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

step "night summary"
$PY - <<'PYEOF' || echo "FAILED summary"
import glob, json
for p in sorted(glob.glob("out/eval_heldout/eval_*/summary_*.json")):
    if "fsa_small" in p:
        continue
    s = json.load(open(p))
    print(p.split("/")[2], s["pool"].split("/")[-1],
          "pass", s["pass_rate"], "any", s["pass_any_rate"], "fmt", s["format_rate"])
PYEOF
echo "ALL_DONE $(date -u +%FT%TZ)"
