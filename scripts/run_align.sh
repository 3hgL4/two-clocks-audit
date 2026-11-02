#!/bin/sh
# Compute-aligned A x11 driver (2026-11-02): s0 -> eval -> s1 -> eval,
# s2 only if a time gate (06:00 local) still leaves room to finish by 09:00.
cd /root/rivermind-data/format-signal-audit/repo || exit 1
PY=/opt/conda/bin/python
BASE=/root/rivermind-data/two-clocks-audit/models/Qwen3-1.7B-Base
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

step() { echo "=== $(date -u +%FT%TZ) $*"; }

run_one() {
  s=$1
  tag=fsa_a_align_s${s}
  step "train $tag"
  $PY src/train.py --config configs/fsa_a_align.yaml --seed $s --tag $tag \
      > /tmp/${tag}.log 2>&1 || echo "FAILED train $tag"
  tail -1 runs/${tag}/metrics.jsonl > /dev/null 2>&1 \
      && echo "attempted_check $tag: $($PY -c "import json;att=sum(json.loads(l).get('completion_tokens_attempted',0) or 0 for l in open('runs/$tag/metrics.jsonl'));print(att)" 2>/dev/null)" || true
  step "eval $tag"
  $PY tools/eval_heldout.py --pool data/heldout_counting_512.jsonl --model $BASE \
      --adapter runs/${tag}/adapters/r010 --n 4 --temperature 1.0 --seed 0 \
      --tag eval_${tag} --out out/eval_heldout > /tmp/eval_${tag}.log 2>&1 \
      && grep -h '"pass_rate"' out/eval_heldout/eval_${tag}/summary_*.json \
      || echo "FAILED eval $tag"
}

for s in 0 1; do
  run_one $s
done

if [ "$(date +%H%M)" -lt 0600 ]; then
  run_one 2
else
  step "SKIP s2 (time gate: past 06:00 local, s2 cannot finish by 09:00)"
fi

step "align summary"
$PY - <<'PYEOF' || echo "FAILED summary"
import glob, json
for p in sorted(glob.glob("out/eval_heldout/eval_fsa_a_align_*/summary_*.json")):
    s = json.load(open(p))
    print(p.split("/")[2], "pass", s["pass_rate"], "any", s.get("pass_any_rate"), "fmt", s["format_rate"])
PYEOF
echo "ALL_DONE $(date -u +%FT%TZ)"
