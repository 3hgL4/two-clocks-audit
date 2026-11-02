#!/bin/sh
# One-click reproduction on a fresh gpuhome cloud box (no prior setup).
#
#   sh scripts/repro_cloud.sh            # = verdict: code -> evidence check ->
#                                        #   frozen-rule judgment (0 GPU, ~1 min after code)
#   sh scripts/repro_cloud.sh smoke      # + unit tests + CPU dry-run (no GPU, ~2 min)
#   sh scripts/repro_cloud.sh gpu-smoke  # + GPU smoke train (fsa_smoke_b x2 + bit-exact
#                                        #   rollout-id audit; ~10 min, needs a GPU card)
#   sh scripts/repro_cloud.sh arm        # + one real arm: fsa_small_a x3 seeds + heldout
#                                        #   evals (~1-2 h GPU) -> numbers comparable to
#                                        #   evidence/evals/eval_fsa_small_a_*
#
# Designed for the gpuhome PyTorch image (/opt/conda/bin/python has torch/vllm).
# github IS reachable from the box but flaky (HTTP2 framing errors / TLS stalls),
# so code arrives via codeload tarball with resume+retry; weights arrive from
# ModelScope via tools/fetch_model.py (parallel ranges, sha256-verified).
# Overridables: REPO_DIR, MODEL_DIR (env).
set -u
REPO_DIR=${REPO_DIR:-/root/fsa-repro}
MODEL_DIR=${MODEL_DIR:-/root/rivermind-data/two-clocks-audit/models/Qwen3-1.7B-Base}
TARBALL=/root/fsa_repo.tar.gz
URL=https://codeload.github.com/3hgL4/two-clocks-audit/tar.gz/refs/heads/fsa-resample
PY=/opt/conda/bin/python; [ -x "$PY" ] || PY=python3

step() { echo "=== $(date -u +%FT%TZ) $*"; }

# --- 1) code ---
if [ ! -f "$REPO_DIR/reproduce.sh" ]; then
  step "fetch code tarball (resume + retry; github is flaky from the box)"
  curl -sSL -C - --retry 20 --retry-delay 2 --retry-all-errors \
       --speed-time 30 --speed-limit 10000 -o "$TARBALL" "$URL"
  rm -rf "$REPO_DIR" && mkdir -p "$REPO_DIR"
  tar xzf "$TARBALL" -C "$REPO_DIR" --strip-components=1
fi
cd "$REPO_DIR" || exit 1
step "code ready (fsa-resample tarball HEAD 9ee4b00)"

# --- 2) evidence integrity ---
step "sha256 check of evidence/ against MANIFEST"
( cd evidence && sha256sum --check MANIFEST.sha256 --quiet ) || { echo "EVIDENCE MISMATCH"; exit 1; }
echo "evidence OK ($(grep -c . evidence/MANIFEST.sha256) files)"

# --- 3) judgment (0 GPU, the pre-registered verdict) ---
step "frozen-rule verdict from evidence/"
$PY analysis/verdict.py

MODE=${1:-verdict}
case "$MODE" in verdict) echo "ALL_DONE (verdict)"; exit 0;; esac

# --- 4) tests + CPU dry-run ---
step "unit tests"
$PY tests/run_all.py
step "CPU dry-run (stub model + fake backend)"
# Many-core boxes (192C) thrash on tiny stub matrices: torch spawns one OMP
# thread per core and the sync overhead turns a <1 min dry-run into 30+ min.
# Cap the threads; the GPU modes below are unaffected.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
$PY src/train.py --config configs/dryrun.yaml --device cpu
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
$PY src/train.py --config configs/dryrun_resample_b.yaml --device cpu
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
$PY src/train.py --config configs/dryrun_resample_c.yaml --device cpu
case "$MODE" in smoke) echo "ALL_DONE (smoke)"; exit 0;; esac

# --- 5) GPU: weights, then smoke train with bit-exactness audit ---
case "$MODE" in gpu-smoke|arm) ;; *) echo "usage: $0 {verdict|smoke|gpu-smoke|arm}"; exit 2;; esac
step "GPU check"
$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  step "fetch Qwen3-1.7B-Base from ModelScope (3.4GB, parallel ranges)"
  mkdir -p "$(dirname "$MODEL_DIR")"
  $PY tools/fetch_model.py --model Qwen/Qwen3-1.7B-Base --out "$MODEL_DIR" --workers 8
fi
step "GPU smoke train: fsa_smoke_b run twice, then bit-exact rollout audit"
$PY src/train.py --config configs/fsa_smoke_b.yaml --tag fsa_smoke_b_a
$PY src/train.py --config configs/fsa_smoke_b.yaml --tag fsa_smoke_b_b
$PY tools/audit_rollout_ids.py runs/fsa_smoke_b_a/rollouts/r000.npz \
     --compare runs/fsa_smoke_b_b/rollouts/r000.npz --group 8
echo "ALL_DONE (gpu-smoke)"
case "$MODE" in gpu-smoke) exit 0;; esac

# --- 6) one real arm + heldout eval (numbers comparable to evidence/) ---
step "train fsa_small_a x3 seeds + heldout eval each"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 0 1 2; do
  tag=fsa_small_a_s$s
  $PY src/train.py --config configs/fsa_small_a.yaml --seed $s --tag $tag
  $PY tools/eval_heldout.py \
      --pool data/heldout_counting_512.jsonl \
      --model "$MODEL_DIR" \
      --adapter runs/${tag}/adapters/r001 \
      --n 4 --temperature 1.0 --seed 0 \
      --tag eval_${tag} --out out/eval_heldout
  grep -h '"pass_rate"' out/eval_heldout/eval_${tag}/summary_*.json || true
done
echo "ALL_DONE (arm)"
