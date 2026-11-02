#!/usr/bin/env bash
# Reproduce the audit.
#
#   ./reproduce.sh verdict  one command: recompute all claims from runs/ (0 GPU, no retraining)
#   ./reproduce.sh dry     local only: tests + CPU dry-run (0 GPU, 0 元, ~1 min)
#   ./reproduce.sh full    9 runs x 4090-24G (expect ¥22-54) + analysis
#   ./reproduce.sh analyse  rebuild the table/figures from existing runs/
#
# `full` refuses to start unless tests pass and the prompt pool hash matches:
# a rented GPU is the most expensive place to discover a broken hash guard.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-dry}"
PY="${PY:-python}"

step() { printf '\n=== %s ===\n' "$*"; }

case "$MODE" in
  verdict)
    # One command to recompute every headline claim from the runs/ data in this
    # repo (no GPU, no retraining). Any failure aborts (set -e): ALL_DONE means
    # all six blocks actually ran.
    step "dependency gate"
    "$PY" -c "import yaml, numpy, pandas" 2>/dev/null || {
      echo "missing python deps -- run: $PY -m pip install -r requirements.txt"
      exit 1
    }
    step "C2-C7 + P1 (residual claims vs paper)"
    "$PY" analysis/residual_claims.py --runs runs
    step "C8 (slope ratio, paper Eq.(5) = 6.1x)"
    "$PY" analysis/slope_ratio.py
    step "C10 (b x mu 2x2 -> mu_is_primary)"
    "$PY" analysis/bmu_2x2.py --runs runs --ab runs_ab2x2
    step "C11 (no-clip ablation -> clip_is_inert)"
    "$PY" analysis/noclip_ablation.py
    step "C12 (R4-A: mu=1 19-rollout continuation grid)"
    "$PY" analysis/r4a_grid.py --runs runs_r4a --base runs
    step "claim trace (16 claims, strict: fail on pending evidence)"
    "$PY" analysis/trace.py --strict
    echo "ALL_DONE (verdict)"
    ;;
  dry)
    step "unit tests"
    "$PY" tests/run_all.py
    step "CPU dry-run (stub model + fake backend)"
    rm -rf runs/dryrun
    "$PY" src/train.py --config configs/dryrun.yaml --device cpu
    step "resume smoke test"
    "$PY" src/train.py --config configs/dryrun.yaml --device cpu --resume --num-rollouts 3
    step "schema gate"
    "$PY" tools/check_schema.py runs/dryrun --strict
    step "analysis on dry-run output"
    "$PY" analysis/make_table.py --runs runs --out analysis/out
    "$PY" analysis/make_figures.py --runs runs --out analysis/out/fig/curves
    step "claim trace"
    "$PY" analysis/trace.py || true
    ;;

  full)
    step "pre-flight: tests"
    "$PY" tests/run_all.py
    step "pre-flight: pool hash"
    "$PY" - <<'EOF'
import sys; sys.path.insert(0, "src")
import data, yaml
cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
pool = data.load_pool(cfg["data"]["pool"], cfg["data"]["sha256"])
print(f"pool ok: {len(pool)} items, sha256 verified")
EOF
    for mu in 1 8 16; do
      for seed in 0 1 2; do
        tag="mu${mu}_s${seed}"
        step "train mu=$mu seed=$seed -> runs/$tag"
        "$PY" src/train.py --config "configs/mu${mu}.yaml" --seed "$seed" --tag "$tag"
        "$PY" tools/check_schema.py "runs/$tag" --strict
      done
    done
    step "analysis"
    "$PY" analysis/make_table.py --runs runs --out analysis/out
    "$PY" analysis/make_figures.py --runs runs --out analysis/out/fig/curves
    "$PY" analysis/trace.py --strict || true
    echo
    echo "now fill REPORT.md from analysis/out/table.md -- do not retype numbers"
    ;;

  analyse)
    "$PY" tools/check_schema.py runs/*/ --strict
    "$PY" analysis/make_table.py --runs runs --out analysis/out
    "$PY" analysis/make_figures.py --runs runs --out analysis/out/fig/curves
    "$PY" analysis/trace.py
    ;;

  *)
    echo "usage: $0 {dry|full|analyse}" >&2
    exit 2
    ;;
esac
