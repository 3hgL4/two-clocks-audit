#!/usr/bin/env bash
# Reproduce the audit.
#
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
