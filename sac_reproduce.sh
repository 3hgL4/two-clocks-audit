#!/usr/bin/env bash
# One command to recompute the SAC-mu audit tables (P1/P2/P3/P5') from the
# runs_sac/, runs_fixed/ and runs/mu16_s*/ data committed to this branch.
# No GPU, no retraining. set -e: ALL_DONE means every table actually ran.
#
#   ./sac_reproduce.sh          # four tables + JSON dump (~seconds)
#   ./sac_reproduce.sh rescore  # + deterministic mu16 rescore rebuild (needs
#                               #   local rollouts/*.npz, tokenizers pkg; the
#                               #   rebuilt mu16_rescored_r6r7.json is committed,
#                               #   so this is an offline verification, not required)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"

step() { printf '\n=== %s ===\n' "$*"; }

step "dependency gate"
"$PY" -c "import yaml, numpy, pandas" 2>/dev/null || {
  echo "missing python deps -- run: $PY -m pip install pyyaml numpy pandas"
  exit 1
}

step "data sanity (runs_sac x9, runs_fixed x6, mu16 baseline json)"
for r in sac_s0 sac_s1 sac_s2 sacS_s0 sacS_s1 sacS_s2 sacR_s0 sacR_s1 sacR_s2; do
  test -f "runs_sac/$r/metrics.jsonl" || { echo "missing runs_sac/$r/metrics.jsonl"; exit 1; }
done
for r in mu32_s0 mu32_s1 mu32_s2 mu64_s0 mu64_s1 mu64_s2; do
  test -f "runs_fixed/$r/metrics.jsonl" || { echo "missing runs_fixed/$r/metrics.jsonl"; exit 1; }
done
test -f analysis/out/mu16_rescored_r6r7.json || { echo "missing mu16_rescored_r6r7.json"; exit 1; }
echo "data OK"

if [ "${1:-}" = rescore ]; then
  step "deterministic mu16 rescore rebuild (offline verification)"
  "$PY" tools/rescore_mu16.py
fi

step "SAC-mu tables: P1 / P2 / P3 / P5'"
"$PY" analysis/sac_p1p3.py

echo "ALL_DONE (sac verdict)"
