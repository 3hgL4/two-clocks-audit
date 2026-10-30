#!/bin/bash
# SAC-mu main grid queue (14 runs). Fail-fast: any run rc!=0 stops the queue.
# Versioned copy: tools/grid_queue.sh (deploy: scp to /root/rivermind-data/, nohup).
# Disk note: data disk expanded to 99G on 2026-10-30 -- adapters are KEPT for
# every run (execution card section 6 "delete adapters after upload" is voided;
# clean only after upload + sha256 verification).
cd /root/rivermind-data/two-clocks-audit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/opt/conda/bin/python
LOGD=/root/rivermind-data/grid_logs
MAIN=/root/rivermind-data/grid.log
mkdir -p $LOGD
echo "grid start $(date -u +%FT%TZ)" >> $MAIN
run() {
  tag=$1; shift
  echo "run start $(date -u +%FT%TZ) tag=$tag cmd=$PY src/train.py $*" >> $MAIN
  $PY src/train.py "$@" > $LOGD/$tag.log 2>&1
  rc=$?
  echo "run end $(date -u +%FT%TZ) tag=$tag rc=$rc" >> $MAIN
  if [ $rc -ne 0 ]; then
    echo "GRID_STOP tag=$tag rc=$rc -- waiting for human" >> $MAIN
    exit 5
  fi
}
for t in sac_s1 sac_s2 sacS_s0 sacS_s1 sacS_s2 sacR_s0 sacR_s1 sacR_s2; do
  run $t --config configs/$t.yaml
done
for s in 0 1 2; do
  run mu32_s$s --config configs/cloud48g_main.yaml --mu 32 --seed $s --tag mu32_s$s
  run mu64_s$s --config configs/cloud48g_main.yaml --mu 64 --seed $s --tag mu64_s$s
done
echo "grid done $(date -u +%FT%TZ) all 14 rc=0" >> $MAIN
