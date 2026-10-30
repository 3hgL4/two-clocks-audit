#!/bin/bash
# P4 -> gate -> smoke chain. Survives session disconnect (nohup).
#
# Repo copy (tools/p4_then_smoke.sh). Deploy by scp to /root/rivermind-data/
# and launch with nohup BEFORE p4_mu32/mu64 finish. Never overwrite the
# script in place on a live instance while it is waiting in the sleep loop:
# bash reads scripts incrementally and an in-place rewrite can corrupt the
# running instance's execution.
#
# v2 change (2026-10-30): record the smoke launch itself in $LOG (pid + cmd).
# v1 only logged "chain start" and "gate_rc", so the fact that sac_s0 was
# auto-launched had no log evidence once the process exited.
cd /root/rivermind-data/two-clocks-audit
PY=/opt/conda/bin/python
LOG=/root/rivermind-data/chain.log
echo "chain start $(date -u +%FT%TZ)" >> $LOG
while true; do
  a=$(grep -c "done:" /root/rivermind-data/p4_mu32.log 2>/dev/null)
  b=$(grep -c "done:" /root/rivermind-data/p4_mu64.log 2>/dev/null)
  if grep -q "Traceback" /root/rivermind-data/p4_mu32.log /root/rivermind-data/p4_mu64.log 2>/dev/null; then
    echo "P4_ERROR traceback seen, chain stops" >> $LOG
    exit 4
  fi
  if [ "$a" -ge 2 ] && [ "$b" -ge 2 ]; then break; fi
  sleep 60
done
$PY tools/p4_gate.py runs/p4_mu32 runs/p4_mu64 --json /root/rivermind-data/p4_gate_verdict.json > /root/rivermind-data/p4_gate.log 2>&1
rc=$?
echo "gate_rc=$rc (0=PASS 3=FAIL)" >> $LOG
if [ $rc -eq 0 ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  $PY src/train.py --config configs/sac_s0.yaml > /root/rivermind-data/sac_s0.log 2>&1 &
  smoke_pid=$!
  echo "smoke start $(date -u +%FT%TZ) pid=$smoke_pid cmd=$PY src/train.py --config configs/sac_s0.yaml" >> $LOG
  wait $smoke_pid
  echo "smoke_rc=$?" >> $LOG
else
  echo "GATE_FAIL: no smoke launch, waiting for human" >> $LOG
fi
