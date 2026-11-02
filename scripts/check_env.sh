#!/bin/sh
# 30-second environment report for a fresh gpuhome box: tells you which
# repro_cloud.sh mode this box can run, BEFORE you try. Single file, no repo
# needed -- paste via:  sh <(curl -sL https://raw.githubusercontent.com/3hgL4/two-clocks-audit/fsa-resample/scripts/check_env.sh)
PY=/opt/conda/bin/python; [ -x "$PY" ] || PY=python3
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tail -1
echo "=== network (each needs ~5s; timeouts mean that route is dead) ==="
for u in "https://codeload.github.com" "https://www.modelscope.cn" "https://pypi.org"; do
  code=$(timeout 8 curl -s -o /dev/null -w '%{http_code}' "$u" 2>/dev/null)
  echo "$u -> ${code:-TIMEOUT}"
done
echo "=== python deps for smoke/arm ==="
$PY -c "import peft, transformers, accelerate; print('deps OK')" 2>&1 | tail -1
echo "=== disk ==="
df -h /root 2>/dev/null | tail -1
df -h /root/rivermind-data 2>/dev/null | tail -1 || echo "no data disk"
echo "=== existing assets ==="
ls /root/rivermind-data/two-clocks-audit/models/Qwen3-1.7B-Base/config.json 2>/dev/null && echo "weights present (saves 3.4GB ModelScope fetch)" || echo "weights absent (will fetch from ModelScope)"
echo "=== verdict: run which mode? ==="
GPU=$($PY -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null)
if [ "$GPU" = "1" ]; then
  echo "GPU box -> verdict + smoke + gpu-smoke + arm all runnable (sh scripts/repro_cloud.sh <mode>)"
else
  echo "no GPU -> verdict + smoke runnable; gpu-smoke/arm need a card"
fi
