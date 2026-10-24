#!/usr/bin/env bash
# cloud_bootstrap.sh - 把一台"智川云 4090-24G / PyTorch-2.8.0-CUDA12.8-Py3.11-Ubuntu22"
# 实例推到"能跑 smoke"的状态。幂等：重复执行只会跳过已完成的步骤。
#
# 用法（在云端项目根目录）：
#   bash tools/cloud_bootstrap.sh                # 全流程（含 vllm）
#   bash tools/cloud_bootstrap.sh --skip-vllm    # 只用 HF 后端时不装 vllm，省 10-15 min
#
# 事实依据：notes/2026-10-24_cloud24g/ENV.md（同一批实测，含五个坑）

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

# 唯一有包的解释器。非交互 ssh 的 PATH 不含 conda，裸 python3 会落到 /usr/bin/python3 3.10（零 ML 包）。
PY="${PY:-/opt/conda/bin/python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/Qwen3-1.7B-Base}"
SKIP_VLLM=0
for a in "$@"; do [[ "$a" == "--skip-vllm" ]] && SKIP_VLLM=1; done

ok()   { printf '[ok]   %s\n' "$*"; }
skip() { printf '[skip] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
die()  { printf '[fail] %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s ==\n' "$*"; }

step "1/6 硬件与解释器"
command -v nvidia-smi >/dev/null || die "没有 nvidia-smi，这不是 GPU 机器"
[[ -x "$PY" ]] || die "$PY 不存在。用 PY=<绝对路径> bash tools/cloud_bootstrap.sh 指定解释器"
"$PY" -c 'import sys; assert sys.version_info[:2] >= (3, 10), sys.version' \
  || die "$PY 版本过低（需要 >=3.10；本次云端验证过的是 3.11.13）"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' \
  || warn "$PY 不是 3.11；代码在 3.10/3.11/3.12 上都过测试，但云端只实测过 3.11"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
df -h . | tail -1
if command -v nvcc >/dev/null; then
  ok "nvcc 存在"
else
  warn "无 nvcc：vllm 必须 enforce_eager（configs/cloud24g*.yaml 已设 rollout.vllm.enforce_eager: true）"
fi

step "2/6 题池哈希闸门（纯标准库，装依赖之前）"
"$PY" tools/verify_pool.py || die "题池哈希不一致：不要继续。先查行尾（DEVLOG D-09）或题池被改"

step "3/6 依赖"
"$PY" -c "import torch, transformers, peft, requests" 2>/dev/null \
  && ok "torch/transformers/peft/requests 已具备" \
  || { echo "[info] 装 requirements.txt"; "$PY" -m pip install -q -r requirements.txt || die "requirements 安装失败"; }
if [[ $SKIP_VLLM -eq 1 ]]; then
  skip "按 --skip-vllm 跳过 vllm（本次主线可用 --rollout hf 跑完）"
elif "$PY" -c "import vllm" 2>/dev/null; then
  ok "vllm 已装：$("$PY" -c 'import vllm; print(vllm.__version__)')"
else
  echo "[info] 装 vllm==0.32.0（会顺带把 torch 换成 2.13.0+cu130，10-15 min）"
  "$PY" -m pip install -q "vllm==0.32.0" || warn "vllm 安装失败；不阻塞，改 --rollout hf 即可"
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

step "4/6 权重（ModelScope CDN，8 路 range，约 13 min / 3.3 GB）"
if [[ -f "$MODEL_DIR/model.safetensors" ]]; then
  skip "已存在 $MODEL_DIR"
else
  "$PY" tools/fetch_model.py --model Qwen/Qwen3-1.7B-Base --out "$MODEL_DIR" --workers 8 \
    || die "权重下载失败：先测 modelscope.cn 连通性，别用 snapshot_download（实测 1 MB/s）"
fi

step "5/6 回归测试（CPU，12 项）"
"$PY" tests/run_all.py || die "回归失败：先修，再花钱跑 GPU"

step "6/6 就绪"
{
  echo "host: $(hostname)"
  echo "date_utc: $(date -u +%FT%TZ)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
  "$PY" -c "import sys,torch,transformers,vllm; print('python',sys.version.split()[0],'torch',torch.__version__,'transformers',transformers.__version__,'vllm',vllm.__version__)" 2>/dev/null \
    || "$PY" -c "import sys,torch,transformers; print('python',sys.version.split()[0],'torch',torch.__version__,'transformers',transformers.__version__,'vllm none')"
} | tee -a env.txt

cat <<EOF

下一步（本次实测跑通的命令）：

  cd "$ROOT" && $PY src/train.py \\
    --config configs/cloud24g_smoke.yaml --rollout hf \\
    --responses-per-rollout 32 --tag smoke_\$(date +%m%d_%H%M)

关机前记得把 runs/*/{metrics.jsonl,samples.jsonl,summary.json,run_meta.json,config.resolved.yaml} 拉回本地。
EOF
