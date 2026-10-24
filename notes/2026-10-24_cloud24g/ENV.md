# 云端环境快照：智川云 RTX 4090-24G（2026-10-24 会话，关机前实测）

> 采集时间：2026-10-24 16:44 UTC（实例 we9h402bfkqfdwgs，hostname `jupyter-we9h4o2bfkqfdwgs`）
> 目的：让"下一次重建实例"从 40 分钟试错降到 3 条命令。所有数字都是 `ssh` 上去实测的，不是推断。

---

## 1. 硬件与平台

| 项 | 实测值 | 采集命令 |
|---|---|---|
| 平台 | 智川云（`/etc/custom-motd`） | — |
| 实例 ID | `we9h402bfkqfdwgs` | — |
| GPU | RTX 4090，24,564 MiB，compute cap **8.9**（sm_89） | `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv` |
| 驱动 / CUDA | 580.119.02 / **CUDA Version 13.0**（驱动层） | `nvidia-smi` |
| **nvcc** | **不存在**（非 Devel 镜像） | `which nvcc` → not found |
| CPU | `nproc` 报 128（宿主机 EPYC 7K62），**cgroup 实际配额 15 核**（`cpu.max = 1500000 100000`） | `nproc` / `cat /sys/fs/cgroup/cpu.max` |
| 内存 | `free -g` 报 total **1007**（宿主机口径），**cgroup 实际上限 100 GiB**（`memory.max = 107374182400`） | `free -g` / `cat /sys/fs/cgroup/memory.max` |
| OS | Ubuntu 22.04.5 LTS，kernel 5.15.0-94-generic，glibc 2.35 | `/etc/os-release` |
| 系统盘 | overlay **30 G**，已用 8.3 G（镜像自带） | `df -hT /` |
| **数据盘** | ext4 **49 G**，挂载点 **`/root/rivermind-data`** | `df -hT` |

> **两个"宿主视角"数字已澄清**：`nproc`/`free` 读的是宿主机（128 核 / 1007 GiB），容器真实配额是 **15 核 / 100 GiB**（cgroup 实测）。CPU 侧不是这批实验的瓶颈（生成全在 GPU 上、单进程），但别按 128 核去设 dataloader 并发。

---

## 2. 解释器：这台机器上有两个，只有一个能用

| 路径 | 版本 | 包情况 |
|---|---|---|
| **`/opt/conda/bin/python`** | **3.11.13**（conda-forge base） | **全部 ML 包都在这里**；`run_meta.json` 记录的 python 就是它 |
| `/usr/bin/python3` | 3.10.12 | **零 ML 包**（torch/numpy/transformers 全 MISSING） |

**坑（实测）**：非交互 ssh（`ssh host 'cmd'`）的 PATH **不含 conda**，`python3` 会落到 3.10.12 那个空环境；交互式登录（`bash -lc`、终端里）才是 `/opt/conda/bin/python`。
→ 脚本与命令里**一律写绝对路径 `/opt/conda/bin/python`**，不要依赖 `python3`。

---

## 3. 关键包版本（装完依赖后的实测态）

| 包 | 版本 | 备注 |
|---|---|---|
| torch | **2.13.0+cu130** | 镜像原本是 2.8.0+cu128，`pip install vllm` 把它换掉了（驱动 580 可承载 CUDA 13.0） |
| vllm | **0.32.0** | 从清华源装；本机无 nvcc → 必须 `enforce_eager` |
| transformers | **5.19.0** | |
| tokenizers | 0.23.3 | |
| peft | 0.21.2 | |
| accelerate | 1.15.0 | |
| modelscope | 1.40.2 | 只用来查文件清单/sha256，**不用它的 downloader** |
| numpy | 2.3.2 | |
| safetensors | 0.8.0 | |
| tiktoken | 0.14.0 | |
| trl / datasets / wandb | **都没装** | GRPO 是自己写的循环；日志自己写 jsonl。这三个不是依赖，别装 |

完整清单见同目录 `pip-freeze.txt`（已剔除 `@ file:///home/conda/...` 这类不可复现的 conda 构建路径与 nvidia-* 子包）。

---

## 4. 网络（决定装配走哪条路，实测）

| 目标 | 结果 |
|---|---|
| `pypi.org` | **直连不通**（curl 000）→ 但镜像已在 `/etc/pip.conf` 预设清华源，装包不用自己配 |
| `github.com` | **22 与 443 都不通** → 代码用 `scp` 上传，**云端没有 `.git` 目录** |
| ModelScope CDN | 通。SDK `snapshot_download` 实测 **~1.0 MB/s**；同源 8 路 range 并发 **8–22 MB/s** → 用 `tools/fetch_model.py`，3.22 GiB / 13 min |

---

## 5. 五个坑（都是这次实际翻的车，重建时按顺序避开）

1. **无 nvcc** → vLLM 的 torch.compile 路径崩，必须 `rollout.vllm.enforce_eager: true`。
2. **`max_model_len` 默认 32768** → 光 KV 就要 ~3.5 GB，24G 上直接 OOM，必须显式 `1024`。
3. **24 GiB 里 vLLM 与 HF 打分不能共存**：`gpu_memory_utilization` 从 0.55→0.50→0.28 三档全 OOM（HF 侧打分要涨到 13.6 GiB）→ 本次改走 **HF 后端**（`--rollout hf`）。
4. **`cap=128` 造成 pass_rate=0 的假象**：完成长度均值 112.6/128，全被截断 → 看起来"模型一题不会"。`cap=512` 后 pass=0.25、zero_var=0.0。
5. **数据盘挂载点是 `/root/rivermind-data`**，不是 AutoDL 习惯的 `/root/autodl-tmp`。写错就是往 30 G 系统盘上灌 3.3 G 权重。

---

## 6. 哪些东西关机就没了（结论：不影响任何结果）

| 对象 | 体积 | 可再生？ | 关机后的处置 |
|---|---|---|---|
| 代码 | <5 MB | 是（本地 4 个 commit 全部已推送 `3hgL4/two-clocks-audit`） | 云端**无 `.git`、无未提交改动** → 不丢 |
| Qwen3-1.7B-Base | 3.3 G | 是，13 min | 丢弃，重建时重拉 |
| LoRA adapter (`runs/_adapter`) | 267 M | 是，一次 smoke 的产物 | 只取了 `AUDIT_ADAPTER_META.json`，权重丢弃 |
| `runs/hfsmoke/state.pt` | 799 M | 只在"续跑"时有用 | 丢弃 |
| **`runs/*/metrics.jsonl` / `samples.jsonl` / `summary.json` / `config.resolved.yaml` / `run_meta.json` / 日志** | <50 KB | **否** | **已全部回本地**（同目录：`metrics.jsonl` `samples.jsonl` `config.resolved.yaml` `run_meta.json` `summary.hfsmoke.json` `calib16.json` `smoke.log` `hfsmoke.log`） |

**唯一不可再生的就是那几十 KB 日志，它们已经在本地了。** 所以"关机"这个动作不损失任何结论；损失的是 40 分钟的装配时间（可用 `tools/cloud_bootstrap.sh` 压到 ~15 min，其中 13 min 是下权重）。

---

## 7. 重建：三条命令

```bash
# 0) 本地：GitHub 不通，代码走 scp（把端口/密码换成当次实例的）
scp -P <port> -r "two-clocks-audit" root@<host>:/root/rivermind-data/

# 1) 云端：一键幂等装配（验题池 → 装依赖 → 拉权重 → 回归测试）
bash /root/rivermind-data/tools/cloud_bootstrap.sh

# 2) 云端：跑 smoke（本次实测可用配置）
cd /root/rivermind-data && /opt/conda/bin/python src/train.py \
  --config configs/cloud24g_smoke.yaml --rollout hf \
  --responses-per-rollout 32 --tag <tag>
```

本次跑通的完整命令与产物在 `run_meta.json`（`argv` 字段）与 `config.resolved.yaml` 里，是最高权威的"当时到底怎么跑的"。

---

## 8. 操作卡 §八"待回填 8 项"的实测答案

| 待回填项 | 实测 |
|---|---|
| 镜像名/ID | `PyTorch/2.8.0-CUDA12.8-Py3.11-Ubuntu22`（装机后 torch 被升到 2.13.0+cu130） |
| Python | 3.11.13 @ `/opt/conda/bin/python`（另有 3.10.12 @ `/usr/bin/python3`，无包） |
| torch / CUDA | 原始 2.8.0+cu128 → 本次实际 2.13.0+cu130；驱动 580.119.02 报 CUDA 13.0 |
| 是否预装 vllm | **否**（本次 `pip install vllm` 装的 0.32.0） |
| 是否预装 transformers | 是（装完为 5.19.0） |
| 数据盘挂载点 | `/root/rivermind-data`（49 G ext4） |
| root / conda | root；conda 在 `/opt/conda`（非交互 PATH 不含） |
| 出网 | 22 端口对 ssh 开放；pypi 直连不通（镜像已配清华源）；github 全不通 |
| 关机策略 | 关机后 **7 天释放**，系统盘/数据盘随之删除（平台提示，用户转述） |
