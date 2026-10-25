# 2026-10-25 云端 48G 验证（RTX 4090-48G，智川云）

> 执行者：AI（用户指示"请你开始验证，跑一下冒烟，记录有价值的信息，估算跑完的时间"）。
> 这是**测量记录**，不是结论报告；判读归用户。
> 数字口径：〔实测〕= 本次真卡跑出来；〔推算〕= 由实测外推，未验证。

## 0. 一句话

48G 上 **vLLM 引擎本身健康**，但**训练步跑不起来**——卡在 `src/model.py:199` 的
`logits = logits.float()`：它把整块 `(B, T, V)` logits 转 fp32，单次分配要 **21.99 GiB**
〔实测〕。退到 HF 后端能跑通，`1.01 s/response`〔实测〕，但按此推 **9 run ≈ 93 GPU-h ≈ ¥194**，
比 24G 的 ¥143 更贵（单价 2.08 vs 1.56，而只快了 10%）。

**结论：48G 并没有解掉预算问题；能解预算的只有 vLLM，而 vLLM 现在被一行代码堵住。**

## 1. 环境（〔实测〕2026-10-25T01:47Z 开箱）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，**49,140 MiB**（47.37 GiB 可用），compute cap 8.9 |
| 驱动 | 580.119.02 |
| nvcc | **无** → vLLM 必须 `enforce_eager: true`（与 24G 同） |
| 容器配额 | `cpu.max 2200000 100000` = **22 核**；`memory.max` = **58 GiB**（`nproc`=192、`free`=503G 是宿主机视图） |
| 数据盘 | **`/root/rivermind-data` 49G**，开箱 3.7M → 干净，无上次会话残留 |
| 解释器 | `/opt/conda/bin/python` 3.11.13（`/usr/bin/python3` 3.10.12 零 ML 包；非交互 ssh 的 PATH 不含 conda） |
| 初始 torch | 2.8.0+cu128（装 vllm 后变 **2.13.0+cu130**，cuda 13.0，`is_available=True`） |
| 装 vllm 后 transformers | **5.19.0**（24G 时是 4.57.x 量级，注意版本漂移） |
| 平台 | 智川云（`/etc/custom-motd`），hostname `jupyter-2xc0vrtudfbcphgw` |
| 系统时钟 | **UTC**（比北京慢 8h）；`date` 显示 "Asia" 但实为 UTC，日志时间要加 8 |

### 与 24G 会话的三个差异
1. **github.com 可达**（`curl → 200`，3.5s）；24G 上是 000 不通。→ 本次**直接 `git clone`**，不再 scp。
2. **bootstrap 353 秒完成**（24G 约 15 min）：vllm 安装与 3.22 GiB 权重下载都显著更快。
3. pypi 走清华镜像（`/etc/pip.conf` 预置），modelscope 302 可达。

## 2. 通过的闸门（〔实测〕）

| 步骤 | 结果 |
|---|---|
| `git clone https://github.com/3hgL4/two-clocks-audit` | ✅ commit `0fe107b`，38 文件 |
| `tools/verify_pool.py`（纯标准库） | ✅ `result: consistent`，`d781f955…`，916,330 bytes，`configs/base.yaml` 匹配 |
| `tests/run_all.py` | ✅ 12/12 passed in 1.85s |

> 这是 CRLF 修复（DEVLOG D-09）在**第 3 个环境**验证通过（Windows 本机 / WSL / 云端 48G）。

## 3. 发现的两个缺陷

### 3.1 权重路径不一致（会让"新机器照着 bootstrap 装完"直接跑挂）
- `tools/cloud_bootstrap.sh:17` 默认 `MODEL_DIR="$ROOT/models/Qwen3-1.7B-Base"` → 落在**仓库内**；
- `configs/cloud24g.yaml:26` 写的是 `/root/rivermind-data/models/Qwen3-1.7B-Base` → 在**仓库外**。
- 本次实测路径是仓库内那一个。24G 那次显然用了 `MODEL_DIR=` 覆盖，但**没有任何文档记录这个覆盖**。
- 后果：任何人 clone 后跑 `cloud_bootstrap.sh` 再 `--rollout vllm`，会因模型目录不存在而失败。
- 本次处理：新写的 `configs/cloud48g.yaml` 指向实际路径；**未改** cloud24g.yaml 与 bootstrap（改动需申报）。

### 3.2 `score()` 的整块 fp32 logits（vLLM 路的真正堵点）
`src/model.py:196-199`：
```python
out = self.model(input_ids=input_ids, attention_mask=attn)
logits = out.logits                    # (B, Lmax, V)
if fp32 and logits.dtype != torch.float32:
    logits = logits.float()            # ← 单次分配 21.99 GiB（实测）
```
- 模块 docstring 自己写过"内存受词表轴支配，micro=8 × ~650 token ≈ 1.6 GB bf16"——但 `.float()`
  会再要一份 fp32 的全量副本，且 `logits` 与 `out` 同时存活。
- 实测反推：单次分配 23,609,737,216 B ÷ (151,669 × 4) = 38,915 个 token 行；而 cap=512、prompt≈200，
  预期行数约 5,700。**差约 7 倍，说明实际序列长度远超 cap，或者 batch 大于 micro。**
  这一点**未查清**（见 §6 待补测），但它不是"分账不够"能解释的。

## 4. 三次 vLLM 尝试（〔实测〕，全部 OOM，失败点不同）

| # | util | vLLM 占用 | 策略侧 OOM 时 | 失败点 |
|---|---|---|---|---|
| 1 | 0.60 | 28.42 GiB（权重 3.38 + KV 24.3） | 18.11 GiB，剩 124 MiB | `train.py:314 score(requires_grad=True)` |
| 2 | 0.45 | 21.31 GiB（KV 17.19） | 25.25 GiB，剩 170 MiB | 同上 |
| 3 | 0.35 | 16.57 GiB | 30.20 GiB，剩 46 MiB | 同上（策略侧继续涨到吃满） |
| 4 | 0.35（Br=512/micro=8） | 16.82 GiB | 单次要 21.99 GiB | `model.py:199 logits.float()` |

**读法**：vLLM 每次都正常起来（KV cache 分配成功，`enforce_eager` 生效），失败一律在策略侧；
且"给多少用多少"（18→25→30）表明内存随 micro 推进而增长，不是固定需求。
**"48G 就能跑 vLLM"这个假设，本次实测不成立。**

## 5. HF 后端跑通（可用基线，〔实测〕）

```
[r0 mu00] pass=0.188 r=+0.233 len=290 kl=+0.00e+00 clip=0.0000 gn=0.034 sig=0.04664 d=0.7356
[r0 mu01] pass=0.188 r=+0.233 len=290 kl=-7.84e-05 clip=0.0013 gn=0.031 sig=0.04288 d=0.7294
[r0] done: rollout 64.6s update 8.3s cum 64 resp, 0.020 GPU-h, ¥0.04
```
- **1.01 s/response**（64.6s / 64）〔实测〕，对比 24G 的 **1.12 s/resp**〔10-24 实测〕→ 只快 10%。
- 三条仪器可信判据成立：µ-step0 的 **KL 恰好 +0.00e+00**、`clip=0.0000`、σ/d 有限非零（0.047/0.736）。
- `pass=0.188`、`len=290`（cap 512 未截断）→ 与 24G 的 `pass=0.25 / len 318` 同量级。

## 6. 9 run 时长与费用估算

单 run = 32,768 responses。每 response 成本 = rollout + update。

| 路径 | s/resp | 单 run | 9 run 墙钟 | 9 run 费用 | 对 ¥100 |
|---|---|---|---|---|---|
| **48G + HF**（〔实测〕1.01 + 0.13 update） | 1.14 | **10.4 h** | **93 h ≈ 3.9 天** | **¥194** | ✗ 超 94% |
| 24G + HF（〔10-24 实测〕1.12） | 1.20 | 10.9 h | 98 h | ¥153 | ✗ 超 53% |
| **48G + vLLM**（〔推算〕0.17 + 0.13） | 0.30 | **2.7 h** | **25 h ≈ 1.0 天** | **¥51** | ✅ 留 ¥49 余量 |
| 48G + vLLM（保守〔推算〕0.25 + 0.13） | 0.38 | 3.5 h | 31 h | ¥65 | ✅ |
| 48G + HF，缩到 6 run（2 seeds） | 1.14 | 10.4 h | 62 h | ¥129 | ✗ 超 29% |

〔推算〕依据：24G 上 vLLM（cap=128）实测 **0.17 s/resp**；48G 的 KV 更宽，吞吐只会更好不会更差，
但**没有在 vLLM 跑通训练步之前，这个 0.17 不能当作 48G 的承诺值**。

**两个反直觉结论**：
1. **48G 的 HF 路径比 24G 更贵**（¥194 vs ¥153）——单价涨 33%，速度只涨 10%。所以"升到 48G"在 HF 路径上是负收益。
2. **预算能否成立，完全取决于 vLLM 那 3.4 倍加速**；vLLM 一旦跑通，9 run 只要 ¥51，留出一半预算给 D2/D6 扩展臂。

## 7. 待补测（本次没做，因为要先改代码）

1. **`logits.float()` 分块或改 bf16 后**，vLLM 路能否跑通、真实 s/resp 是多少。这是唯一能定预算的测量。
2. 实测序列长度分布（`samples.jsonl` 里 `resp_len`）——用来解释 21.99 GiB 的 7 倍偏差。
3. `micro_batch_responses` 的安全上界（本次 micro=8 必爆，micro=2 也爆，但爆在更晚）。

## 8. 本次花费

开机 2026-10-25T01:47Z → 收工约 02:12Z，**约 25 分钟 ≈ ¥0.9**（¥2.08/h）。
其中四次 GPU 运行累计不到 6 分钟，其余是装环境与下载。**实例仍在计费，不用请关机。**
