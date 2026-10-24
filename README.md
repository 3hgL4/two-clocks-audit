# two-clocks-audit

对 arXiv:2610.18853（*Two Clocks*, CMU / U Toronto）的一次**可复现审计**，不是复现、不是改进。

一句话：**论文用两座"钟"（drift clock、progress clock）给出"每批数据该做几次更新"的处方，但这个处方建立在一个被显式搁置的假设上——它忽略了同一批 rollout 被反复用于多次更新造成的策略陈旧（stale）代价。本项目在 ¥100 预算内，把这个代价测出来。**

---

## 1. 增量在哪（不是"复现原文"）

| 原文做了什么 | 本项目做什么 |
|---|---|
| Eq.(3) 定义 drift clock `d = κ‖ḡ‖/σ` | 逐 micro-batch 实测 `‖ḡ‖` 与 `σ`，得到 `d` 的轨迹 |
| Eq.(5) 给出 `N_τ = N_min(1 + Br/B*)`，预测 µ↑ → `N_τ` 变大（6.1×） | 在同一 `Br`、同一 token 预算下，实测 µ ∈ {1, 8, 16} 的 `N_τ` 比值 |
| **假设 A2 原文**：*"For progress we treat the mini-batches within a rollout as fresh and ignore policy staleness, whose effect on importance ratios and clipping we model separately below."* | 直接测量 A2 忽略的那一项：`KL(π_θ ‖ π_rollout)` 在 µ 个更新步内的累积 |
| 全文 **0 处**报告响应长度 | 记录 `resp_len_mean` 随 µ 的漂移 |
| 报告 clip 占 0.5% token 却占 50–70% 梯度范数 | 记录 `clip_frac_*` 与 `‖g_clip‖/‖g‖` 随 µ 的变化 |

**措辞纪律**：本项目只说"测到多少 / 与预测差多少"，不说"论文错了 / 我提出了改进"。A2 是作者自己写明的简化，量化它的代价不等于推翻它。

---

## 2. 目录

```
configs/         base.yaml + mu{1,8,16}.yaml（arm 只写"改了什么"，靠 extends 继承）
src/
  config.py      配置与硬校验（b=Br/µ、M≥2 才能估计 σ，违反直接拒绝启动）
  data.py        冻结题池 + sha256 闸门 + 逐 rollout 确定性采样（配对的前提）
  model.py       Qwen3-1.7B-Base + LoRA，逐 token log-prob；含 CPU 用的 stub 模型
  rollout.py     三种后端：fake（本地干跑）/ hf（小样本冒烟）/ vllm（生产）
  grpo.py        组内优势 + token 级 GRPO（双段 clip 0.2/0.28 + dual clip 10）+ 梯度统计
  accounting.py  三币种记账：responses / tokens / FLOPs，另记墙钟与 ¥
  metrics.py     JSONL schema + 断点续跑用的截断
  train.py       主循环：rollout → µ 次更新 → 逐步落盘 → 存 state
  gen_counting.py, verify.py   从 S1 原样复用（counting 任务与验证器）
tools/           make_pool.py（造池+冻结）/ check_schema.py（数据闸门）/ calibrate_pool.py
analysis/        make_table.py（含自助 CI）/ make_figures.py（无 matplotlib 也出 CSV）/ trace.py（claim→证据）
tests/           12 个测试：损失正确性（标量参考实现 + 有限差分）、配对性、记账、配置闸门
runs/            每次运行一个子目录（git 忽略）
```

---

## 3. 三条命令

```bash
# 1) 零成本自检：不联网、不下载、不用 GPU（约 10 秒）
python tests/run_all.py
python src/train.py --config configs/dryrun.yaml --device cpu

# 2) 数据闸门
python tools/check_schema.py runs/*/ --strict

# 3) 云端（Linux + 24G 显存）
python src/train.py --config configs/mu8.yaml --seed 0
python analysis/make_table.py && python analysis/make_figures.py
```

`configs/dryrun.yaml` 用一个 2 层 / hidden 64 的随机小模型顶替 Qwen3-1.7B，用 `fake` 后端顶替 vLLM。**跑的是同一套代码路径**：rollout → 优势 → log-prob 前向 → 反向 → 梯度累积 → σ → clip → 优化器 → 指标校验 → 存盘 → 续跑。区别只有张量尺寸。

---

## 4. 数据流（本地 / 云端各放什么）

| 内容 | 放哪 | 通道 |
|---|---|---|
| 代码、configs、题池、MANIFEST、测试 | GitHub 公开仓 | git |
| 权重（Qwen3-1.7B-Base） | 只在云端缓存 | ModelScope `snapshot_download`（HF 常 429） |
| `metrics.jsonl` / `samples.jsonl` / `state.pt` | 云盘 → 回传本地 | ModelScope 私有 dataset 仓 `modelscope upload --repo-type dataset` |
| 最终冻结的分析产物 | GitHub 仓 `analysis/out/` | git（跑完之后才提交） |

两条纪律：

1. **不在云端改代码。** 任何改动在本地写、本地过测试、push，云端只 pull。
2. **云端盒子当一次性计算器。** `runs/<tag>/state.pt` 每个 rollout 落盘一次（含 optimizer / RNG / LoRA 权重），关机重开用 `--resume` 续跑；`metrics.jsonl` 里已写入的 rollout 会被自动截断重放，不会出现重复键。

```bash
# 云端开机后第一句（把上次进度接上）
python src/train.py --config configs/mu8.yaml --seed 0 --resume
```

---

## 5. 复现边界（明确不做的）

- **不做多卡、不做 DeepSpeed**：单卡 24G 是预算决定的实验条件，不是能力上限。
- **不追论文绝对数值**：论文是 4×A6000 / ~2000 GPU-h，本项目是单卡 9 run。κ 是量纲约定，绝对标度不可比，**只比比值**。
- **不做 Tobit / Kaplan-Meier**：`N_τ` 的删失情况（"9 次里 6 次在 8 个 rollout 内达到阈值"）如实上报，不拟合一个报告不出来的时间常数。
- **不改任务难度去凑好看的曲线**：题池冻结后不重采样。

---

## 6. 预算

| 项 | 数 |
|---|---|
| 单 run | 32,768 responses（8 rollout × 4096），约 1.6–4 h |
| 单价 | ¥1.56/h（4090-24G；48G 是 ¥2.08/h） |
| 单 run 成本 | ¥2.5–6.2 |
| 计划 9 run（3 个 µ × 3 seeds） | **¥23–56** |
| 首场 session 时间盒 | 2 小时 / ¥3，强制在 T+40min 用实测 s/step 重算总预算 |

价格与硬件见 `configs/base.yaml` 的 `accounting`；每个 run 的 `summary.json` 里记着实测墙钟、FLOPs 与 ¥，账不是估的，是记的。

---

## 7. 引用

- Two Clocks: <https://arxiv.org/abs/2610.18853>
- 上游 S1 / Bauer（counting 任务的生成器与验证器来源）：`../s1-grpo-rft/`
- 论文索引（每条公式对应的原文位置）见 `REPORT.md` §2。
