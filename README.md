# two-clocks-audit

对 arXiv:2610.18853（*Two Clocks*, CMU / U Toronto）的一次**可复现审计**，不是复现、不是改进。

一句话：**论文用两座"钟"（drift clock、progress clock）给出"每批数据该做几次更新"的处方，但这个处方建立在一个被显式搁置的假设上——它忽略了同一批 rollout 被反复用于多次更新造成的策略陈旧（stale）代价。本项目在 ¥100 预算内，把这个代价测出来。**

---

## 1. 增量在哪（不是"复现原文"）

| 原文做了什么 | 本项目做什么 |
|---|---|
| Eq.(3) 定义 drift clock `d = κ‖ḡ‖/σ` | 逐 micro-batch 实测 `‖ḡ‖` 与 `σ`，得到 `d` 的轨迹 |
| Eq.(5) 给出 `N_τ = N_min(1 + Br/B*)`，因 `B* ∝ µ`，预测 µ↑ → `N_τ` **变小**（即 µ=1 需要 µ=16 的 6.1 倍样本） | 在同一 `Br`、同一 token 预算下，实测 µ ∈ {1, 8, 16} 的 `N_τ` 比值 |
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
  metrics.py     JSONL schema + 断点续跑用的截断（含 group 级口径的 4 个 optional 字段）
  prompt_log.py  per-prompt 归档：prompt id + 8 位正确性 + 8 个 reward；组级统计（含 ICC）
  train.py       主循环：rollout → µ 次更新 → 逐步落盘 → 存 state
  gen_counting.py, verify.py   从 S1 原样复用（counting 任务与验证器）
tools/           make_pool.py（造池+冻结）/ check_schema.py（数据闸门）/ calibrate_pool.py
                 offline_sigma.py（C9 的离线 σ 3×3 分解，需 base 权重 + adapter + GPU）
analysis/        make_table.py（含自助 CI）/ make_figures.py（无 matplotlib 也出 CSV）/ trace.py（claim→证据）
                 slope_ratio.py（C8 的 N_τ 比：斜率比 + 窗口区间 + 统计误差 + µ8/µ16 自检）
                 bmu_2x2.py（C10 的 b×µ 双因子：主效应 + 配对 CI + 预算匹配口径）
                 noclip_ablation.py（C11 的 dual-clip 消融：三种 pass 口径 + ratio_p95 余量）
                 r4a_grid.py（C12 的 19-rollout 网格判定：首次命中 vs 下界支）
                 residual_claims.py（C2–C7 + P1 一次复算，自带 8/19 rollout 守卫）
                 group_grain.py（72 个已归档 group 的组级分析：全对/全错/ICC/zvR vs zvC）
                 data_review.py / prompt_overlap.py（现有九 run 的复核与题池复用实测）
tests/           37 个测试：损失正确性（标量参考实现 + 有限差分）、配对性、记账、配置闸门、
                 归档校验、per-prompt 归档与组级统计（含手算 ANOVA 的 ICC 对照）
runs/            每次运行一个子目录。9 个正式 run 的 metrics/samples/summary/run_meta/config
                 入库（约 1.3 MB）；state.pt、adapters/、rollouts/*.npz 只在本机（git 忽略）
runs_ab2x2/      C10 的两个新预算格（mu8@b256、mu16@b512），指标入库
runs_r4a/        C12 的 19-rollout 续跑（µ=1），指标入库；与 runs/mu1_* 的 8-rollout 版**不是同一份数据**
REPORT.md        预注册审计报告（事实源，C1–C7 + P1，含逐判据结果与有效性威胁）
REPORT.html      同上，单文件版（图内嵌，离线可读，由 tools/build_report_html.py 生成）
ATTRIBUTION.md   事后归因报告（C8–C12，与 REPORT 的分工见其 §0）
ATTRIBUTION.html 同上，单文件版
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
python analysis/slope_ratio.py --json analysis/out/slope_ratio.json   # C8 的 N_τ 比
```

`configs/dryrun.yaml` 用一个 2 层 / hidden 64 的随机小模型顶替 Qwen3-1.7B，用 `fake` 后端顶替 vLLM。**跑的是同一套代码路径**：rollout → 优势 → log-prob 前向 → 反向 → 梯度累积 → σ → clip → 优化器 → 指标校验 → 存盘 → 续跑。区别只有张量尺寸。

---

## 3.5 复算报告里的数字（不需要 GPU、不需要重训）

**每条判据的输入都在仓库里**（21 个 run 的 metrics/samples/summary/run_meta/config，合计约 4 MB）：

| 目录 | 内容 | 支撑哪条 |
|---|---|---|
| `runs/mu{1,8,16}_s{0,1,2}/` | 主审计 9 run，各 8 rollout | C1–C8、P1 |
| `runs_ab2x2/` | b×µ 的两个新预算格（mu8@b256、mu16@b512） | C10 |
| `runs_noclip/` | dual-clip 消融（`clip_delta=null`） | C11 |
| `runs_r4a/` | µ=1 的 19-rollout 续跑 | C12 |

克隆下来直接跑：

```bash
git clone https://github.com/3hgL4/two-clocks-audit && cd two-clocks-audit

python analysis/residual_claims.py --runs runs              # C2–C7 + P1，一条命令全出
python analysis/slope_ratio.py                              # C8，末行打印 6.136
python analysis/bmu_2x2.py    --runs runs --ab runs_ab2x2    # C10，HEADLINE mu_is_primary
python analysis/noclip_ablation.py                           # C11，HEADLINE clip_is_inert
python analysis/r4a_grid.py   --runs runs_r4a --base runs     # C12，hit 3/3
python analysis/make_table.py --runs runs --out analysis/out
python analysis/trace.py --strict                            # 16 条 claim 的证据链
```

**C9 是唯一不能复算的一条**：它要 base 权重 3.2 GiB + 3 个 adapter（各 266 MiB）+ 一块 GPU，三者都不入库。仓库给的是脚本、命令与落盘数值，不是可重跑的结果。

**两个目录纪律**（都不是洁癖，是踩过的坑）：

- `runs_r4a/mu1_s*` 与 `runs/mu1_s*` **同名却不是同一份数据**（19 vs 8 rollout）。放进同一个目录会触发 `make_table.py` 自带的 duplicate 检测。
- `runs_noclip/noclip_mu16_s*` 的 µ=16、b=256，与 `mu16_s0` **同键**。若塞进 `runs/`，凡按 metrics.jsonl 内部 µ/b 分组的脚本都会把它折叠进 µ16 臂——而两臂的差只有 0.x pp（C11 测的就是"不可区分"），**从结果上没有人能看出来**。`bmu_2x2.py` 与 `r4a_grid.py` 因此各带一道目录名白名单守卫，遇到意料之外的目录直接报错退出。

仓库里没有的是 `state.pt`（837 MB/run）、`adapters/`、`rollouts/*.npz`——那三样只影响"能不能继续训"，不影响"结论对不对"。

> 实测（2026-10-26，干净克隆 + 无 GPU）：`slope_ratio.py` 直接输出 `measured 6.136 in [3.594, 9.17]`，与 `REPORT.md` §6 一致。
> 实测（2026-10-27，全部走仓库内路径）：C10 四格 `0.4593 / 0.5168 / 0.4383 / 0.5671`、C11 `+0.60pp` 与 `ratio_p95 1.0906`、C12 `hit 3/3` 与 grid 比 `3.58 / 3.89`，与 `ATTRIBUTION.md` 逐位一致。

---

## 4. 数据流（本地 / 云端各放什么）

| 内容 | 放哪 | 通道 |
|---|---|---|
| 代码、configs、题池、MANIFEST、测试 | GitHub 公开仓 | git |
| 9 个正式 run 的 `metrics/samples/summary/run_meta/config.resolved` | **GitHub 公开仓 `runs/mu*/`** | git（约 1.3 MB，见 §3.5） |
| 权重（Qwen3-1.7B-Base） | 只在云端缓存 | ModelScope `snapshot_download`（HF 常 429） |
| `state.pt` / `adapters/` / `rollouts/*.npz` | 云盘 → 回传本地 | ModelScope 私有 dataset 仓（27 GB，只在本地） |
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
- **不做 Tobit / Kaplan-Meier**：`N_τ` 的删失情况如实上报（本次实测 **9/9 run 全部右删失**，比值因此不存在，见 `REPORT.md` §5.2），不拟合一个报告不出来的时间常数。
- **不改任务难度去凑好看的曲线**：题池冻结后不重采样。

---

## 6. 预算：计划 vs 实际

| 项 | 计划（开跑前） | 实际 |
|---|---|---|
| 单 run | 32,768 responses（8 rollout × 4096），约 1.6–4 h | 同规格，墙钟约 1.7–1.8 h |
| 单价 | ¥1.56/h（4090-24G） | **¥2.08/h（4090-48G，49,140 MiB）** |
| 单 run 成本 | ¥2.5–6.2 | **¥3.5316–3.6749** |
| 主审计 9 run（3 µ × 3 seeds） | ¥23–56 | **¥32.4019** |
| 全项目（21 次启动 / 774,144 responses） | — | **¥86.68** = **¥0.112 / 1k responses** |
| 首场 session 时间盒 | 2 小时 / ¥3，T+40min 用实测 s/step 重算 | 按此执行 |

价格与硬件见 `configs/base.yaml` 的 `accounting`；每个 run 的 `summary.json` 里记着实测墙钟、FLOPs 与 ¥，**账不是估的，是记的**。

> 预注册 C7 原文写的是"4090-24G @ ¥1.50/h"，与实际租用条件不符（且 ¥1.50 也不是 24G 的真实价 ¥1.56）。2026-10-27 已按作者授权修正为 48G @ ¥2.08/h，**阈值 ¥6/run、¥55/九次未动**。修订方向对结论不利（更贵的卡在 ¥6 上限下允许的墙钟更短），偏差对照保留在 `REPORT.md` 附录 C 第 1–2 条。

---

## 7. 引用

- Two Clocks: <https://arxiv.org/abs/2610.18853>
- 上游 S1 / Bauer（counting 任务的生成器与验证器来源）：`../s1-grpo-rft/`
- 论文索引（每条公式对应的原文位置）见 `REPORT.md` §2。
