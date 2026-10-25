# REPORT — 待数据（模板）

> 状态：**未跑数据**。带 `TODO` 的格子必须在有 `runs/*/metrics.jsonl` 之后才能填，
> 填的时候只允许引用 `analysis/out/table.json` 里的字段，不允许手抄数字。
> 预注册声明见 `analysis/claims.yaml`（跑之前写定，跑完只改 status）。

---

## 0. 一句话结论

TODO（一句话，含"实测 / 预测"两个数和一个比值，不含量化词）：

> 在同一 32k responses 预算下，µ 从 1 提到 16，`N_τ` 实测放大 __×（自助 95% CI __–__），
> 论文 Eq.(5) 在假设 A2（忽略陈旧性）下预测 6.1×；同一批数据中 `KL(π_θ‖π_rollout)`
> 累计到 __，即 A2 忽略的那一项在 µ=16 时已经达到 __ 的量级。

---

## 1. 预注册

| 项 | 值 |
|---|---|
| 主结论 | `analysis/claims.yaml: main_claim = C3` |
| 数据口径 | `Br=4096`、`G=8`、`32,768 responses/run`、`max_completion_length=512` |
| 臂 | µ ∈ {1, 8, 16}（`b = Br/µ` ∈ {4096, 512, 256}） |
| seeds | 3（0/1/2），配对比，不合并 |
| 判定口径 | 主结论用 seed 配对差值 + 逐 rollout 自助 CI；其余标为探索性 |
| 删失处理 | 未达到阈值即记 `censored`，不插补、不外推时间常数 |

---

## 2. 原文索引（每条公式/数字的位置）

> 依据 `jd-lab/discussions/2026-10-24_TwoClocks精读_核心发现与局限.md` 的阅读记录。
> **写终稿前必须回 PDF 复核页码**（本表目前的"位置"是章节级，不是页码级）。

| 量 | 原文位置 | 原文表述（摘要） | 本项目的用法 |
|---|---|---|---|
| drift clock | Eq.(3) | `d = κ‖ḡ‖/σ`，batch-invariant | `src/grpo.py:drift_clock` |
| progress clock | Eq.(4) | `r(b) = r₀ − C_curv·b` | 不直接测量，只用于解释 |
| 静止点 | Eq.(5) | `N_τ = N_min(1 + Br/B*)` | 主结论的"预测值"一侧 |
| `B*` 闭式 | Eq.(5) 下方 | 含 `1/κ`，与 McCandlish(2018) 不同 | 用于核对 6.1× 的量级 |
| 失败时钟 | Eq.(7)(8) | `N_fail = C√b/η` | L 档失败窗口的观察对象 |
| 重要性比对数展开 | Eq.(9) | µ 次更新的 off-policy 修正 | 对应 `kl_per_rollout_policy` |
| 假设 A2 | §假设列表 | *"we treat the mini-batches within a rollout as fresh and ignore policy staleness…"* | **被测量的对象** |
| Table 1 | §实验设置 | S/M/L/XL：`P=64/192/512/1024`、`Br=512/1536/4096/8192`、`G=8`、`µ=8` | 本项目取 L 档 `Br=4096` |
| Table 2 | §结果 | `α=0.15/0.28/0.44`；`N_.15` 的 L/S=1.40、1.72、2.34；Instruct 0.67；7B 0.36 | 只作参照，不作为目标 |
| 种子噪声 | §结果 | `D_AUC` 0.005–0.009 | 判定"变化是否超过噪声"的门槛依据 |
| `B*` 反解 | Remark 7 | 9.3k / 4.0k / 1.9k | 与实测 `N_τ` 比值对照 |
| 失败律拟合 | §失败时钟 | rms 1.53（p≈0.29）vs 2.32（p<1e-4）；Tobit β=0.47、γ=0.72 | 写法参照（本项目不做 Tobit） |
| clip 现象 | §结果 | clip 占 0.5% token，却占 50–70% 梯度范数 | `clip_frac_any` + `grad_norm_clipped/raw` |
| 计算量 | App. B | 每 run 4×A6000，全程约 2000 GPU-h | 预算对比的分母 |
| 响应长度 | 全文 | **0 处报告** | 本项目新增观测项 |
| 代码 | — | 仓库 404（三重核对） | 因此只能重建，不能复用 |

---

## 3. 实验矩阵（9 run）

| arm | µ | b=Br/µ | M（每步 micro-batch 数） | 优化步数/run | seeds |
|---|---|---|---|---|---|
| mu1 | 1 | 4096 | 512 | 8 | 0,1,2 |
| mu8 | 8 | 512 | 64 | 64 | 0,1,2 |
| mu16 | 16 | 256 | 32 | 128 | 0,1,2 |

三臂共享：同一题池（sha256 冻结）、同一 prompt 序列（`prompt_batch(seed, rollout)` 是纯函数）、同一 `Br`、同一 token 预算。**µ 是唯一被操纵的变量。**

---

## 4. 与原文的差异（每条都要说理由）

| # | 项 | 原文 | 本项目 | 理由 | 是否影响主结论 |
|---|---|---|---|---|---|
| 1 | 硬件 | 4×A6000/run | 1×4090-24G | ¥100 预算 | 否（只比比值） |
| 2 | 计算量 | ~2000 GPU-h | 9 run × ≤4 h ≈ ¥22–54 | 同上 | 否 |
| 3 | 模型 | 未在正文固定为单一模型（多规模对照） | Qwen3-1.7B-Base | 24G 单卡 + LoRA 可训 | 部分：绝对 `N_τ` 不可比 |
| 4 | 训练方式 | 未说明是否 LoRA | LoRA r=64/α=16 | 与 S1 一致；24G 显存 | 可能影响 σ 量级 → 只比 µ 之间的比值 |
| 5 | 任务 | 未在正文报告本文所用任务细节（待复核） | counting（depth 1–7） | 复用 S1，失败率可控 | 部分 |
| 6 | 响应长度上限 | 未报告 | 512 | 24G KV 预算 | 影响 `resp_len` 观测 |
| 7 | 题池 | 未报告 | 2000 条，冻结 | 可复现 | 否 |
| 8 | κ | 量纲约定 | 取 1.0 | `d` 的绝对值无意义 | 否（比值中消掉） |
| 9 | 损失归一化 | 未明确（DAPO 式总 token 归一） | 逐 token 均值 | 实现简单、三臂一致 | 等价于改 lr，三臂相同 → 否 |
| 10 | 失败律拟合 | 两套律做似然比检验 + Tobit | 只报删失计数 | 9 run 的样本量撑不起拟合 | 是（第 6 条局限） |
| 11 | seeds | 报告了种子噪声区间 | 3 seeds，配对 | 预算 | 是（CI 会宽） |
| 12 | 早停/选择性报告 | 未说明 | 预注册主结论，其余标探索性 | 防事后选指标 | 否 |
| 13 | vLLM 与训练的权重同步 | 未说明 | 每步导出 LoRA adapter + 新 `LoRARequest` | 唯一公开 API 路径 | 否 |
| 14 | 离线打分精度 | 未说明 | 逐 token log-prob 用 fp32 计算 logsumexp | bf16 会让 KL 误差达 1e-3，与信号同量级 | 否（更准） |
| 15 | 响应长度记录 | 无 | 有 | 新增观测 | 否 |
| 16 | clip 梯度份额 | 定性报告 50–70% | 逐记录 `grad_norm_clipped/raw` | 新增观测 | 否 |
| 17 | 题池难度校准 | 未报告 | 上云后先跑 `tools/calibrate_pool.py` 确认通过率落在 10–90% | 1.7B base 可能全错，全错则无梯度 | **是（前置门）** |
| 18 | 续跑 | 未说明 | 每 rollout 存 state，`--resume` | 租卡关机即失数据 | 否 |
| 19 | 代码 | 404 | 本仓库 | — | 否 |
| 20 | 前向块大小 | 未报告（微批=8 由 σ 的定义隐含） | `micro_batch_responses: 4`，σ 组仍取 8 | 48 GiB 单卡放不下 micro=8 + 并置的 vLLM；组边界与 σ 口径不变 | 否（D-15 消融：g8m8 与 g8m4 逐位相同） |
| 21 | G 条的采样种子 | 未说明（默认 G 条独立采样） | 每 response 独立 seed；另关前缀缓存/chunked prefill、固定 token 预算 | 共享 seed 会让同 prompt 的 G 条退化成 1 条（实测 1.2/8 distinct、零方差组 87.5%）；固定批次形状让运行可逐位重现 | **否，但是前置门（D-18/D-19）** |

---

## 5. 结果

### 5.1 主表

TODO：直接贴 `analysis/out/table.md` 的输出，不要手工排版。

### 5.2 三张图

| 图 | 说明 | 文件 |
|---|---|---|
| KL 随 µ_step | A2 忽略的量到底有多大 | `analysis/out/fig/curves.png`（左） |
| d 随 rollout | drift clock 轨迹，与 `N_τ` 的阈值交叉 | 右上 |
| pass_rate 随 rollout | 学习曲线，确认三臂在同一难度上 | 左下 |
| clip 比例随 µ_step | 与论文"0.5% token"的对照 | 右下 |

### 5.3 与预测的对照

| 量 | 论文 Eq.(5) 预测（A2 下） | 实测 | 差值 | 说明 |
|---|---|---|---|---|
| `N_τ(µ=16)/N_τ(µ=1)` | 6.1× | TODO | TODO | 主结论 |
| `N_τ(µ=8)/N_τ(µ=1)` | TODO（用同一 `B*` 反解） | TODO | TODO | 次要 |
| `KL` 累计 @µ=16 | 0（A2 假设） | TODO | — | 直接反证假设 |

---

## 6. 局限（写报告时必须原样保留）

1. **单卡、单模型、单任务**：任何"通用性"表述都超出口径。
2. **3 seeds**：配对差值的 CI 会很宽；报告里不允许出现"显著"字样，只说区间。
3. **删失**：`N_τ` 若在 8 个 rollout 内未达到阈值，只记 censored，不外推。
4. **κ 不可比**：`d` 的绝对标度与原文不可比，只比比值。
5. **LoRA 改变了可训练子空间的几何**：`σ` 与 `ḡ` 的量级受此影响，故所有结论都限制在"µ 之间的比较"。
6. **未做失败律拟合**：`N_fail = C√b/η` 只有观察，没有检验。
7. **题池通过率未校准前，学习曲线不可解释**（见差异表 #17）。

---

## 7. 复现命令

```bash
python tests/run_all.py
python tools/check_schema.py runs/*/ --strict
python analysis/make_table.py --runs runs --out analysis/out
python analysis/make_figures.py --runs runs --out analysis/out/fig/curves
python analysis/trace.py --strict
```

`reproduce.sh` 把上述命令和 9 次训练串在一起，本地 `./reproduce.sh dry` 只跑干跑部分（0 GPU 0 元）。
