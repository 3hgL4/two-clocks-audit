# vLLM 采样：两个独立问题，都已实测定位并修好

2026-10-25 第三会话（同一台 48G 实例，UTC 04:04–04:33）。九个 512/1024 条的真实运行，
全部走 `Qwen3-1.7B-Base` + LoRA r=64 + `enforce_eager` + `util 0.45` + `max_num_seqs 256`。
证据在同目录 `det/<tag>/`（`rollouts/r000.npz` + `audit.json` + `metrics.jsonl` + `samples.jsonl`）
与 `probe_det1.log` / `probe_det2.log`；判读工具 `tools/audit_rollout_ids.py`。

---

## 摘要

| # | 问题 | 判据 | 修复前 | 修复后 |
|---|---|---|---|---|
| A | 同一 prompt 的 G=8 条生成退化成 1 条 | 组内 distinct 完成数、`zero_var_group_frac` | **1.22–1.36 / 8**，`zero_var` 0.80–0.875 | **8.00 / 8**（min 8），`zero_var` 0.000–0.016 |
| B | 同配置同 seed 两次运行数据不同（D-16） | 512/1024 行逐行比 id | **355 / 512 行不同** | **0 / 512、0 / 1024 行不同** |

两者的修法互不替代，campaign 配置必须同时带（`configs/cloud48g_main.yaml` 已加四项 sampler 设置）。

---

## A. 组内退化：G 条样本其实只有 1 条

**现象（修复前，`det_base_a`，512 条 = 64 组）**

```
[diversity] 64 groups of 8: unique 1.25/8 (min 1), distinct first token 1.016,
            within-group len sd 6.77 vs overall 163.58 (ratio 0.041)
zero_var_group_frac = 0.875        sigma = 0.02209
```

- 组内平均只有 **1.25 条互不相同**，首 token 平均只有 **1.016 种**（即 8 条几乎全从一个 token 出发）。
- 组内长度标准差 6.77，总体 163.58 —— 组内几乎没有波动，波动全在组间。
- 旁证：旧 run 的 `samples.jsonl`（每 rollout 前 8 条，正好是 p0×8）在**所有** vLLM run 里
  `distinct=1`、完成文本逐字相同；同一次实验的 HF 后端 `hf48g` 是 `distinct=8`。

**后果**：`grpo.group_advantages` 对零方差组的 advantage 恰好置 0（`grpo.py:43`），
所以 **87.5% 的 response 对梯度没有任何贡献**；σ 是在"一个样本重复 8 次"的组上量的；
`pass_rate` 的有效样本是 64 个 prompt 而不是 512 条 response。

**原因**：`rollout.py` 把**一次 rollout 的所有请求**用同一个 `SamplingParams.seed`。
同 prompt + 同权重 + 同 seed → 同一条轨迹；少数组出现差异，只是批内位置不同带来的浮点微扰
让轨迹在平局处岔开。

**修法**：`rollout.vllm.seed_per_request: true`，每请求 `seed*1_000_003 + 行号`
（`rollout.py` 的 `generate()`，默认 `false` 保留旧行为）。

**效果（`det_seed_a`，同配置同数据量）**

```
unique 8.00/8 (min 8), distinct first token 7.05, within/overall len sd 0.932
zero_var_group_frac = 0.000        sigma = 0.06589     pass = 0.189
```

σ 从 0.0209 涨到 0.0659（**3.15×**），`d = κ‖ḡ‖/σ` 相应降到 1/3。
这不是"σ 变好了"，而是**以前的 σ 量的不是它声称的东西**。清单里所有 pilot 的 σ/d 绝对值
（0.0204–0.0306 / 0.12–0.18 一带）不能与新口径直接比较，只能各自内部比。

---

## B. 跨进程不可复现：D-16 的根因与修法

**修复前（`det_base_a` vs `det_base_b`，同配置同 seed，vLLM 默认）**

```
rows_differing = 355/512 (69.3%), mean_len 322.17 vs 322.50
pass 0.238 vs 0.188
```

这是 D-16 的 token 级证据：不是"聚合指标有出入"，是 7 成的行 token 都不同。

**修复后（`det_fix_a` vs `det_fix_b`，关前缀缓存 + 关 chunked prefill + 固定 token 预算）**

```
identical = True, rows_differing = 0/512, pass 0.334 vs 0.334
unique 1.359/8（此时 seed 仍是共享的，见 A）
```

`configs/cloud48g_det_fixed.yaml` 的三项：`enable_prefix_caching: false`、
`enable_chunked_prefill: false`、`max_num_batched_tokens: 2048`。

**机制（与假设一致）**：一个 prompt 的 G 条共享 ~100 token 的前缀。前缀缓存命中与否
取决于调度顺序，命中与全量 prefill 走不同的 attention 路径，数值不同 → 平局翻 token →
整条轨迹岔开。关掉缓存、关掉 chunked prefill、把每步 token 预算钉死后，调度器对同一份输入的
决策变成确定函数，于是**逐位重现**。

**代价**：没测出代价。512 条时修复组 rollout 51.0 s / update 44.3 s，基线 56.0 / 47.4 s；
但两次的样本集不同（平均长度 301 vs 322），所以这不是干净对比，只能说"没有可见的变慢"。

---

## C. 两者组合（候选 campaign 配置）

`configs/cloud48g_det_seedfix.yaml`（= A 的修法 + B 的修法）

| 数据量 | 逐行相同 | 组内 distinct | pass | zero_var | σ | d |
|---|---|---|---|---|---|---|
| 512 条（64 组） | **0/512 不同** | 8.00/8 | 0.205 / 0.205 | 0.016 | 0.06346 / 0.06346 | 0.1359 / 0.1358 |
| 1024 条（128 组） | **0/1024 不同** | 8.00/8 | 0.225 / 0.225 | 0.008 | 0.06852 / 0.06853 | 0.1008 / 0.1008 |

**做到 1024 而不是只做 512 的原因**：512 条只够铺满两个调度波次，而正式 run 是 Br=4096
（十六个波次）。"固定形状"只有在调度决策仍是输入的确定函数时才有用，所以至少要在更大的批上验一次。
1024 条是"不再是单双波次、又只花 ¥0.1"的最小点。**Br=4096 仍未验**（一次 1.8 h），
但 512→1024 的重现没有退化，且该机制的来源（缓存命中）与批大小无关。

**吞吐**：1024 条 rollout 84.8 s + update 89.8 s = **0.1705 s/response**，比 512 条的 0.189 略好
（批更大、warm-up 摊薄）。按此外推单 run（32768 条）≈ 1.6 h，9 run ≈ 14–16 h ≈ ¥30–34，
与 D-17 的 16.5 h / ¥34 一致，预算结论不变。

---

## D. 对 campaign 与报告的四个影响

1. **`configs/cloud48g_main.yaml` 必须四项齐全**：`seed_per_request: true` +
   三项固定形状。少任何一半，要么 σ 量的不是噪声，要么跑完没法复核。
2. **σ 绝对值重标定**：旧 pilot 的 0.020–0.031 与新口径 0.063–0.069 不可混用。
   `claims.yaml` C4 只比 µ 之间的方向与比值，方向性不受影响；但任何"σ 的量级"叙述要重写。
3. **有效样本量变了**：零方差组从 87.5% 降到 0.8–1.6%，同一 Br 下真正参与梯度的数据量升了约 8 倍。
   这对"1.7B base 在 counting 上能不能学动"是实质性利好，也让 `pass_rate` 的置信区间
   （512 条时 ±0.019）真正成立。
4. **D-16 可以不再写成 limitation**：它从"vLLM 采样不可复现"降级为
   "vLLM 默认批次形状让采样不可复现，固定形状后可逐位复现（0/512、0/1024 行不同）"。
   `rollouts/r{idx}.npz` 的全量落盘仍然保留——它现在是可复核性的常规保障，不再是唯一退路。

## E. 开销

九个运行（2×512 + 2×512 + 1×512 + 2×512 + 2×1024 ≈ 6.1k responses）≈ **¥1.3**，
含引擎启动。9 run 仍未启动。
