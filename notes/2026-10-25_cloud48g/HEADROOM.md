# 48G 上"余量只剩 3.1 GiB"是不是真风险，以及 micro=4 的代价能不能消掉

2026-10-25 第二会话（同一台 48G 实例，UTC 02:37–03:00）。全部数字为本次真卡实测，
出处见同目录 `chunk512.log` / `m8.log` / `m8b.log` / `vram_*.csv`，探针
`tools/probe_memory.py`（全卡单跑，含 score → grpo_loss → backward → 梯度缓冲
→ write_back → opt.step 的完整一步）。

> **⚠ 2026-10-25 晚追注：本文件里所有 `σ` / `d` 数值作废，口径已变（D-18）。**
> 它们是在"一次 rollout 的全部请求共享同一个 `SamplingParams.seed`"下量的：同 prompt 的
> G=8 条生成退化成 1 条（组内 distinct 1.25/8，`zero_var_group_frac` 0.875），σ 是在
> "一个样本重复 8 次"的组上算出来的。改成每请求独立 seed 后 σ 由 0.0209 升到 0.0659（3.15×），
> d 相应降到约 1/3。**显存、吞吐、墙钟三类数字不受影响**（它们与 σ 无关），仍可引用；
> 任何跨轮次的 σ/d 比较必须在新口径下重测。

---

## 一、先纠正我上一轮自己的一个错误

上一轮说"若 completion 普遍接近 cap 512，L_max 会到 ~712，峰值再涨 20%"。
这个 712 是拿 **Lp=200** 估的。实测 prompt 只有 **72–102 token**（计数题），
所以结构上界是 `prompt_max + 512 ≈ 614`，不是 712。

**实测 L_max 分布**（`[lmax]` 行，128 次 grad 调用，micro=4，512 responses）：

```
min=210  p50=586  p90=610  p99=614  max=614
histogram(50): 200:3 250:3 300:1 350:1 400:5 450:10 500:3 550:74 600:28
```

conclusion：**上一轮那次 46,034 MiB 的峰值，已经是在最坏长度下测的**——因为
completion 被 cap 512 硬截断，L_max 从结构上就到不了 712。所谓"没实测的 +20% 风险"
不存在，它已经被那次运行覆盖了。这是我用错 Lp 造成的虚警，记在这里。

---

## 二、余量的真实账（探针，全卡，Lmax = prompt+completion）

| 变体 | micro | Lmax | 前向峰值 | 含反传峰值 | reserved |
|---|---|---|---|---|---|
| 现版 score（`.float()` 整块） | 4 | 584 | 16.21 | 18.19 | 21.27 |
| 现版 | 4 | 712 | 19.58 | 21.99 | 25.17 |
| 现版 | 8 | 712 | 35.06 | 39.89 | 45.45 |
| `fp32=False` | 8 | 712 | 30.23 | 33.45 | 35.79 |
| 分块 logsumexp（chunk） | 4 | 712 | 16.65 | **18.04** | 19.31 |
| chunk | 8 | 584 | 24.83 | 27.04 | 28.73 |
| chunk | 8 | 712 | 29.20 | 31.98 | 33.73 |

校准：`current/micro=4/584` 的 18.19 GiB 与上一会话 `probe_micro_batch.py` 的 18.20
逐位吻合，两套探针互证。

**结论**：分块 logsumexp 让"最坏长度 712"的代价（18.04）低于现版"平均长度 584"
的代价（18.19）。也就是说改法不是"少涨 20%"，是**长度这一维基本不再影响峰值**。

---

## 三、数值上只有 chunk 可用

`[numeric]` 行（B=2, Lmax=584）：

| 对比 | max abs Δ | mean abs Δ |
|---|---|---|
| fp32 vs bf16（`fp32=False`） | **1.218e-01** | 3.119e-02 |
| fp32 vs chunk | **1.907e-06** | 2.484e-08 |

`fp32=False` 差 0.03–0.12 nats，而 GRPO 的 ratio = exp(Δlogp)，这个量级直接污染
ratio / clip / KL；而且它连 `grpo_loss` 都过不去（`quantile()` 拒绝 bf16）。
chunk 每 64 位仍升到 fp32 再归约，所以与现版等价到 1e-6——端到端证据是三次真实
512 响应运行的指标几乎逐位相同（`sig` 0.0306→0.03063，`d` 0.1323→0.1322）。

---

## 四、三次真实 512 响应运行（mu=1, seed=0, cap=512）

| 方案 | micro | util | max_num_seqs | rollout | update | 合计 s/resp | 峰值 MiB | σ / d（作废，口径已变 D-18） |
|---|---|---|---|---|---|---|---|---|
| 基线（现版 score） | 4 | 0.45 | 256 | 56.4s | 43.5s | 0.1951 | 46,034 | 0.0306 / 0.1323 |
| chunk | 4 | 0.45 | 256 | 56.5s | 46.7s | 0.2016 | **43,650** | 0.03063 / 0.1322 |
| chunk | 8 | 0.32 | 128 | 83.1s | 49.3s | 0.2586 | 46,344 | 0.02041 / 0.1357 |

第三行需要 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 才过：不开时在
26.17 GiB 已分配 + **4.22 GiB 保留未用**的状态下，请求 1.33 GiB 失败，只剩 1.22 GiB——
是分配器碎片，不是需求不够（`m8.log`）。

## 五、三个此前未验证的判断，这次被推翻/证实

1. **"micro=4 的 update 变慢"是错的**。同卡同批量实测：micro=4 用 46.7s，
   micro=8 用 49.3s。micro=4 反而更快。原先"update 随 M 翻倍变慢、mu=1 时占 44%"
   这一条作废——M 从 64 到 128 并没有带来 update 代价，代价只在 rollout 侧
   （util 0.32 + max_num_seqs 128 让 rollout 慢 47%）。
2. **micro=8 在 48G 上确实能跑**，但要用 util 0.32 + 128 并发 + expandable_segments，
   总时长 +28%（0.1951 → 0.2586 s/resp）。9 run 从 ~11.8h 涨到 ~15h（¥25 → ¥31）。
3. **σ 的 √m 律得到跨批次确认**：micro=8 的 σ=0.02041，micro=4 的 0.03063，
   比值 1.501（理论 √2=1.414）；D-13 那次 dryrun 是 1.365。两次都在同一侧，
   说明 m 只改变尺子，不改变被量的东西。
   **作废，口径已变（D-18）**：两个 σ 都在共享 seed 的退化组上量，比值虽同向，
   但"被量的东西"本身不是采样噪声；这条结论要等新口径下重测才能引用。

---

## 六、代价最小的那一步（未实施，待拍板）

σ 的定义（`grpo.py:131`）只依赖 **组边界**：`sigma² = (Σ||g_i||² − M||ḡ||²)/(M−1)`。
它完全不需要"一次前向就把 8 条塞进去"。所以可以把两件事拆开：

* **前向块** micro=4（受显存约束，无所谓）
* **统计组** group=8（设计值，决定 M 与 σ）

做法：组内累加两个子块的梯度，得到与 micro=8 数学等价的 g_i（按 token 数加权：
`grpo_loss` 用 `token_mean` 归一，所以子块梯度乘回 n_i，组末除 N），再喂给
`GradStats`。代价：多一个 0.26 GiB 的 LoRA 级 fp32 缓冲（69.73M 可训练参数），
梯度范数计算从 128 次降到 64 次，显存与时间都持平或略优。

若采用，`n_micro_batches` 回到 64、σ 回到设计口径，第五节第 3 条的"尺子换算"
不再需要写进方法节。若不做，替代方案是报 σ̃ = σ·√m（m 不变则同乘常数，
所有 arm 排序不变），但要在方法节显式声明 m。

---

## 七、落账

* 新增 `tools/probe_memory.py`（显存探针，含三变体）、`tools/run_with_chunk.py`
  （用分块 score 跑真实流水并记录每次调用的 L_max）、`configs/cloud48g_m8.yaml`。
* 未改动 `src/model.py` / `src/train.py`；chunk 目前只以 monkeypatch 形式存在于
  `tools/run_with_chunk.py`，等拍板后再决定是否收进 `Policy.score`。

---

## 八、落地：chunk 进 `Policy.score` + 解耦 group/micro（D-15）

代码改动（已并入 `src/`，见 DEVLOG D-15）：

| 文件 | 改动 |
|---|---|
| `src/model.py` | 新增 `chunked_logsumexp()`；`score(chunk=...)` 默认走 64 位切片 + checkpoint 重算；`score(chunk=0)` 保留旧整块路径；`score(fp32=False)` 删除；`score_memory_bytes` 改按切片估 |
| `src/config.py` | 新增 `model.logit_chunk`（64）、`train.micro_group_responses`（None ⇒ = micro）；`groups_per_step`；校验 group 必须是 micro 的整数倍且整除 b |
| `src/train.py` | 两层的组/块循环；组内按 token 数累加、组末除一次；`metrics` 记 `n_groups` 与 `n_forward_chunks` |
| `tests/test_pipeline.py` | 新增 3 条：分块归约与整块等价（含梯度）、group/chunk 可分离的配置校验、token 加权代数 |

### 8.1 消融：解耦是精确的（stub + fake rollout，同一份数据）

`configs/ab_g8m8.yaml` / `ab_g4m4.yaml` / `ab_g8m4.yaml`，本机与云端各 16/16 测试通过。

| arm | group | micro | M | chunks | ‖ḡ‖ | σ† | d† |
|---|---|---|---|---|---|---|---|
| ab_g8m8 | 8 | 8 | 4 | 4 | 0.03697 | 0.06405 | 0.57724 |
| **ab_g8m4** | 8 | 4 | 4 | 8 | **0.03697** | **0.06405** | **0.57724** |
| ab_g4m4 | 4 | 4 | 8 | 8 | 0.03705 | 0.08745 | 0.42362 |

* `σ(g8m4)/σ(g8m8) = 1.0000`，4 步全等，‖ḡ‖ 逐位相同 → **解耦没有代价**
* `σ(g4m4)/σ(g8m8) = 1.4249` ≈ √2 → D-14 记的那个 1.41 是真实存在的，只是可以被消掉

### 8.2 真实模型：group 只改 σ，不改梯度

同一次 rollout、只改 `micro_group_responses`：

| 数据 | group | M | chunks | ‖ḡ‖ | σ† | d† |
|---|---|---|---|---|---|---|
| A（pass 0.188） | 4 | 128 | 128 | 0.003 | 0.02772 | 0.1192 |
| A | 8 | 64 | 128 | 0.003 | 0.02087 | 0.1606 |
| B（pass 0.238） | 4 | — | — | 0.004 | 0.03063 | 0.1322 |
| B | 8 | 64 | 128 | 0.004 | 0.02209 | 0.1812 |

σ 比值 1.328 / 1.386，方向与量级对，偏差来自单步 + M 只有 64/128 的有限样本误差。

† 两张表的 `σ` / `d` 列**作废，口径已变（D-18）**：全部在共享 `SamplingParams.seed` 的退化组上量。
`‖ḡ‖` 列不受影响（它只依赖 token 与 loss 归一化，不依赖组内多样性），
所以 "group 改不改 ‖ḡ‖" 这个结论仍然成立，"σ 的 √2 律"必须在新口径下重测。

### 8.3 显存：分组不占额外显存，`expandable_segments` 白捡 5 GiB

同一份数据（pass 0.188，`[lmax]` 分布逐位相同：min 246 / p50 586 / max 610）：

| 配置 | 峰值 MiB |
|---|---|
| micro=4 group=4 | 45,140 |
| micro=4 **group=8** | **45,140**（完全一致 → 组缓冲被分配器吃掉，零成本） |
| micro=4 group=8 + `expandable_segments` | **40,172**（另一份数据，pass 0.238） |

不加 `expandable_segments` 时，两份数据之间本身有 ~1.5 GiB 的波动（43,650 @ max 614 vs
45,140 @ max 610）。**结论：campaign 启动时带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
预期峰值 ~40–41.5 GiB，余量 6–7 GiB。**

### 8.4 一个必须先说清的限制（D-16）

同配置同 seed 连跑两次，`pass_rate` 0.188 / 0.238 → **vLLM rollout 不可复现**。
所以 8.3 的"同一份数据"是事后按 pass 与 L_max 分布配对出来的，不是保证；
8.1 的消融之所以可靠，是因为它用 stub + fake rollout，数据由 rng 构造、逐位相同。
