# DEVLOG — 决策与修正记录

格式：日期 / 决定 / 理由 / 影响面。只记会影响代码或结论的条目，不记过程。

---

## 2026-10-24（本地代码日）

### D-01 训练器：自写循环（A1），不用 TRL GRPOTrainer

**先前判断作废**：我一度认为"TRL 表达不了 `b = Br/µ`，必须自写"。
核对 `trl 1.16.0` 源码后**该理由不成立**：`GRPOConfig.num_iterations`（默认 1，
docstring 原文 "denoted as μ in the algorithm"，`grpo_config.py:176-188`、`689-713`）
就是 µ；`epsilon_high` / `delta` / `importance_sampling_level="token"` 也都在。

**真正理由**：审计需要的是**观测**，不是表达能力。`num_iterations` 下，
per-µ-step 的 `KL(π_θ‖π_rollout)`、clip 的梯度范数份额、micro-batch 梯度散度 σ
都拿不到（TRL 只把聚合标量交给 callback）。这三项恰好就是被测量的对象。

**成本**：约 700 行。**对冲**：`tests/test_grpo.py` 用独立标量实现（1e-9 对齐）+
有限差分（1e-6 对齐）自证，损失语义按 `grpo_trainer.py:3095-3180` 逐行对照。

### D-02 任务复用 S1 的 counting，题池重建为 2000 条并冻结

`gen_counting.py` / `verify.py` 拷入 `src/`（前者唯一改动见 D-09 的第 3 条），题池新建
（`data/prompts_mixed2000.jsonl`，seed=20261024，
sha256 `d781f955…a798b`，见 D-09 的哈希更正）。

两个必须处理的细节：
1. **去重**：初版 2000 条里 3 条 prompt 文本重复。同一 prompt 若在一个 rollout 内
   出现两次，8 条生成会被拆进两个组，组内方差被污染 → 去重后才冻结（`make_pool.py:dedupe`）。
2. **池子必须 > L 档需求**：`Br=4096, G=8` 每个 rollout 要 512 条不同 prompt，
   S1 的 500 条比需求少 12 条，会静默在单个 rollout 内复用。2000 条足够。

### D-03 模型 Qwen3-1.7B-Base（用户指定），并接受随之而来的显存变化

1.7B 的 KV 是 **112 KB/token**（2×28 层×8 KV heads×128×2B），
不是 Qwen2.5-1.5B 的 28 KB。512 并发 × 650 token ≈ 37 GB，**塞不进 24G**。
vLLM 会自动降并发（不 OOM，只变慢），因此：
- `max_num_seqs` 设 256，`gpu_memory_utilization` 0.45；
- 保留 `rollout.vllm.offload_policy` 开关（生成前把 HF 策略挪到 CPU），
  作为 vLLM + HF 双份权重同时驻留时的兜底。
**写报告时必须交代**：单 run 墙钟会明显高于按 1.5B 估的值，¥ 上限据此上调。

### D-04 落盘：ModelScope 私有 dataset 仓（HF 常 429）

核对魔搭文档后确认可用：`modelscope create <id> --repo_type dataset --visibility private`、
`modelscope upload <id> <local> --repo-type dataset`、
SDK `HubApi().upload_folder(..., repo_type='dataset')`。
顺带解决权重下载：Qwen 系列在魔搭有官方镜像，`snapshot_download` 不必绕 hf-mirror。

纪律不变：**云端只 pull 代码，只 push 小文件**（metrics / samples / 最终 adapter）。

### D-05 三个实现缺陷（都是"看起来对、数字也合理"的那类）

1. **`_grad_norm` 传了参数而不是梯度** → 算出来的是 ‖θ‖ 不是 ‖∇‖，
   数值 17.33 而非 0.06，下游表现是 `sigma = 0`。修法：`_grad_norm` 接受
   Parameter 时读 `.grad`。局部性教训：这个 bug 在真机上不会报错，只会让
   drift clock 变成常数。
2. **σ 的方差公式在 float32 下会翻符号** → `Σ‖g_i‖² − M‖ḡ‖²` 是两个大数相减。
   改用 `sum(dtype=torch.float64)` 累加。
3. **续跑时累计成本从零重启** → `CostMeter` 未继承上次的 totals，
   `cost_cny_cum` 会掉下去。修法：`CostMeter.base`，续跑时从 `state.pt` 注入；
   `check_schema.py` 把"累计计数下降"直接判为失败。

### D-06 阈值交叉按删失上报，不拟合时间常数

`N_τ` 在 8 个 rollout 内没达到阈值就记 `censored`（`make_table.py:first_crossing`）。
9 个 run 的样本量撑不起 Tobit / KM 拟合；原文用了似然比检验和 Tobit
（β=0.47、γ=0.72），本项目明确不做，写进 REPORT §6 局限。

### D-07 本地干跑保留，且跑的是同一套代码

`configs/dryrun.yaml`：stub 模型（2 层 / hidden 64 / vocab 512，字符级 tokenizer）
+ fake 后端。**不下载、不联网、0 GPU**，但 forward / backward /
梯度累积 / σ / clip / 存盘 / 续跑全部真实执行。
`--rollout fake` 的 `acc` 随 depth 衰减，因此组内方差、zero-variance 组
这些路径也被真实触发。

### D-08 预注册写死在 `analysis/claims.yaml`

跑之前写定 11 条（主结论 C3、其余探索性），`analysis/trace.py` 校验
"标 verified 的必须有证据文件且字段存在"。跑完只允许改 status 和数值。

### D-09 行尾符：第一次提交把题池存成 LF，却把 CRLF 的哈希写进了 MANIFEST

**发现**（上推前核查）：`data/MANIFEST.json` 记录 `670b2854…`，那是**磁盘上 CRLF 版**
的哈希；而 `git cat-file -p HEAD:data/prompts_mixed2000.jsonl` 得到的是 **LF 版**，
哈希 `d781f955…`。本机 Windows `core.autocrlf=true`，所以 `git status` 干净、12/12 测试
全过、干跑也正常——**这个错误只在换机器时引爆**：云端 Linux 默认 `autocrlf=false`，
clone 出来是 LF，`src/data.py:56-67` 的 `load_pool(expect_sha256=...)` 直接
`ValueError: prompt pool hash mismatch`，训练启动即中止。

**根因不止 autocrlf**，还有一处代码：
1. `gen_counting.save_jsonl` 与 `data.write_manifest` 都用文本模式 `open(..., "w")`，
   Windows 下 `\n` 被翻译成 `\r\n` —— 生成端就写错了字节，git 只是把它"修"回去了。
2. `calibrate_pool.py` 写 report 同样问题（云端会生成，必须同口径）。

**修法**（内容零改动，只统一换行）：
1. 磁盘归一 LF：`918,330 → 916,330` 字节；哈希 `670b2854…` → **`d781f955…`**。
   这个新值**恰好等于 HEAD 里已有 blob 的 sha256**，即内容与已提交版本逐字节相同，
   只是 MANIFEST 从"记错了"变成"记对了"。
2. 三个写盘点加 `newline="\n"`（`gen_counting.py:save_jsonl`、`data.py:write_manifest`、
   `calibrate_pool.py`）——这是跨 OS 可复现的前提，不是格式偏好。
3. 新增 `.gitattributes`：`* text=auto eol=lf`；仓库级 `core.autocrlf=false`、`core.eol=lf`。

**影响面**：`data/prompts_mixed2000.jsonl` 的 git blob **不变**；只 MANIFEST.json 变；
D-02 的哈希跟着改；题池 2000 条的 prompt / answer / depth 分布完全不变。

**教训（写进 REPORT）**：这是本项目第二次出现"本机全绿、换机必挂"的缺陷（D-05 是第一次）。
两处共同点——**校验条件本身依赖了未被显式固定的环境**（梯度范数依赖 `.grad` 的隐式语义；
哈希依赖 OS 的换行策略）。凡"闸门"都要在**目标环境**里验一次，本机通过不算。

### D-10 第一次上云（4090-24G）暴露的五个问题，全部有实测数字

会话：2026-10-24 23:43–01:0x，¥2.08/h，镜像 `PyTorch/2.8.0-CUDA12.8-Py3.11-Ubuntu22`，
数据盘挂载 `/root/rivermind-data`（49 GiB），系统盘 30 GiB overlay，cgroup 15 CPU / 100 GiB。

1. **云端取不到 GitHub 也连不上 pypi.org**（`curl` 均 000 超时）。可用的是
   清华 pypi 镜像（2.2 s）与 modelscope（0.43 s）。→ 代码改用 `scp` 上传，
   仓库的"git = 同步通道"在云端这一侧不成立，README §4 的说法要改。

2. **`modelscope.snapshot_download` 只有 1.0 MB/s**（3.44 GB 要 57 min），
   而同一 CDN 端点上单条 range 请求 6.3 MB/s、4 路并发 22 MB/s。
   → 写了 `tools/fetch_model.py`（并行 range + 按 API 的 sha256 校验），实测 8 路
   ~10 MB/s，13 min 拉完，**全部文件 sha256 对得上**。瓶颈在 SDK 不在网络。

3. **镜像无 nvcc，vLLM 0.32.0 起不来**：
   `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`
   （抛在它的 torch.compile 路径里）。`pip install vllm` 还把 torch 2.8.0+cu128
   换成了 **2.13.0+cu130**（驱动 580.119.02 撑得住 cu130，所以能跑）。
   → 加 `rollout.vllm.enforce_eager`，默认 False，云端开 True。
   **非 Devel 镜像省了 20 GB 盘，代价是丢掉 nvcc**，这个取舍现在有据可查。

4. **24 GiB 装不下 vLLM(LoRA) + HF 策略共存**（本项目最硬的一条约束）：
   - `util=0.55`（12.9 GiB）→ 打分时 OOM（HF 侧已占 10.97 GiB）
   - `util=0.50`（11.8 GiB）→ 仍 OOM（HF 侧涨到 13.58 GiB，因为 grad+Adam 起来了）
   - `util=0.28`（6.9 GiB）+ `micro=2` + `cap=512` → 仍 OOM，free 仅 14 MB
   - 另：`max_model_len` 默认取模型 32768，KV 预留 3.5 GiB，0.28 时只有 2.61 GiB 可用，
     vLLM 直接拒启动 → 必须显式给 `max_model_len`
   → **结论：24 GiB 上"vLLM 常驻 + HF 同卡"这条路走不通**，需要 vLLM sleep mode
   （`enable_sleep_mode` + `llm.sleep()`）或 48G 卡。写进 REPORT 的硬件章节。

5. **`cap=128` 会让整个测量归零**（本次最有价值的一条）：
   第一次 smoke 得到 `pass_rate=0.000, zero_var_group_frac=1.0`，看着像"模型不会做"。
   实际是题面要求"think step by step"，平均长度 112.6/128 正好卡在截断上，
   答案行根本没输出。`cap=512` 后同池实测 **pass=0.25**（16 题探针 4/16 对得上），
   `zero_var_group_frac` 从 1.0 掉到 **0.0**。

### D-11 HF 后端跑通完整闭环（这是第一个"真权重 + 真更新"的成功运行）

vLLM 与 HF 抢显存时，退到 `--rollout hf` 反而先拿到了完整链路（32 responses, µ=2, cap=512）：

```
[r0 mu00] pass=0.250 len=318 kl=+0.00e+00 ratio=1.0000 clip=0.0000 gn=0.0366 sig=0.0491 d=0.7452
[r0 mu01] pass=0.250 len=318 kl=+1.46e-04 ratio=1.0008 clip=0.0039 gn=0.0466 sig=0.0647 d=0.7198
[r0] done: rollout 35.8s update 3.9s cum 32 resp, 0.011 GPU-h, ¥0.02
```

三个"仪器可信"的判据同时成立：**µ-step 0 的 KL 恰好 0**、**µ-step 0 的 ratio 恰好 1.0**、
**σ 与 d 都是有限非零数**且量级合理（σ≈0.05，d≈0.72）。这不是在 stub 上验的。
吞吐对照：HF 后端 1.12 s/response，vLLM 后端（cap=128 那次）0.17 s/response，差 6 倍。

**证据存档**：`notes/2026-10-24_cloud24g/`（metrics / samples / 两份 log / 校准探针）。


### D-12 单价记错了：24G 不是 2.08（2026-10-25 用户核实）

D-10 与 `configs/cloud24g*.yaml` 把单价写成 **¥2.08/h**，那是 **48G** 的价；
**4090-24G 实为 ¥1.56/h**。代码侧已改（`base.yaml` / `cloud24g.yaml` / `src/config.py`
默认值 / 两处 smoke 注释 / README 预算表）。

**不改写的东西**：`notes/2026-10-24_cloud24g/` 里的 `summary.hfsmoke.json`、
`config.resolved.yaml`、`vllmsmoke/*` 是**当时的配置实况**，作为证据保持原样。
读这些数时 `cost_cny × 0.75`（1.56/2.08）才是真实花销：那次成功运行实为 **¥0.017**，
不是 0.0229；整场 61 min 约 **¥1.6**。

**外推跟着变**（同一 1.12 s/response 基线）：9 run ¥191 → **≈¥143**，仍超 ¥100，
"24G 跑不完 9 run" 的判断不变；vLLM cap=512 那条 ¥84–94 → **¥63–70**。
48G 路线的 ≈¥23 不变（2.08 本来就是对的）。

**故意没改**：`analysis/claims.yaml` C7（"单 run ≤¥6，九次 ≤¥55"，写在 ¥1.50/h 的假设下）。
按 1.56 重算，区间上限是 ¥6.2 / ¥56 —— 这条预注册会因为价格口径而不是科学原因证伪。
改预注册要留痕（加 revision 行），等用户拍板，AI 不代改。


### D-13 `micro_batch_responses` 被当成"切几块"，导致四次 vLLM OOM（2026-10-25）

**现象**：48G 上 vLLM 四次尝试全 OOM，失败点都在策略侧（`model.py:196` / `:199`），
且降 `gpu_memory_utilization`（0.60 → 0.45 → 0.35）完全无效——策略侧占用"给多少用多少"
（18.11 → 25.25 → 30.20 GiB）。

**根因**：`train.py:72` 写的是 `np.array_split(s, micro)`，把每个 optimizer step 切成
**micro 份**。但 `micro_batch_responses` 的语义是**每块多大**，这一点在另外三处都有明证：
`config.py:40` 注释（`logits memory ~ micro*len*vocab`）、`train.py` 的预检
`score_memory_bytes(micro, ...)`、以及 `config.py:125`
`micro_batches_per_step = responses_per_step // micro`（这里算的是**块数**）。

**可复算的算术闭合**（V=151669）：

| | 公式 | 值 | 日志 |
|---|---|---|---|
| 预检 | `8 × 640 × V × 4` | 2.89 GiB | `est. logits memory ~2.89 GiB` |
| 实际 | `64 × 608 × V × 4` | **21.99 GiB** | `Tried to allocate 21.99 GiB` |
| 比值 | B 从 8 变 64 | **7.60** | — |

`plan_partition(512, mu=1, micro=8)` 旧实现给出 **8 块 × 64 个**，应是 **64 块 × 8 个**。
所以"策略侧需要 19–25 GiB"是假象，真实需求约 11 GiB。

**修复**（`train.py`，一行）：按每块 `micro` 切，块数由 `b // micro` 决定。

**修复不改梯度，只改 σ**。同一份 `configs/dryrun.yaml`（Br=64, mu=2, micro=4, b=32）在
旧/新实现下各跑一次，CPU，同 seed=7：

| | n_micro_batches | grad_norm_raw | sigma（4 个 step） | drift_d |
|---|---|---|---|---|
| 旧（4 块 × 8 个） | 4 | 0.037 / 0.027 | 0.0641 0.0536 0.0610 0.0557 | 0.5772 0.5083 0.5225 0.4742 |
| 新（8 块 × 4 个） | 8 | **0.037 / 0.027** | 0.0875 0.0764 0.0874 0.0822 | 0.4236 0.3563 0.3665 0.3226 |

**`grad_norm_raw` 逐位相同**，确认梯度数值等价；σ 放大 **1.365×**（理论 √(8/4)=1.414，
差额来自 σ 本身的估计噪声），d 按同一因子反向缩放（实测 0.734，理论 0.707）。

这个缩放是物理必然，不是可选：σ 本来就是"当前 minibatch 大小的梯度噪声"，改 m 就改 σ。
48G 真实场景 m 从 64 变 8，σ 约放大 **√8 = 2.83**，d 同比缩小。因此：

- `codes/` 与 `notes/` 里任何修复前的 d/sigma 绝对数值**作废**，不再引用；
- `claims.yaml` C1（"d 随 mu 增大而下降"）是**相对趋势**，所有 arm 用同一 micro=8，
  缩放因子对所有 arm 相同，**结论不受影响**；
- 与论文报的 d 数值做跨文献绝对值比较时需要论文也报 micro 大小，否则口径不可比。

**连带作废的旧结论**：D-10 那张"24G 上 vLLM 三档（0.55/0.50/0.28）全 OOM"的表，
失败原因同样是这个 bug，**不能作为"24G 跑不动 vLLM"的证据**。24G 能否跑通需重新实测。

**为什么 12/12 测试没抓到**：`tests/test_pipeline.py` 原有断言只检查
`cfg.micro_batches_per_step == 64`（配置层算术），从未断言 `plan_partition` 真实切出的块
大小，于是"配置说 64 块、代码给 8 块"长期并存。已补
`test_plan_partition_micro_is_batch_size_not_count`，同时锁块数与块大小，并验证 mu 只移动边界、并集仍是全排列（arm 配对的前提）。现 13/13。


### D-14 D-13 之后仍然 OOM：`score()` 的真实显存是每行 ~3.2 GiB，micro 必须从 8 降到 4（2026-10-25）

**第一层（已修）**：D-13 的切分 bug 让 `score()` 跑 64 行而不是 8 行，这是四次 OOM 的直接原因，
修完预检和实际一致（2.89 GiB）。

**第二层（本次暴露）**：修好切分后，48G 上 vLLM 仍然起不来，而且症状反过来了——
vLLM 拿得越少，策略占得越多：

| util | vLLM | 策略 OOM 时 allocated |
|---|---|---|
| 0.60 | 28.41 GiB | 18.37 GiB |
| 0.45 | 21.31 GiB | 25.08 GiB |

差值 6.71 GiB 对 vLLM 的 7.10 GiB，看起来像"分配器自适应"，其实**不是**：单跑一个不含 vLLM 的
探针（`notes/2026-10-25_cloud48g/probe_micro_batch.py`，全 47 GiB 可用），B=8 照样要 29.01 GiB。
两次 OOM 的区别只是**在哪一点撞墙**：util 0.60 时在 18.37 就没了，util 0.45 时撑到 25.08。

**探针实测**（L_max=584，`requires_grad=True`，逐 B 重置 peak）：

| micro (B) | 前向峰值 | 含反传峰值 | 该图形的 fp32 logits |
|---|---|---|---|
| 8 | 29.01 GiB | 32.96 GiB | 2.64 GiB |
| 4 | 16.22 GiB | 18.20 GiB | 1.32 GiB |
| 2 | 9.93 GiB | 10.92 GiB | 0.66 GiB |

拟合 `peak ≈ 3.5 + 3.19·B`：3.5 GiB 是 1.7B 参数（bf16），**每行 response 约 3.2 GiB**。
这不是显存泄漏，也不是 gradient checkpointing 没生效——探针确认
`is_gradient_checkpointing=True` 且 `.model._gradient_checkpointing_func` 已是
`partial(checkpoint, use_reentrant=False)`。成本来自 1.7B 模型的逐层激活加上 151,669 词表的
logits（bf16 一份 + fp32 一份 + 反传梯度）。

**结论**：48 GiB 减去 vLLM 的 21.3 GiB，只剩 26 GiB，装不下 micro=8 的 33 GiB。
**micro 必须降到 4**（18.2 GiB，util 0.45 下余 ~7.8 GiB）。micro=2 也测得可行（10.9 GiB），
保留为 util 0.60 时的备选。

**首次跑通**（`configs/cloud48g_vllm64.yaml`，util 0.45，micro=4，Br=64/mu=1 → M=16）：

```
[r0 mu00] pass=0.125 r=+0.150 len=291 kl=+0.00e+00 clip=0.0000 gn=0.000 sig=0 d=None
[r0] done: rollout 27.9s update 5.5s cum 64 resp, 0.009 GPU-h, ¥0.02
峰值 44,734 MiB
```

`gn=0 / sig=0 / d=None` 是数据巧合，不是故障：64 个 response = 8 prompt × G=8，pass=0.125 恰好
是"其中 1 个 prompt 全对、其余全错"，8 个组内都没有混合，GRPO 的组内优势全为 0。判据是
`_accumulate` 没抛"no gradient"，即梯度张量存在但数值为 0。

**吞吐（512 responses，`configs/cloud48g_throughput.yaml`，已足以让 256 槽调度器饱和）**：

```
[r0 mu00] pass=0.238 r=+0.273 len=322 kl=+0.00e+00 clip=0.0000 gn=0.004 sig=0.0306 d=0.1323
[r0] done: rollout 56.4s update 43.5s cum 512 resp, 0.028 GPU-h, ¥0.06
峰值 46,034 MiB（余 3.1 GiB）
```

- rollout **0.110 s/response**（HF 后端在 48G 上是 ~1.03 → **快 9.4 倍**）
- update **0.085 s/response**（M=128 块，0.34 s/块）
- 合计 **0.195 s/response**，对照 HF 的 1.14 → **5.8 倍**

三条仪器判据同时成立：µ-step0 的 **KL 恰好 +0.00e+00**、**clip=0**、**σ=0.0306 与 d=0.1323 有限非零**。
（σ=0.0306 / d=0.1323 这两个数值**作废，口径已变（D-18）**：当时整次 rollout 共享一个
`SamplingParams.seed`，组内 G=8 条退化成 1 条，σ 是在"一个样本重复 8 次"上算的；
新口径 σ≈0.0659、d≈0.13/3。KL=0 与 clip=0 这两条判据不受影响。）

**9 run 外推**（单 run 32,768 responses；update 成本随 M=b/micro 变，故三个 mu 不同）：

| arm | M/rollout | rollout | update | 单 run | 3 seeds |
|---|---|---|---|---|---|
| mu=1 (b=4096) | 1024 | 451 s | 348 s | 1.78 h | 5.33 h |
| mu=8 (b=512) | 128 | 451 s | 44 s | 1.10 h | 3.30 h |
| mu=16 (b=256) | 64 | 451 s | 22 s | 1.05 h | 3.15 h |

**合计 ≈ 11.8 GPU-h ≈ ¥25**（¥2.08/h），保守取整 **12–16 h / ¥25–33**。对照 D-12 的 HF 口径
（93 h / ¥194）：**预算从"超 94%"变成"用掉 1/3"**。

**⚠️ 修正（D-17，2026-10-25 晚）**：上表 `M/rollout` 与 `update` 两列在 mu=8 / mu=16 上算错了，
`合计 11.8 h / ¥25` 因此偏低。每 rollout 的前向块数是 `Br/micro = 1024`，**与 mu 无关**（mu 只改
"每步几块"和"几步"，乘积不变），表里却拿 `b/micro` 当作每 rollout 块数、只算了一步。修正后三臂
单 run 都 ≈1.83 h，**9 run ≈16.5 h ≈ ¥34**。预算结论（<¥100）不变，差的 4.7 h 记在 D-17。

**这次为了跑通付出的代价，必须记进 deviation**：

1. **micro 8 → 4 是一处偏离预注册**。`base.yaml` 写的是 8。σ 依赖微批大小（σ ∝ 1/√m），
   相对设计值放大 **√2 = 1.41**，与 D-13 那次（√8）方向相同、量级更小。因为所有 arm 共用同一
   micro，`claims.yaml` C1 的**相对趋势仍然成立**，但 d 的绝对值不可与 micro=8 的旧数比较。
2. **update 成本随 M 线性增长**。micro 减半 → 块数翻倍 → update 时间翻倍，mu=1 尤其明显
   （348 s/rollout，占总时长 44%）。这是 micro=4 的直接代价，已含在上表里。
3. **余量只剩 3.1 GiB**（peak 46,034 / 49,140）。主实验的 completion 若普遍接近 cap 512，
   L_max 会从 322 升到 ~712，峰值约再涨 20%。这一条**尚未实测**，9 run 首跑必须盯第一轮的峰值。

**证据**：`notes/2026-10-25_cloud48g/` 下 `vllm_util060_oom.log`、`vllm_util045_oom.log`、
`vllm_micro4_pass.log`、`vllm_micro4_512_throughput.log`、`vram_throughput.csv`、
`probe_micro_batch.py`。

### D-15 logits 归约分块 + σ 的组边界与前向块解绑（2026-10-25，本条作废 D-14 的第 1–3 条代价）

D-14 记了 micro 8→4 的三笔代价。三笔现在都有实测结论，其中两笔是错的、一笔可以消掉。

**一、code：`score()` 不再把整块 logits 升 fp32**

`model.py` 新增模块级 `chunked_logsumexp(logits, chunk)`：每 `chunk`（默认 64，配置
`model.logit_chunk`）个位置升 fp32 归约一次，backward 用 `torch.utils.checkpoint` 重算
而不是留着。`score(chunk=0)` 保留旧的整块路径，只用于等价性测试。
`score_memory_bytes` 跟着改成按切片的峰值，预检数字重新和代码一致。

数值：`full vs chunk` 的 logp 差 **max 1.9e-06 / mean 2.5e-08 nats**（探针），
真实 512 响应运行里 σ 0.0306→0.03063、d 0.1323→0.1322。
（这两个值的**绝对值作废，口径已变（D-18）**，但"改 chunk 后两位小数不变"这个**差值**判据仍成立——
两次运行都在同一套退化采样下，chunk 只改归约顺序。）
`score(fp32=False)` 那条路**已被删除**：它与 fp32 差 max 0.122 nats，且 `grpo_loss` 的
`quantile()` 直接拒绝 bf16 张量。

显存（探针，全卡单跑，含优化器与梯度缓冲）：

| | micro=4 Lmax=712 | micro=8 Lmax=712 | micro=8 Lmax=584 |
|---|---|---|---|
| 旧整块 | 21.99 GiB | 39.89 | 28.73 (reserved) |
| 分块 | **18.04** | 31.98 | 27.04 |

**二、code：σ 只看组边界，不看一次前向能装几条**

`sigma² = (Σ‖g_i‖² − M‖ḡ‖²)/(M−1)` 只依赖组怎么切，不依赖前向块多大。所以新增
`train.micro_group_responses`（默认 = micro，即 D-15 之前的行为），前向块仍由
`micro_batch_responses` 决定。组内把各块的梯度**按 token 数加回未归一的和**、组末除一次
（`grpo_loss` 用 `token_mean`，每块已被自己的 token 数除过，必须乘回来），得到与
"一次前向塞 group 条"数学等价的 g_i。

`metrics.jsonl` 现在**同时记两个数**：`n_groups`（σ 的 M）与 `n_forward_chunks`（实际前向块数），
`n_micro_batches` 保留为 `n_forward_chunks` 的旧别名。

**三、ablation（`configs/ab_g8m8 / ab_g4m4 / ab_g8m4`，stub + fake rollout，CPU，同一份数据）**

| arm | group | micro | M | chunks | ‖ḡ‖ | σ | d |
|---|---|---|---|---|---|---|---|
| ab_g8m8 | 8 | 8 | 4 | 4 | 0.03697 | 0.06405 | 0.57724 |
| **ab_g8m4** | 8 | 4 | 4 | 8 | **0.03697** | **0.06405** | **0.57724** |
| ab_g4m4 | 4 | 4 | 8 | 8 | 0.03705 | 0.08745 | 0.42362 |

`σ(ab_g8m4)/σ(ab_g8m8) = 1.0000`（4 步全等，‖ḡ‖ 逐位相同）；
`σ(ab_g4m4)/σ(ab_g8m8) = 1.4249`（√2 = 1.414，4 步）。**解耦是精确的，不是近似。**

真实模型上（同一份 rollout，只改 group）：σ 0.02772 → 0.02087（比 1.328）、d 0.1192 → 0.1606、
‖ḡ‖ 不变（0.003）；另一次采样 σ 0.03063 → 0.02209（比 1.386）。M=64 / chunks=128 已落进
metrics。单步估计下比值在 1.33–1.39 之间波动，是 σ 的有限样本误差，不是口径错。

**四、D-14 三笔代价的现判**

1. ~~σ ∝ 1/√m 放大 1.41，d 不可比~~ → **已消除**：group 保持 8，σ 回到设计口径（ratio 1.0000）。
2. ~~update 随 M 翻倍变慢、mu=1 占 44%~~ → **测错了**。同卡同数据：micro=4 的 update 46.8–49.4 s，
   micro=8 的 update 49.3 s，micro=4 反而略快。代价不在 update 侧。
3. ~~余量 3.1 GiB、L_max 可能到 712 未实测~~ → **虚警 + 已实测**。prompt 只有 72–102 token，
   completion 被 cap 512 硬截断，所以 L_max 的结构上界是 `prompt_max + 512 ≈ 614`；实测
   128 次调用的 max 就是 614（p50 586、p90 610）。那次 46,034 MiB 本来就是最坏长度下测的。
   并且加上 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后峰值 **40,172 MiB**
   （同一份数据不开是 43,650），余量从 3.1 GiB 变成 8.9 GiB。

**五、micro=8 在 48G 上仍可跑，只是不值**

`util 0.32 + max_num_seqs 128 + expandable_segments` 能过（见 D-14 后续会话），
但 rollout 83.1 s vs 56.5 s，总时长 +28%。既然 σ 可以零成本解耦，没有理由付这 28%。

**证据**：`notes/2026-10-25_cloud48g/HEADROOM.md`、`reg_g4.log`、`g8.log`、`g4b.log`、`g8c.log`、
`chunk512.log`、`vram_*.csv`；探针 `tools/probe_memory.py`；对照 `tools/compare_runs.py`。

### D-16 vLLM 后端的 rollout 不可复现：同配置同 seed 两次跑出不同数据（2026-10-25）

`rollout.py:239` 确实把 seed 传给了 `SamplingParams(seed=seed*7919+rollout_idx)`，但
**同配置、同 seed 连跑两次，pass_rate 0.188 vs 0.238**（`det_a.log` 与 `det_b.log`，
512 responses，仅差每次运行的调度批次）。所以这个 seed 不足以让 rollout 逐位重现。

影响三条，都要写进方法节或限制说明：

1. **跨运行的 σ/d 不可直接相减**。两边数据不同，差异里混着采样噪声。D-15 的对照因此
   只能在 stub + fake rollout（数据由 rng 构造、逐位相同）上做，那是唯一干净的仪器。
2. **臂内配对仍然成立**。`plan_partition` 让所有 mu 臂共用同一次 rollout，所以 `claims.yaml`
   的配对比较不受影响——受影响的是"重跑一次验证"这种跨运行复现。
3. **3 seed 实际是 3 次独立抽样**，比"3 个可控种子"的方差更大，估 σ 的离散时应按此解释。

未做的验证：不确定这是 vLLM 版本/`max_num_batched_tokens`/前缀缓存中的哪一个造成的。
若论文要声称可复现，只有两个选择：固定 vLLM 内部批次形状，或改用 HF 后端（`torch.manual_seed`
在 `rollout.py:127` 已设，代价是慢 5.8 倍）。

**→ 2026-10-25 晚已结案：见 D-19。** 根因是前缀缓存 + chunked prefill + 未固定的 token
预算共同让批次形状随调度漂移；三者关掉后两次运行 0/512、0/1024 行 token 完全相同。
本条的三条影响里，第 1 条（跨运行不可比）随之失效，第 2、3 条仍成立但原因要改成
"每请求独立 seed"（D-18）而非"后端不可复现"。

### D-17 9 run 定档：选 A（micro 4 / group 8），修一处外推错误，把"可复现"落到数据层（2026-10-25）

三件事一起记，因为它们互相依赖：选哪套配置决定钱，钱决定能不能先做探针，探针决定 D-16 怎么写。

**一、外推错误（改 D-13 的表）**

D-13 的 9 run 外推表把 update 成本记成"随 `M = b/micro` 变"，于是 mu=8 记 44 s、mu=16 记 22 s。
错在分子：**每 rollout 的前向块数 = `mu × (b/micro) = Br/micro = 1024`，与 mu 无关**。表里只算了
一步的块数，漏乘 mu。修正：

| arm | 块数/rollout | rollout | update | 单 run | 3 seeds |
|---|---|---|---|---|---|
| mu=1 / mu=8 / mu=16 | 都 1024 | 446 s | 378 s | **1.83 h** | 5.5 h |

单块成本取实测 0.37 s（`g8c.log`：47.3 s / 128 块）。**9 run = 16.5 h ≈ ¥34**（原记 11.8 h / ¥25）。
预算结论不变（<¥100，留 66% 余量给校准门与重跑），但差 4.7 h，不能留在文档里。
教训：凡"每 rollout / 每 step"的量，写表时必须把分母写出来，否则 mu 一进表就错。

**二、配置定档：A**

用户 2026-10-25 拍板用 A。三套候选的实测与取舍：

| | A micro4/group8/util .45/mns256 | B micro8/group8/util .32/mns128 | C micro8/group8/util .45 |
|---|---|---|---|
| rollout / update（512 条） | 56.0 / 47.3 s | 83.1 / 49.3 s | 四次全 OOM |
| s/resp | **0.2018** | 0.2586 (+28%) | — |
| 峰值 | **40,172 MiB** | 46,344 MiB | — |
| 余量 | **8.9 GiB** | 2.8 GiB | — |
| σ 口径 | group=8 = 设计值 | 同 | — |
| 9 run | **16.5 h ≈ ¥34** | 21.2 h ≈ ¥44 | 死路 |

- C 是死路：探针 micro=8 在 L_max 712 下训练侧 31.98 GiB，util .45 的 vLLM 占 21.31 GiB，
  合计超可用量 5.9 GiB。除非 logits 降 bf16（已否决：与 fp32 差 0.03–0.12 nats）。
- B 唯一的好处是字面上不用写 micro 偏离，代价是 +4.7 h / +¥10 / 余量掉到 2.8 GiB / 并发砍半，
  而 σ 口径 A 已经保住了。**用"字面一致"换"少 3 倍余量"不划算。**
- A 要申报的偏离只有一条：`micro 8 → 4`，且仅指前向块大小；`micro_group_responses: 8` 保持
  预注册值，`metrics.jsonl` 同时记 `n_groups`（=b/8）与 `n_forward_chunks`（=b/4），事后不会混。
  等价性证据见 D-15 §三（stub 上 g8m8 与 g8m4 逐位相同；真机 ‖ḡ‖ 三位小数不变）。

落地文件：`configs/cloud48g_main.yaml`（A 本体，8×4096）、`scripts/run_9.sh`（9 次循环 + 三个前置门，
含 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与每次清空 `runs/_adapter`）。**未启动。**

**三、"可复现"落到数据层（L0）**

D-16 说采样不可复现。但"跑完发现不可复现"真正的杀伤力不是采样随机，而是**事后无法回答
"这个 σ 是从哪批 response 算出来的"**：`samples.jsonl` 每 rollout 只写 8 条
（`train.py` `log_rollout_samples`），32768 条里 8 条。

`train.py:save_rollout_ids()` 现在把每 rollout 的完整 `prompt_ids` / `completion_ids` 写进
`runs/<tag>/rollouts/r{idx}.npz`（ragged → flat uint32 + lengths，压缩后约 25–30 MiB/run），
`rollouts/index.jsonl` 记 sha256，同一 sha 也进每行 metrics（`rollout_data_sha`，schema 里可选）。
**同一指数第二次写不覆盖第一次**：resume 会重新采样，那是不同数据，必须让它可见而不是被盖掉。

`tools/audit_rollout_ids.py` 提供两件查询：`--compare` 逐行比两份 archive（L1 探针的判据），
`--group G` 量组内多样性（见四）。

测试：`tests/test_rollout_archive.py`（5 条，纯 numpy），全套 21/21 通过。

**四、D-16 的两个待验假设，和 L1 探针**

1. **批次形状**（最可能）：`enable_prefix_caching=True` + `enable_chunked_prefill=True` 是
   `det_a`/`det_b` 逐字相同的引擎配置。一个 prompt 的 G=8 条共享 ~100 token 前缀，命中缓存与
   否取决于调度顺序，两条路径的 attention 数值不同 → 平局附近翻 token → 轨迹发散。
   `VLLMBackend` 现在能透传三件事：`enable_prefix_caching` / `enable_chunked_prefill` /
   `max_num_batched_tokens`（默认全不设 = 老行为）。探针配置 `configs/cloud48g_det_fixed.yaml`
   全关 + 2048，与 `cloud48g_det_base.yaml` 各跑两次（4×512 条 ≈ ¥0.4），`--compare` 逐位判。
2. **组内样本相关性**（比 1 更该先查）：`rollout.py` 给一次 rollout 的所有请求传**同一个**
   `SamplingParams.seed`。若 vLLM 按请求独立播种同值，同 prompt 的 G 条会共享随机流，
   有效样本从 512 掉到 64 个 prompt，σ 与所有 pass 的置信区间都要重算。
   独立证据：三次 512 条 pilot 的 pass = **0.188 / 0.238 / 0.357**，固定 n 时二项 sd 只有 0.019，
   实测散布 ≈ 9σ → 不是抽样噪声，是分布真变了。
   `--group 8` 直接量：组内 distinct 完成数、首 token 不同数、组内长度 sd / 总体 sd。
   `seed_per_request` 开关（默认关 = 老行为）已加，等证据再决定是否翻。

含义（若假设 2 成立）：`σ` 的绝对量级要重标定，`claims.yaml` C3 的 CI 要按 prompt 数而非
response 数算，且 pilot 阶段（512 条）的 pass 不能当结论——正式 run 每 rollout 4096 条，
同一个效应被稀释 8 倍。

**五、开销账**

L0 代码 + 测试：0 元。L1 探针 4×512 条 ≈ ¥0.4。云端测试与同步 ≈ ¥0.2。9 run 未启动。

### D-18 vLLM 的 G 条样本其实只有 1 条：共享 `SamplingParams.seed` 让组内退化（2026-10-25）

**这是一条比 D-16 严重的 bug，而且它一直在所有 vLLM 运行里。**

现象（`det_base_a`，512 条 = 64 组，全量 archive 而非 8 条样本）：
组内互不相同的完成平均 **1.25/8**（min 1），首 token 平均 **1.016 种**，组内长度 sd 6.77 /
总体 163.58（ratio 0.041）。旧 run 的 `samples.jsonl`（前 8 条正好是 p0×8）在**每个** vLLM run
都是逐字相同的 8 条；同一实验的 HF 后端是 8 条各不相同。

后果不是"样本少一点"，而是设计本身被抽空：

- `group_advantages` 对零方差组直接把 advantage 置 0（`grpo.py:43`），
  实测 `zero_var_group_frac = 0.875` → **87.5% 的 response 对梯度零贡献**；
- σ 是在"一个样本重复 8 次"的组上量的，所以它既不反映采样噪声、也不反映 micro-batch 之间的差异；
- `pass_rate` 的有效样本是 64 个 prompt，不是 512 条 response。

原因：`rollout.py` 给一次 rollout 的所有请求同一个 `SamplingParams.seed`。同 prompt + 同权重 +
同 seed → 同一条轨迹。少数组会有差异，只是批内位置不同带来的浮点微扰在平局处岔开。

修法：`rollout.vllm.seed_per_request: true`，每请求 `seed*1_000_003 + 行号`；默认 `false`
保留旧行为，便于已跑数据仍能解释（它们都是 `false` 跑的）。

| | 共享 seed（旧） | 每请求 seed（新） |
|---|---|---|
| 组内 distinct | 1.22–1.36 / 8 | **8.00 / 8**（min 8） |
| 首 token 种数 | 1.02 | 7.05 |
| 组内/总体长度 sd | 0.04–0.06 | 0.93 |
| `zero_var_group_frac` | 0.797–0.875 | **0.000–0.016** |
| σ | 0.0204–0.0306 | **0.0635–0.0685** |
| d | 0.12–0.18 | 0.10–0.14 |

σ 涨 3.15×，`d` 相应降到约 1/3。**清单里所有 pilot 的 σ/d 绝对值都与新口径不可比**——
以前的 σ 量的不是它声称的东西。`claims.yaml` C4 只比 µ 之间的方向与比值，方向性不受影响；
但"σ 的量级"这类叙述要全部重写。

### D-19 D-16 结案：不可复现的根因是批次形状，固定形状后可逐位复现（2026-10-25）

D-16 记"同配置同 seed 两次跑出 0.188 / 0.238，原因未定位"。现在有了 token 级判据
（`tools/audit_rollout_ids.py --compare`）：

| 配置 | 两次运行逐行比较 | pass |
|---|---|---|
| vLLM 默认（前缀缓存 on、chunked prefill on、无 token 预算） | **355 / 512 行不同（69.3%）** | 0.238 vs 0.188 |
| `enable_prefix_caching: false` + `enable_chunked_prefill: false` + `max_num_batched_tokens: 2048` | **0 / 512 行不同** | 0.334 vs 0.334 |
| 再加 `seed_per_request`（512 条） | **0 / 512 行不同** | 0.205 vs 0.205 |
| 再加 `seed_per_request`（1024 条） | **0 / 1024 行不同** | 0.225 vs 0.225 |

机制：一个 prompt 的 G 条共享 ~100 token 前缀，前缀缓存命中与否由调度顺序决定，命中与全量
prefill 走不同 attention 路径 → 数值不同 → 平局翻 token → 轨迹整条岔开。三者都关掉后，
调度器对同一份输入的决策成为确定函数。

为什么补测了 1024：512 条只铺满两个调度波次，正式 run 是 Br=4096（十六个波次）。
"固定形状"只有在调度决策仍是输入的确定函数时才有用，所以要在更大的批上验一次没有退化。
**Br=4096 仍未验**（单次 1.8 h）；机制与批大小无关，且 512→1024 没有退化，接受这个残余风险，
写进方法节。

吞吐：1024 条 rollout 84.8 s + update 89.8 s = 0.1705 s/response（512 条时 0.189），
单 run 外推 ≈1.6 h，9 run 14–16 h ≈ ¥30–34，与 D-17 一致。

**D-16 因此降级**：不再写"vLLM 采样不可复现"这条 limitation，改成"vLLM 默认批次形状让采样
不可复现，固定形状后逐位复现"。`rollouts/r{idx}.npz` 全量落盘保留，身份从"唯一退路"变成常规保障。

**证据**：`notes/2026-10-25_cloud48g/DETERMINISM.md`、`det/*/`（九个运行的 archive、audit、
metrics、samples）、`probe_det1.log`、`probe_det2.log`；配置
`configs/cloud48g_det_{base,fixed,seed,seedfix,seedfix1024}.yaml`。

### 会话开销（2026-10-25 第三会话）

九个探针运行 ≈ ¥1.3（6.1k responses 含引擎启动）。9 run **未启动**，等用户放行。

### D-20 14–16 h 是怎么算出来的，缺哪些项，以及数据完整性审计（2026-10-25）

用户问"14–16 h / ¥30–34 确定吗"。答：**不确定，它是一条外推带，不是余量。** 下面把推导摊开，
并列出四类没有计入的开销。

**一、两个实测点（mu=1 / seed=0 / 固定批次形状 / 每请求 seed / step 0，resp_len 283）**

| 条数 | rollout | update | 合计 | s/resp |
|---|---|---|---|---|
| 512 | 50.7 / 49.6 s | 46.1 / 46.1 s | 96.8 / 95.7 s | 0.1891 |
| 1024 | 84.8 / 83.2 s | 89.8 / 89.5 s | 174.6 / 172.7 s | **0.1705** |

两次同配置逐位可复现，所以这四组数的重复精度 <2%（`probe_det2.log`；1024 那发的
`summary.json` 记 `wall_hours_measured = 0.048523` h = 174.7 s，与日志一致）。

拆成固定项 + 边际项（N = responses）：
`rollout = 16.6 + 0.0666·N`，`update = 2.4 + 0.0854·N`（s）。
update 里含"old-policy logprob 整遍前向 + 梯度遍"两次全量前向，所以边际 0.085 s 是两次，
不是一次。两次前向的块数都是 `Br/micro`，**与 mu 无关**——这就是 D-17 修正的依据。

**二、单 run 时长**

* 只用边际（对 4096 条外推）：`32768 × 0.1705 = 5587 s = 1.55 h`
* 带固定项：`8 × (16.6 + 0.1520×4096) = 5120 s = 1.42 h`
* 每 run 未进 meter 的开销：启动 ~70 s（两次 probe run 的间隔 252 s − 174.6 s）、
  `state.pt` 8×~2 s（837 MB，`dd` 实测盘速 730 MB/s）、rollout npz 8×0.3 s、
  adapter 导出+重载 8×~5 s ≈ **130 s**
* ⇒ 单 run **≈1.59 h**，九次 **≈14.3 h ≈ ¥29.7**（¥2.08/h）
* 悲观口径：取本卡 A 配置上**最慢的一次实测** 0.2018 s/resp（`g8c`，512 条，
  那是固定形状之前的采样器，比新口径慢 6%），即假设 4096 条完全退回小批效率：
  **1.87 h/run → 16.9 h → ¥35**

所以 `14–16 h` 的区间 = 「1024 条边际外推」到「512 条最慢点外推」，**中间没有人为预留**。

**三、四类未计入的开销**

1. **长度漂移**。成本与 `resp_len` 成正比（rollout 是 decode、update 是 token 前向）。现在 283；
   旧 run 实测 283–322（+14%）→ 单 run ±0.2 h。训练推进后长度会变，方向未知。
2. **持续负载**。探针只跑 3 分钟。16 h 连续跑的温度/时钟与邻租户影响**完全没测**，
   按 5–10% 记 1–1.6 h（这是估计，不是测量）。
3. **mu=8/16 与 mu=1 同价**是假设：块数 `Br/micro` 与 mu 无关（已由代码确认），
   但每 step 的 `zero_grad` + 梯度范数（66 M 个 LoRA 参数）+ `opt.step` 会多出 120 步，
   量级应是分钟级。**这一条没有任何实测**——三臂都只在 mu=1 上报过吞吐。
4. **重跑与掉线**：0 计入。单 run 全重跑 +1.6 h；掉线有 `state.pt`（每 rollout 写）兜底，
   最坏损失一个 rollout ≈ 11.6 分钟。

**四、数据完整性审计（当前代码跑完一次，能不能做完全部事后检查）**

已落盘（每 run）：`config.resolved.yaml`、`run_meta.json`、`metrics.jsonl`（每 optimizer step 一
行，39 字段）、`rollouts/r{idx}.npz`（全量 prompt/completion ids，1024 条 = 0.26 MiB，
4096 条 ≈1 MiB）、`rollouts/index.jsonl`（sha256/字节数，重复采样写 `_v1` 不覆盖）、
`samples.jsonl`（8 条文本）、`state.pt`（837 MB）、`summary.json`。

可事后重建：每条 response 的 reward / correct（`completion_ids` → decode → `verify`，
用 prompt ids 反查题池取答案）⇒ **pass_rate 的 prompt 级自助 CI 可以算**，不依赖落盘的 8 条样本。

拿不全的四项：

1. **`metrics.jsonl` 的 `wall_ms_update` 恒为 0**（`train.py:534` 写死 `0.0`，真实值在
   `train.py:550` 之后才算出，只进了 tee 的文本日志与 meter）。⇒ 结构化的每步耗时缺失，
   事后"这一步为什么慢"查不到，只能 grep 日志。
2. **`run_meta.json` 没有 git commit、没有 vLLM 版本**（`_run_meta` 只记
   torch / transformers / cuda / python / gpu / argv / cwd / platform）。D-17 里写过"补 git
   commit 与 vllm 版本"，**实际没落地**；vLLM 版本只以 `v0.32.0` 存在于引擎初始化那行日志里。
3. **没有 per-rollout 权重快照**：`state.pt` 每 rollout 覆盖，`runs/_adapter` 每 run 清。
   ⇒ 事后无法重放某一步的 logprob / ratio（只能看聚合值）。
4. **C6 的后半句测不了**：claims 要求"被 clip 的梯度范数份额高于其 token 份额"，
   但 `grpo.py` 只返回 `clip_frac_low/high/any`（token 份额），没有按 token 计的梯度范数份额，
   metrics 里也没有这个字段。要么加仪器，要么在 REPORT 里把 C6 后半句标为未测。

第 1、2 项各约 10 行改动，第 3 项要把 adapter 按 rollout 落盘（+1.3 GB/run），
第 4 项要额外一次 backward hook。**四项都未改**，等拍板。

### D-21 开跑前补三个观测缺口：每步墙钟、溯源信息、C6 的后半句（2026-10-25）

D-20 审计出四个缺口，用户拍板"现在补"。补了三个，第四个（per-rollout 权重快照）没补。

**一、`wall_ms_update` 恒为 0（结构性 bug）**

旧代码在 mu 循环**内部**就写 metrics 行（`train.py` rec 字面量里 `"wall_ms_update": 0.0`），
而这一步的墙钟要等整轮 rollout 的 update 跑完才算得出来（`upd_ms` 在循环之后）。
后果不止一列是 0：`cost_cny_cum` 也是在同一位置取的 `meter.totals()`，那时 meter 里的
update 墙钟还是 0，所以**每一行都少算了它所在 rollout 的 update 成本**，只有 `summary.json`
是对的。

修法：每 rollout 的 rec 先放进 `pending`，等 `upd_ms` 出来后按 `(k+1)/mu` 补进
`wall_ms_update` 与 `cost_cny_cum`，再一次性 flush。第 k 行在自己的时间点上确实只吸收了
`(k+1)/mu` 的 update 墙钟，与 meter 的分摊口径一致，所以 `cost_cny_cum` 单调且末行等于 summary。

代价（必须说清）：崩溃发生在 rollout 中途时，这一轮的 mu 行**整批丢失**，而旧行为是留下半批。
但 resume 本来就会 `truncate_to_rollout(start_rollout)` 把这一轮的行全删掉，所以丢失的行
不会污染账本，只是少了一段"半途观测"。mu=1 时两者等价。

**二、`run_meta.json` 没有溯源**（D-17 说要补但没落地）

新增 `git_commit` / `git_dirty` / `vllm` / `numpy` / `nvidia_driver`。全部走失败即空的兜底
（`_sh()` 5 秒超时，`_pkg_version()` 捕获异常），provenance 不值得让 16 小时的 run 死掉。

**三、C6 后半句（"0.5% token 占 50–70% 梯度范数"）**

先确认原方案是错的：`grad_norm_clipped/grad_norm_raw` 是**整步**的裁剪前后范数比，不是
"被 clip 的 token 占了多少范数"，两者不能互相代替。
再确认一个更根本的事实：**被 clip 的 token 在 PPO/GRPO 里梯度恰为 0**（`min` 走了 clamp 那支），
所以"被 clip 的 token 的梯度范数份额"只能对**未 clip 的代理损失**发问：
每 token 的 `dL/dlogp` 在未裁剪处正是 `|ratio·adv|`，被 clip 的那些 token 占这份质量的多少，
就是裁剪丢掉了多少。

实现：`grpo.py` 在已有的 `no_grad` 诊断块里多算一个 `clip_grad_mass_share`
（`Σ_clip|ratio·adv| / Σ_all|ratio·adv|`），纯读已有张量，不加 forward、不改图。
metrics 里是**可选**字段（老行仍过 schema）。REPORT 第 16 行改了定义，并写明
**参数空间范数份额没测**（要逐 token 反传），报告里不得等同。

顺带记一个会影响 C6 解释的事实：`mu=1` 时整轮只有一步，`old_lp` 就是当步策略，`ratio ≡ 1`，
所以 **mu=1 的 clip_frac 结构性为 0**（实测 0.0），C6 的数据只会来自 mu=8/16 的 `mu_step>0` 行。

**四、验证（云端，¥0.1）**

* 本地与云端 `tests/run_all.py` **22/22**（新增 `test_clip_grad_mass_share`：无裁剪→0、
  全裁剪→1、padding 不计入）。
* 回归探针 `det_sf_c` 用 `det_sf_a` 的**同一份配置**重跑 512 条：
  `prompts_identical: true`、`completions_identical: true`、**rows_differing 0 / 512**、
  `pass_rate 0.205078 → 0.205078`、`resp_len 285.3652 → 285.3652`。
  即插桩对采样与结果透明。
* 残留差异：σ 0.06346013 → 0.06345711（相对 4.7e-05）、d 0.13586728 → 0.13585257。
  这**不是补丁造成的**（mass 计算在 `no_grad` 里、在 loss 之后、不碰计算图），是 update
  路线上浮点归约本身不逐位（与 D-19 记的 0.06852/0.06853 同一现象）。所以口径要改：
  **采样可逐位复现，训练侧度量复现到 ~1e-5 相对**。
* 新字段实测落盘：`wall_ms_update = 46434.411`（旧 0.0）、`cost_cny_cum = 0.0559`、
  `clip_grad_mass_share = 0.0`（mu=1 结构上就是 0，符合上面第三条）、
  `meta.git_commit / git_dirty / vllm=0.32.0 / numpy=2.3.2 / nvidia_driver=580.119.02` 全部有值。

**五、没补的一项**

per-rollout 权重快照（+1.3 GB/run，9 run +12 GB，盘上剩 32 GB）。目前"重放某一步"做不到，
但 token 级 archive 已经能支撑数据层复现，暂不需要。

### D-22 按 rollout 存 LoRA 快照；以及为什么"参数空间逐 token 梯度份额"不该做（2026-10-25）

用户要求"条件允许就尽可能把实验数据拿全"，并问这两件事会不会让云上 OOM。

**一、per-rollout adapter 快照：可以补，且不占显存**

实测体积：`runs/_adapter` 里是 392 个张量、**F32、266 MiB**（不是先前估的 132 MiB——PEFT 存
float32，而这正是优化器持有的 dtype；转 bf16 会让快照不等于"被训练的那个值"）。
`export_adapter` 走 PEFT `save_pretrained`，只对 LoRA 那 69.7 M 参数做 CPU 侧落盘，
**不分配新的显存**，所以与 40.2 GiB 峰值无关。

新增 `logging.save_adapter_every_rollout`（默认开）与 `train.snapshot_adapter()`：
在 `backend.generate()` 返回后**立刻**拷一份到 `runs/<tag>/adapters/r{idx}`（此时 `_adapter`
里的权重正是生成这批 response 的那一版），写 `adapters/index.jsonl`（字节数、safetensors
sha256、adapter 版本号）。同一 idx 二次快照写 `_v1` 不覆盖，与 rollout ids 的规则一致。

磁盘账（重算过，之前估错了）：
| 项 | 单价 | 9 run |
|---|---|---|
| adapter 快照 | 266 MiB × 8 | **19.1 GiB** |
| `state.pt` | 837 MiB × 1 | 7.5 GiB |
| rollout ids | ~1 MiB × 8 | 0.1 GiB |
| 合计 | — | **26.7 GiB** |

盘上原本剩 31 GiB，加上 `det_*` 九个探针 run 的 `state.pt`（7.5 GiB，只是 checkpoints，
判据不依赖它）删掉后剩约 38 GiB。`snapshot_adapter` 自带余量闸：可用空间低于 max(4 GiB,
本次需求×1.5) 就跳过并把原因写进 index，**不会因为磁盘满而搞死 16 小时的 run**。

**二、backward hook：不会 OOM，但拿不到想要的东西**

结论先说：**不做**。理由不是显存，是定义。

* 精确的"参数空间逐 token 梯度份额"需要每个 token 一次反传：`g_t = ∂L_t/∂θ`。
  一个 rollout 116 万个 token，等于 116 万次 backward；用 vmap 一次算出来要物化
  116 万份 69.7 M 参数的梯度缓冲，量级 10^5 GiB。**不可行**。
* 唯一"便宜"的 hook 位置是 lm_head 的 grad_output，形状 `(B,T,V)`：
  micro=4、T=614、V=151669、bf16 → **745 MiB**，加 fp32 副本 1.5 GiB。
  而当前峰值 40,172 MiB / 可用 47,370 MiB，余 7 GiB，**所以不会 OOM**——
  但它算出来的就是 `c_t·(p−onehot)`，即已经记下的 `clip_grad_mass_share` 的同一份系数，
  **没有新信息**。
* 还有一条硬伤：`model.gradient_checkpointing=true`，hook 会在重算图里二次触发，
  拿到的是错误的中间值。

所以 ④ 不是"要不要花显存"的问题，是"这个量在合理成本内不存在"。已经记的
`clip_grad_mass_share`（D-21）就是这个侧证的上限，REPORT 里已写明参数空间份额未测。

**三、代价与状态**

* 额外耗时：每次快照 = 266 MiB 拷贝 + sha256，实测盘速 730 MB/s → 约 0.8 s，
  8 次/run ≈ **6 s/run**（0.1%）。
* 补测试：`tests/test_adapter_snapshot.py` 3 条（拷贝+摘要、二次快照不覆盖、缺 adapter
  只记录不致命），全套 **25/25**。
* 代价：为改这一版，13:03 启动的战役在跑完第一个 rollout 前被停掉，浪费约 **¥0.05 / 2 分钟**。
  重跑前删掉 `runs/mu1_s0`（否则 `run_9.sh` 会因"有目录无 state.pt"拒绝启动）。

### D-23 第一次重启后 25 分钟：run_meta 记了一个错的 commit，改完再启动（2026-10-25）

**症状**：campaign 跑完第一个 rollout 后，`runs/mu1_s0/run_meta.json` 写
`git_commit = 0fe107b…`、`git_commit_source = "git"`、`git_dirty = true`。
0fe107b 是**几轮前的提交**，实际部署的代码是 a73dc91。也就是说溯源字段在说谎——
比没有这个字段更糟。

**根因**：机器上还留着一个**过期的 `.git`**（早前某次整仓拷贝留下的），而
(a) `_run_meta` 优先信 `git rev-parse`，(b) `run_9.sh` 的 preflight 会把
`git rev-parse HEAD` 写回 `COMMIT` 文件。两步都把过期值当权威。
我之前的生产笔记记的是"云端无 .git"——**错了**，这次实测确认有。

**修法**（三处）：
1. `_run_meta` 改成**文件优先**：COMMIT 与 git HEAD 不一致时取文件，并把两者都记下来
   （`git_commit` / `git_commit_source` / `git_head_in_checkout`），让分歧可见而不是被静默解决。
2. `run_9.sh` preflight：COMMIT 必须是 40 位十六进制且非全零，否则**直接拒绝启动**；
   若 `.git` 存在且与其不一致，打警告并声明"以 stamp 为准"，不再回写。
3. 顺手修日志：`export PYTHONUNBUFFERED=1`。上一次重启时日志停在 `[info] arm …` 十分钟不动，
   我一开始怀疑 tee 死了，其实是 stdout 接管道后是**块缓冲**，8 KiB 才刷一次；
   而 `metrics.jsonl` 是逐行落盘的，所以"数据在走、日志不动"。监控要以 metrics 为准。

**代价**：这一轮又停了，浪费约 25 分钟 / ¥0.9（首个 rollout 已完成：rollout 290.3 s、
update 367.1 s、¥0.38）。删掉 `runs/mu1_s0` 与过期 `.git` 后重启。

**顺带拿到的真实数字**（首个 rollout，Br=4096、micro=4、group=8、mu=1）：
`rollout 290.3 s`（D-20 预测 289.4 s，误差 0.3%）、`update 367.1 s`（预测 351.8 s，+4.3%）、
合计 **657.4 s/rollout**、`resp_len 292.9`（预测用的 283，+3.5%，正好解释 update 那 4%）、
`pass 0.218`、`zero_var 0.0`、`σ 0.0663`、`d 0.0723`、`¥0.38/rollout`。
按 8 rollouts + 93 s 固定开销外推：**单 run ≈1.49 h，九次 ≈13.4 h ≈ ¥28**——
落在 14–16 h 区间的下沿（因为 rollout 完全命中、update 只慢 4%）。

### D-24 首个 rollout 落地：链路全通，但成本记账少记 17%（old-policy 前向在计时之外）（2026-10-25）

**一、实测两行（mu1_s0，Br=4096 / micro=4 / group=8 / mu=1）**

| | r0 | r1 |
|---|---|---|
| rollout | 288.35 s | 291.57 s |
| update（`wall_ms_update`） | 368.21 s | 373.99 s |
| `resp_len_mean` | 292.93 | 299.52 |
| `pass_rate` | 0.218018 | 0.229980 |
| `zero_var_group_frac` | **0.000** | 0.003906 |
| σ / d | 0.06627 / 0.07232 | 0.06550 / 0.07144 |
| `cost_cny_cum` | 0.3793 | 0.7639 |
| `ts` | 1792906736.708 | 1792907530.212 |

链路全部验证通过：`rollout_data_sha` 两行不同（不同批数据）、`n_groups=512` / `n_forward_chunks=1024`
（D-15 的解耦口径如实落盘）、`adapters/r000` 267 MB 落盘（`sha256=88f2b66c…`，源 `_adapter` 的
`adapter_model.safetensors` mtime 05:25:54 = 训练起始，即"生成这批 response 的那一版"，语义正确）、
`state.pt` 799 MB、`zero_var ≈ 0` 说明 D-18 的 per-request seed 修复在真机上成立。

**二、发现：`wall_ms_update` 不含 old-policy logprob 前向，成本少记 17%**

`train.py:513-527` 的 old_lp 整遍前向在 `upd_t0`（`:534`）**之前**，所以不在 `wall_ms_update` 里。
用 `ts` 反推：`Δts = 793.504 s`，减去 rollout 291.57 与 update 373.99，**剩 128.0 s** 是
old_lp 前向（1024 块 × 4096 条）+ npz/adapter 落盘。r1 也是 128 s，**不是首次编译开销，
是每轮常驻**。

后果：`[r0] done` 打的 `0.182 GPU-h`、`cost_cny_cum`、以及 meter 都只按 rollout+update 计，
**每 rollout 真实墙钟 793.5 s 而不是 657 s，成本少记 17%**（0.3793 vs 实际 0.4584）。

**三、ETA 修正**

| | 旧 | 新 |
|---|---|---|
| 单 rollout | 657 s | **793.5 s** |
| 单 run | 1.49 h | **1.775 h** |
| 9 run | 13.4 h | **15.98 h** |
| 完赛 | 北京次日 02:49 | **北京次日 05:22** |
| 成本 | ¥28 | **¥33.2**（仍 < C7 的 ¥55/9、¥6/单 run） |

**四、为什么不回改（用户拍板前）**

`ts` 是每行真实写入时刻，相邻差就是真实墙钟，**这批 run 的成本可以事后精确重算**，
所以不需要为了记账停掉 16 小时的战役。D-20 里"update 含 old-policy 整遍前向 + 梯度遍
两次前向"的说法被证伪：`wall_ms_update` 只含**梯度遍 + opt.step**（一次前向）；D-20 由
512/1024 两点拟合的 update 斜率（0.0854 s/resp）本身是对的（368 s vs 预测 352 s，+4.5%），
漏的是**拟合之外**的 old_lp 那一项。

### D-25 把 old-policy 前向纳入计时，并给 16 小时战役装一个显存安全阀（2026-10-25）

用户拍板两件事：D-24 的记账 bug 现在就改（接受停机损失）；`resp_len` 继续涨时，余量跌破
阈值允许自动降级。

**一、计时口径（改 `upd_t0` 位置）**

`upd_t0` 从 old_lp 段之后（原 `:534`）移到之前，于是 `wall_ms_update` 现在包含
"old-policy 整遍前向 + mu 次梯度遍 + opt.step"，与 meter、`cost_cny_cum`、`[rN] done`
的 GPU-h 全部对齐。

同时新增 `wall_ms_old_logp`（只记在每 rollout 的第 0 行，schema 里是**可选**字段，老行仍过
校验）。理由：不单记的话，"update 496 s"里混着两个性质不同的东西——一次 no_grad 前向和一次
梯度前向+step。这两项随 mu 变化的规律不同（old_lp 每 rollout 一次、与 mu 无关；梯度遍是 mu 次），
混在一起会让 mu=1/8/16 三臂的耗时无法拆解，而"mu 是否同价"正是 D-17 里唯一没有实测的假设。
所以既纳入计时（成本准），又单独可见（可拆解）。

`[rN] done` 打印改成三段：`rollout 288.3s oldlogp 128.0s grad 368.2s update 496.2s`。

**二、"降 max_num_seqs"单独不解决问题（改了方案）**

涨的是**训练侧**：vLLM 在 Br=4096 下按 util 0.45 预留 21,317 MiB 并一直占着；峰值的其余
约 19.7 GiB 是策略侧（logits 按 64 分块的 T×151,669 + attention 的 T²）。实测 `resp_len`
292.9 → 299.5 时峰值 40,172 → 40,974 MiB，增量全在训练侧。**`max_num_seqs` 只缩小 KV 的
工作集，不缩小预留**，单独降它在这个 OOM 路径上几乎无效；真正把内存还回去的是
`gpu_memory_utilization`。

所以 `configs/cloud48g_lowmem.yaml` 同时降两项：util 0.45 → **0.38**（还 3.3 GiB），
mns 256 → **192**（让调度器不超额使用变小的 KV 预算）。代价预估 rollout 慢 10–20%
（288 s → 320–345 s），九次多约 1.5 h——作为应急可以，不值得预防性采用。

**三、安全阀 `tools/vram_guard.sh`**

每 20 s 读一次 `nvidia-smi memory.free`，**连续 2 次**低于 2048 MiB 才动作（单次抖动不动手）。
动作：停 `run_9.sh` → 停 `train.py` → 停 `EngineCore` → 用 `CFG=cloud48g_lowmem.yaml`
重启 `run_9.sh`。因为 `run_9.sh` 会 skip 有 `summary.json` 的、resume 有 `state.pt` 的，
所以这是**续跑**而不是重跑，最坏丢一个 rollout（`state.pt` 每 rollout 写一次）。

**只降级一次**：降完若又触底，只记日志然后退出，不允许把 16 小时的战役拖进重启循环。
`runs/_vram_guard_downgraded` 是那个一次性标记。日志 `runs/_logs/vram_guard.log`。

**四、丢掉的三个 rollout**

为统一账本口径，`runs/mu1_s0` 删掉从头跑（已完成的 3 行是旧口径，混在一起会让
`cost_cny_cum` 出现一段台阶）。这不损失任何科学信息：D-19 已证明同配置两次运行
0/512 行不同，重跑得到的是同一批 response，只是重花约 30 分钟 / ¥1。

本次停机总代价：3 个 rollout（05:53 → 06:13，约 ¥0.7）+ 重跑 30 分钟。

**五、未验证项**

本地 `conda ai` 环境的 `transformers` 坏了（`import transformers` 抛
`NameError: name 'nn' is not defined`，与本次改动无关），所以本地 CPU dryrun 跑不了，
新字段的落盘验证放在云端做。单元测试本地 25/25 通过（纯 numpy，不碰 transformers）。

**六、D-25 验证（真机，06:21:17Z 重启，commit `f04df9ff`）**

| 项 | 值 |
|---|---|
| `[r0] done` | `rollout 285.5s oldlogp 124.3s grad 367.9s update 492.2s` |
| `wall_ms_old_logp` / `wall_ms_update` | 124,275 / 492,213 ms（124.3 + 367.9 = 492.2 ✅） |
| `cost_cny_cum` / GPU-h | **0.4494** / 0.216（旧口径 0.3793 / 0.182，**+18.5%**） |
| `resp_len_mean` / `pass_rate` | **292.925 / 0.218018** |
| σ / d | 0.06626243 / 0.07232314 |
| 显存 | 40,568 used / **7,941 free** MiB |

关键交叉验证：`resp_len` 292.925 与 `pass_rate` 0.218018 与前一次运行（重启前删掉的那个
`mu1_s0` 的 r0）**完全一致**，σ 只差 5.6e-7（0.06626613 → 0.06626243）、d 差 2.2e-6。
即 D-19 的确定性在**真机上、跨进程、跨代码改动**复现了——删掉重跑确实没有损失信息，
只是重花 30 分钟。

ETA 用本次实测（每 rollout 285.5 + 492.2 + 落盘 ≈ 783 s）：单 run **1.75 h**，
九次 **15.75 h**，完赛 **北京次日 06:06**（比 D-24 的 05:22 晚，因为中间重跑了一次）。
`resp_len` 若继续涨，按每 +10% 加 1.4 h 记。

**七、两次自伤要记住**

1. `run_9.sh` 的 preflight 要求 `COMMIT` 是**非全零 40 位 hex**，我先写了短 hash `f04df9f`，
   被 preflight 正确拦下（`[error] COMMIT file is 'f04df9f' -- stamp it`），campaign 空转
   一轮。这个闸是对的，错的是我。
2. 这次会话 SSH 极不稳（`kex_exchange_identification: Connection reset` 与
   `Connection closed` 轮替，一分钟内多次），`scp` 会直接失败。改用
   `ssh ... "bash -s" < script.sh` 单连接送脚本后稳定；`setsid -w` + `ServerAliveInterval`
   是必需的。

---

### D-26 显存安全阀改为“停机取证、不自动降级”（2026-10-25）

用户指令原话：**“guard 触发降级后停下来判断”**。

**一、旧行为为什么必须改**

D-25 的 guard 在余量跌破 2 GiB 时做两件事：停战役，然后**自动用
`configs/cloud48g_lowmem.yaml`（util 0.38 / mns 192）重启 `run_9.sh`**，把剩下 8 个 run 跑完。
问题不在降级本身，在于**谁做这个决定**：一个脚本，在无人值守的时段，替人决定了“剩下八个 run
换一套配置跑完”。后果是同一个战役里 µ 臂不再同配置可比，而唯一的痕迹是
`runs/_logs/vram_guard.log` 里一行字。16 小时的产物如果中途换过仪器，事后无法证伪——
这正是本项目的核心指控对象。显存安全阀的职责是**保护已有数据**，不是挑下一个实验。

**二、新行为**

```
余量 < 2048 MiB 连续 2 次
  -> tools/guard_snapshot.py 落一份现场快照（runs/_vram_guard_halt.md + .json）
  -> 停 run_9.sh（先掐循环，防止 train.py 一死就进下一个 run）
  -> SIGTERM train.py，25 s 不退则 SIGKILL
  -> 停 EngineCore
  -> HALTED，退出。**不重启、不改配置、不推进任何 run**
```

快照内容（决策真正需要的）：free/used/total + 每进程显存、四个进程的 pid/etime/完整命令行、
每个 run 的 summary 有无 / metrics 行数 / rollout npz 数 / state.pt 大小与年龄 / 最后一行的
resp_len·pass_rate·zero_var·cost、当前 run 最近 4 个 rollout 轨迹与 **resp_len 斜率**、
当前 run 日志尾 12 行、磁盘余量、代码 commit。文件末尾列出三个续跑命令（A 原配置、
B `cloud48g_util042`、C `lowmem`），**只是文本，不执行**。

**三、新增两个文件，都只是给选项，不改现状**

- `tools/guard_snapshot.py`（只读，训练中可跑）。斜率算不出来时明说“只有 N 个 rollout，
  趋势还不可观测”，而不是含糊成“flat”——这条是干跑时发现的措辞 bug，已修。
- `configs/cloud48g_util042.yaml`：util 0.45→0.42（预留 21,317→19,895 MiB，返 1.4 GiB）、
  mns 256→224（KV 少 6.7%，上限不降会招 preempt，preempt 改批次形状正好撞 D-19）。
  预期 rollout 慢 5–7%，九次约 +0.5 h / +¥1。**不生效，除非有人在报告上选 B。**

**四、验证（全部在真机，06:42–06:44Z）**

| 检查 | 结果 |
|---|---|
| `bash -n tools/vram_guard.sh` | OK |
| 三配置解析 | main 0.45/256、util042 **0.42/224**、lowmem 0.38/192，micro=4 group=8 均不变 |
| `DRY_RUN=1 THRESH_MIB=99999` 干跑 | 快照正常生成到 `/tmp`，**没有杀任何进程**，战役继续 |
| 真实路径写入 | `runs/_vram_guard_halt.md` 3,373 B / `.json` 3,119 B 生成→核对→删除 |
| 三文件 md5 | 与本地**逐一相等**（`e8f60dda` / `5a01e99a` / `bfc48714`） |
| 换阀 | 旧 guard 49863 停，新 guard **55457** 起，日志三行 banner 含 `NO auto-relaunch (D-26)` |
| 训练未受影响 | 换阀前后 `mu1_s0` 同一 pid 50313 / EngineCore 50836 未重启，free 7,583 MiB |

**五、代价，写在明处**

触发后机器**空转且按小时计费**（¥1.56/h），直到有人看到 `runs/_vram_guard_halt.md` 并选一条
续跑命令。旧行为不会空转，但它会用一个未经批准的配置把剩下 8 个 run 跑完——那是更贵的错误。
若长时间无人，正确处置是关机，不是让它挂着。

**六、训练代码没动**

本次只改 guard（独立脚本）+ 新增工具/配置，`src/train.py` 与 `src/rollout.py` 一字未改，
所以云端 `COMMIT=f04df9ff697d0d4fb22d39a31f405a1def0789c1` 对 `run_meta.json` 仍然准确。

**七、拍板（2026-10-25 14:45 北京）**

用户选定**「只报告，等你定」**：guard 触发后落盘即止，**不按任何默认值续跑**。
另一个选项（“报告 + 自动按 A/B/C 里某个默认值续跑”）被明确排除。因此 `vram_guard.sh` 里
**没有** `AUTO_RESUME` 分支，这不是“暂时没实现”，而是拍板结果——将来若要加强自动续跑，
必须重新拍板，不能援引“当时没说不要”。

`32420f0` 已推 `origin/main`（push 从 WSL 执行，GitHub 密钥只在 WSL；Windows 侧仓库无密钥）。

---

### D-27 `clip_grad_mass_share` 全 0 是真 0：µ=1 臂结构上不可裁剪（2026-10-25）

用户要求当场核验 16:15 报告里的疑点（mu1_s0 八行 `clip_grad_mass_share` 全 0.0000）。

**结论：不是指标坏，是 µ=1 臂的结构性恒等。** 证据链三层：

1. **实现有护栏**（`tests/test_grpo.py:137`，25/25 里）：ratio=1 → 0.0；全体越界 → 1.0；
   padding 不带质量。恒 0 的坏表过不了这三条。
2. **真机数据**（mu1_s0 全 8 行）：`ratio_mean = 1 ± 9.6e-9`、`ratio_p95 = 1.00000000`、
   `kl_per_rollout_policy = 0.00e+00`、`clip_frac_any = 0`。ratio 贴着 1 到浮点噪声量级，
   离裁剪带（0.80 / 1.28）差 8 个数量级。
3. **结构性原因**（`src/train.py` update 循环）：µ=1 每 rollout 只有一次 `opt.step()`，
   且全部 micro-batch 前向都发生在 step 之前 → loss 前向权重 ≡ 算 `old_logp` 的权重 →
   `ratio = exp(logp − old_logp) ≡ 1`（仅浮点噪声）。KL 同理恒 0。

**对解读的影响**：
- µ=1 臂（run1–3, 6–9）里 PPO 裁剪与 KL 正则是**空操作**，唯一约束步长的是
  `grad_clip`（norm 剪裁）与 lr——这不是 bug，是该臂定义（每 rollout 一步）的直接推论。
- `clip_grad_mass_share` / `clip_frac_*` / `kl_per_rollout_policy` 只在 **µ=8 臂**（run4–5）
  有信息量：一个 rollout 内第 2 步起的权重已更新、与 `old_logp`（θ₀）不同，ratio 才会离开 1。
  预期这三列在 µ=8 行非零；若届时仍全 0 才需要回头怀疑实现。
- 对 drift/stability 时钟无影响：σ 与 d 只依赖梯度统计，不经过 ratio。

**验证时间**：2026-10-25 08:20Z（真机 mu1_s0 metrics 全列）。

### D-28 报告成文：写的时候又抓出四个口径问题（2026-10-26）

`REPORT.md` 从"待数据"模板改成正式实验报告（约 480 行），原模板里仍然有效的两张表
（原文索引、与原文的差异）保留为 §2 与附录 A。同时新增两个分析脚本：

- `analysis/report_stats.py` —— 报告引用的派生量（按臂聚合、d0 拆解、C2 单调性诊断、
  C6 两种份额、d 的穿越窗口）。**故意不 import `make_table`**，这样两者一致是检验而不是
  同义反复；输出 `analysis/out/report_stats.json`。
- `analysis/plot_report.py` —— 四张 PNG。与 `make_figures.py` 的分工：后者不依赖
  matplotlib（无图也能读结果），前者在 matplotlib 可用时出图，缺失就打印一行并退出 0。

**写作过程中发现的四个口径问题（都不改数据，只改措辞或加限制）**

1. **`claims.yaml` 的 C3 把比值方向写反了。** C3 写 `N_τ(µ=16)/N_τ(µ=1) = 6.1×`，
   但 6.1× 来自 `B* ∝ µ` 外推：`N_τ/N_min` = 9.2 / 2.0 / 1.5（µ=1/8/16），所以
   6.1× 对应的是 `N_τ(µ=1)/N_τ(µ=16)`。方向相反。**没有改预注册**——修订要加
   revision 行并留痕，属研究诚信决定；在 REPORT §1.1 加了脚注，并列入 §8.4 待拍板。
   对判定无影响：两端都右删失，任何方向的比值都不存在。

2. **C5 的"基本平"不准确。** 之前只看了全程均值（298.3/288.3/285.4）。逐 rollout 看：
   µ1 前半 295.1 → 后半 296.4（+1.3），µ8 292.5 → 277.2（−15.3），µ16 290.3 → 272.2
   （−18.1，−6.2%）。判据仍通过（预注册只禁止上涨），但"µ 越大训练后期回答越短"是个
   未预注册的方向，报告里如实写出并明确不据此解释。

3. **"同预算 pass 排序"不能读成"µ 提高样本效率"。** 这是最容易误导人的一条。
   每 token 只被 update 一次，µ 只决定"切成几次更新"，所以三臂的总 responses 相同、
   总 FLOPs 也几乎相同（1.84 / 1.78 / 1.76e17，4% 的差来自生成长度）。差异来自
   优化步数 8 / 64 / 128。真正的样本效率要看"达到同一 pass 各需多少 responses"，
   而三臂在 8 个 rollout 内都还在上升、没有交点。报告 §6 把这条写成探索性结果，
   并把这层解释放在显眼位置——否则它会被读成对 Eq.(6) 的支持。

4. **σ@r0 有两个口径，报告换成了自洽的那个。** 验收文档用的是**全 run 均值**
   （σ 0.0659 / 0.0762 / 0.0805，涨 1.22×）；报告改用 **rollout 0 的均值**
   （0.06596 / 0.06652 / 0.06733，涨 1.02×）。理由：`d0` 本身是 rollout 0 的 d 均值，
   而 d 逐行等于 `‖ḡ‖/σ`，拆解必须与 d0 同源，否则乘积对不上。换成 r0 口径后
   C4 的机制叙述更干净：**σ 几乎不动（1.02×），d0 的 2.63× 全部由 ‖ḡ‖ 的 2.69× 解释**，
   而这与 `micro_group_responses=8` 固定 σ 估计窗口的实现直接相关。旧口径不算错，
   但它混入了训练过程中的变化，不能用来解释"初始"的 d。

**没动的东西**：`claims.yaml` 的任何 status；`notes/2026-10-24_cloud24g/` 与
`notes/2026-10-26_pull/` 里归档的旧口径证据（它们记录的是当时的实况）。

**校验**：`trace.py --strict` 11 条全 `[ok]`（pending 7 / verified 1 / default 3）；
`tests/run_all.py` 25/25 通过；报告引用的 12 个文件路径全部存在。
