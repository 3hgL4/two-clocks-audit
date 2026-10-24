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

