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

`gen_counting.py` / `verify.py` 原样拷入 `src/`，题池新建
（`data/prompts_mixed2000.jsonl`，seed=20261024，
sha256 `670b2854…b437398`）。

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
