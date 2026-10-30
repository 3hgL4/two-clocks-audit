# runs_p4/ — P4 成本探针证据（2026-10-30，云端拉回）

来源实例：gpuhome `jupyter-cl2zfu43xyr9vkps`（4090-48G，`sh01-ssh.gpuhome.cc:30669`），
`/root/rivermind-data/`。实例可随时释放，本目录是这些数据的唯一持久副本。

## 内容

- `p4_gate_verdict.json` — 门判最终版（**两分量口径 rollout+update**，gate_commit
  `b666c26`，含两份 metrics.jsonl 的 sha256 与 utc/host）
- `p4_gate_verdict.threecomponent_9fbaf34.json` — 历史留档：三分量旧口径
  （948.9/909.8），`11ff431` 把 old_logp 错当独立分量叠加（实际 update 已含
  old_logp），每周期虚增 ~121 s；被 `b666c26` 回退（v2 Revision A-4）
- `p4_gate_verdict.chain_consumed.json` — 自动链实际消费的最初版 verdict
  （无 provenance 字段；三分量数值，判为 PASS 的结论本身没错——两个口径都 PASS）
- `p4_gate.log` / `chain.log` — 门判输出与链日志（chain start 2026-10-30T07:42:22Z，gate_rc=0）
- `p4_mu32/`、`p4_mu64/` — 各含 config.resolved.yaml / metrics.jsonl /
  prompts.jsonl / run_meta.json / samples.jsonl / summary.json

## 刻意未入库（体积原因，留云端正本）

每个 run 的 `state.pt`（837 MB）、`adapters/`、`rollouts/` 未拉回。
若实例释放，这些不可再生——但在 P4 判据范围内不需要：门判只依赖
metrics.jsonl（sha256 已锚定在 verdict 里）。

## 复算

```
python tools/p4_gate.py runs_p4/p4_mu32 runs_p4/p4_mu64
# 期望：PASS，p4_mu32 [846.4, 808.1] 均值 827.3；
#       p4_mu64 [782.4, 793.5] 均值 788.0（两分量口径，2026-10-30 实测一致）
```

⚠ 口径：周期墙钟 = `wall_ms_rollout + wall_ms_update`。**`wall_ms_old_logp`
是 update 的子集（train.py 记账 update = old_logp + grad），不可叠加**——
叠了就是三分量旧口径（948.9/909.8），已作废（v2 Revision A-4）。
metrics sha256（两口径一致）：
- p4_mu32: `29f5e20725c84b242e11b0739c876f96c2bae37a94a9b4615d9fd766e1bf5291`
- p4_mu64: `7815bb15c359bc724d57fad3a686e554c9b90bb2bdd9b37fff5b87567309769e`

## 判读口径（勿越界）

两臂均值差 5.0%（827.3 vs 788.0 s）：update 段两臂几乎相同（恒 ~485 s，
这是 F2"每周期恒 4096 responses"成立的证据），差全落在 rollout 采样段
（361.5 vs ~297 s，采样噪声内）。结论只能写"周期成本与 µ 无关（差在采样
噪声内）"，**不得写"µ 增大降低成本/µ 大更省"**。

带值注记（A-4）：890 s ±25% 冻结带不动；带中心比实测节拍（~771–827 s）
高约 15%，实际判别上界约为 +44% 而非 ±25%。

## 成本口径

890 s（及实测 ~827 s）是"计划 4096 responses 满跑"的单周期成本。SAC 臂
早停时单周期 ≈ 该值 × executed/µ_plan；8 周期 ≈ 2 GPU-h 是**上界**，不是
预期值，报告直接乘 8 会高估。另：metrics 的 `cost_cny_cum` 只计
rollout+update，不含 vllm 加载/checkpoint/adapters 快照（sac_s1 实测
elapsed 2604 s vs 计费 2247 s，差 ~14%）——报告引用成本必须标口径。
