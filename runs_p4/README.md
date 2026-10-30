# runs_p4/ — P4 成本探针证据（2026-10-30，云端拉回）

来源实例：gpuhome `jupyter-cl2zfu43xyr9vkps`（4090-48G，`sh01-ssh.gpuhome.cc:30669`），
`/root/rivermind-data/`。实例可随时释放，本目录是这些数据的唯一持久副本。

## 内容

- `p4_gate_verdict.json` — 门判最终版（带 provenance，gate_commit `9fbaf34`，
  含两份 metrics.jsonl 的 sha256 与 utc/host）
- `p4_gate_verdict.chain_consumed.json` — 自动链实际消费的旧版 verdict
  （无 provenance 字段，verdict/数值与最终版逐位一致，仅留档溯源）
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
# 期望：PASS，p4_mu32 [967.9, 929.9]，p4_mu64 [903.9, 915.7]
```

拉回当日已实测复算一致。metrics sha256（与 verdict 内一致）：
- p4_mu32: `29f5e20725c84b242e11b0739c876f96c2bae37a94a9b4615d9fd766e1bf5291`
- p4_mu64: `7815bb15c359bc724d57fad3a686e554c9b90bb2bdd9b37fff5b87567309769e`

## 判读口径（勿越界）

两臂均值差 4.1%（948.9 vs 909.8 s）：update 段几乎相同（485.1 vs 485.0 s）、
old_logp 相同（121.8 vs 122.2 s），差全落在 rollout 采样段（317.8 vs 304.5 s，
cycle 1）。结论只能写"周期成本与 µ 无关（差在采样噪声内）"——这是 P4
公平性（F2）成立的证据，**不是**"µ 大更省"的效率结论。

8 周期 × 890 s ≈ 2 GPU-h 是"计划 4096 responses 满跑"的**上界**：
SAC 臂早停时单周期成本 ≈ 890 s × executed/µ_plan，直接乘 8 会高估。
