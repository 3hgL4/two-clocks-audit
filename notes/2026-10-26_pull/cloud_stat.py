#!/usr/bin/env python3
"""在云端直接读原始 metrics.jsonl 复算，不经任何本地看板/汇总脚本。

口径与 analysis/make_table.py（修复后）一致：
  - "后半程" = 按 rollout_idx 排序后的末 4 个 rollout（每 run 共 8 个）
  - 每个 rollout 的 pass = 该 rollout 内所有 mu_step 行的均值（与脚本 rollout_means 对齐）
同时输出另一口径（末 4 个 rollout 的全部行池化）供对照。
"""
import json
import glob
import os
import statistics as st

RUNS = "/root/rivermind-data/two-clocks-audit/runs"
TAGS = ["mu1_s0", "mu1_s1", "mu1_s2", "mu8_s0", "mu8_s1", "mu8_s2",
        "mu16_s0", "mu16_s1", "mu16_s2"]

out = {"runs": [], "env": {"host": os.uname().nodename, "cwd": RUNS}}
for tag in TAGS:
    p = os.path.join(RUNS, tag, "metrics.jsonl")
    rows = [json.loads(l) for l in open(p) if l.strip()]
    mu = int(rows[-1]["mu"])
    seed = int(rows[-1]["seed"])
    by_r = {}
    for r in rows:
        by_r.setdefault(int(r["rollout_idx"]), []).append(r)
    idxs = sorted(by_r)
    per_roll = {i: [x["pass_rate"] for x in by_r[i]] for i in idxs}
    roll_mean = {i: sum(v) / len(v) for i, v in per_roll.items()}
    half = max(1, len(idxs) // 2)
    keep = idxs[-half:]
    pooled = [v for i in keep for v in per_roll[i]]
    last4 = idxs[-4:]
    half_vals = [roll_mean[i] for i in keep]
    d = {
        "tag": tag, "mu": mu, "seed": seed,
        "rows": len(rows), "rollouts": len(idxs),
        "rows_expected": 8 * mu,
        "pass_half_mean": round(sum(half_vals) / len(half_vals), 6),
        "pass_last4_mean": round(sum(roll_mean[i] for i in last4) / len(last4), 6),
        "pass_pooled_half": round(sum(pooled) / len(pooled), 6),
        "pass_all": round(sum(r["pass_rate"] for r in rows) / len(rows), 6),
        "pass_final_rollout": round(roll_mean[idxs[-1]], 6),
        "keep_rollouts": keep,
        "resp_len_first": round(rows[0]["resp_len_mean"], 2),
        "resp_len_last": round(rows[-1]["resp_len_mean"], 2),
        "kl_first": float(rows[0]["kl_per_rollout_policy"]),
        "kl_last": float(rows[-1]["kl_per_rollout_policy"]),
        "d_first": round(rows[0]["drift_d"], 6),
        "d_last": round(rows[-1]["drift_d"], 6),
        "responses_cum": rows[-1]["responses_cum"],
        "flops_cum": float(rows[-1]["flops_cum"]),
        "cost_cny_cum": round(rows[-1]["cost_cny_cum"], 4),
        "ratio_dev_max": round(max(abs(r["ratio_mean"] - 1.0) for r in rows), 12),
    }
    out["runs"].append(d)

# 配对差（用 half 口径）
by = {(r["mu"], r["seed"]): r["pass_half_mean"] for r in out["runs"]}
for m in (8, 16):
    deltas = []
    for s in (0, 1, 2):
        if (1, s) in by and (m, s) in by:
            deltas.append(round(by[(m, s)] - by[(1, s)], 6))
    out[f"paired_mu{m}_minus_mu1"] = deltas

print(json.dumps(out, indent=1, ensure_ascii=False))
