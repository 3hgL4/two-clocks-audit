> **入库副本说明**：本文件是 `xiangmu/SAC-mu/云端执行卡.md`（工作副本，单一编辑入口）的版本控制快照，每次改动工作副本后同步到此并 commit。判据冻结稿在 `jd-lab/discussions/2026-10-30_SAC-mu_实验方案v2_冻结稿.md`。

# SAC-µ 云端执行卡（租卡后照此执行）

> **依据**：`jd-lab/discussions/2026-10-30_SAC-mu_实验方案v2_冻结稿.md`（判据已冻结，改判据必须留 revision 行）。
> **代码**：`two-clocks-audit` 仓，分支 **`sac-analysis`**（含 sac-mu 全部历史）。执行侧代码 = `31682a6`（控制器 + 修复 `9211933`/`89c30e9`，59/59 测试，云端在跑的就是它 + `11ff431` 门脚本散件）；判读侧：`e149d4c`（审计方 `analysis/sac_arms.py` 守卫）、`9fbaf34`（门脚本 provenance）、`f730ec3`（P4 证据入库 `runs_p4/`）、`5961290`（链脚本入库+启动日志）。**主网格开跑前的全量重上传改对准 sac-analysis 当时 HEAD**（§1）。
> **平台参考**：`gpt2/TwoClocks云端操作卡.md`（三层落地、--resume、关机检查清单全部沿用）。定价 4090-48G ¥2.08/h；预算新花 ≈31 GPU-h ≈ ¥65，上限 ¥104。
>
> **环境一致性（2026-10-30 两次收紧，A-2 → A-3）**：本实例与原 campaign 的环境比对以 `runs/mu16_s0/run_meta.json` + `notes/2026-10-26_pull/cloud_only/env.txt` 为准——python 3.11.13 / torch 2.13.0+cu130 / transformers 5.19.0 / vllm 0.32.0 / numpy 2.3.2 / driver 580.119.02 / cuda 13.0 七项逐位一致；**原唯一未记录项 peft 已于 10-30 闭合**：campaign 原实例无卡启动抓取 pip freeze（入库 `notes/2026-10-26_pull/cloud_only/pip_freeze_2026-10-30.txt`），peft==0.21.2 / accelerate==1.15.0 与本实例 run_meta 逐位一致（v2 Revision A-3）。**µ16 历史基线（F4=0.517）复用成立，八项全对齐，不需复跑；`31682a6` 起 run_meta 记录 peft/accelerate，此后跨环境比对只看 run_meta。**

---

## 0 租卡前自查（本卡完成即勾掉）

- [x] 判据 P1–P5' 冻结（v2 §4），revision 行 R-1/R-2/R-3 已留
- [x] 控制器实现 + 59/59 测试（Windows conda ai 实跑）
- [x] **12 个** run 配置文件（sac×3 / sacS×3 / sacR×3 / p4×2 / dryrun×1；"15" 指主网格 run 总数 = SAC/S/R/µ32/µ64 各×3，此前括号计数笔误已修）
- [x] off-by-one 语义钉死（v2 §1.2）；阶梯替代连续 AIMD（Br=4096 整除约束）
- [x] 消融 R 危险率表冻结（HAZARD_M16，24 周期首触分布）
- [x] SAC 臂 LR 视界读码核实（2026-10-30 审计项 2）：`nominal_mu=max(LADDER)=32`，`lr_base=r_idx×32`，`total_steps=R×32=256`，每周期 LR 索引恒进 µ_cap、与 mu_executed 无关——与 v2 §1.5 冻结口径逐位一致（train.py L436-465, L548-553, L731）

## 1 上传（GitHub 云端不通，走 scp）

```bash
# 本地（Git Bash，在 two-clocks-audit 目录）
git archive --format=tar.gz -o /tmp/sac-mu.tgz sac-mu
scp -P <port> /tmp/sac-mu.tgz root@<host>:/root/rivermind-data/
```

云端：**必须解压到 `/root/rivermind-data/two-clocks-audit`**（configs 里的模型绝对路径依赖这个位置）。

```bash
cd /root/rivermind-data && tar xzf sac-mu.tgz -C two-clocks-audit
# ⚠ git archive 的包没有顶层前缀，不要加 --strip-components=1（2026-10-30 踩过：
#   加了会把 tools/、src/ 一级目录剥平，verify_pool 直接找不到文件）。
# 覆盖旧文件前确认：这是全新实例则直接解；若复用旧盘先备份 runs/
```

**主网格开跑前必须全量重上传一次**（审计纪律：云端 = 本地 HEAD 逐位一致，不接受散件补丁叠 Archive）：

```bash
git archive --format=tar.gz -o /tmp/sac-full.tgz HEAD   # 本地，对准当时 HEAD
scp -P <port> /tmp/sac-full.tgz root@<host>:/root/rivermind-data/
# 云端解包覆盖 + 重跑 verify_pool + tests/run_all.py（59/59）
```

## 2 环境与自检（复用上轮工具链）

```bash
cd /root/rivermind-data/two-clocks-audit
/opt/conda/bin/python tools/verify_pool.py          # 第一条：题池闸门，必须 result: consistent
bash tools/cloud_bootstrap.sh                        # 依赖 + 魔搭拉权重 + 回归
/opt/conda/bin/python tests/run_all.py               # **期望 59/59**（37 旧 + 22 新；少于 59 = 代码不对版）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 每个训练会话都要
```

## 3 SAC dry-run（CPU 级，~1 min，0 额外成本）

```bash
/opt/conda/bin/python src/train.py --config configs/sac_dryrun.yaml --device cpu
# 第二步（必做，非可选）：U=1.05 定向触发变体——sac_dryrun 默认 U=1.0738 下
# toy 数据 max_p95≈1.05 永远不触发，验收必须人为压阈值把触发分支跑出来。
# 2026-10-30 的 break 级 bug（触发后周期不停、慢环永远看不到触发）就是
# 靠这一步暴露的；59/59 单测拦不住它（单测不执行 main() 的 sac 分支）。
sed -e "s/tag: sac_dryrun/tag: sac_dryrun2/" -e "s/U: 1.0738/U: 1.05/" \
    configs/sac_dryrun.yaml > configs/sac_dryrun2.yaml
/opt/conda/bin/python src/train.py --config configs/sac_dryrun2.yaml --device cpu
rm -rf runs/sac_dryrun runs/sac_dryrun2   # 验完即删或移 runs_quarantine/，不得混入正式 run
```

**验收清单（两个变体合起来看）**：
1. 变体 1（U=1.0738，burn_in=1）：burn-in 周期无 `[SAC TRIGGER]`；`state.pt` 可加载（含 sac 状态）；burn-in 越阈不触发（门生效）。
2. 变体 2（U=1.05）：出现触发行；**触发后周期立即结束**（该周期 rows = 触发 step+1，`mu_executed` = 触发 step）；**下一周期 plan 降一档**（慢环看到触发）；history 末条 `(cycle, plan, executed, True)`。
3. dry-run 的 `burn_in_cycles: 1` 是有意的快验配置（主配置为 2）——dry-run 不完全复刻主路径，此差异已知、可接受。

**失败则停，贴输出回来。**

## 4 P4 成本门（~1 GPU-h）

```bash
/opt/conda/bin/python src/train.py --config configs/p4_mu32.yaml
/opt/conda/bin/python src/train.py --config configs/p4_mu64.yaml
```

判据（v2 §5，预注册）：每周期总耗时（rollout+old_logp+update，从 metrics.jsonl 逐周期求和）在 **890 s ± 25%**（即 668–1113 s）内。
- **判读用脚本（2026-10-30 两次加固，断会话不影响）**：`python tools/p4_gate.py runs/p4_mu32 runs/p4_mu64 --json /root/rivermind-data/p4_gate_verdict.json`（stdlib，exit 0=PASS / 3=FAIL）。**周期墙钟 = rollout + old_logp + update 三分量之和**（metrics.jsonl 的 `wall_ms_rollout` + `wall_ms_old_logp` + `wall_ms_update`；⚠ `done:` 日志行的分段和含重叠，不得用）。校准（commit `11ff431` 修掉漏加 old_logp 的 bug 后）：历史 mu16_s0/s1/s2 = **840–939 s，均值 ~888–892**，与冻结带中心 890 s 吻合。⚠ 初版校准（727–813 s"偏保守"）是漏项下的低估值，已作废。
- **自动链**：`/root/rivermind-data/p4_then_smoke.sh`（nohup）等两个 P4 run 完成 → 跑门脚本 → PASS 自动启动 §5 冒烟（sac_s0）；FAIL 或 Traceback 则停链等人工。chain 日志：`/root/rivermind-data/chain.log`。
- **过** → 固定 µ32/64 臂按 R=8 跑（§6 原表）；
- **不过** → µ32 臂按等壁钟折算周期数并披露采样不对齐；**µ64 臂砍掉**；
- 每个 run 跑完立刻回传（§8），别攒。
- **P4 实测结果（2026-10-30）**：mu32 [967.9, 929.9] 均值 948.9；mu64 [903.9, 915.7] 均值 909.8；带 [667.5, 1112.5] → **PASS**。证据（verdict 带 provenance + 两 run 文件）已入库 `runs_p4/`（commit `f730ec3`）。
- **判读措辞禁令（审计 2026-10-30）**：两臂差 4.1%，分解后 update 段几乎相同（485.1 vs 485.0 s）、old_logp 相同（121.8 vs 122.2 s），差全落在 rollout 采样段（317.8 vs 304.5 s，cycle 1）。**只能写"周期成本与 µ 无关（差在采样噪声内）"，不得写"µ 增大降低成本/µ 大更省"。** 这个差是 P4 公平性（F2）成立的证据，不是效率结论。
- **成本预估口径（审计 2026-10-30）**：890 s 是"计划 4096 responses 满跑"的单周期成本。SAC 臂早停时实际处理 b × executed < 4096，单周期成本 ≈ 890 s × executed/µ_plan；**8 周期 × 890 s ≈ 2 GPU-h 是上界，不是预期值**，报告直接乘 8 会高估。

## 5 冒烟行为门（sac_s0，~2 GPU-h）

```bash
/opt/conda/bin/python src/train.py --config configs/sac_s0.yaml
```

判据（v2 §6，四条全过才继续）：
1. burn-in 两周期 µ=16 无触发；
2. 周期 2–7 触发率 20%–80%；
3. 阶梯轨迹在 [4,32]；
4. 无 NaN、checkpoint 正常、metrics 行数与 mu_executed 自洽。

快速验收命令（云端或拉回本地跑）：

```bash
/opt/conda/bin/python - <<'EOF'
import json, collections
rows=[json.loads(l) for l in open("runs/sac_s0/metrics.jsonl")]
by=collections.defaultdict(list)
for r in rows: by[r["rollout_idx"]].append(r)
for k in sorted(by):
    v=sorted(by[k], key=lambda r:r["mu_step"])
    print(f"r{k}: plan={v[0]['mu_plan']} executed={v[0]['mu_executed']} "
          f"trigger={any(r['sac_trigger'] for r in v)} rows={len(v)}")
EOF
```

**门失败 → 停，回校准卡；不许改 U 硬跑**（红线 3）。

## 6 主实验网格（门过后，~29 GPU-h）

顺序（先 SAC 补齐，再消融，再固定臂——任何一步失败都不阻塞已完成的）：

```bash
for t in sac_s1 sac_s2 sacS_s0 sacS_s1 sacS_s2 sacR_s0 sacR_s1 sacR_s2; do
  /opt/conda/bin/python src/train.py --config configs/$t.yaml
done
for s in 0 1 2; do
  /opt/conda/bin/python src/train.py --config configs/cloud48g_main.yaml --mu 32 --seed $s --tag mu32_s$s
  /opt/conda/bin/python src/train.py --config configs/cloud48g_main.yaml --mu 64 --seed $s --tag mu64_s$s
done
```

- 用 `tmux`/`screen` 挂会话（WSL 惯例：留持久客户端）；中断用 `--resume` 续；
- **磁盘预算**：每 run 产物 ≈2.5 GB（adapter 2.1G + rollouts npz + metrics）；49G 数据盘装不下 15 个 run 的全量——**每个 run 回传校验通过后立刻 `rm -rf runs/<tag>/adapters`**（原件在魔搭私有仓）；
- `save_adapter_every_rollout` 保持与原 campaign 一致（provenance 纪律），靠"跑完即清"控盘。

## 7 回传与校验（每 run 收尾立刻做）

```bash
modelscope upload <user>/sac-mu-logs ./runs/<tag> --repo-type dataset --private
# 或 rclone；校验（§11.5 惯例）：
find runs/<tag> -type f -exec sha256sum {} \; >> MANIFEST.sha256
```

全部结束后拉回本地归档 + `sha256sum -c`。

## 8 成本与止损

| 阶段 | 累计 GPU-h | 判据 |
|---|---|---|
| 环境+dry-run | ~0.3（装机不计卡时则 0） | 59/59 |
| P4 | ~1.3 | ±25% 门 |
| sac_s0 | ~3.3 | 行为门四条 |
| 主网格 | ~32 | P1/P2/P3（判读归用户） |

止损点：P4 失败 → 只跑 µ32 折算臂 + SAC/消融，省 ~6h；行为门两次失败 → 全停（已花 ≤4h）。

## 9 跑完之后（不在本卡范围）

判读：`训练结果分析决策模板` 流程，AI 只供数与框架；分析脚本（配对 bootstrap、P3 判定、P5' 诊断表）在数据齐后写 `analysis/sac_*`，以 v2 §4 的冻结口径为准。

## 10 卫生纪律（2026-10-30 审计后固化）

1. **runs/ 残留**：非冻结参数或带 bug 代码产生的目录一律移 `/root/rivermind-data/runs_quarantine/`（附 README 说明），归档时不得混入正式 run。
2. **云端一致性**：云端代码 = 本地 HEAD 逐位（全量 git archive），散件 scp 补丁只允许存在于两次全量上传之间，且必须记 commit 号。
3. **门判定离线化**：凡是"跑完等人判"的门，判读脚本必须提前部署（stdlib、可离线跑），断会话不产生 GPU 空转窗口。
4. **run_meta 自文档**：`31682a6` 起 run_meta.json 记录 peft/accelerate——跨环境比对以后只看 run_meta，不靠回忆。
5. **链脚本（2026-10-30 二次记录）**：① ~~不在仓内~~ **已修**——入库 `tools/p4_then_smoke.sh`（commit `5961290`），并新增 smoke 启动日志行（`smoke start <utc> pid=… cmd=…`）。**本次部署的链仍是旧版**：sac_s0（PID 5535）确由链启动，但 chain.log 只有 `chain start` + `gate_rc=0` 两行，启动事实仅当时 `ps` 可证、进程退出后无实据——已作遗留注记，下次部署换用入库版。**严禁在链等待期间原地覆盖云端脚本**（bash 增量读脚本，原地重写可致执行错乱）。② 等待循环无超时——mu64 若以非 Traceback 方式死掉（如 OOM-kill）链会无限空等，加 `--timeout` 兜底仍是待办。
6. **门口径防误读**：周期墙钟只用 metrics 三分量；`done:` 行的分段打印（rollout/oldlogp/grad/update）含重叠记账，加总 ≠ 周期耗时，不得进门判定。
7. **术语表（防 10-27 "把行数当 rollout 数" 重犯）**：
   - `num_rollouts` = **周期数**（P4 探针=2、SAC 臂=8、历史 campaign=8），不是采样 batch 数；
   - `rollout_idx` = 第几个采样 batch，即**周期索引**（0 起）；
   - 每个周期有 **µ 行** metrics（每行 = 一次更新步，`mu_step` 0..µ-1）；行数 ≠ 周期数，`half = len(rows)//2` 一类按行数切周期是错的（10-27 实际踩过，修于 ce664a2）。
8. **判读侧交接约束（审计 2026-10-30）**：SAC 臂 metrics 在 burn-in 期与 `mu16_s0` 字段完全相同（µ16/b256/Br4096）且**无 tag 字段**——所以：① **禁止用任何按 (µ, b) 建键的脚本分析 SAC 臂**（`bmu_2x2.load_cells` 就是这种设计，目录名白名单救不了它，因为键来自 metrics 内部字段）；② 拉回的 SAC 数据一律放 **`runs_sac/`**，与 `runs/`、`runs_p4/`、`runs_ab2x2/` 物理隔离；③ 交叉分析前必须调用 `analysis/sac_arms.py`（分支 `sac-analysis`，`e149d4c`）的 `assert_no_collision(paths)`。
9. **P4 证据入库**：verdict（带 provenance：utc/host/gate_commit/输入 sha256/band_source）+ 两 run 小文件在仓 `runs_p4/`（`f730ec3`）；`state.pt`/`adapters/`/`rollouts/` 留云端正库未入库（体积），实例释放即失，但门判只依赖 metrics（sha256 已锚定）。
