# 2026-10-26 关机前的完整拉取

实例将在关机后被回收、7 天后释放数据盘。本地此前的 `runs/` 只有 75 MB
（每条 run 的 metrics / samples / rollouts / summary），**72 个 LoRA adapter
与 9 个 optimizer state 只在云端**。这个目录记录这次把它们全部搬回本地之后
的状态，以及"本地看到的数据与云端是否一致"的核对结果。

## 一、云端资产在哪

```
/root/rivermind-data/two-clocks-audit      <- 这次的 48G 实例（容器启动 2026-10-25T01:45Z）
```

注意与 10-24 那次 24G 实例不同：那台的产物在 `/root/rivermind-data/runs/`，
没有 `two-clocks-audit/` 这一层。这次的 `runs/` 在项目目录里面。

云端 `COMMIT` 文件记的是 `f04df9f`，比本地当时的 HEAD（`ce664a2`）落后 5 个提交。
逐文件比对确认**云端代码没有本地缺失的内容**：`src/`、`configs/`、`tests/`、
`scripts/`、`tools/vram_guard.sh` 全部一致；`DEVLOG.md` 云端 497 行 vs 本地 1031 行、
`REPORT.md` 云端 143 行 vs 本地 156 行，差异方向均为"云端是旧版"（云端 DEVLOG
没有一行是本地缺的；REPORT 的三行旧口径 —— 24G / ≤4h / ¥22–54 —— 本地已改写成
48G / ≈1.6h / ¥30–34）。

## 二、拉了什么

| 类别 | 文件数 | 体积 | 位置 |
|---|---|---|---|
| LoRA adapters（每 rollout 一份，fp32 safetensors） | 72 | **19 GB** | `runs/*/adapters/r000..r007/` |
| optimizer / RNG state | 9 | **7.2 GB** | `runs/*/state.pt` |
| rollout token id（npz） + 小文件 | 144 | 68 MB | `runs/*/` |
| 训练 stdout 日志 | 10 | 200 KB | `runs/_logs/` |
| 工作 adapter（`_adapter`） | — | 267 MB | `runs/_adapter/` |
| 主控日志 `campaign.log` 等 /tmp 产物 | 9 | 228 KB | `cloud_tmp/` |
| 数据盘顶层的探针 log / vram csv | 24 个 log + 7 个 csv | 2.8 MB | `cloud_tmp/` |
| 云端独有、仓库里从来没有的脚本 | 4 | 8 KB | `cloud_only/` + `tools/run_with_chunk.py` |

**没拉的**：`models/` 3.3 GB 权重（`tools/fetch_model.py` 13 分钟可从魔搭重下）、
`~/.cache/pip` 3.8 GB、以及 6 个吞吐/确定性探针 run 的 `state.pt`
（每个 799 MB，那些 run 只测显存与速度，没有可复现的优化轨迹）。

## 三、怎么保证"拉全了"

云端现场生成三份清单，拉回来后本地逐文件核对：

- `MANIFEST.nostate.sha256` — 432 条（9 个 run 的全部非 state 文件）
- `MANIFEST.state.sha256` — 9 条（`state.pt`）
- `MANIFEST.sizes.tsv` — 441 条（字节数，用于先按大小快速筛掉传输截断）

核对结果在 `verify_result.txt`（生成方式与命令都在文件头）：
**441/441 一致（432 个非 state 文件 + 9 个 `state.pt`），0 mismatch，0 missing，
每个 run 49/49。** 判定标准不是"看着一样"，是逐条 sha256 相符；任何一条不符即
视为失败（`tools/verify_archive.py` 的退出码非 0）。

"没漏东西"另有一条更强的证据：云端 `runs/` 目录下**全部 599 个非 state 文件**
（含 6 个吞吐/确定性探针 run、`_logs`、`_adapter`）都生成了字节数清单
（`MANIFEST.all.sizes.tsv`）与 sha256 清单（`MANIFEST.misc.sha256`，167 条），
本地逐条比对：**599/599 存在且大小一致，167/167 sha256 相符**。
加上 9 个 `state.pt`，共 **608 个文件全部经 sha256 校验通过，无一条缺失或改动**。

传输方式：`tar` 流经 ssh 直通（`ssh 'tar cf - dir' | tar xf -`），逐 run 分 9 次，
每次失败自动重试 3 次。实测吞吐 9.4–12.3 MB/s（约 90 Mbps），
单 run 2.1 GB 用 170–230 s。**未使用压缩**：safetensors 是已压缩的浮点数据，
`-C` 只会吃 CPU。

## 四、云端复算 vs 本地看板（这是本次核对的重点）

### 结论 1：数据本身没有偏差

**432 个文件（除 state.pt 外全部）与云端 sha256 逐条比对，432 条一致、0 条不符。**
构成：72 个 `adapter_model.safetensors` + 216 个 adapter 附属文件
（每个 adapter 的 `adapter_config.json` / `README.md` / `AUDIT_ADAPTER_META.json`）
+ 144 个 run 小文件（72 个 npz、9 个 metrics、9 个 samples、9 个 summary、
9 个 run_meta、9 个 config、18 个 index、9 个 audit）。
所以看板读到的数字就是云端的数字，不存在"本地留了一份旧数据"这种情况。

### 结论 2：但"算"错过两次，都在 µ>1 的臂上

1. **`analysis/make_table.py` 的"后半程"口径**（已在 `ce664a2` 修）。
   `half = max(1, len(rows)//2)` 里的 `len(rows)` 是 metrics **行数**而不是 rollout 数：
   µ=1 时 8 行 → 取后 4 个 rollout（碰巧正确）；µ=8 时 64 行 → `keep_from` 变成 −24，
   8 个 rollout 全部进入，于是"后半程"实际是"全程"。后果是 µ8/µ16 的 pass 被
   低估约一半，配对差也腰斩。
2. **`tools/dashboard.html` 的 µ-step 显示**（已在 `ffd2778` 修）。
   feed 的 `KEEP` 白名单不含 `mu_step`，表格里每一行都显示 0。
   修法是用"组内落盘顺序"当序号（同 rollout 的行按 ts 严格递增追加），
   C1 自检反证了这不是猜测：72/72 个 rollout 的位置 0 行都是 ratio=1.000、clip=0。

### 结论 3：云端直接复算的数字

`cloud_stat.py` 在云端用 `/opt/conda/bin/python` 直接读 `/root/rivermind-data/.../metrics.jsonl`，
不 import 任何本地分析代码、不经过看板。输出见 `cloud_stat_out.json`。要点：

| arm | rows | pass（末 4 rollout 均值） | pass（全程） | d0 | KL@末行 |
|---|---|---|---|---|---|
| µ1 s0/s1/s2 | 8 | 0.2623 / 0.2656 / 0.2585 | 0.2458 / 0.2501 / 0.2446 | 0.072 / 0.068 / 0.066 | 0 / 0 / 0 |
| µ8 s0/s1/s2 | 64 | 0.4290 / 0.4427 / 0.4320 | 0.3516 / 0.3640 / 0.3538 | 0.138 / 0.141 / 0.141 | 3.9e-5 / 1.0e-4 / 3.2e-5 |
| µ16 s0/s1/s2 | 128 | 0.4883 / 0.4771 / 0.4862 | 0.3886 / 0.3872 / 0.3892 | 0.201 / 0.174 / 0.185 | −8.7e-8 / 6.5e-5 / 1.3e-4 |

配对差（同一 seed，末 4 rollout 口径）：µ8−µ1 = **+0.1666 / +0.1771 / +0.1735**，
µ16−µ1 = **+0.2260 / +0.2115 / +0.2277**。与修复后的本地 `analysis/out/table.csv`
逐位一致（那是三个小数位的 0.167/0.177/0.174 与 0.226/0.212/0.228）。

"pass（全程）"与"pass（末 4 rollout）"两列并列放在这里，就是为了让上面那个
口径 bug 的后果一目了然：**同一批数据，差 0.07–0.10 个绝对通过率**。

## 五、云端独有、此前不在仓库的文件

- `tools/run_with_chunk.py` — 分块 logsumexp 评分器包装（D-25 期间写的），
  已随 `ffd2778` 进仓库；它依赖已在仓库里的 `tools/probe_memory.py`。
- `cloud_only/_dbg_patch.py` — 一次 monkey-patch 调试的临时脚本，只归档不提交。
- `cloud_only/vram_guard.sh.d25.bak` — D-25 改 guard 之前的备份，只归档不提交。
- `cloud_only/env.txt` / `cloud_only/COMMIT_at_shutdown.txt` — 云端自报的环境与提交号。

`/tmp/campaign.log`（190 KB）是整场 9 run 的主控日志，逐 rollout 记录
pass / len / kl / clip / gn / sigma / d 与每步的 GPU-h、CNY。9 个 run 全部以
`[OK]` 收尾，全程记账 **¥33.58**（含所有探针；9 个正式 run 合计 ¥32.40）。

## 六、本地是唯一副本

`runs/*` 在 `.gitignore` 里（19 GB 不进 git），所以 **72 个 adapter + 9 个
state.pt 现在只有本地这一份**。要长期保留请自行备份（例如挪到移动硬盘或对象存储）。

## 七、复现

```bash
# 校验本地归档与云端一致性（需要本目录三份 MANIFEST）
cd <repo>
python tools/verify_archive.py            # 见 verify_result.txt 头部说明
# 重新计算汇总表
python analysis/make_table.py --runs runs --out analysis/out
```
