# 2026-10-24 首次云端会话（4090-24G）证据存档

实例：新疆 4090-24G，记账单价 ¥2.08/h（**错：2.08 是 48G 的价，24G 是 ¥1.56/h，2026-10-25 用户核实后已改 `configs/`**。
本目录里 `summary.hfsmoke.json` / `config.resolved.yaml` / `vllmsmoke/*` 仍写着 2.08 —— 那是**当时的配置实况**，作为证据不改写；
读这些数时把 `cost_cny` 乘以 **1.56/2.08 = 0.75** 才是真实花销，即本次成功运行实为 ¥0.017 而非 ¥0.0229），
镜像 `PyTorch/2.8.0-CUDA12.8-Py3.11-Ubuntu22`，数据盘 `/root/rivermind-data`（49 GiB），
系统盘 30 GiB，容器配额 15 CPU / 100 GiB（cgroup，非 `nproc`/`free` 报的 128 / 1007）。

| 文件 | 是什么 |
|---|---|
| `metrics.jsonl` | **成功运行**（HF 后端，32 responses，µ=2，cap=512）的逐 µ-step 指标，2 行 |
| `samples.jsonl` | 同一运行的 8 条 rollout 样本（prompt / completion / reward / correct） |
| `config.resolved.yaml` / `run_meta.json` | 该运行解析后的完整配置与环境元数据（`argv` 是当时真实的命令行） |
| `summary.json` → `summary.hfsmoke.json` | 该运行的汇总（FLOPs 三分账、GPU 小时、¥0.02） |
| `hfsmoke.log` | 该运行完整日志（35.8 s rollout / 3.9 s update） |
| `smoke.log` | **失败运行**的日志：vLLM 0.32.0 + 24 GiB 的 OOM 现场（D-10 第 4 条） |
| `vllmsmoke/` | 同一次失败运行的 run 目录：`run_meta` + `config.resolved` 有值，`metrics/samples` 是 **0 字节**（OOM 在第 1 步之前） |
| `calib16.json` | 16 题 × 1 样本的通过率探针（cap=512），总 4/16 |
| **`ENV.md`** | **关机前采集的完整环境快照 + 五个坑 + 重建三条命令** |
| **`pip-freeze.txt`** | `/opt/conda/bin/python` 的 207 个包（已剔除不可复现的 conda 构建路径） |

归档完整性：8 个产物与云端 `sha256sum` 逐字节一致（2026-10-25 00:47 核对，`summary.hfsmoke.json` 首次用 shell 重定向抓取时尾部差 1 字节，改用 `scp -O` 后一致）。

## 为什么这几条数字值得留

1. `metrics.jsonl` 里 `mu_step=0` 的 `kl_per_rollout_policy` 与 `ratio_mean`
   分别是 `0.0` 和 `1.0` —— 精确相等，不是"接近"。这是"测量器没接错线"的最强证据。
2. 同一运行 `mu_step=1` 的 KL=1.46e-4、clip_frac=0.39%、σ=0.065、d=0.72：
   三个被测量对象都在动，且量级落在 Two Clocks 原文报告的范围附近。
3. `smoke.log` 里 `pass_rate=0.000 / zero_var_group_frac=1.0` 是**假象**，
   根因是 cap=128 截断（resp_len_mean=112.6/128）。cap=512 后 pass=0.25、
   zero_var=0.0。这条如果没记下来，后面会把"测量归零"误诊成"模型不行"。

## 尚未解决

vLLM 与 HF 策略在 24 GiB 上无法共存（D-10 第 4 条）。下一步要么开 vLLM
sleep mode，要么用 48G 卡。当前 `configs/cloud24g.yaml` 的 vLLM 参数
（0.28 / 64 / max_model_len 1024 / enforce_eager）是已经验证到"能启动"、
但**未能验证到"能跑完整一步更新"**的设置。

## 关机与重建

实例于 2026-10-25 00:5x 关机（平台提示"关机后 7 天释放"）。云端**没有 `.git`**（GitHub 不通，代码走 scp），
所以没有"云端未推送的改动"这种东西；本地 4 个 commit 全部已推远端。
唯一不可再生的是上面那几十 KB 日志，已全部回本地。
重建走 `tools/cloud_bootstrap.sh`（幂等，装环境 + 拉权重 + 12 项回归，约 15 min，其中 13 min 是下权重）。
