# 2026-10-24 首次云端会话（4090-24G）证据存档

实例：新疆 4090-24G，¥2.08/h，镜像 `PyTorch/2.8.0-CUDA12.8-Py3.11-Ubuntu22`，
数据盘 `/root/rivermind-data`，系统盘 30 GiB，cgroup 15 CPU / 100 GiB。

| 文件 | 是什么 |
|---|---|
| `metrics.jsonl` | **成功运行**（HF 后端，32 responses，µ=2，cap=512）的逐 µ-step 指标，2 行 |
| `samples.jsonl` | 同一运行的 8 条 rollout 样本（prompt / completion / reward / correct） |
| `config.resolved.yaml` / `run_meta.json` | 该运行解析后的完整配置与环境元数据 |
| `hfsmoke.log` | 该运行完整日志（35.8 s rollout / 3.9 s update / ¥0.02） |
| `smoke.log` | **失败运行**的日志：vLLM 0.32.0 + 24 GiB 的 OOM 现场（D-10 第 4 条） |
| `calib16.json` | 16 题 × 1 样本的通过率探针（cap=512），总 4/16 |

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
