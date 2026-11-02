# format-signal-audit

GRPO 训练中"过滤-重抽"机制（DAPO 式 accuracy 过滤）的三臂受控审计。**预注册判读规则**（结果可见前冻结），**负结果收口**：等预算下未检出过滤重抽的样本效率优势。

本分支是 [two-clocks-audit](https://github.com/3hgL4/two-clocks-audit)（主项目：对 arXiv:2610.18853 的可复现审计）的 `fsa-resample` 分支。

---

## 一键复现

### 判读复算（0 GPU，任何机器，秒级）

```bash
git clone -b fsa-resample https://github.com/3hgL4/two-clocks-audit fsa && cd fsa
./reproduce.sh verdict    # 从 evidence/ 复算全部预注册判读（纯 stdlib）
./reproduce.sh verify     # sha256 校验 evidence/ 79 文件完整性
```

`verdict` 输出四组实验的逐 seed 配对差 + prompt 级 bootstrap 95% CI + 预注册规则自动裁决。实测与冻结判读**逐位一致**（已在 4 台独立云盒的全新副本上验证）。

### 云端训练复现（gpuhome 盒，四档）

```bash
sh scripts/check_env.sh               # 30 秒自检：GPU/网络/依赖 -> 能跑哪个档
sh scripts/repro_cloud.sh             # verdict：判读复算（0 GPU）
sh scripts/repro_cloud.sh smoke       # 单测 + CPU 干跑（含 B/C 臂重抽路径）
sh scripts/repro_cloud.sh gpu-smoke   # GPU 冒烟训练 x2 + rollout 逐位审计（~10 min）
sh scripts/repro_cloud.sh arm         # 真实训练一臂 fsa_small_a x3 + 评测（~1-2 h GPU）
```

自包含：代码（github 拉取，失败时打印 SFTP 兜底指引）→ 依赖（pypi 自动补装）→ 模型（[魔搭 ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B-Base) 并行分段下载 + sha256 校验，无需任何在盘权重）→ 判读 → 训练。many-core 盒自动限 OMP 线程。

### 本地开发验证

```bash
./reproduce.sh dry    # 单测 38 项 + CPU 干跑（stub 模型，~1 min，many-core 需 OMP_NUM_THREADS=4）
```

---

## 判读结论（预注册规则，负结果）

| 实验 | 设计 | 结果（3 seed） | 预注册裁决 |
|---|---|---|---|
| E2 计数族 | B×8 − A×8（等 kept 预算） | +0.29/+1.81/+1.17pp，均值 +1.09pp（p≈0.13） | 弱正，**不作主张** |
| E3 mod 族 | B − A（等 kept 预算，模运算第二族） | −0.34/−0.54/+0.35pp，均值 −0.18pp | 规则 2 命中：**成本收益负结果** |
| E4 算力对齐 | A×11 − B×8（等 rollout 生成预算） | A11−B8 = +2.64/+0.83/+0.83pp，均值 +1.43pp | D_s 全 <+0.5pp：规则 2 命中，**全项目负结果收口** |
| E0 small | 3 臂 × 2 周期（机制定位） | b−a +0.26pp / c−a +0.73pp | 仅定位用 |

**表述边界**：E2 禁写"有效"（不跨任务族稳健 + 功效不足）；"35% 算力混杂"框架已被独立审计证伪（B 的 kept token 反少 3.4%）；E4 的资源差异是设计而非混杂。完整禁写清单见 `docs/`。

## 证据与审计链

- `evidence/`：79 个文件（25 个评测目录 + 32 份训练 metrics + MANIFEST.sha256）——判读的全部输入，`reproduce.sh verify` 可随时校验
- `analysis/verdict.py`：stdlib-only 判读复算（CI 与裁决）
- `docs/探索期日志_2026-10-31.md`：预注册规则原文（第九/十一节）、修订链 R-1～R-3（全部在结果可见前落盘）、全部读数与事故记录
- 独立外部审计：6 项证实 / 6 项证伪；关键证伪（"35% 算力混杂"）经执行侧独立重算后接受并修订判读措辞

## 复现训练全链（约 60-80 GPU-h，4090-48G）

`reproduce.sh full` 文档化云端驱动顺序：`run_fsa_small.sh`（E0）→ `run_night.sh`（E2 B8 + E3）→ `run_align.sh`（E4）→ `verdict.py`。判读数据无需重训，`evidence/` 即正本。
