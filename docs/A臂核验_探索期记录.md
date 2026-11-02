# A 臂六项核验（探索期记录，2026-10-31 21:40）

> 状态：**探索期暂定结论**（两阶段模型；确认阶段启动时誊入 HYPOTHESIS.md）。核验对象 = `two-clocks-audit` 本地仓 `runs/mu16_s0/1/2`（SAC-µ 战役基线）。

## 逐项结果

| # | 项 | 结果 | 证据 |
|---|---|---|---|
| ① | 代码版本 | **有条件通过** | 三 run 记录 `git_commit=f04df9ff`（是本地 HEAD `f8adc3d` 的祖先 ✓）。`f04df9f..HEAD` 的 `src/train.py` 差异（+211/−30）经分级核对**全部位于 SAC 触发/统计路径**（ratio 统计累积、触发跳步、日志），固定 µ 路径的 backward→clip→step 序列逐语句等价；`src/sac.py` 为新增文件不影响固定 µ run。**限定**：B/C 臂代码基底见下方"待决" |
| ② | 训练配置 | **通过** | 三 run `config.resolved.yaml` 除 `tag`/`seed` 外**逐行一致**；µ=16、num_rollouts=8（周期）、responses_per_rollout=4096、micro_batch_responses=4、micro_group_responses=8、lr 5e-5、warmup 0.1、LoRA 开、max_model_len 1024 |
| ③ | 数据顺序 | **通过** | pool = `data/prompts_mixed2000.jsonl`，sha256 前缀 `d781f955` 与历史冻结引用一致；置换由 `data.prompt_batch`（纯函数）从 seed 导出 |
| ④ | seed | **通过** | s0/s1/s2 三个 seed，与暂定设计一致 |
| ⑤ | 预算口径 | **通过（含注）** | 采样量锁死（每周期 4096 响应）；历史 run 无显式预算字段，成本以墙钟/¥ 决算——与主口径（completion token 计数）兼容，token 计数可从 npz 离线补算 |
| ⑥ | 评测协议 | **不通过 → 设计缺口** | **配置与数据目录中不存在任何 held-out 测试集**（`data/` 只有训练池；SAC-µ 冻结稿的"末段 pass"= 训练 rollout pass 率）。B/C 臂重抽改变训练 prompt 流 → 末段训练 pass 臂间不可比。**方案中"冻结测试集 pass@1"目前无实现载体，必须新建** |

## 由 ⑥ 触发的设计动作（探索期需补）

新建 **held-out 冻结测试池**：同一生成器固定 seed 生成（或从现有池留出），建议 256–512 prompt，sha 冻结；评测 = 训练前 + 末段各一次（纯推理，4090 上成本低）。主判据 P1 改为该池上的 pass@1。此变更同时使"末 2 周期窗口"定义适用于训练侧诊断、测试集 eval 独立于窗口。

## ①的待决项（确认阶段前定）

B/C 臂训练代码基底二选一：
- **(i) 从 `f04df9f` 拉 resample 分支**（推荐）：三臂代码基底完全一致，① 直接过；放弃 HEAD 训练路径里的 SAC 统计埋点（不影响本 R2，分析在本地离线做）；
- (ii) 基于 HEAD：复用现有埋点，但需单测证明固定 µ 模式下 HEAD ≡ f04df9f 的更新行为（多一道验证工序）。

## 环境核验（⑥之外的补充，2026-10-31 云端勘察）

- 战役环境（无卡实例实测 + run_meta）：torch 2.13.0（PyPI 构建，run_meta 标 +cu130）/ transformers 5.19.0 / peft 0.21.2 / accelerate 1.15.0 / tokenizers 0.23.3 / numpy 2.3.2 / py 3.11.13；freeze 全量 322 包已存 `out/env_pip_freeze_sac-mu_2026-10-31.txt`。
- 新实例（hz01）：同型号 4090-48G，但**初始环境不匹配**（torch 2.8.0+cu128、无 transformers 等）→ 已按 freeze 精确 pin 重装（进行中）。download.pytorch.org 不可达，走清华 PyPI（torch 2.13.0 PyPI 构建与战役 freeze 一致）。
- 模型权重：老数据盘有 `Qwen3-1.7B-Base`（model.safetensors 3.44GB），old→new 直连 scp 中；**注意配置内路径** = `/root/rivermind-data/two-clocks-audit/models/Qwen3-1.7B-Base`，传输落点需对齐（当前落 `/root/rivermind-data/models/`，完成后移动）。
