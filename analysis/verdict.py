#!/usr/bin/env python3
"""format-signal-audit 判读复算（一键）。

从 evidence/evals/ 读评测明细（items jsonl，prompt 级配对），重算全部组间配对差
与 prompt 级 bootstrap 95% CI，并按预注册规则（探索期日志第九/十一节 + R-3）输出
规则命中情况。

用法：
    python analysis/verdict.py [--evidence EVIDENCE_DIR] [--draws N] [--seed N]

依赖：仅 Python 标准库。

口径备忘（勿改动，出处见探索期日志与 R-2/R-3）：
- 配对单位 = prompt（512 个），样本级 is_correct 为 n=4 内嵌列表，先按行取均值。
- 差值方向：A11-B8 / B-A 等，均在输出中显式标注方向。
- bootstrap = prompt 级重抽（反映测试任务抽样不确定性；seed 间不确定性由
  逐 seed 列表呈现，n=3 不做合并推断——见禁写 7）。
- E2 的 mu1 A 臂评测与 B8 评测、E4 的 B8 评测若不在 evidence/ 中，输出
  MISSING 并跳过该组，不得视为零效应。
"""
import argparse
import json
import random
from pathlib import Path

B8_SUMMARY = {0: 0.2598, 1: 0.2769, 2: 0.2695}  # fsa_b_full 摘要值（云端记账）


def load_prompts(path):
    """items jsonl -> {prompt_key: prompt级正确率}。"""
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = r.get("prompt_id", r.get("prompt_idx", r.get("id")))
        v = r.get("is_correct", r.get("correct", 0))
        if isinstance(v, list):
            v = sum(float(x) for x in v) / len(v)
        else:
            v = float(v)
        d.setdefault(key, []).append(v)
    return {k: sum(v) / len(v) for k, v in d.items()}


def paired_ci(diff_pairs, draws, rng):
    n = len(diff_pairs)
    boots = sorted(
        sum(diff_pairs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)
    )
    return boots[int(0.025 * draws)] * 100, boots[int(0.975 * draws)] * 100


def evaluate_pair(evals, name_a, name_b, rng, draws):
    """返回 [(seed, mean_diff_pp, lo, hi)]，diff = A - B（按传入顺序）。"""
    out = []
    for s in range(3):
        pa = list((evals / f"{name_a}{s}").glob("items_*.jsonl"))
        pb = list((evals / f"{name_b}{s}").glob("items_*.jsonl"))
        if not pa or not pb:
            return None
        A = load_prompts(pa[0])
        B = load_prompts(pb[0])
        keys = sorted(set(A) & set(B))
        pd = [A[k] - B[k] for k in keys]
        m = sum(pd) / len(pd) * 100
        lo, hi = paired_ci(pd, draws, rng)
        out.append((s, m, lo, hi, len(keys)))
    return out


def report(title, rows, direction, rule=None):
    print(f"\n=== {title} ===")
    if rows is None:
        print("MISSING（评测数据不在 evidence/ 中——该组结论未经本地复算，不得引用）")
        return
    ds = []
    for s, m, lo, hi, n in rows:
        print(f"  s{s}: {direction} = {m:+.2f}pp  bootstrap95%CI [{lo:+.2f}, {hi:+.2f}]  (n={n})")
        ds.append(m)
    mean = sum(ds) / len(ds)
    print(f"  3seed 均值 = {mean:+.2f}pp（seed 间不确定性以逐 seed 列表为准，n=3 不做合并检验）")
    if rule:
        print(f"  预注册命中：{rule}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=str(Path(__file__).resolve().parent.parent / "evidence"))
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    evals = Path(args.evidence) / "evals"
    rng = random.Random(args.seed)

    print(f"evidence = {evals}")
    found = sorted(p.name for p in evals.iterdir() if p.is_dir())
    print(f"可用评测目录 {len(found)} 个")

    # E3：mod 族，等 kept 梯度预算（B - A）
    rows = evaluate_pair(evals, "eval_fsa_mod_b_s", "eval_fsa_mod_a_s", rng, args.draws) \
        if all((evals / f"eval_fsa_mod_{a}_s0").exists() for a in "ab") else None
    report("E3 mod 族（B 过滤 - A 无过滤，等 kept 预算，heldout_mod_512）", rows, "B-A")
    print("  预注册：规则2 = 三 seed 均 <+0.5pp 或方向分裂（实测 −0.34/−0.54/+0.35pp，命中，负结果收口）")

    # E4：算力对齐（A11 - B8）
    have_b8 = all((evals / f"eval_fsa_b_full_s{s}").exists() for s in range(3))
    if have_b8:
        rows = evaluate_pair(evals, "eval_fsa_a_align_s", "eval_fsa_b_full_s", rng, args.draws)
        report("E4 算力对齐（A×11 - B×8，等 rollout 生成预算，heldout_counting_512）", rows, "A11-B8")
        print("  预注册：规则2 = D_s(B8-A11) 三 seed 均 <+0.5pp 或分裂（实测 −2.64/−0.83/−0.83pp，命中，全项目负结果收口）")
    else:
        print("\n=== E4 算力对齐（A×11 - B×8） ===")
        print("MISSING：eval_fsa_b_full_s{0,1,2} 不在 evidence/ 中（云端独占数据，待拉回）")
        print("  已判读值（探索期日志第十二节）：A11-B8 = +2.64/+0.83/+0.83pp，均值 +1.43pp，规则 2 命中")

    # E2：mu1（A×8 复用基线） vs B8，等 kept 预算
    if all((evals / f"eval_mu1_s{s}").exists() for s in range(3)) and have_b8:
        rows = evaluate_pair(evals, "eval_fsa_b_full_s", "eval_mu1_s", rng, args.draws)
        report("E2 计数族（B×8 - A×8 mu1，等 kept 预算，heldout_counting_512）", rows, "B-A")
        print("  已判读值：+0.29/+1.81/+1.17pp，均值 +1.09pp（t=2.48, df=2, p≈0.13，弱正不作主张）")
    else:
        print("\n=== E2 计数族（B×8 - A×8） ===")
        print("MISSING：eval_mu1_s* / eval_fsa_b_full_s* 不在 evidence/ 中（云端独占，待拉回）")
        print("  已判读值（探索期日志第六节）：B-A = +0.29/+1.81/+1.17pp，均值 +1.09pp，p≈0.13")

    # E0：small 三臂（机制定位，无分辨力，仅描述）
    if all((evals / f"eval_fsa_small_{a}_s0").exists() for a in "abc"):
        for x, y in (("b", "a"), ("c", "a")):
            rows = evaluate_pair(evals, f"eval_fsa_small_{x}_s", f"eval_fsa_small_{y}_s", rng, args.draws)
            report(f"E0 small（{x} - {y}，2周期×512，仅机制定位）", rows, f"{x}-{y}")

    print("\n完成。注意：所有 CI 只反映测试任务抽样不确定性；禁写清单见 jd-lab discussions 简历模板 v1.0。")


if __name__ == "__main__":
    main()
