"""Counterfactual (would-be) filter trigger rates from an A-arm samples.jsonl.

The campaign A-arm runs (mu1_s*) never executed the resample path, but their
per-response records (prompt, reward, correct) let us compute EXACTLY what the
B (accuracy: group all-correct or all-wrong) and C (variance: reward std
ddof=1 < 1e-8, the grpo zeroing threshold) filters WOULD have done at every
rollout -- the treatment dose curve, offline and free.

Grouping: responses of one group share (rollout_idx, prompt text) and sit
prompt-major in the file; group size is asserted constant (= train G).

Outputs: <out>/<tag>_trigger.json with per-rollout
  n_groups, wouldbe_b_reject_rate, all_wrong_frac, all_correct_frac,
  wouldbe_c_reject_rate, kept_batch_pass   (mean correctness of B-kept groups)
plus an overall summary.

  python tools/trigger_wouldbe.py runs/mu1_s0/samples.jsonl --tag mu1_s0
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("samples")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="out/trigger_wouldbe")
    ap.add_argument("--var-threshold", type=float, default=1e-8)
    args = ap.parse_args()

    groups = defaultdict(list)
    with open(args.samples, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            groups[(str(r["rollout_idx"]), r["item"])].append(r)

    sizes = {len(v) for v in groups.values()}
    if len(sizes) != 1:
        raise SystemExit(f"inconsistent group sizes {sorted(sizes)} -- check grouping key")
    gsize = sizes.pop()

    per = defaultdict(lambda: defaultdict(int))
    for (ri, _item), rows in groups.items():
        corr = [bool(r["correct"]) for r in rows]
        rews = [float(r["reward"]) for r in rows]
        d = per[ri]
        d["n_groups"] += 1
        allc, allw = all(corr), not any(corr)
        d["all_correct"] += allc
        d["all_wrong"] += allw
        if allc or allw:
            d["b_reject"] += 1
        if statistics.variance(rews) < args.var_threshold:
            d["c_reject"] += 1
        else:
            d["kept_correct"] += sum(corr)
            d["kept_responses"] += gsize
        d["all_correct_resp"] += sum(corr)
        d["n_responses"] += gsize

    per_out = {}
    for ri in sorted(per, key=int):
        d = per[ri]
        n = d["n_groups"]
        kept = n - d["b_reject"]
        per_out[ri] = {
            "n_groups": n,
            "wouldbe_b_reject_rate": round(d["b_reject"] / n, 4),
            "all_wrong_frac": round(d["all_wrong"] / n, 4),
            "all_correct_frac": round(d["all_correct"] / n, 4),
            "wouldbe_c_reject_rate": round(d["c_reject"] / n, 4),
            "kept_batch_pass": round(d["kept_correct"] / max(1, d["kept_responses"]), 4),
            "batch_pass": round(d["all_correct_resp"] / d["n_responses"], 4),
        }
    rates = [v["wouldbe_b_reject_rate"] for v in per_out.values()]
    kept = [v["kept_batch_pass"] for v in per_out.values()]
    batch = [v["batch_pass"] for v in per_out.values()]
    out = {
        "tag": args.tag or Path(args.samples).parent.name,
        "samples": str(args.samples),
        "group_size": gsize,
        "n_rollouts": len(per_out),
        "per_rollout": per_out,
        "summary": {
            "b_reject_rate_mean": round(sum(rates) / len(rates), 4),
            "b_reject_rate_first_half": round(sum(rates[: len(rates) // 2]) / max(1, len(rates) // 2), 4),
            "b_reject_rate_second_half": round(sum(rates[len(rates) // 2:]) / max(1, len(rates) - len(rates) // 2), 4),
            "kept_batch_pass_mean": round(sum(kept) / len(kept), 4),
            "batch_pass_mean": round(sum(batch) / len(batch), 4),
        },
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{out['tag']}_trigger.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    s = out["summary"]
    print(f"[wouldbe] {out['tag']}: G={gsize}, {len(per_out)} rollouts | "
          f"B reject {s['b_reject_rate_mean']:.4f} "
          f"(1st half {s['b_reject_rate_first_half']:.4f} -> 2nd half {s['b_reject_rate_second_half']:.4f}) | "
          f"kept-batch pass {s['kept_batch_pass_mean']:.4f} vs batch {s['batch_pass_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
