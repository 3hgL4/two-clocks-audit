"""Counterfactual (would-be) filter trigger rates from a campaign A-arm run dir.

The campaign archive's npz keeps full completion token ids (samples.jsonl is
only an 8-response-per-rollout SAMPLE -- not usable for group statistics).
This tool replays the frozen pipeline offline:

    pool (sha-checked) -> data.prompt_batch (pure fn of seed/rollout_idx)
    -> npz completion tokens -> detokenize -> verify.reward / is_correct
    -> per-group would-be B (accuracy: all-correct/all-wrong) and
       C (variance: reward std ddof=1 < 1e-8, the grpo zeroing threshold)

Cross-check: the recomputed per-rollout batch pass rate must equal the
run's metrics.jsonl pass_rate EXACTLY (same bit replay the rescore_mu16 tool
validated with 0/512 mismatches); any mismatch aborts with a report.

CPU only, no GPU, no model weights -- just the tokenizer json.

  python tools/trigger_wouldbe_npz.py RUN_DIR --seed 0 --tokenizer TOKENIZER_JSON \
      --pool POOL_JSONL --out out/trigger_wouldbe [--rollouts 0-18]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# two-clocks-audit repo root (owns data.py / verify.py the campaign ran with)
TCA = Path(r"C:\Users\Administrator\Desktop\work sapce 1\jd-lab\projects\two-clocks-audit")
sys.path.insert(0, str(TCA / "src"))

import data as data_mod  # noqa: E402
import verify as verify_mod  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tokenizer", default=r"C:\Users\Administrator\Desktop\work sapce 1\jd-lab\projects\s1-grpo-rft\models\Qwen3-4B-Base\tokenizer.json")
    ap.add_argument("--pool", default=str(TCA / "data" / "prompts_mixed2000.jsonl"))
    ap.add_argument("--pool-sha", default="d781f955ad9773cc0107fd171b2371ea3bec262a2021b6e8e481da81f64a798b")
    ap.add_argument("--out", default="out/trigger_wouldbe")
    ap.add_argument("--metrics", default=None,
                    help="metrics.jsonl to cross-check against (default RUN_DIR/metrics.jsonl). "
                         "For the R4A-era campaign runs the canonical metrics live in "
                         "runs_r4a/<run> while the npz archives live in runs/<run>/rollouts "
                         "(the pre-crash execution's metrics sit in runs/<run> -- do NOT "
                         "cross-check against those).")
    ap.add_argument("--rollouts", default="0-7")
    ap.add_argument("--var-threshold", type=float, default=1e-8)
    args = ap.parse_args()

    r0, r1 = args.rollouts.split("-")
    rollout_idxs = list(range(int(r0), int(r1) + 1))

    pool = data_mod.load_pool(args.pool, expect_sha256=args.pool_sha)
    tok = Tokenizer.from_file(args.tokenizer)
    run_dir = Path(args.run_dir)

    metrics = {}
    mpath = Path(args.metrics) if args.metrics else run_dir / "metrics.jsonl"
    if mpath.exists():
        with open(mpath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("mu_step") == 0:
                    metrics[str(row["rollout_idx"])] = row["pass_rate"]

    per = {}
    mismatch = []
    for ri in rollout_idxs:
        npz = run_dir / "rollouts" / f"r{ri:03d}.npz"
        d = np.load(npz, allow_pickle=True)
        clen, cflat = d["completion_len"], d["completion_flat"]
        n_resp = len(clen)
        G = 8
        n_groups = n_resp // G
        assert n_resp % G == 0, f"r{ri}: {n_resp} responses not divisible by G={G}"
        prompts = data_mod.prompt_batch(pool, seed=args.seed, rollout_idx=ri, n=n_groups)
        offs = np.concatenate([[0], np.cumsum(clen)])
        corr, rews = [], []
        for g in range(n_groups):
            for j in range(G):
                text = tok.decode(cflat[offs[g * G + j]:offs[g * G + j + 1]].tolist())
                ans = int(prompts[g]["answer"])
                corr.append(verify_mod.is_correct(ans, text))
                rews.append(verify_mod.reward(ans, text))
        n_corr = 0
        b_reject = c_reject = allc = allw = 0
        kept_corr = kept_resp = 0
        for g in range(n_groups):
            gc = corr[g * G:(g + 1) * G]
            gr = rews[g * G:(g + 1) * G]
            n_corr += sum(gc)
            ac, aw = all(gc), not any(gc)
            allc += ac
            allw += aw
            if ac or aw:
                b_reject += 1
            else:
                kept_corr += sum(gc)
                kept_resp += G
            m = sum(gr) / G
            if statistics.variance(gr) < args.var_threshold:
                c_reject += 1
        recomputed = n_corr / n_resp
        logged = metrics.get(str(ri))
        if logged is not None and abs(recomputed - logged) > 1e-9:
            mismatch.append((ri, recomputed, logged))
        per[str(ri)] = {
            "n_groups": n_groups,
            "wouldbe_b_reject_rate": round(b_reject / n_groups, 4),
            "all_wrong_frac": round(allw / n_groups, 4),
            "all_correct_frac": round(allc / n_groups, 4),
            "wouldbe_c_reject_rate": round(c_reject / n_groups, 4),
            "kept_batch_pass": round(kept_corr / max(1, kept_resp), 4),
            "batch_pass": round(recomputed, 6),
        }
        print(f"r{ri:03d}: B reject {per[str(ri)]['wouldbe_b_reject_rate']:.4f} "
              f"(allw {per[str(ri)]['all_wrong_frac']:.4f} allc {per[str(ri)]['all_correct_frac']:.4f}) "
              f"C reject {per[str(ri)]['wouldbe_c_reject_rate']:.4f} "
              f"kept-pass {per[str(ri)]['kept_batch_pass']:.4f} batch {per[str(ri)]['batch_pass']:.4f}",
              flush=True)

    if mismatch:
        raise SystemExit(f"PASS MISMATCH vs metrics.jsonl (replay not bit-exact): {mismatch}")

    rates = [v["wouldbe_b_reject_rate"] for v in per.values()]
    kept = [v["kept_batch_pass"] for v in per.values()]
    batch = [v["batch_pass"] for v in per.values()]
    h = len(rates) // 2
    out = {
        "run_dir": str(run_dir),
        "seed": args.seed,
        "group_size": 8,
        "n_rollouts": len(per),
        "per_rollout": per,
        "summary": {
            "b_reject_rate_mean": round(sum(rates) / len(rates), 4),
            "b_reject_rate_first_half": round(sum(rates[:h]) / max(1, h), 4),
            "b_reject_rate_second_half": round(sum(rates[h:]) / max(1, len(rates) - h), 4),
            "c_reject_rate_mean": round(
                sum(v["wouldbe_c_reject_rate"] for v in per.values()) / len(per), 4),
            "kept_batch_pass_mean": round(sum(kept) / len(kept), 4),
            "batch_pass_mean": round(sum(batch) / len(batch), 4),
        },
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_dir.name}_trigger.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    s = out["summary"]
    print(f"[wouldbe] {run_dir.name}: B reject {s['b_reject_rate_mean']:.4f} "
          f"(1st half {s['b_reject_rate_first_half']:.4f} -> 2nd half {s['b_reject_rate_second_half']:.4f}) | "
          f"C reject {s['c_reject_rate_mean']:.4f} | "
          f"kept-pass {s['kept_batch_pass_mean']:.4f} vs batch {s['batch_pass_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
