"""Reconstruct per-prompt correctness for the archived mu16 baseline runs (r6, r7).

The campaign archive kept only 1 prompt-group per cycle in samples.jsonl, so the
frozen P1 caliber (prompt-clustered bootstrap, v2 section 4) has no per-prompt
ground truth for the baseline arm. This tool rebuilds it exactly:

    pool (frozen, sha-checked) -> data.prompt_batch (pure fn of pool/seed/idx)
    -> npz completion tokens -> detokenize -> verify.is_correct -> 512x8 bits.

Validated bit-exact on sac_s0 cycle 0 against its live-logged prompts.jsonl
(0/512 mismatches, 2026-10-31). Same pool sha256 (d781f955...) is asserted for
every run scored here.

Requires the `tokenizers` package (offline detokenize only; no model, no GPU):

    python tools/rescore_mu16.py --json analysis/out/mu16_rescored_r6r7.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import data as data_mod  # noqa: E402
import verify as verify_mod  # noqa: E402

TOKENIZER = r"C:\Users\Administrator\Desktop\work sapce 1\jd-lab\projects\s1-grpo-rft\models\Qwen3-4B-Base\tokenizer.json"
RUNS = {
    "mu16_s0": os.path.join(REPO, "runs", "mu16_s0"),
    "mu16_s1": os.path.join(REPO, "runs", "mu16_s1"),
    "mu16_s2": os.path.join(REPO, "runs", "mu16_s2"),
}
CYCLES = (6, 7)
POOL = os.path.join(REPO, "data", "prompts_mixed2000.jsonl")
POOL_SHA = "d781f955ad9773cc0107fd171b2371ea3bec262a2021b6e8e481da81f64a798b"


def rescore_run(tok: Tokenizer, run_dir: str, seed: int, cycle: int) -> dict:
    pool = data_mod.load_pool(POOL, expect_sha256=POOL_SHA)
    prompts = data_mod.prompt_batch(pool, seed=seed, rollout_idx=cycle, n=512)
    d = np.load(os.path.join(run_dir, "rollouts", f"r{cycle:03d}.npz"), allow_pickle=True)
    clen = d["completion_len"]
    cflat = d["completion_flat"]
    assert len(clen) == 4096, f"{run_dir} r{cycle}: expected 4096 responses, got {len(clen)}"
    offs = np.concatenate([[0], np.cumsum(clen)])
    bits = []
    for g in range(512):
        s = "".join(
            "1" if verify_mod.is_correct(prompts[g]["answer"],
                                         tok.decode(cflat[offs[g * 8 + j]:offs[g * 8 + j + 1]].tolist()))
            else "0"
            for j in range(8)
        )
        bits.append(s)
    return {
        "run": os.path.basename(run_dir),
        "seed": seed,
        "rollout_idx": cycle,
        "n_prompts": len(bits),
        "group_size": 8,
        "correct_bits": bits,
        "pass_rate": sum(b.count("1") for b in bits) / (len(bits) * 8),
        "source": os.path.join(run_dir, "rollouts", f"r{cycle:03d}.npz"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(REPO, "analysis", "out", "mu16_rescored_r6r7.json"))
    args = ap.parse_args()

    tok = Tokenizer.from_file(TOKENIZER)
    out = {"caliber": "pool->prompt_batch->npz->verify, bit-exact validated on sac_s0 r0",
           "pool_sha256": POOL_SHA, "runs": []}
    for run, path in RUNS.items():
        seed = int(run.rsplit("_s", 1)[1])
        for cycle in CYCLES:
            rec = rescore_run(tok, path, seed, cycle)
            out["runs"].append(rec)
            print(f"{run} r{cycle}: pass={rec['pass_rate']:.4f} "
                  f"({rec['n_prompts']}x{rec['group_size']})")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", args.json)


if __name__ == "__main__":
    main()
