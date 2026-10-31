"""Held-out evaluation entry for format-signal-audit (2026-10-31).

Evaluates a model (optionally + LoRA adapter) on a frozen held-out pool under
the SAME conditions as training rollouts: raw prompt, no chat template,
temperature/top_p/max_completion_length from the training config defaults,
judged by src/verify.py -- the exact function the GRPO reward uses for
correctness, so a training-side pass rate and an eval-side pass rate are the
same statistic on different prompt sets.

Sampling discipline mirrors the campaign's seed_per_request finding (cloud48g
config header: one seed per request; n>1 under a single seed produced near
identical samples). Each prompt is expanded into k independent requests with
deterministic per-(prompt, sample) seeds: seed*1_000_003 + prompt_idx*k + j.
Re-running with the same seed reproduces the completions bit for bit (vLLM
greedy/sampled determinism caveats apply, same as the training smoke).

Outputs (under --out):
  <tag>/items_<poolstem>.jsonl   one row per prompt: per-sample correctness,
                                 parsed values, format flags, pool metadata
  <tag>/summary_<poolstem>.json  aggregate pass rate, format rate, breakdowns
                                 by depth and aggregation, sha256 of the pool

  python tools/eval_heldout.py --pool data/heldout_mod_512.jsonl \
      --model /path/Qwen3-1.7B-Base --n 4 --temperature 1.0 \
      --tag base_mod --out out/eval_heldout
  python tools/eval_heldout.py --pool data/heldout_counting_512.jsonl \
      --model /path/Qwen3-1.7B-Base --adapter runs/fsa_small_b_s0/adapters/r001 \
      --n 4 --tag fsa_small_b_s0_eval --out out/eval_heldout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

import verify  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pool(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", action="append", required=True,
                    help="held-out pool jsonl; repeat for several pools")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (PEFT layout)")
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--n", type=int, default=4, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N prompts (smoke)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="out/eval_heldout")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    args = ap.parse_args(argv)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_req = None
    llm_kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_lora=args.adapter is not None,
        max_lora_rank=args.max_lora_rank if args.adapter is not None else None,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=args.enforce_eager,
    )
    llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}
    if args.adapter:
        lora_req = LoRARequest("adapter", 1, args.adapter)
    t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    print(f"[eval] engine up in {time.perf_counter() - t0:.1f}s")

    sp = SamplingParams(
        n=1, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, stop=None,
    )

    overall = {}
    for pool_path in args.pool:
        pool_path = Path(pool_path)
        items = load_pool(pool_path)
        if args.limit:
            items = items[: args.limit]
        k = args.n

        # Prompt-major expansion: prompt i -> k requests with per-(i, j) seeds.
        prompts, seeds, owner = [], [], []
        for i, it in enumerate(items):
            for j in range(k):
                prompts.append(it["prompt"])
                seeds.append(args.seed * 1_000_003 + i * k + j)
        sps = []
        for s in seeds:
            sp2 = SamplingParams(
                n=1, temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens, seed=int(s),
            )
            sps.append(sp2)
        t1 = time.perf_counter()
        outs = llm.generate(prompts, sps, lora_request=lora_req, use_tqdm=True)
        gen_s = time.perf_counter() - t1

        rows = []
        for i, it in enumerate(items):
            texts = [outs[i * k + j].outputs[0].text for j in range(k)]
            values, fmt = zip(*(verify.parse_answer(t) for t in texts))
            correct = [verify.is_correct(int(it["answer"]), t) for t in texts]
            rows.append({
                "prompt_idx": i,
                "answer": int(it["answer"]),
                "depth": it.get("depth"),
                "aggregation": it.get("aggregation"),
                "filters": it.get("filters"),
                "transforms": it.get("transforms"),
                "values": list(values),
                "format_ok": [v is not None for v in values],
                "correct": correct,
                "pass_any": any(correct),
                "texts": texts,
            })

        n = len(rows)
        n_samples = n * k
        pass_rate = sum(sum(r["correct"]) for r in rows) / n_samples
        pass_any = sum(r["pass_any"] for r in rows) / n
        fmt_rate = sum(sum(r["format_ok"]) for r in rows) / n_samples

        def breakdown(key):
            agg = defaultdict(lambda: [0, 0])
            for r in rows:
                kk = r.get(key)
                if kk is None:
                    continue
                if isinstance(kk, list):
                    kk = ">".join(map(str, kk))
                agg[kk][0] += sum(r["correct"])
                agg[kk][1] += k
            return {kk: {"pass_rate": round(c / m, 4), "n_samples": m}
                    for kk, (c, m) in sorted(agg.items())}

        summary = {
            "pool": str(pool_path),
            "pool_sha256": sha256_file(pool_path),
            "model": args.model,
            "adapter": args.adapter,
            "n_prompts": n,
            "samples_per_prompt": k,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "judge": "src/verify.is_correct (same as GRPO reward correctness)",
            "pass_rate": round(pass_rate, 4),
            "pass_rate_std_by_prompt": round(
                (sum((sum(r["correct"]) / k - pass_rate) ** 2 for r in rows) / max(1, n - 1)) ** 0.5, 4),
            "pass_any_rate": round(pass_any, 4),
            "format_rate": round(fmt_rate, 4),
            "by_depth": breakdown("depth"),
            "by_aggregation": breakdown("aggregation"),
            "generate_seconds": round(gen_s, 1),
        }
        stem = pool_path.stem
        items_path = out_dir / f"items_{stem}.jsonl"
        with open(items_path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary_path = out_dir / f"summary_{stem}.json"
        with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        overall[stem] = summary
        print(f"[eval] {stem}: pass {pass_rate:.4f} (any {pass_any:.4f}), "
              f"format {fmt_rate:.4f}, {n} prompts x {k} samples, {gen_s:.0f}s")

    print("[eval] summaries:")
    for stem, s in overall.items():
        print(json.dumps({kk: s[kk] for kk in
                          ("pool", "pass_rate", "pass_any_rate", "format_rate")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
