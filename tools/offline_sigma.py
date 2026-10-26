#!/usr/bin/env python
"""Offline sigma decomposition -- claim C9.

Question this answers
---------------------
Training shows sigma growing with mu: r0 gives 0.0663 / 0.0695 / 0.0700 across
arms (5% spread) but r7 gives 0.0663 / 0.0830 / 0.0958 (44%). Two explanations are
available and the training runs cannot separate them:

  (H1) split effect   -- sigma depends on how the 4096 responses are cut into
                         steps of b = Br/mu. Then sigma is a ruler artefact.
  (H2) staleness      -- sigma grows because the policy drifts away from the one
                         that produced the data, and mu sets how far it drifts
                         inside a rollout. Then sigma is a real off-policy signal.

Why the obvious experiment cannot work
--------------------------------------
The original plan was "re-cut archived data at different b and see if the sigma
curve matches training". It cannot: sigma is `std(||g_i||) / ||g_bar||` over groups
of `micro_group_responses` = 8 responses, and b only changes the *number* of such
groups (512 / 64 / 32), not their size. A coefficient of variation's expectation
does not depend on the sample count, so re-cutting a fixed data set at different b
must return (very nearly) the same sigma. It would have produced a flat line and
"refuted" H1 by construction.

What this script does instead
-----------------------------
Fix one archived rollout D (mu1_s0/r000, 4096 responses produced by the initial
policy) and vary the *policy* instead of the split: load the per-rollout LoRA
snapshots r000 / r003 / r007, i.e. policies that have taken 1 / 4 / 8 update steps
away from the one that generated D. Every policy is evaluated on the same D with
the same advantages, so the only thing changing between rows is staleness.

For each policy the script reports sigma at all three splits in ONE backward pass:
step boundaries are nested (a mu=16 step is 32 groups, a mu=8 step is 64, mu=1 is
all 512), so three gradient accumulators are kept and flushed at their own
boundaries. This costs one pass, not three.

Reading the result
------------------
  * columns nearly equal within a row  -> H1 false: the split does not move sigma
  * sigma rising down the rows         -> H2 supported: staleness does
  * C9 in analysis/claims.yaml states the thresholds: same-adapter spread < 5%,
    and sigma(r007)/sigma(r000) >= 1.20.

Self-check (refuses to report numbers if it fails)
--------------------------------------------------
Re-deriving the rewards from the archived token ids must reproduce the r0 row of
metrics.jsonl: reward_mean 0.2631, pass_rate 0.2180. If it does not, the prompt
batch or the scorer has drifted and nothing below is comparable to the runs.

    /opt/conda/bin/python tools/offline_sigma.py \
        --run runs/mu1_s0 --adapters 000 003 007 --json analysis/out/offline_sigma.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as config_mod          # noqa: E402
import data as data_mod              # noqa: E402
import grpo as grpo_mod              # noqa: E402
import verify as verify_mod          # noqa: E402
from model import Policy             # noqa: E402

# train.py guards its entry point with __main__, so importing it does not run a
# campaign. Reusing its buffer helpers is the point: sigma must be computed by
# the same code that produced the training numbers, not by a lookalike.
import train as train_mod            # noqa: E402

# Split name -> groups per step. b = groups * micro_group_responses.
SPLITS = {"mu1": 512, "mu8": 64, "mu16": 32}


def _unflatten(flat: np.ndarray, lens: np.ndarray) -> list[list[int]]:
    out, off = [], 0
    for n in lens:
        n = int(n)
        out.append([int(x) for x in flat[off:off + n]])
        off += n
    return out


def load_adapter(policy: Policy, path: Path) -> str:
    """Load a snapshotted LoRA directory into the policy.

    `export_adapter` writes PEFT's own layout, whose key names may or may not carry
    the `base_model.` prefix depending on the PEFT version. Try the trainer's own
    loader first (exact key match against model.state_dict), fall back to PEFT's
    adapter loader, and refuse silently-wrong loads: a partially matched state dict
    would leave part of the previous adapter in place and quietly corrupt every row.
    """
    from safetensors.torch import load_file

    st = load_file(str(path / "adapter_model.safetensors"))
    try:
        policy.load_trainable_state_dict(st)
        return "direct"
    except ValueError:
        from peft import set_peft_model_state_dict

        res = set_peft_model_state_dict(policy.model, st)
        missing = [k for k in (res.missing_keys or []) if "lora_" in k]
        if missing:
            raise RuntimeError(
                f"{path.name}: {len(missing)} lora tensors did not load, e.g. {missing[:3]}"
            )
        return "peft"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloud48g_main.yaml")
    ap.add_argument("--run", default="runs/mu1_s0", help="run whose archived rollout is used")
    ap.add_argument("--rollout", type=int, default=0, help="rollout index of the fixed data D")
    ap.add_argument("--adapters", nargs="+", default=["000", "003", "007"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--limit-groups", type=int, default=None,
                    help="debug: stop after N groups (does not produce a reportable sigma)")
    args = ap.parse_args()

    cfg = config_mod.load_config(args.config)
    run_dir = Path(args.run)
    G = int(cfg["train.num_generations"])
    grp = cfg.micro_group_responses
    eps_low = float(cfg["train.eps_low"])
    eps_high = float(cfg["train.eps_high"])
    clip_delta = cfg["train.clip_delta"]
    norm_name = str(cfg["train.loss_norm"])

    # ---- 1. the fixed data D -------------------------------------------------
    pool = data_mod.load_pool(cfg["data.pool"], cfg["data.sha256"])
    items = data_mod.prompt_batch(pool, args.seed, args.rollout, cfg.prompts_per_rollout)
    npz_path = run_dir / "rollouts" / f"r{args.rollout:03d}.npz"
    if not npz_path.exists():
        raise SystemExit(f"missing archive: {npz_path}")
    z = np.load(npz_path)
    pids = _unflatten(z["prompt_flat"], z["prompt_len"])
    cids = _unflatten(z["completion_flat"], z["completion_len"])
    N = len(cids)
    if N != cfg["train.responses_per_rollout"]:
        raise SystemExit(f"archive has {N} responses, config expects {cfg['train.responses_per_rollout']}")
    if N % grp:
        raise SystemExit(f"{N} responses not divisible by group size {grp}")

    policy = Policy(cfg, args.device)
    policy.eval()

    # ---- 2. rebuild rewards, check against the run's own r0 row --------------
    texts = [policy.tokenizer.decode(c, skip_special_tokens=True) for c in cids]
    items_x = [it for it in items for _ in range(G)]
    if len(items_x) != N:
        raise SystemExit(f"prompt batch expands to {len(items_x)} != {N}")
    rewards = torch.tensor(
        [verify_mod.reward(int(it["answer"]), t) for it, t in zip(items_x, texts)],
        dtype=torch.float32,
    )
    correct = [verify_mod.is_correct(int(it["answer"]), t) for it, t in zip(items_x, texts)]
    adv, adv_stats = grpo_mod.group_advantages(rewards, G)

    import json as _json
    row = None
    with open(run_dir / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = _json.loads(line)
            if rec.get("rollout_idx") == args.rollout:
                row = rec
                break
    checks = {}
    if row is not None:
        checks = {
            "reward_mean_archive": float(rewards.mean()),
            "reward_mean_run": float(row["reward_mean"]),
            "pass_rate_archive": float(np.mean(correct)),
            "pass_rate_run": float(row["pass_rate"]),
        }
        dr = abs(checks["reward_mean_archive"] - checks["reward_mean_run"])
        dp = abs(checks["pass_rate_archive"] - checks["pass_rate_run"])
        print(f"[check] reward_mean {checks['reward_mean_archive']:.4f} vs run "
              f"{checks['reward_mean_run']:.4f} (d={dr:.5f})")
        print(f"[check] pass_rate   {checks['pass_rate_archive']:.4f} vs run "
              f"{checks['pass_rate_run']:.4f} (d={dp:.5f})")
        if dr > 1e-4 or dp > 1e-4:
            raise SystemExit(
                "reward rebuild does not reproduce the run: the prompt batch or the "
                "scorer has drifted, so every sigma below would be measured on "
                "different data than the training runs. Refusing to report."
            )
    else:
        print(f"[warn] no metrics row for rollout {args.rollout}; skipping the self-check")

    # ---- 3. old log-probs under the policy that GENERATED D ------------------
    # Rollout 0 is sampled before any optimizer step, so the producing policy is
    # the freshly built one (PEFT initialises lora_B to zero => the base model).
    n_groups = N // grp
    if args.limit_groups:
        n_groups = min(n_groups, args.limit_groups)
    print(f"[old] scoring {n_groups * grp} responses under the initial policy ...")
    old_by_group: list[torch.Tensor] = []
    with torch.no_grad():
        for gi in range(n_groups):
            sl = slice(gi * grp, (gi + 1) * grp)
            old_by_group.append(policy.score(pids[sl], cids[sl]).detach().float().cpu())
    print(f"[old] done, {len(old_by_group)} groups")

    # ---- 4. per-adapter sweep ------------------------------------------------
    results: dict[str, dict] = {}
    for tag in args.adapters:
        adir = run_dir / "adapters" / f"r{tag}"
        if not adir.exists():
            raise SystemExit(f"missing adapter snapshot: {adir}")
        how = load_adapter(policy, adir)
        policy.eval()
        print(f"\n[adapter r{tag}] loaded via {how}")

        bufs = {k: train_mod._grad_buffer(policy) for k in SPLITS}
        seg_norms = {k: [] for k in SPLITS}   # norms of the current segment
        seg_sigma: dict[str, list[float]] = {k: [] for k in SPLITS}
        g_norms: list[float] = []

        n_groups = N // grp
        if args.limit_groups:
            n_groups = min(n_groups, args.limit_groups)

        for gi in range(n_groups):
            sl = slice(gi * grp, (gi + 1) * grp)
            mask = policy.mask_for(cids[sl])
            lp = policy.score(pids[sl], cids[sl], requires_grad=True)
            old = old_by_group[gi].to(policy.device)
            a = adv[sl].to(policy.device)
            loss, _ = grpo_mod.grpo_loss(
                lp, old, a, mask,
                eps_low=eps_low, eps_high=eps_high,
                clip_delta=clip_delta, norm=norm_name,
            )
            policy.model.zero_grad(set_to_none=True)
            loss.backward()
            gn = train_mod._grad_norm(train_mod._trainable(policy))
            g_norms.append(gn)
            del lp, loss, mask

            for k, size in SPLITS.items():
                train_mod._accumulate(bufs[k], policy, 1.0)
                seg_norms[k].append(gn)
                if len(seg_norms[k]) == size:
                    train_mod._rescale(bufs[k], 1.0 / size)
                    g_bar = train_mod._grad_norm(bufs[k])
                    gs = grpo_mod.GradStats()
                    for x in seg_norms[k]:
                        gs.add(x)
                    s = gs.sigma(g_bar)
                    if s is not None:
                        seg_sigma[k].append(float(s))
                    train_mod._rescale(bufs[k], 0.0)
                    seg_norms[k] = []

            if (gi + 1) % 32 == 0:
                done = {k: len(v) for k, v in seg_sigma.items()}
                print(f"  [{gi + 1}/{n_groups}] groups  steps so far {done}", flush=True)

        row_out = {"adapter": f"r{tag}", "load": how, "n_groups": n_groups,
                   "sigma": {}, "sigma_sd": {}, "n_steps": {}}
        for k in SPLITS:
            v = seg_sigma[k]
            if not v:
                row_out["sigma"][k] = None
                continue
            row_out["sigma"][k] = float(np.mean(v))
            row_out["sigma_sd"][k] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            row_out["n_steps"][k] = len(v)
        results[f"r{tag}"] = row_out
        torch.cuda.empty_cache()

    # ---- 5. report -----------------------------------------------------------
    keys = list(results)
    print("\n=== sigma table (rows = policy staleness, cols = split) ===")
    print(f"{'adapter':<10}" + "".join(f"{k:>10}" for k in SPLITS) + "     row spread")
    for r in keys:
        vals = [results[r]["sigma"][k] for k in SPLITS]
        have = [v for v in vals if v is not None]
        spread = (max(have) - min(have)) / (sum(have) / len(have)) if len(have) > 1 else 0.0
        print(f"{r:<10}" + "".join(f"{v:>10.4f}" if v is not None else f"{'--':>10}"
                                   for v in vals) + f"   {spread * 100:8.1f}%")

    verdict = {}
    h1_ok = all(
        (lambda vals: (max(vals) - min(vals)) / (sum(vals) / len(vals)) < 0.05)(
            [results[r]["sigma"][k] for k in SPLITS if results[r]["sigma"][k] is not None])
        for r in keys
        if sum(results[r]["sigma"][k] is not None for k in SPLITS) == len(SPLITS)
    )
    verdict["H1_split_free"] = bool(h1_ok)
    if len(keys) >= 2:
        first, last = results[keys[0]]["sigma"], results[keys[-1]]["sigma"]
        for k in SPLITS:
            if first[k] and last[k]:
                verdict[f"ratio_{keys[-1]}_over_{keys[0]}_{k}"] = last[k] / first[k]
        ratios = [last[k] / first[k] for k in SPLITS if first[k] and last[k]]
        verdict["H2_staleness"] = bool(ratios and min(ratios) >= 1.20)
        print("\n=== verdict (thresholds from C9) ===")
        print(f"  H1 split-free   (row spread < 5%)      : {verdict['H1_split_free']}")
        for k in SPLITS:
            if first[k] and last[k]:
                print(f"  sigma({keys[-1]})/sigma({keys[0]}) at {k:<5}: {last[k] / first[k]:.3f}")
        print(f"  H2 staleness    (ratio >= 1.20)        : {verdict['H2_staleness']}")
        if verdict["H1_split_free"] and verdict["H2_staleness"]:
            print("  => sigma's mu dependence is off-policy degree, not the split.")
        elif not verdict["H1_split_free"]:
            print("  => the split itself moves sigma; the original premise holds after all.")
        else:
            print("  => neither explains it; sigma stays an unexplained confound.")

    payload = {
        "claim": "C9",
        "run": str(run_dir),
        "rollout": args.rollout,
        "config": args.config,
        "n_responses": N,
        "group_size": grp,
        "adapters": args.adapters,
        "self_check": checks,
        "adv_stats": adv_stats,
        "results": results,
        "verdict": verdict,
        "splits": SPLITS,
    }
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
