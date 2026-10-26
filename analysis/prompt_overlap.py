"""How much of the prompt pool does each run actually see, and does it repeat?

Why this exists: the pool is 2000 items and every rollout draws 512 of them
without replacement, so a run is *not* 8 disjoint batches. Two rollouts overlap
by ~131 prompts in expectation, and by r7 each prompt has been seen ~2x. Any
claim of the form "pass kept rising" has to survive the objection that the
policy is re-reading prompts it has already been trained on, and that objection
grows linearly with the horizon -- which matters before extending a run to 19
rollouts (projected ~4.9x).

Measured, not assumed: the archived `rollouts/rNNN.npz` files hold the exact
token ids that were sampled, so a prompt is identified by hashing those ids.
Two rollouts that drew the same pool item produce the same token sequence and
therefore the same hash -- no tokenizer needed, and no trust in the sampler's
bookkeeping.

    python analysis/prompt_overlap.py [--runs runs]

Needs numpy. Pure analysis: writes nothing.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np


def rollout_prompt_hashes(npz_path: str) -> list[str]:
    """One hash per *distinct* prompt in this rollout (512 of them at Br=4096, G=8)."""
    z = np.load(npz_path)
    flat = z["prompt_flat"]
    lens = z["prompt_len"]
    off = np.concatenate([[0], np.cumsum(lens)])
    h = [hashlib.sha1(flat[off[i]:off[i + 1]].tobytes()).hexdigest()[:16]
         for i in range(len(lens))]
    # prompt_ids is laid out per *response*, so every prompt appears G=8 times
    # back to back. Dedupe in order: this function is about prompts, not rows.
    seen, uniq = set(), []
    for x in h:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    print(f"{'run':10s} {'rollouts':>8s} {'slots':>6s} {'distinct':>8s} "
          f"{'reuse':>6s} {'ovl(r,r+1)':>11s} {'ovl(r0,r7)':>11s}")
    out = {}
    for d in sorted(glob.glob(os.path.join(a.runs, "mu*_s*"))):
        tag = os.path.basename(d)
        per_roll = {}
        for f in sorted(glob.glob(os.path.join(d, "rollouts", "r*.npz"))):
            base = os.path.basename(f)
            if "_v" in base:            # a resumed run re-sampled this index
                continue
            per_roll[int(base[1:4])] = rollout_prompt_hashes(f)
        if not per_roll:
            continue
        keys = sorted(per_roll)
        slots = [h for k in keys for h in per_roll[k]]
        union = set(slots)
        consec = [len(set(per_roll[keys[i]]) & set(per_roll[keys[i + 1]])) / 512
                  for i in range(len(keys) - 1)]
        r0r7 = (len(set(per_roll[keys[0]]) & set(per_roll[keys[-1]])) / 512
                if len(keys) > 1 else float("nan"))
        rec = {"rollouts": len(keys), "slots": len(slots), "distinct": len(union),
               "reuse": len(slots) / len(union), "overlap_consec": consec,
               "overlap_r0_r7": r0r7}
        out[tag] = rec
        print(f"{tag:10s} {len(keys):8d} {len(slots):6d} {len(union):8d} "
              f"{rec['reuse']:6.2f} {sum(consec)/len(consec):11.3f} {r0r7:11.3f}")

    # what a 19-rollout extension would cost in reuse, holding the pool fixed
    pool = 2000
    print(f"\nprojection for a fixed pool of {pool} (birthday model, 512/rollout):")
    for n in (8, 12, 19):
        slots = n * 512
        distinct = pool * (1 - (1 - 1 / pool) ** slots)
        print(f"  {n:2d} rollouts = {slots:5d} slots -> ~{distinct:6.0f} distinct "
              f"({distinct/pool*100:.1f}% of pool), reuse {slots/distinct:.2f}x")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
