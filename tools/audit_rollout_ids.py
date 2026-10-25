#!/usr/bin/env python
"""Audit an archived rollout: group diversity, and bit-for-bit comparison.

Two questions this answers, both of which used to be unanswerable after a run
finished because only 8 of 32768 responses were ever written to disk:

1. **Is the data auditable at all?**  `--compare` diffs two archives row by row.
   That is the L1 determinism probe (DEVLOG D-16): run the same config twice,
   then ask whether the completion ids are identical. Aggregates like pass rate
   are not enough -- they can agree while the underlying tokens differ.

2. **Are the G generations of one prompt actually G independent samples?**  If
   the sampler shares one random stream across a group (all requests in a
   rollout get the same `SamplingParams.seed`, see rollout.py), the samples
   collapse towards copies and the *effective* sample size is prompts, not
   responses -- which would bias sigma and every pass-rate confidence interval.
   `--group` measures exactly that: distinct completions, distinct first tokens,
   and the within-group spread of lengths against the overall spread.

Usage
-----
    python tools/audit_rollout_ids.py runs/mu1_s0/rollouts/r000.npz --group 8
    python tools/audit_rollout_ids.py runs/det_base_a/rollouts/r000.npz \
           --compare runs/det_base_b/rollouts/r000.npz --group 8

Exit code is 0 when the audit ran, 1 when `--compare` found a difference
(so a shell can gate on it), 2 on a usage/IO error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[list[list[int]], list[list[int]]]:
    z = np.load(path)
    p_flat, p_len = z["prompt_flat"], z["prompt_len"]
    c_flat, c_len = z["completion_flat"], z["completion_len"]
    prompts, comps = [], []
    po = co = 0
    for pl, cl in zip(p_len.tolist(), c_len.tolist()):
        prompts.append(p_flat[po : po + pl].tolist())
        po += pl
        comps.append(c_flat[co : co + cl].tolist())
        co += cl
    return prompts, comps


def digest(prompts: list[list[int]], comps: list[list[int]]) -> str:
    """Same digest as src/train.py:save_rollout_ids -- prompt flat, prompt len,
    completion flat, completion len, little endian."""
    h = hashlib.sha256()
    for seqs in (prompts, comps):
        flat = np.fromiter((t for s in seqs for t in s), dtype=np.uint32)
        lens = np.fromiter((len(s) for s in seqs), dtype=np.int32, count=len(seqs))
        h.update(np.ascontiguousarray(flat, dtype="<u4").tobytes())
        h.update(np.ascontiguousarray(lens, dtype="<i4").tobytes())
    return h.hexdigest()


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def diversity(comps: list[list[int]], group: int) -> dict:
    n = len(comps) - len(comps) % group
    uniq, first_tok, dup_groups, all_same = [], [], 0, 0
    within_sd, group_means = [], []
    for g0 in range(0, n, group):
        rows = [tuple(c) for c in comps[g0 : g0 + group]]
        u = len(set(rows))
        uniq.append(u)
        if u < len(rows):
            dup_groups += 1
        if u == 1:
            all_same += 1
        first_tok.append(len({r[0] if r else -1 for r in rows}))
        lens = np.array([len(r) for r in rows], dtype=np.float64)
        within_sd.append(float(lens.std(ddof=1)) if len(lens) > 1 else 0.0)
        group_means.append(float(lens.mean()))
    total_sd = float(np.std([len(c) for c in comps], ddof=1))
    out = {
        "groups": n // group,
        "group_size": group,
        "mean_unique": round(float(np.mean(uniq)), 3),
        "min_unique": int(np.min(uniq)),
        "groups_with_duplicates": dup_groups,
        "groups_all_identical": all_same,
        "mean_unique_first_token": round(float(np.mean(first_tok)), 3),
        "len_sd_within_group_mean": round(float(np.mean(within_sd)), 2),
        "len_sd_overall": round(total_sd, 2),
    }
    out["within_over_total_len_sd"] = (
        round(out["len_sd_within_group_mean"] / total_sd, 3) if total_sd else None
    )
    return out


def compare(a: tuple[list[list[int]], list[list[int]]],
            b: tuple[list[list[int]], list[list[int]]]) -> dict:
    pa, ca = a
    pb, cb = b
    if not (len(pa) == len(pb) == len(ca) == len(cb)):
        return {"comparable": False,
                "reason": f"row count {len(ca)} vs {len(cb)} (prompts {len(pa)} vs {len(pb)})"}
    p_same = [x == y for x, y in zip(pa, pb)]
    c_same = [x == y for x, y in zip(ca, cb)]
    n_diff = sum(not s for s in c_same)
    diff_rows = [i for i, s in enumerate(c_same) if not s]
    detail = []
    for i in diff_rows[:5]:
        detail.append({
            "row": i,
            "len_a": len(ca[i]),
            "len_b": len(cb[i]),
            "common_prefix_tokens": common_prefix(ca[i], cb[i]),
        })
    return {
        "comparable": True,
        "rows": len(pa),
        "prompts_identical": all(p_same),
        "completions_identical": n_diff == 0,
        "rows_differing": n_diff,
        "frac_rows_differing": round(n_diff / max(1, len(pa)), 6),
        "first_rows_differing": detail,
        "mean_len_a": round(float(np.mean([len(c) for c in ca])), 2) if ca else 0.0,
        "mean_len_b": round(float(np.mean([len(c) for c in cb])), 2) if cb else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path)
    ap.add_argument("--compare", type=Path, default=None,
                    help="second archive; report row-by-row identity")
    ap.add_argument("--group", type=int, default=8,
                    help="responses per prompt (G) in this archive")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        a = load(args.npz)
    except Exception as e:  # noqa: BLE001
        print(f"[audit] cannot read {args.npz}: {e}", file=sys.stderr)
        return 2

    report: dict = {
        "npz": str(args.npz),
        "n_responses": len(a[1]),
        "sha256_recomputed": digest(*a),
        "prompt_tokens": int(sum(len(p) for p in a[0])),
        "completion_tokens": int(sum(len(c) for c in a[1])),
        "mean_completion_len": round(float(np.mean([len(c) for c in a[1]])), 2),
    }
    # the digest stored next to the archive is what the metrics row carries
    idx = args.npz.parent / "index.jsonl"
    if idx.exists():
        for line in idx.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("npz") == args.npz.name:
                report["sha256_recorded"] = rec.get("sha256")
                report["sha_matches_record"] = rec.get("sha256") == report["sha256_recomputed"]
                break

    if args.group and args.group > 1:
        report["diversity"] = diversity(a[1], args.group)

    rc = 0
    if args.compare:
        try:
            b = load(args.compare)
        except Exception as e:  # noqa: BLE001
            print(f"[audit] cannot read {args.compare}: {e}", file=sys.stderr)
            return 2
        cmp = compare(a, b)
        cmp["other"] = str(args.compare)
        report["comparison"] = cmp
        if cmp.get("comparable") and not cmp["completions_identical"]:
            rc = 1

    if not args.quiet:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        d = report.get("diversity")
        if d:
            print(f"\n[diversity] {d['groups']} groups of {d['group_size']}: "
                  f"unique {d['mean_unique']}/{d['group_size']} (min {d['min_unique']}), "
                  f"distinct first token {d['mean_unique_first_token']}, "
                  f"within-group len sd {d['len_sd_within_group_mean']} vs overall "
                  f"{d['len_sd_overall']} (ratio {d['within_over_total_len_sd']})")
            if d["mean_unique"] < 0.9 * d["group_size"]:
                print("[diversity] WARNING: groups are far from G distinct samples -> "
                      "effective n is closer to #prompts than #responses. Check how "
                      "SamplingParams.seed is shared before trusting sigma or any CI.")
        c = report.get("comparison")
        if c and c.get("comparable"):
            if c["completions_identical"]:
                print(f"\n[compare] IDENTICAL: {c['rows']} rows, prompts "
                      f"{'same' if c['prompts_identical'] else 'DIFFER'} -> the sampler "
                      f"repeats under this config.")
            else:
                print(f"\n[compare] DIFFER: {c['rows_differing']}/{c['rows']} rows "
                      f"({100*c['frac_rows_differing']:.1f}%), mean len "
                      f"{c['mean_len_a']} vs {c['mean_len_b']}")
                for row in c["first_rows_differing"]:
                    print(f"          row {row['row']}: len {row['len_a']} vs "
                          f"{row['len_b']}, common prefix {row['common_prefix_tokens']} tokens")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
