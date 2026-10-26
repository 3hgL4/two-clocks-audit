"""Per-prompt archive: the grain the aggregate metrics threw away.

`metrics.jsonl` records one row per (rollout, mu-step) with quantities already
averaged over 4096 responses. That is enough for the clocks -- they are means by
construction -- but it makes a whole family of questions unanswerable after the
fact: how many prompts are all-wrong, how correlated are the eight generations
of one prompt, does accuracy depend on whether the prompt was seen before.

The nine archived runs cannot answer them. `samples.jsonl` kept the first eight
responses of each rollout, which by layout is *one prompt's whole group*, so the
archive holds 72 groups in total, not 36,864. This module exists so the next set
of runs does not have the same hole, and it is written now, before that data
exists, rather than after seeing what the numbers would be.

Two things live here:

  `group_summary`  turns the flat per-response `correct` list into the four group
                   statistics that go into the metrics row. The important one is
                   `zero_var_correct_frac`: the existing `zero_var_group_frac`
                   counts groups whose *reward* is identical, and reward is
                   continuous (format bonuses -0.1 / 0.05 / 0.1 on wrong answers,
                   1.0 / 1.05 / 1.1 on right ones), so a group of eight wrong
                   answers with two different formats is not counted. That makes
                   the old field measure "all-wrong *and* format-stable" on a weak
                   policy and "all-right" on a strong one -- not one thing.

  `build_record`   one JSONL line per rollout carrying prompt ids and per-prompt
                   outcomes. ~50 KiB per rollout at 512x8, against 25-30 MiB for
                   the token-level npz archive.

    python -c "from src import prompt_log"   # importable, stdlib only
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# group statistics
# ---------------------------------------------------------------------------


def group_summary(correct: Sequence[bool], group_size: int) -> dict:
    """Four group-level statistics from a flat per-response correctness list.

    `correct` is in prompt-major order: [p0_g0..p0_g{ G-1}, p1_g0, ...], which is
    what `rollout._expand` produces and what the trainer scores.

    Definitions:
      group_all_correct_frac  share of groups with all G responses right
      group_all_wrong_frac    share with all G wrong
      zero_var_correct_frac   sum of the two -- "no signal in this group" when
                              signal is defined on correctness rather than reward
      group_icc               intraclass correlation of the 0/1 outcomes,
                              one-way random effects: (MSB-MSW)/(MSB+(G-1)MSW).
                              0 means the eight samples behave like independent
                              draws; 1 means they are one draw repeated. This is
                              the number that says how much of the nominal 4096
                              responses the run actually bought.
    """
    n = len(correct)
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    if n % group_size:
        raise ValueError(f"{n} responses not divisible by group_size {group_size}")
    n_groups = n // group_size

    all_right = all_wrong = 0
    group_sums: list[int] = []
    for i in range(n_groups):
        chunk = [1 if c else 0 for c in correct[i * group_size:(i + 1) * group_size]]
        s = sum(chunk)
        group_sums.append(s)
        if s == group_size:
            all_right += 1
        elif s == 0:
            all_wrong += 1

    # Grand mean on the 0/1 scale. group_sums counts successes per group, so it
    # must be divided by the *responses* per group, not by the group count --
    # getting this wrong scales SSB by G^2 and silently turns ICC into nonsense.
    grand = sum(group_sums) / (n_groups * group_size)
    ssb = 0.0
    for s in group_sums:
        gm = s / group_size
        ssb += group_size * (gm - grand) ** 2

    # SSW for 0/1 values has a closed form: sum over responses of (x - group
    # mean)^2 = s*(1-gm)^2 + (G-s)*gm^2, and gm = s/G, so it is s*(G-s)/G.
    ssw = sum(s * (group_size - s) / group_size for s in group_sums)

    msb = ssb / (n_groups - 1) if n_groups > 1 else 0.0
    msw = ssw / (n_groups * (group_size - 1))
    denom = msb + (group_size - 1) * msw
    icc = None if abs(denom) < 1e-12 else (msb - msw) / denom

    return {
        "group_all_correct_frac": all_right / n_groups,
        "group_all_wrong_frac": all_wrong / n_groups,
        "zero_var_correct_frac": (all_right + all_wrong) / n_groups,
        "group_icc": icc,
    }


# ---------------------------------------------------------------------------
# per-rollout archive
# ---------------------------------------------------------------------------


def _pack_bits(values: Iterable[bool]) -> str:
    return "".join("1" if v else "0" for v in values)


def _unpack_bits(s: str) -> list[bool]:
    return [c == "1" for c in s]


def build_record(*, rollout_idx: int, arm: str, seed: int, mu: int,
                 prompt_idx: Sequence[int], correct: Sequence[bool],
                 reward: Sequence[float], group_size: int) -> dict:
    """One JSONL line: 512 prompt ids, 4096 correctness bits, 4096 rewards.

    `correct` and `reward` are flat in prompt-major order; `prompt_idx` has one
    entry per prompt. The three lengths are checked because a silent length
    mismatch here would mislabel every group and nothing downstream would notice.
    """
    n_prompts = len(prompt_idx)
    n_resp = len(correct)
    if len(reward) != n_resp:
        raise ValueError(f"reward has {len(reward)} entries, correct has {n_resp}")
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2, got {group_size}")
    if n_resp != n_prompts * group_size:
        raise ValueError(
            f"{n_resp} responses != {n_prompts} prompts x {group_size} generations")

    bits = _pack_bits(correct)
    return {
        "rollout_idx": int(rollout_idx),
        "arm": arm,
        "seed": int(seed),
        "mu": int(mu),
        "n_prompts": n_prompts,
        "group_size": int(group_size),
        "prompt_idx": [int(i) for i in prompt_idx],
        "correct": [bits[i * group_size:(i + 1) * group_size] for i in range(n_prompts)],
        "reward": [[round(float(r), 4) for r in reward[i * group_size:(i + 1) * group_size]]
                   for i in range(n_prompts)],
    }


def write_record(fh, rec: dict) -> None:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_records(path: str | Path) -> list[dict]:
    p = Path(path)
    out: list[dict] = []
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def truncate_to_rollout(path: str | Path, rollout_idx: int) -> int:
    """Keep rows with rollout_idx < k. Returns the number of rows dropped.

    Same contract as `MetricsWriter.truncate_to_rollout`, for the same reason: a
    resumed run replays the rollout it was killed in, so without this the archive
    would carry two rows for that index and every per-prompt aggregate would
    double-count it. Call it before opening the file for append.
    """
    p = Path(path)
    if not p.exists():
        return 0
    keep: list[str] = []
    dropped = 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1          # torn last line from a killed process
                continue
            if int(rec.get("rollout_idx", -1)) < rollout_idx:
                keep.append(line)
            else:
                dropped += 1
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for line in keep:
            f.write(line + "\n")
    return dropped


def flat_correct(rec: dict) -> list[bool]:
    """Undo the per-group packing back to the flat prompt-major list."""
    out: list[bool] = []
    for s in rec["correct"]:
        out.extend(_unpack_bits(s))
    return out


def unseen_vs_seen(records: Sequence[dict]) -> dict:
    """Accuracy on first-sight vs repeat prompts, given repeated records.

    The pool is smaller than the number of prompts a long run draws (2,000 items
    against 512 per rollout), so by rollout 19 the run is mostly revising: the
    measured 8-rollout coverage is already 89.9-91.5%. Whether that inflates the
    learning curve is a question only per-prompt ids can answer, which is the
    other reason this archive exists.
    """
    seen: set[int] = set()
    first = [0, 0]
    again = [0, 0]
    for rec in sorted(records, key=lambda r: r["rollout_idx"]):
        G = rec["group_size"]
        bits = flat_correct(rec)
        for i, pid in enumerate(rec["prompt_idx"]):
            chunk = bits[i * G:(i + 1) * G]
            hit = sum(chunk)
            if pid in seen:
                again[0] += hit
                again[1] += len(chunk)
            else:
                first[0] += hit
                first[1] += len(chunk)
                seen.add(pid)
    return {
        "first_sight_responses": first[1],
        "first_sight_pass": (first[0] / first[1]) if first[1] else None,
        "repeat_responses": again[1],
        "repeat_pass": (again[0] / again[1]) if again[1] else None,
        "n_distinct_prompts": len(seen),
    }
