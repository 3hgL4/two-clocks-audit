"""Metrics records: one JSONL line per (rollout, mu-step).

Design rule from the protocol: *record raw quantities, derive later.* The
writer therefore stores the ingredients of every clock, not the conclusions:

  drift clock      d = kappa * ||g_bar|| / sigma        (Eq. 3 ingredients:
                   grad_norm_raw = ||g_bar||, grad_noise_sigma = sigma)
  progress clock   r(b) needs the reward/response-length trajectory, stored as
                   pass_rate and resp_len_mean per mu-step
  staleness cost   kl_per_rollout_policy = mean(old_logp - logp) over the
                   completion tokens: exactly the quantity Assumption A2 of
                   arXiv:2610.18853 sets aside
  clipping budget  clip_frac_* and grad_norm_clipped/grad_norm_raw
                   (the paper's "0.5% of tokens but 50-70% of grad norm")

Nothing here is aggregated across rollouts; the analysis scripts do that.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# field -> (type, required)
SCHEMA: dict[str, tuple[type | tuple[type, ...], bool]] = {
    "ts": (float, True),
    "run_id": (str, True),
    "arm": (str, True),
    "seed": (int, True),
    "rollout_idx": (int, True),
    "mu": (int, True),
    "mu_step": (int, True),
    "global_step": (int, True),
    "b_responses": (int, True),
    "br_responses": (int, True),
    "prompts_per_rollout": (int, True),
    "n_generations": (int, True),
    # rollout-side
    "resp_len_mean": (float, True),
    "completion_tokens": (int, True),
    "reward_mean": (float, True),
    "pass_rate": (float, True),
    # `zero_var_group_frac` counts groups whose *reward* is identical, and reward
    # carries format bonuses, so it measures "all-wrong and format-stable" on a
    # weak policy and "all-right" on a strong one. The four fields below are the
    # correctness-defined version and are optional so rows written before
    # 2026-10-26 still validate; see src/prompt_log.py:group_summary.
    "zero_var_group_frac": (float, True),
    "zero_var_correct_frac": (float, False),
    "group_all_correct_frac": (float, False),
    "group_all_wrong_frac": (float, False),
    "group_icc": ((float, type(None)), False),
    "adv_abs_mean": (float, True),
    "wall_ms_rollout": (float, True),
    # D-24: old-policy forward time, booked once per rollout on row 0. Part of
    # wall_ms_update; optional so rows written before the fix still validate.
    "wall_ms_old_logp": (float, False),
    # update-side
    "loss": (float, True),
    "lr": (float, True),
    "ratio_mean": (float, True),
    "ratio_p95": (float, True),
    "kl_per_rollout_policy": (float, True),
    "clip_frac_low": (float, True),
    "clip_frac_high": (float, True),
    "clip_frac_any": (float, True),
    # C6 second half: share of the *logprob-space* gradient mass |ratio*adv| that
    # the clipped tokens would have carried. Optional so pre-D-21 rows still pass.
    "clip_grad_mass_share": (float, False),
    "grad_norm_raw": (float, True),
    "grad_norm_clipped": (float, True),
    "grad_noise_sigma": (float, True),
    "n_groups": (int, True),
    "n_forward_chunks": (int, True),
    "n_micro_batches": (int, True),  # legacy alias of n_forward_chunks
    # digest of rollouts/r{idx}.npz (DEVLOG D-17). Optional: records written
    # before the archive existed have no such field.
    "rollout_data_sha": (str, False),
    "drift_d": ((float, type(None)), True),
    "wall_ms_update": (float, True),
    # cumulative
    "responses_cum": (int, True),
    "flops_cum": (float, True),
    "cost_cny_cum": (float, True),
}


class SchemaError(ValueError):
    pass


def check_record(rec: dict, *, allow_extra: bool = True) -> None:
    missing = [k for k, (_, req) in SCHEMA.items() if req and k not in rec]
    if missing:
        raise SchemaError(f"record missing fields: {missing}")
    for key, (expected, _) in SCHEMA.items():
        if key not in rec:
            continue
        val = rec[key]
        if val is None and expected is not type(None):
            continue  # nulls allowed for nullable fields
        if not isinstance(val, expected):
            raise SchemaError(
                f"field {key!r}: expected {expected}, got {type(val).__name__} ({val!r})"
            )
    if not allow_extra:
        extra = set(rec) - set(SCHEMA)
        if extra:
            raise SchemaError(f"unexpected fields: {sorted(extra)}")


class MetricsWriter:
    """Append-only JSONL with resume support.

    `truncate_to_rollout(k)` drops any record belonging to rollout >= k, which is
    what a resume needs: a rollout that was interrupted mid-mu must be replayed
    from scratch (its optimizer state was never checkpointed), otherwise the
    (rollout, mu_step) keys would be duplicated.
    """

    def __init__(self, path: str | Path, *, fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self.n_written = 0

    def write(self, rec: dict, *, validate: bool = True) -> None:
        if validate:
            check_record(rec)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n_written += 1
        if self.fsync:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- resume helpers -----------------------------------------------------
    def truncate_to_rollout(self, rollout_idx: int) -> int:
        """Keep only records with rollout_idx < k. Returns number kept."""
        if not self.path.exists():
            return 0
        self._fh.close()
        keep: list[str] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a killed process
                if int(rec.get("rollout_idx", -1)) < rollout_idx:
                    keep.append(line)
        with open(self.path, "w", encoding="utf-8") as f:
            for line in keep:
                f.write(line + "\n")
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        return len(keep)


def read_metrics(path: str | Path) -> list[dict]:
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def now_ts() -> float:
    return time.time()
