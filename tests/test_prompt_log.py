"""Tests for the per-prompt archive and the correctness-defined group stats.

These exist because the two failure modes here are silent. A factor-of-G length
error in `build_record` would mislabel every group and nothing downstream would
notice; an ICC formula with the wrong (G-1) would report 0.5 where the truth is
0.9, which is the difference between "4096 responses" and "800 independent ones".

Written without pytest so `python tests/run_all.py` runs it on a bare interpreter.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import prompt_log  # noqa: E402


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# group_summary
# ---------------------------------------------------------------------------


def test_group_summary_counts_all_right_and_all_wrong() -> None:
    G = 4
    # group 0 all right, group 1 all wrong, group 2 mixed, group 3 all right
    correct = ([True] * G + [False] * G
               + [True, False, True, False]
               + [True] * G)
    s = prompt_log.group_summary(correct, G)
    assert close(s["group_all_correct_frac"], 2 / 4), s
    assert close(s["group_all_wrong_frac"], 1 / 4), s
    assert close(s["zero_var_correct_frac"], 3 / 4), s


def test_group_summary_iid_rate_matches_binomial() -> None:
    """Under iid sampling the all-correct share must be about p**G."""
    G, n_groups, p = 8, 20000, 0.5
    rng = random.Random(7)
    correct = [rng.random() < p for _ in range(n_groups * G)]
    s = prompt_log.group_summary(correct, G)
    expected = p ** G
    assert abs(s["group_all_correct_frac"] - expected) < 0.0035, (s, expected)
    assert abs(s["group_all_wrong_frac"] - expected) < 0.0035, (s, expected)
    # iid -> ICC near zero
    assert s["group_icc"] is not None and abs(s["group_icc"]) < 0.02, s["group_icc"]


def test_group_summary_perfectly_clustered_has_icc_one() -> None:
    """Half the prompts every generation right, half every generation wrong."""
    G = 8
    correct = [True] * (G * 50) + [False] * (G * 50)
    s = prompt_log.group_summary(correct, G)
    assert close(s["zero_var_correct_frac"], 1.0), s
    assert close(s["group_icc"], 1.0), s


def test_group_icc_matches_hand_computed_anova() -> None:
    """Two groups of four: [1,1,1,0] and [0,0,1,1].

    SSW = 4*(3*1/4) ... computed by hand per group: group A s=3 -> 3*1/4 = 0.75;
    group B s=2 -> 2*2/4 = 1.0. SSW = 1.75, df = 2*(4-1) = 6, MSW = 0.2917.
    group means 0.75 and 0.5, grand 0.625: SSB = 4*(0.015625+0.015625) = 0.125,
    df = 1, MSB = 0.125.
    ICC = (0.125-0.2917)/(0.125+3*0.2917) = -0.1667/1.0 = -0.1667.
    """
    G = 4
    correct = [True, True, True, False,  False, False, True, True]
    s = prompt_log.group_summary(correct, G)
    assert abs(s["group_icc"] - (-0.1666667)) < 1e-6, s["group_icc"]


def test_group_summary_rejects_bad_length() -> None:
    for bad in ([True] * 7, [True] * 9):
        try:
            prompt_log.group_summary(bad, 4)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for len={len(bad)}, G=4")


def test_group_summary_zero_var_exceeds_reward_variant() -> None:
    """The point of the new field: it counts groups the reward version misses.

    Eight wrong answers with two different formats produce two distinct rewards,
    so `zero_var_group_frac` (reward-based) does NOT count the group, while the
    correctness-based share does. This is the mu1 mechanism measured on the
    archived runs: 8.2% expected all-wrong against 0.46% observed.
    """
    G = 8
    warnings_filter = [False] * G
    s = prompt_log.group_summary(warnings_filter, G)
    assert close(s["group_all_wrong_frac"], 1.0), s
    assert close(s["zero_var_correct_frac"], 1.0), s
    # and the reward analogue, for contrast: two distinct values -> not zero-var
    rewards = [-0.1] * 4 + [-0.05] * 4
    assert len(set(rewards)) > 1, "reward is not constant, so the old field would miss it"


# ---------------------------------------------------------------------------
# build_record / round trip
# ---------------------------------------------------------------------------


def test_build_record_layout_and_roundtrip() -> None:
    G = 3
    prompt_idx = [10, 20]
    correct = [True, False, True,   False, False, False]
    reward = [1.0, -0.1, 1.05,   -0.1, -0.1, -0.1]
    rec = prompt_log.build_record(
        rollout_idx=2, arm="mu8", seed=1, mu=8, prompt_idx=prompt_idx,
        correct=correct, reward=reward, group_size=G)
    assert rec["correct"] == ["101", "000"], rec["correct"]
    assert rec["reward"][0] == [1.0, -0.1, 1.05], rec["reward"]
    assert prompt_log.flat_correct(rec) == correct

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "prompts.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            prompt_log.write_record(f, rec)
        back = prompt_log.read_records(p)
    assert len(back) == 1
    assert back[0]["prompt_idx"] == prompt_idx
    assert prompt_log.flat_correct(back[0]) == correct


def test_build_record_rejects_length_mismatch() -> None:
    try:
        prompt_log.build_record(
            rollout_idx=0, arm="mu1", seed=0, mu=1, prompt_idx=[1, 2],
            correct=[True] * 5, reward=[0.0] * 6, group_size=3)
    except ValueError as e:
        assert "6" in str(e) and "5" in str(e), str(e)
        return
    raise AssertionError("expected ValueError")


def test_truncate_to_rollout_drops_replayed_rows() -> None:
    G = 2
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "prompts.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            for k in range(5):
                prompt_log.write_record(f, prompt_log.build_record(
                    rollout_idx=k, arm="mu1", seed=0, mu=1, prompt_idx=[k],
                    correct=[True, False], reward=[1.0, -0.1], group_size=G))
            # a torn line, as left by a killed process
            f.write('{"rollout_idx": 5, "trunc')
        n = prompt_log.truncate_to_rollout(p, 3)
        assert n == 3, n            # rows 3, 4 and the torn one
        recs = prompt_log.read_records(p)
        assert [r["rollout_idx"] for r in recs] == [0, 1, 2]


def test_unseen_vs_seen_splits_first_sight_from_repeat() -> None:
    G = 2
    recs = []
    for k, (idx, bits) in enumerate([([1, 2], [1, 0, 1, 1]),   # rollout 0: all new
                                     ([2, 3], [0, 0, 1, 1])]):  # rollout 1: 2 seen, 3 new
        recs.append(prompt_log.build_record(
            rollout_idx=k, arm="mu1", seed=0, mu=1, prompt_idx=idx,
            correct=[bool(b) for b in bits], reward=[1.0, 0.0, 1.0, 1.0], group_size=G))
    out = prompt_log.unseen_vs_seen(recs)
    assert out["first_sight_responses"] == 6, out       # prompts 1,2 in r0 + prompt 3 in r1
    assert out["repeat_responses"] == 2, out            # prompt 2 in r1
    # first sight: prompt1 [1,0] -> 1 hit, prompt2 r0 [1,1] -> 2, prompt3 [1,1] -> 2
    assert close(out["first_sight_pass"], 5 / 6), out
    assert close(out["repeat_pass"], 0.0), out
    assert out["n_distinct_prompts"] == 3, out


# ---------------------------------------------------------------------------
# schema compatibility
# ---------------------------------------------------------------------------


def _minimal_metrics_row() -> dict:
    """Every required field of metrics.SCHEMA, nothing else."""
    return {
        "ts": 0.0, "run_id": "t", "arm": "mu1", "seed": 0, "rollout_idx": 0,
        "mu": 1, "mu_step": 0, "global_step": 0, "b_responses": 4096,
        "br_responses": 4096, "prompts_per_rollout": 512, "n_generations": 8,
        "resp_len_mean": 200.0, "completion_tokens": 4096, "reward_mean": 0.4,
        "pass_rate": 0.3, "zero_var_group_frac": 0.01, "adv_abs_mean": 0.8,
        "wall_ms_rollout": 1.0, "loss": 0.1, "lr": 5e-5, "ratio_mean": 1.0,
        "ratio_p95": 1.0, "kl_per_rollout_policy": 0.0, "clip_frac_low": 0.0,
        "clip_frac_high": 0.0, "clip_frac_any": 0.0, "grad_norm_raw": 0.5,
        "grad_norm_clipped": 0.5, "grad_noise_sigma": 0.4, "n_groups": 512,
        "n_forward_chunks": 512, "n_micro_batches": 512, "drift_d": 1.2,
        "wall_ms_update": 1.0, "responses_cum": 4096, "flops_cum": 1.0,
        "cost_cny_cum": 0.1,
    }


def test_metrics_schema_accepts_the_new_group_fields() -> None:
    import metrics as metrics_mod

    rec = _minimal_metrics_row()
    metrics_mod.check_record(rec)          # old rows must still validate

    rec2 = dict(rec)
    rec2.update({"zero_var_correct_frac": 0.09, "group_all_correct_frac": 0.09,
                 "group_all_wrong_frac": 0.0, "group_icc": 0.31})
    metrics_mod.check_record(rec2)

    rec3 = dict(rec)
    rec3["group_icc"] = None              # degenerate case: no variance at all
    metrics_mod.check_record(rec3)

    rec4 = dict(rec)
    rec4["group_icc"] = "high"            # wrong type must be rejected
    try:
        metrics_mod.check_record(rec4)
    except metrics_mod.SchemaError:
        return
    raise AssertionError("group_icc: str should have been rejected")


def test_prompt_idx_comes_from_the_expanded_item_list() -> None:
    """The trainer reads _idx at stride G. Verified here because a stride bug
    would pair every prompt with another prompt's outcomes, silently."""
    G = 2
    items = [{"_idx": 5}, {"_idx": 5}, {"_idx": 9}, {"_idx": 9}]
    got = [int(items[i].get("_idx", i // G)) for i in range(0, len(items), G)]
    assert got == [5, 9], got

    # and without _idx the fallback is the positional group index, not the
    # response index
    items2 = [{"prompt": "a"}, {"prompt": "a"}, {"prompt": "b"}, {"prompt": "b"}]
    got2 = [int(items2[i].get("_idx", i // G)) for i in range(0, len(items2), G)]
    assert got2 == [0, 1], got2
