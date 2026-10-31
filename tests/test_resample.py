"""Unit tests for the resample arms (B/C): verdicts, refill, determinism.

Everything runs on CPU with scripted/fake backends -- no GPU, no vLLM. The
determinism contract under test: the whole procedure is a pure function of
(seed, rollout_idx, pool, mode, cap), and round 1 sees exactly the arm-A
prompt batch.

    python tests/run_all.py     (or: pytest tests/test_resample.py -q)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config as config_mod  # noqa: E402
import data as data_mod  # noqa: E402
import resample  # noqa: E402
from rollout import FakeBackend, Rollout  # noqa: E402


def _pool(n: int) -> list[dict]:
    return [{"prompt": f"pool item {i}", "answer": i, "depth": 1 + i % 3} for i in range(n)]


class ScriptBackend:
    """Deterministic scripted backend: per-prompt group behaviour by callback.

    kind(item, attempt, pos_in_round) -> 'all_correct' | 'all_wrong' | 'mixed'
    'mixed' alternates correct/wrong inside the group, so the composed reward
    has variance 1.0 and accuracy 0.5 under either filter.
    Records every call's items for the round-1 pairing assertions.
    """

    name = "script"

    def __init__(self, kind_fn):
        self.kind_fn = kind_fn
        self.calls: list[list[dict]] = []

    def generate(self, items, G, rollout_idx, seed=0, attempt=0, response_offset=0):
        self.calls.append(list(items))
        texts, pids, cids = [], [], []
        for i, it in enumerate(items):
            kind = self.kind_fn(it, attempt, i)
            for g in range(G):
                gold = int(it["answer"])
                if kind == "all_correct":
                    val = gold
                elif kind == "all_wrong":
                    val = gold + 7
                else:
                    val = gold if g % 2 == 0 else gold + 1
                texts.append(f"Reasoning here.\nAnswer: {val}")
                pids.append([1 + (hash(it["prompt"]) % 100), 2, 3])
                cids.append([attempt * 10_000 + i * 10 + g, response_offset % 97])
        return Rollout(
            items=[it for it in items for _ in range(G)],
            prompt_ids=pids,
            completion_ids=cids,
            completion_texts=texts,
            group_size=G,
            wall_ms=1.0,
            backend="script",
        )


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def test_verdicts_accuracy():
    rewards = [[1.1] * 4, [-0.1] * 4, [1.1, 0.1, 1.1, 0.1]]
    correct = [[True] * 4, [False] * 4, [True, False, True, False]]
    keep = resample.group_verdicts(rewards, "accuracy", correct)
    assert keep == [False, False, True], keep


def test_verdicts_accuracy_needs_correct():
    try:
        resample.group_verdicts([[1.0, 0.0]], "accuracy", None)
    except ValueError:
        return
    raise AssertionError("accuracy mode without group_correct must raise")


def test_verdicts_variance():
    # all-correct group: composed rewards identical (1.1 each) -> zero variance
    # all-wrong group: -0.1 each -> zero variance (format bonus does NOT rescue
    # acc=0 groups into nonzero variance; that is exactly why B needs the
    # correctness counter and C uses the reward directly)
    rewards = [[1.1] * 4, [-0.1] * 4, [1.1, 0.1, -0.1, 1.1], [0.9, 0.9, 0.9, 0.95]]
    keep = resample.group_verdicts(rewards, "variance", None)
    assert keep == [False, False, True, True], keep


def test_verdicts_none_keeps_all():
    assert resample.group_verdicts([[0.0], [1.0, 2.0]], "none", None) == [True, True]


def test_verdicts_var_threshold_matches_grpo():
    import grpo
    import torch

    # a group whose std is just under/over 1e-8: the filter and the advantage
    # zeroing must agree bit for bit
    g = torch.tensor([1.0, 1.0 + 1e-9, 1.0, 1.0])
    std = g.std(unbiased=True).item()
    keep = resample.group_verdicts([g.tolist()], "variance", None)
    zeroed = std < 1e-8
    assert keep[0] == (not zeroed), (std, keep, zeroed)
    adv, _ = grpo.group_advantages(torch.cat([g, g, g, g]), 4)
    assert float(adv.abs().sum()) == 0.0 if zeroed else float(adv.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# permutation sync with data.prompt_batch
# ---------------------------------------------------------------------------


def test_permutation_prefix_matches_prompt_batch():
    pool = _pool(50)
    for seed in (0, 7, 12345):
        for r in (0, 3, 17):
            perm = data_mod.rollout_permutation(len(pool), seed, r)
            batch = data_mod.prompt_batch(pool, seed, r, 10)
            assert [p["prompt"] for p in batch] == [pool[i]["prompt"] for i in perm[:10]]


# ---------------------------------------------------------------------------
# refill loop with a scripted backend
# ---------------------------------------------------------------------------


def _pattern_backend(cycle="all_wrong", mixed_period=3):
    """Every `mixed_period`-th prompt (by global pool name order) is mixed."""

    def kind(it, attempt, pos):
        idx = int(it["prompt"].split()[-1])
        return "mixed" if idx % mixed_period == 0 else cycle

    return ScriptBackend(kind)


def test_refill_fills_target_and_counts():
    pool = _pool(60)
    be = _pattern_backend()
    merged, stats = resample.rollout_with_resample(
        pool, seed=0, rollout_idx=0, G=4, n_target_groups=8,
        backend=be, mode="accuracy", max_extra_prompts=100, log=lambda *_: None,
    )
    assert merged.n_prompts == 8
    assert merged.n_responses == 32
    assert stats["resample_kept_groups"] == 8
    assert stats["resample_attempted_responses"] == 32 + stats["resample_discarded_responses"]
    assert stats["resample_attempted_prompts"] == sum(len(c) for c in be.calls)
    assert stats["resample_attempted_responses"] == 4 * stats["resample_attempted_prompts"]
    assert not stats["resample_capped"]
    # every kept group really is mixed under the accuracy filter
    G = 4
    for gi in range(8):
        chunk = merged.completion_texts[gi * G : (gi + 1) * G]
        corr = [t.endswith(str(int(t.split()[-1]))) for t in chunk]  # unused; sanity parse below
        vals = [int(t.rsplit("Answer: ", 1)[1]) for t in chunk]
        golds = [int(it["answer"]) for it in merged.items[gi * G : (gi + 1) * G]]
        acc = sum(v == g for v, g in zip(vals, golds)) / G
        assert 0.0 < acc < 1.0, (gi, acc)


def test_round1_is_arm_a_batch():
    pool = _pool(60)
    be = _pattern_backend()
    resample.rollout_with_resample(
        pool, seed=5, rollout_idx=3, G=4, n_target_groups=8,
        backend=be, mode="accuracy", max_extra_prompts=100, log=lambda *_: None,
    )
    arm_a = data_mod.prompt_batch(pool, 5, 3, 8)
    assert be.calls[0] == arm_a


def test_refill_comes_from_later_positions_and_is_pairwise_distinct():
    pool = _pool(60)
    be = _pattern_backend()
    resample.rollout_with_resample(
        pool, seed=5, rollout_idx=3, G=4, n_target_groups=8,
        backend=be, mode="accuracy", max_extra_prompts=100, log=lambda *_: None,
    )
    perm = data_mod.rollout_permutation(len(pool), 5, 3)
    first = {p["prompt"] for p in be.calls[0]}
    for round_items in be.calls[1:]:
        for p in round_items:
            assert p["prompt"] not in first, "refill must not repeat the initial batch"
    # all prompts across rounds follow the one permutation, in order
    drawn = [pool[perm_i]["prompt"] for perm_i in perm[: sum(len(c) for c in be.calls)]]
    flattened = [p["prompt"] for c in be.calls for p in c]
    assert drawn == flattened


def test_determinism_scripted():
    pool = _pool(60)
    outs = []
    for _ in range(2):
        be = _pattern_backend()
        merged, stats = resample.rollout_with_resample(
            pool, seed=11, rollout_idx=2, G=4, n_target_groups=8,
            backend=be, mode="variance", max_extra_prompts=100, log=lambda *_: None,
        )
        outs.append((merged.completion_ids, merged.prompt_ids, dict(stats)))
    assert outs[0][0] == outs[1][0]
    assert outs[0][1] == outs[1][1]
    s0, s1 = outs[0][2], outs[1][2]
    for k in s0:
        if k.startswith("discarded_"):
            continue
        assert s0[k] == s1[k], k


def test_cap_triggers_and_reports():
    pool = _pool(60)
    be = _pattern_backend(cycle="all_wrong", mixed_period=5)  # heavy rejection
    merged, stats = resample.rollout_with_resample(
        pool, seed=0, rollout_idx=0, G=4, n_target_groups=20,
        backend=be, mode="accuracy", max_extra_prompts=2, log=lambda *_: None,
    )
    assert stats["resample_capped"]
    assert merged.n_prompts < 20
    assert merged.n_prompts > 0
    assert stats["resample_kept_groups"] == merged.n_prompts


def test_discarded_rows_returned():
    pool = _pool(60)
    be = _pattern_backend()
    merged, stats = resample.rollout_with_resample(
        pool, seed=0, rollout_idx=0, G=4, n_target_groups=8,
        backend=be, mode="accuracy", max_extra_prompts=100, log=lambda *_: None,
    )
    assert stats["resample_discarded_responses"] == len(stats["discarded_completion_ids"])
    assert stats["resample_discarded_responses"] == len(stats["discarded_completion_texts"])
    assert stats["resample_attempted_responses"] == (
        merged.n_responses + stats["resample_discarded_responses"]
    )


# ---------------------------------------------------------------------------
# determinism with the real FakeBackend (rng-driven, not scripted)
# ---------------------------------------------------------------------------


def test_determinism_fake_backend():
    cfg = config_mod.Config({})
    pool = _pool(80)

    def run():
        be = FakeBackend(cfg, None, acc=0.4)
        merged, stats = resample.rollout_with_resample(
            pool, seed=9, rollout_idx=1, G=4, n_target_groups=10,
            backend=be, mode="accuracy", max_extra_prompts=200, log=lambda *_: None,
        )
        return merged.completion_ids, merged.completion_texts, stats["resample_rounds"]

    a, b = run(), run()
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


if __name__ == "__main__":
    for k, fn in sorted(list(globals().items())):
        if k.startswith("test_") and callable(fn):
            fn()
            print("ok", k)
