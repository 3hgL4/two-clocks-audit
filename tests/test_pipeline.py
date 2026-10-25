"""Pipeline-level tests: pool determinism, arm pairing, accounting, config guards.

Written without pytest so the suite runs on a bare Python (`python tests/run_all.py`).
Every function still starts with `test_` and uses bare asserts, so `pytest tests`
works as well once the cloud image has pytest.

The pairing test is the important one. The audit compares mu=1 against mu=8 and
mu=16, and the comparison is only paired if every arm sees byte-identical prompt
sets at the same rollout index. That follows from `data.prompt_batch` being a
pure function of (seed, rollout_idx); it is asserted here rather than trusted.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import accounting  # noqa: E402
import config as config_mod  # noqa: E402
import data as data_mod  # noqa: E402


@contextmanager
def raises(exc_type, match: str = ""):
    try:
        yield
    except exc_type as e:
        if match and match not in str(e):
            raise AssertionError(f"expected message containing {match!r}, got {e!r}") from e
    else:
        raise AssertionError(f"expected {exc_type.__name__} was not raised")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------


def test_prompt_batch_is_deterministic_and_paired() -> None:
    pool = [{"prompt": f"q{i}", "answer": i} for i in range(50)]
    a = [p["prompt"] for p in data_mod.prompt_batch(pool, seed=3, rollout_idx=5, n=16)]
    b = [p["prompt"] for p in data_mod.prompt_batch(pool, seed=3, rollout_idx=5, n=16)]
    assert a == b, "same (seed, rollout_idx) must give the same prompts -- resume depends on it"
    c = [p["prompt"] for p in data_mod.prompt_batch(pool, seed=3, rollout_idx=6, n=16)]
    assert a != c, "different rollout must draw a different prompt set"
    assert len(set(a)) == len(a), "no duplicate prompt inside one rollout"
    # another seed must move the prompt set too
    d = [p["prompt"] for p in data_mod.prompt_batch(pool, seed=4, rollout_idx=5, n=16)]
    assert a != d


def test_prompt_batch_reuses_when_pool_is_small() -> None:
    pool = [{"prompt": f"q{i}", "answer": i} for i in range(4)]
    out = data_mod.prompt_batch(pool, seed=0, rollout_idx=0, n=10)
    assert len(out) == 10
    assert len({p["prompt"] for p in out}) == 4, "reuse must be explicit, not silently truncated"


def test_pool_hash_guard() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pool.jsonl"
        p.write_text('{"prompt": "a", "answer": 1}\n', encoding="utf-8")
        h = data_mod.sha256_file(p)
        assert len(data_mod.load_pool(p, h)) == 1
        with raises(ValueError, "hash mismatch"):
            data_mod.load_pool(p, "0" * 64)


def test_accounting_tri_currency() -> None:
    m = accounting.CostMeter(n_params=1_000_000, price_cny_per_hour=2.0)
    m.add_rollout(rollout_idx=0, mu=1, n_responses=10, prompt_tokens=100,
                  completion_tokens=500, trained_tokens=500,
                  wall_ms_rollout=3_600_000.0, wall_ms_update=0.0)
    t = m.totals()
    assert t["responses"] == 10
    assert close(t["gpu_hours_billed"], 1.0)
    assert close(t["cost_cny"], 2.0)
    # FLOPs: gen 2N*500 + score 2N*(100+500) + train 6N*500
    assert close(t["flops_total"], 2e6 * 500 + 2e6 * 600 + 6e6 * 500, tol=1.0)
    # a resumed meter carries the previous totals instead of restarting at zero
    m2 = accounting.CostMeter(n_params=1_000_000, price_cny_per_hour=2.0, base=t)
    assert close(m2.totals()["cost_cny"], 2.0)
    assert m2.totals()["responses"] == 10


def test_config_rejects_undefined_sigma() -> None:
    cfg = config_mod.Config({
        "train": {"responses_per_rollout": 64, "mu": 16, "micro_batch_responses": 8,
                  "num_generations": 8},
    })
    with raises(ValueError, "groups per step"):
        cfg.validate()   # b = 4 < group = 8 -> M = 0, sigma undefined


def test_config_derived_quantities() -> None:
    cfg = config_mod.Config({"train": {"responses_per_rollout": 4096, "mu": 8,
                                       "micro_batch_responses": 8, "num_generations": 8}})
    cfg.validate()
    assert cfg.prompts_per_rollout == 512
    assert cfg.responses_per_step == 512
    assert cfg.micro_batches_per_step == 64


def test_plan_partition_micro_is_batch_size_not_count() -> None:
    """`micro_batch_responses` is a chunk SIZE; plan_partition must not invert it.

    Regression for the 48G OOM of 2026-10-25. The old body was
    `np.array_split(s, micro)`, which yields `micro` chunks of b/micro responses
    each. With micro=8 and b=512 that put 64 rows through score() while
    `score_memory_bytes(micro=8, 640)` had promised room for 8 -- 21.99 GiB
    requested against a 2.89 GiB pre-flight, so every vLLM arm died in
    score(). Nothing caught it because the only existing assertion
    (test_config_derived_quantities) checks config's arithmetic, never the
    chunking that actually runs.

    Two independent invariants, both derived from config rather than hard-coded:
    the chunk count must equal `cfg.micro_batches_per_step`, and every chunk must
    hold exactly `micro` responses.
    """
    import numpy as np

    import train as train_mod

    cfg = config_mod.Config({"train": {"responses_per_rollout": 4096, "mu": 8,
                                       "micro_batch_responses": 8, "num_generations": 8}})
    cfg.validate()
    micro = int(cfg["train.micro_batch_responses"])

    parts = train_mod.plan_partition(4096, mu=8, micro=micro,
                                     rng=np.random.default_rng(1))
    assert len(parts) == 8, f"mu=8 must give 8 optimizer steps, got {len(parts)}"
    for si, step in enumerate(parts):
        assert len(step) == cfg.micro_batches_per_step, (
            f"step {si}: expected {cfg.micro_batches_per_step} micro-batches per config, "
            f"got {len(step)} -- micro was probably treated as a count, not a size")
        sizes = [len(c) for c in step]
        assert sizes == [micro] * cfg.micro_batches_per_step, (
            f"step {si}: every micro-batch must hold exactly {micro} responses, got {sizes}")

    # mu only moves step boundaries; the union must still be a permutation,
    # otherwise the arms are not paired (the whole audit depends on that).
    seen = sorted(int(i) for step in parts for c in step for i in c)
    assert seen == list(range(4096)), "partition must cover every response exactly once"

    # and the degenerate case: mu=1 collapses to a single step, chunking unchanged.
    one = train_mod.plan_partition(512, mu=1, micro=8, rng=np.random.default_rng(0))
    assert len(one) == 1 and len(one[0]) == 64
    assert [len(c) for c in one[0]] == [8] * 64


def test_chunked_logsumexp_agrees_with_full() -> None:
    """D-15 moved the fp32 (B, T, V) upcast into per-slice reductions.

    The reduction must be numerically the same thing, otherwise the log-probs feed
    a different ratio into GRPO. Checked on values and on gradients: the fp32
    upcast path is the reference.
    """
    import torch

    import model as model_mod

    torch.manual_seed(0)
    B, T, V = 2, 40, 97
    x = torch.randn(B, T, V, dtype=torch.float32, requires_grad=True)
    ref = torch.logsumexp(x, dim=-1)
    got = model_mod.chunked_logsumexp(x, 7)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max().item()

    # the gradient is what flows back to the LM head; a chunking bug there would
    # still produce plausible log-probs but wrong updates.
    g_ref = torch.autograd.grad(ref.sum(), x, retain_graph=False)[0]
    g_got = torch.autograd.grad(got.sum(), x, retain_graph=False)[0]
    assert torch.allclose(g_ref, g_got, atol=1e-6), (g_ref - g_got).abs().max().item()

    # a chunk larger than T must degrade to the single-shot path, not crash
    assert torch.allclose(model_mod.chunked_logsumexp(x, 10_000), ref, atol=1e-6)


def test_config_group_and_chunk_are_separable() -> None:
    """The sigma group may be larger than the forward chunk (D-15).

    micro is a memory budget, group is a statistics budget: sigma only sees group
    boundaries, so a group of 8 can be run as two chunks of 4. Config must reject
    combinations that do not tile.
    """
    cfg = config_mod.Config({"train": {
        "responses_per_rollout": 512, "mu": 1,
        "micro_batch_responses": 4, "micro_group_responses": 8,
        "num_generations": 8,
    }})
    cfg.validate()
    assert cfg.responses_per_step == 512
    assert cfg.micro_batches_per_step == 128, "forward chunks are an execution detail"
    assert cfg.micro_group_responses == 8
    assert cfg.groups_per_step == 64, "M in sigma must stay at the pre-registered 64"

    # default: group == chunk, i.e. the pre-D-15 behaviour
    same = config_mod.Config({"train": {"responses_per_rollout": 512, "mu": 1,
                                        "micro_batch_responses": 4, "num_generations": 8}})
    same.validate()
    assert same.micro_group_responses == 4 and same.groups_per_step == 128

    # a group that is not a whole number of chunks cannot be assembled
    bad = config_mod.Config({"train": {"responses_per_rollout": 512, "mu": 1,
                                       "micro_batch_responses": 8,
                                       "micro_group_responses": 12, "num_generations": 8}})
    with raises(ValueError, "must be a multiple"):
        bad.validate()

    # nor can a step that does not tile into whole groups
    bad2 = config_mod.Config({"train": {"responses_per_rollout": 100, "mu": 1,
                                        "micro_batch_responses": 4,
                                        "micro_group_responses": 8, "num_generations": 8}})
    with raises(ValueError, "divisible by micro_group_responses"):
        bad2.validate()


def test_group_accumulation_reproduces_the_group_mean() -> None:
    """Chunks are normalised individually by grpo_loss; a group must undo that.

    grpo_loss divides by its own chunk's token count, so summing two chunk
    gradients with equal weight gives a mean of means, not the group mean. The
    loop in train.py therefore scales each chunk by its token count and divides
    once per group. This checks that algebra directly, and that the equal-weight
    alternative really does differ (otherwise the weighting would be pointless).
    """
    import torch

    import train as train_mod

    torch.manual_seed(0)
    per_token = [torch.randn(5, dtype=torch.float64) for _ in range(6)]
    counts = [2, 4]                                  # unequal on purpose
    chunks = [sum(per_token[:2]) / counts[0], sum(per_token[2:]) / counts[1]]

    buf = [torch.zeros(5, dtype=torch.float64)]
    for g, n in zip(chunks, counts):
        train_mod._add_into(buf, [g], float(n))
    train_mod._rescale(buf, 1.0 / sum(counts))

    expected = sum(per_token) / sum(counts)
    assert torch.allclose(buf[0], expected, atol=1e-12)

    equal = (chunks[0] + chunks[1]) / 2.0
    assert not torch.allclose(equal, expected, atol=1e-6), (
        "with unequal chunk sizes the unweighted mean must differ, otherwise the "
        "token weighting in train.py is a no-op")



def test_config_extends() -> None:
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "base.yaml").write_text(
            "train:\n  mu: 1\n  lr: 0.5\nrun:\n  tag: base\n", encoding="utf-8")
        (Path(td) / "arm.yaml").write_text(
            "extends: base.yaml\nrun:\n  tag: arm\n", encoding="utf-8")
        cfg = config_mod.load_config(Path(td) / "arm.yaml")
        assert cfg["run.tag"] == "arm"
        assert close(cfg["train.lr"], 0.5)
