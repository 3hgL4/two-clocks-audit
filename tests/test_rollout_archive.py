"""Rollout archive tests: `rollouts/r{idx}.npz` + `tools/audit_rollout_ids.py`.

The archive exists because vLLM's sampler does not reproduce across processes
(DEVLOG D-16), so the responses a run was scored on have to be *stored* or the
numbers cannot be re-derived afterwards. These tests pin the two properties the
audit depends on:

* the digest written next to the archive is the digest of the ids inside it,
  computed the same way on both sides (a silent mismatch would make every
  "is this the same data?" answer worthless);
* a second write for the same rollout index never overwrites the first, because
  a resumed run samples *different* data for the same index and that has to stay
  visible rather than be papered over.

Pure numpy + torch: no model, no tokenizer, no GPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "tools"))

import audit_rollout_ids as audit  # noqa: E402
import train  # noqa: E402


def _fake_rollout(completions: list[list[int]], prompts: list[list[int]] | None = None):
    prompts = prompts or [[1, 2, 3, i] for i in range(len(completions))]
    return SimpleNamespace(prompt_ids=prompts, completion_ids=completions,
                           n_responses=len(completions))


def test_archive_digest_matches_the_audit_tool():
    comps = [[10, 11, 12], [13], [], [14, 15]]
    rb = _fake_rollout(comps)
    with tempfile.TemporaryDirectory() as td:
        info = train.save_rollout_ids(Path(td), 0, rb, seed=0, backend="fake")
        assert (Path(td) / "r000.npz").exists(), "archive not written"

        prompts, completions = audit.load(Path(td) / "r000.npz")
        assert prompts == rb.prompt_ids, "prompt ids did not round-trip"
        assert completions == comps, "completion ids did not round-trip"
        assert audit.digest(prompts, completions) == info["sha256"], (
            "train.py and tools/audit_rollout_ids.py disagree on the digest")
        assert info["max_seq_len"] == max(len(p) + len(c)
                                         for p, c in zip(rb.prompt_ids, comps))

        recs = [json.loads(x) for x in
                (Path(td) / "index.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(recs) == 1 and recs[0]["sha256"] == info["sha256"]
        assert recs[0]["variant"] == 0


def test_second_write_for_the_same_rollout_is_kept_apart():
    """A resume re-samples the rollout; that must not look like the first write."""
    with tempfile.TemporaryDirectory() as td:
        first = train.save_rollout_ids(Path(td), 3, _fake_rollout([[1, 2, 3]]),
                                       seed=0, backend="vllm")
        second = train.save_rollout_ids(Path(td), 3, _fake_rollout([[9, 9]]),
                                        seed=0, backend="vllm")
        assert second["variant"] == 1, "the second write reused variant 0"
        assert second["sha256"] != first["sha256"]
        assert (Path(td) / "r003.npz").exists() and (Path(td) / "r003_v1.npz").exists()
        # the original still holds its own ids
        assert audit.load(Path(td) / "r003.npz")[1] == [[1, 2, 3]]
        assert audit.load(Path(td) / "r003_v1.npz")[1] == [[9, 9]]


def test_diversity_flags_a_collapsed_group():
    """G=4: group 0 is four copies, group 1 is four distinct samples."""
    comps = [[5, 6, 7]] * 4 + [[5, 6, 7], [5, 6], [5, 9, 7, 1], [8, 6, 7]]
    rep = audit.diversity(comps, group=4)
    assert rep["groups"] == 2
    assert rep["groups_all_identical"] == 1, "the collapsed group was not detected"
    assert rep["groups_with_duplicates"] == 1
    assert rep["mean_unique"] == (1 + 4) / 2
    # all four members of group 0 share the first token, in group 1 the first
    # tokens are 5,5,5,8 -> two distinct
    assert rep["mean_unique_first_token"] == (1 + 2) / 2
    # within-group spread (group 0 contributes 0) is smaller than the overall one
    assert rep["len_sd_within_group_mean"] > 0
    assert 0 < rep["within_over_total_len_sd"] < 1


def test_compare_pinpoints_the_first_differing_rows():
    a = ([[1, 2], [3], [4]], [[7, 8, 9, 10], [11, 12], [13]])
    b = ([[1, 2], [3], [4]], [[7, 8, 9, 10], [11, 99], [13, 14]])
    rep = audit.compare(a, b)
    assert rep["comparable"] and rep["prompts_identical"]
    assert not rep["completions_identical"]
    assert rep["rows_differing"] == 2
    assert rep["first_rows_differing"][0]["row"] == 1
    assert rep["first_rows_differing"][0]["common_prefix_tokens"] == 1

    same = audit.compare(a, a)
    assert same["completions_identical"], "identical archives reported as different"
    assert same["rows_differing"] == 0


def test_compare_refuses_ragged_archives():
    rep = audit.compare(([[1]], [[2]]), ([[1]], [[2], [3]]))
    assert rep["comparable"] is False and "row count" in rep["reason"]