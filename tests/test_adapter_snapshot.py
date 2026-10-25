"""The adapter snapshot is the only surviving copy of a rollout's weights.

`state.pt` and `runs/_adapter` are both overwritten every rollout, so if this
helper silently misbehaves (wrong directory, same name twice, truncated copy) the
loss is discovered only when someone asks "what did the policy look like at step
3" -- i.e. after the box is gone. Hence the tests.

No torch, no GPU: it is a file copy plus a digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import train  # noqa: E402


def _fake_adapter(root: Path, payload: bytes = b"\x00" * 4096) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_model.safetensors").write_bytes(payload)
    (root / "adapter_config.json").write_text('{"r": 64, "lora_alpha": 16}')
    (root / "AUDIT_ADAPTER_META.json").write_text('{"version": 3, "base": "stub"}')
    return root


def test_snapshot_copies_weights_and_records_the_digest() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _fake_adapter(td / "src")
        run_dir = td / "run"

        rec = train.snapshot_adapter(run_dir, 0, src)
        dst = run_dir / "adapters" / "r000"

        assert not rec["skipped"], rec
        assert (dst / "adapter_model.safetensors").read_bytes() == (src / "adapter_model.safetensors").read_bytes()
        assert rec["sha256"] == hashlib.sha256((src / "adapter_model.safetensors").read_bytes()).hexdigest()
        assert rec["meta"]["version"] == 3
        assert rec["bytes"] == sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
        # the source is left alone: vLLM is still serving from it
        assert src.exists()
        line = json.loads((run_dir / "adapters" / "index.jsonl").read_text().strip())
        assert line["rollout_idx"] == 0 and line["dst"] == "r000"


def test_second_snapshot_of_the_same_rollout_does_not_overwrite_the_first() -> None:
    """A resumed rollout re-samples and re-trains; both versions must survive."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run_dir = td / "run"
        train.snapshot_adapter(run_dir, 3, _fake_adapter(td / "a", b"A" * 128))
        rec2 = train.snapshot_adapter(run_dir, 3, _fake_adapter(td / "b", b"B" * 128))

        assert rec2["variant"] == 1 and rec2["dst"] == "r003_v1"
        assert (run_dir / "adapters" / "r003" / "adapter_model.safetensors").read_bytes() == b"A" * 128
        assert (run_dir / "adapters" / "r003_v1" / "adapter_model.safetensors").read_bytes() == b"B" * 128


def test_missing_adapter_is_recorded_not_fatal() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rec = train.snapshot_adapter(td / "run", 0, td / "nope")
        assert rec["skipped"] is True and "no adapter" in rec["reason"]
        assert (td / "run" / "adapters" / "index.jsonl").exists(), "the skip must still be logged"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} tests passed")
