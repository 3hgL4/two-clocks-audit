"""Config loading with dotted access and hard validation.

Every value that the audit depends on is declared here so that a run can be
fully reconstructed from `config.resolved.yaml` alone.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "run": {
        "tag": "dev",              # used for runs/<tag>/
        "seed": 0,
        "out_dir": "runs",
        "note": "",
    },
    "data": {
        "pool": "data/prompts_mixed2000.jsonl",
        "sha256": "",              # if set, load fails on mismatch
        "prompt_pad_token": None,  # None -> use tokenizer pad
    },
    "model": {
        "name": "Qwen/Qwen3-1.7B-Base",
        "stub": False,                 # True -> tiny random model, no download (dry-run)
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "gradient_checkpointing": True,
        "lora": {"r": 64, "alpha": 16, "dropout": 0.0},
    },
    "train": {
        "num_rollouts": 8,
        "num_generations": 8,          # G
        "responses_per_rollout": 4096,  # Br
        "mu": 1,                       # optimizer steps per rollout
        "micro_batch_responses": 8,    # forward/backward chunk; logits memory ~ micro*len*vocab
        "max_completion_length": 512,
        "temperature": 1.0,
        "top_p": 1.0,
        "lr": 5e-5,
        "weight_decay": 0.0,
        "warmup_ratio": 0.1,
        "grad_clip": 1.0,
        "eps_low": 0.2,
        "eps_high": 0.28,
        "clip_delta": 10.0,            # dual-clip upper bound (None disables)
        "loss_norm": "token_mean",     # token_mean | rollout_token
    },
    "rollout": {
        "backend": "fake",             # fake | hf | vllm
        "vllm": {
            "gpu_memory_utilization": 0.45,
            "max_num_seqs": 256,
            "enable_lora": True,
            "adapter_dir": "runs/_adapter",
            "offload_policy": False,   # move the HF policy to CPU during generation
        },
    },
    "accounting": {
        "price_cny_per_hour": 1.56,    # RTX 4090-24G (48G = 2.08)
        "gpu_type": "4090-24G",
    },
    "logging": {
        "flush_every": 1,
        "save_state_every_rollouts": 1,
        "log_rollout_samples": 8,
    },
    "drift": {
        "kappa": 1.0,                  # unit convention only; ratios are kappa-free
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, raw: dict, path: str | None = None):
        self.raw = _deep_merge(DEFAULTS, raw or {})
        self.path = path

    def __getitem__(self, key: str) -> Any:
        cur: Any = self.raw
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"missing config key: {key!r} (failed at {part!r})")
            cur = cur[part]
        return cur

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        cur = self.raw
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    # -- derived quantities ---------------------------------------------------
    @property
    def prompts_per_rollout(self) -> int:
        return int(self["train.responses_per_rollout"]) // int(self["train.num_generations"])

    @property
    def responses_per_step(self) -> int:
        """b = Br / mu : responses consumed by exactly one optimizer step."""
        return int(self["train.responses_per_rollout"]) // int(self["train.mu"])

    @property
    def micro_batches_per_step(self) -> int:
        return self.responses_per_step // int(self["train.micro_batch_responses"])

    def validate(self) -> None:
        G = int(self["train.num_generations"])
        Br = int(self["train.responses_per_rollout"])
        mu = int(self["train.mu"])
        mb = int(self["train.micro_batch_responses"])
        errs: list[str] = []
        if G < 2:
            errs.append("num_generations must be >= 2 (GRPO group baseline)")
        if Br % G:
            errs.append(f"responses_per_rollout ({Br}) must be divisible by num_generations ({G})")
        if Br % mu:
            errs.append(f"responses_per_rollout ({Br}) must be divisible by mu ({mu})")
        b = Br // mu
        if b % mb:
            errs.append(f"b = Br/mu ({b}) must be divisible by micro_batch_responses ({mb})")
        if b // mb < 2:
            errs.append(
                f"b/micro = {b // mb} micro-batches per step; need >= 2 to estimate the "
                "gradient-noise scale sigma (otherwise the drift clock is undefined)"
            )
        if str(self["train.loss_norm"]) not in {"token_mean", "rollout_token"}:
            errs.append("loss_norm must be 'token_mean' or 'rollout_token'")
        if int(self["train.num_rollouts"]) < 1:
            errs.append("num_rollouts must be >= 1")
        if errs:
            raise ValueError("invalid config:\n  - " + "\n  - ".join(errs))

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False, allow_unicode=True)


def _load_raw(path: str | Path, seen: set | None = None) -> dict:
    """Load a YAML config, resolving `extends: base.yaml` chains.

    Inheritance keeps exactly one source of truth for the arm definition: the
    arm files only state what they change (mu, tag), so a deviation can never be
    introduced by copying a config and editing it.
    """
    path = Path(path)
    seen = seen or set()
    key = path.resolve()
    if key in seen:
        raise ValueError(f"config extends cycle at {path}")
    seen.add(key)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent:
        base = _load_raw(path.parent / parent, seen)
        raw = _deep_merge(base, raw)
    return raw


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    raw = _load_raw(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    cfg = Config(raw, str(path))
    cfg.validate()
    return cfg
