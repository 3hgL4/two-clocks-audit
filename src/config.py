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
        "logit_chunk": 64,             # positions per log-sum-exp slice; 0 = pre-D-15 whole-T upcast
        "lora": {"r": 64, "alpha": 16, "dropout": 0.0},
    },
    "train": {
        "num_rollouts": 8,
        "num_generations": 8,          # G
        "responses_per_rollout": 4096,  # Br
        "mu": 1,                       # optimizer steps per rollout
        "micro_batch_responses": 8,    # forward/backward chunk; logits memory ~ micro*len*vocab
        "micro_group_responses": None,  # sigma's group size; None = same as the chunk (D-15)
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
        # SAC-mu controller (v2 doc). enabled=False keeps every fixed arm
        # bit-identical to the archived campaign.
        "sac": {
            "enabled": False,
            "U": 1.0738,               # frozen: 12/24 hits on the 24 mu16_b256 cycles
            "burn_in_cycles": 2,       # cycles 0..1 fixed mu=16, checks disabled
            "slow_loop": True,         # False = fast loop only (ablation S)
            "random_trigger": False,   # True = ablation R (HAZARD_M16 coin)
        },
        # Explicit per-rollout mu list (ablation R). Fixed schedule: no trigger
        # checks, no slow loop. Mutually exclusive with sac.enabled.
        "mu_schedule": None,
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
        """Forward/backward chunks per optimizer step (an execution detail)."""
        return self.responses_per_step // int(self["train.micro_batch_responses"])

    @property
    def micro_group_responses(self) -> int:
        """Size of one gradient group, i.e. what sigma is measured over.

        sigma^2 = (sum_i ||g_i||^2 - M||g_bar||^2)/(M-1) only depends on where the
        group boundaries fall, not on how many responses one forward pass can
        hold. Keeping the group at the pre-registered size while shrinking the
        forward chunk separates the statistics from the memory budget (D-15).
        """
        v = self["train.micro_group_responses"]
        return int(v) if v is not None else int(self["train.micro_batch_responses"])

    @property
    def groups_per_step(self) -> int:
        return self.responses_per_step // self.micro_group_responses

    def validate(self) -> None:
        G = int(self["train.num_generations"])
        Br = int(self["train.responses_per_rollout"])
        mu = int(self["train.mu"])
        mb = int(self["train.micro_batch_responses"])
        grp = self.micro_group_responses
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
        if grp % mb:
            errs.append(
                f"micro_group_responses ({grp}) must be a multiple of "
                f"micro_batch_responses ({mb}); a group is split into whole chunks"
            )
        if b % grp:
            errs.append(f"b = Br/mu ({b}) must be divisible by micro_group_responses ({grp})")
        if b // grp < 2:
            errs.append(
                f"b/group = {b // grp} groups per step; need >= 2 to estimate the "
                "gradient-noise scale sigma (otherwise the drift clock is undefined)"
            )
        if str(self["train.loss_norm"]) not in {"token_mean", "rollout_token"}:
            errs.append("loss_norm must be 'token_mean' or 'rollout_token'")
        if int(self["train.num_rollouts"]) < 1:
            errs.append("num_rollouts must be >= 1")

        sac = self.get("train.sac") or {}
        sched = self.get("train.mu_schedule")
        if sac.get("enabled") and sched is not None:
            errs.append("train.sac.enabled and train.mu_schedule are mutually exclusive")
        if sac.get("enabled"):
            from sac import LADDER, START_RUNG

            try:
                sac_U = float(sac.get("U"))
                # P0 card section 3.4: the measured full range on the 2x2 cells
                # is [1.0000, 1.0931]; a U outside [1.05, 1.08] has no trigger
                # headroom (v1's 1.10 fired 0/96 cycles). Frozen at 1.0738.
                if not (1.05 <= sac_U <= 1.08):
                    errs.append(
                        f"sac.U must be in [1.05, 1.08] (P0 measured range), got {sac_U}"
                    )
            except (TypeError, ValueError):
                errs.append("sac.U must be a number")
            for rung in LADDER:
                if Br % rung:
                    continue
                b_r = Br // rung
                if b_r % mb or b_r % grp or b_r // grp < 2:
                    errs.append(
                        f"sac ladder rung mu={rung} invalid: b={b_r} must be divisible "
                        f"by micro ({mb}) and group ({grp}) with >=2 groups"
                    )
            if Br % START_RUNG:
                errs.append(f"sac requires Br divisible by {START_RUNG} (start rung)")
        if sched is not None:
            if not isinstance(sched, list) or len(sched) != int(self["train.num_rollouts"]):
                errs.append(
                    f"mu_schedule must be a list of length num_rollouts "
                    f"({int(self['train.num_rollouts'])})"
                )
            else:
                for i, v in enumerate(sched):
                    v = int(v)
                    if v < 1:
                        errs.append(f"mu_schedule[{i}] must be >= 1")
                        continue
                    b_r = Br // v if Br % v == 0 else None
                    if b_r is None:
                        errs.append(f"mu_schedule[{i}]={v}: Br ({Br}) not divisible")
                    elif b_r % mb or b_r % grp or b_r // grp < 2:
                        errs.append(
                            f"mu_schedule[{i}]={v} invalid: b={b_r} must be divisible "
                            f"by micro ({mb}) and group ({grp}) with >=2 groups"
                        )

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
