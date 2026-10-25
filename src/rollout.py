"""Rollout backends.

Three implementations behind one interface, because they are used for three
different jobs:

  fake  -- no model, no GPU, no network. Used by the local dry-run, the schema
           test and the CPU unit tests. Produces reward-diverse, parseable
           answers so that the advantage/zero-variance/logging paths are all
           exercised without renting anything.
  hf    -- transformers `generate`. Correct but slow; useful for a 20-response
           smoke test of the real weights before paying for a full run.
  vllm  -- offline vLLM engine with a LoRA adapter per update. The production
           path on the cloud box. The adapter is re-exported after each
           optimizer step and passed as a new `LoRARequest`, which is the only
           weight-sync mechanism that stays inside the public vLLM API.

Nothing here knows about clocks or metrics: a backend's job is to turn prompts
into (prompt_ids, completion_ids, text), and to report how long it took.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


@dataclass
class Rollout:
    items: list[dict]
    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    completion_texts: list[str]
    group_size: int
    wall_ms: float
    backend: str = "fake"
    extra: dict = field(default_factory=dict)

    @property
    def n_responses(self) -> int:
        return len(self.completion_texts)

    @property
    def n_prompts(self) -> int:
        return self.n_responses // self.group_size


def _expand(items: list[dict], G: int) -> list[dict]:
    """Repeat each prompt G times, prompt-major order: [p0 x G, p1 x G, ...]."""
    out: list[dict] = []
    for it in items:
        out.extend([it] * G)
    return out


# ---------------------------------------------------------------------------
# fake backend
# ---------------------------------------------------------------------------


class FakeBackend:
    """Deterministic synthetic rollouts. Zero dependencies beyond the tokenizer.

    `acc` is the per-item probability of producing the correct number, and it
    decays with the item's depth so that accuracy is heterogeneous like a real
    policy -- which matters because group variance (and therefore the
    zero-variance-group rate) is a function of accuracy.
    """

    def __init__(self, cfg, policy, *, acc: float = 0.45, vocab_hint: int | None = None):
        self.cfg = cfg
        self.policy = policy
        self.acc = acc
        tok = getattr(policy, "tokenizer", None)
        self.vocab = vocab_hint or (len(tok) if tok is not None else 1000)
        self.name = "fake"

    def generate(self, items: list[dict], G: int, rollout_idx: int, seed: int = 0) -> Rollout:
        t0 = time.perf_counter()
        rng = random.Random(seed * 7919 + rollout_idx)
        expanded = _expand(items, G)
        prompt_ids, completion_ids, texts = [], [], []
        max_len = int(self.cfg["train.max_completion_length"])
        for it in expanded:
            depth = float(it.get("depth", 1))
            p = float(self.acc) * (0.9 ** (depth - 1))
            gold = int(it["answer"])
            ok = rng.random() < p
            value = gold if ok else gold + rng.choice([-3, -2, -1, 1, 2, 3, 10])
            n_steps = rng.randint(2, 6)
            body = " ".join(f"step{i}" for i in range(n_steps))
            text = f"Let me work through it. {body}\nAnswer: {value}"
            if rng.random() < 0.05:
                text = f"{body} no final line"      # format failure path
            pids = self.policy.encode_prompts([it["prompt"]])[0] if self.policy and self.policy.tokenizer else [1, 2, 3]
            cids = [rng.randrange(self.vocab) for _ in range(min(max_len, 24 + 4 * n_steps))]
            prompt_ids.append(pids)
            completion_ids.append(cids)
            texts.append(text)
        return Rollout(
            items=expanded,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_texts=texts,
            group_size=G,
            wall_ms=(time.perf_counter() - t0) * 1000.0,
            backend="fake",
        )


# ---------------------------------------------------------------------------
# HF backend
# ---------------------------------------------------------------------------


class HFBackend:
    def __init__(self, cfg, policy, *, batch_size: int = 32):
        self.cfg = cfg
        self.policy = policy
        self.batch_size = batch_size
        self.name = "hf"

    def generate(self, items: list[dict], G: int, rollout_idx: int, seed: int = 0) -> Rollout:
        import torch

        t0 = time.perf_counter()
        torch.manual_seed(seed * 7919 + rollout_idx)
        expanded = _expand(items, G)
        texts_all, pids_all, cids_all = [], [], []
        tok = self.policy.tokenizer
        max_new = int(self.cfg["train.max_completion_length"])
        self.policy.eval()
        for i in range(0, len(expanded), self.batch_size):
            chunk = expanded[i : i + self.batch_size]
            enc = tok(
                [c["prompt"] for c in chunk],
                return_tensors="pt",
                padding=True,
                padding_side="left",
            ).to(self.policy.device)
            with torch.no_grad():
                out = self.policy.model.generate(
                    **enc,
                    do_sample=True,
                    temperature=float(self.cfg["train.temperature"]),
                    top_p=float(self.cfg["train.top_p"]),
                    max_new_tokens=max_new,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1] :]
            for j in range(gen.shape[0]):
                ids = gen[j].tolist()
                # trim at first eos
                if tok.eos_token_id in ids:
                    ids = ids[: ids.index(tok.eos_token_id) + 1]
                cids_all.append(ids)
                pids_all.append(enc["input_ids"][j][enc["attention_mask"][j].bool()].tolist())
                texts_all.append(tok.decode(ids, skip_special_tokens=True))
        self.policy.train()
        return Rollout(
            items=expanded,
            prompt_ids=pids_all,
            completion_ids=cids_all,
            completion_texts=texts_all,
            group_size=G,
            wall_ms=(time.perf_counter() - t0) * 1000.0,
            backend="hf",
        )


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------


class VLLMBackend:
    """Offline vLLM engine, colocated with the trainer on one GPU.

    Weight sync: export the LoRA adapter to disk, bump `LoRARequest` id, pass it
    on every request. vLLM caches by id, so a new id = new weights.
    """

    def __init__(self, cfg, policy, *, batch_size: int | None = None):
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
            from vllm.lora.request import LoRARequest  # noqa: F401
        except ImportError as e:  # pragma: no cover - cloud only
            raise ImportError(
                "vLLM backend requested but vllm is not installed. Local Windows dev "
                "cannot run this; use --rollout fake locally, or run on the Linux cloud box."
            ) from e
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest
        self.cfg = cfg
        self.policy = policy
        self.name = "vllm"
        vcfg = cfg["rollout.vllm"]
        self.adapter_dir = vcfg["adapter_dir"]
        self._lora_id = 0
        # D-16 determinism knobs. All unset (None) means "vLLM default", which is
        # what every measurement before 2026-10-25 ran on: prefix caching on,
        # chunked prefill on, no explicit token budget. Setting them is how the
        # L1 probe asks whether a fixed batch shape makes the sampler repeat.
        knobs = {}
        if vcfg.get("enable_prefix_caching") is not None:
            knobs["enable_prefix_caching"] = bool(vcfg["enable_prefix_caching"])
        if vcfg.get("enable_chunked_prefill") is not None:
            knobs["enable_chunked_prefill"] = bool(vcfg["enable_chunked_prefill"])
        if vcfg.get("max_num_batched_tokens") is not None:
            knobs["max_num_batched_tokens"] = int(vcfg["max_num_batched_tokens"])
        self.max_num_batched_tokens = knobs.get("max_num_batched_tokens")
        self.seed_per_request = bool(vcfg.get("seed_per_request", False))
        self.llm = LLM(
            model=cfg["model.name"],
            dtype=cfg["model.dtype"],
            gpu_memory_utilization=float(vcfg["gpu_memory_utilization"]),
            max_num_seqs=int(vcfg["max_num_seqs"]),
            enable_lora=bool(vcfg["enable_lora"]),
            max_lora_rank=int(cfg["model.lora.r"]),
            # The non-Devel CUDA image has no nvcc, and vLLM 0.32 aborts engine
            # startup inside its torch.compile path with "Could not find nvcc".
            # enforce_eager skips that path entirely; it costs some decode speed
            # but none of what this project measures. Default False keeps the
            # fast path available on a Devel image. See DEVLOG D-10.
            enforce_eager=bool(cfg.get("rollout.vllm.enforce_eager", False)),
            # vLLM sizes the KV cache for max_model_len (default = the model's
            # 32768), and refuses to start when the reservation cannot hold one
            # such request: at gpu_memory_utilization 0.28 on 24 GiB it wanted
            # 3.5 GiB and only had 2.61. Our sequences are prompt (~200 tokens)
            # + max_completion_length, so 1024 is generous and keeps the cache
            # small enough to coexist with the training policy.
            max_model_len=cfg.get("rollout.vllm.max_model_len", None),
            trust_remote_code=True,
            **knobs,
        )
        self.batch_size = batch_size or int(cfg["train.responses_per_rollout"])

    def _sync(self) -> "object":
        self.policy.export_adapter(self.adapter_dir)
        self._lora_id += 1
        return self._LoRARequest(f"policy_v{self._lora_id}", self._lora_id, self.adapter_dir)

    def generate(self, items: list[dict], G: int, rollout_idx: int, seed: int = 0) -> Rollout:
        t0 = time.perf_counter()
        expanded = _expand(items, G)
        base_seed = int(seed * 7919 + rollout_idx)

        def _params(s: int):
            return self._SamplingParams(
                temperature=float(self.cfg["train.temperature"]),
                top_p=float(self.cfg["train.top_p"]),
                max_tokens=int(self.cfg["train.max_completion_length"]),
                seed=int(s),
                n=1,
            )

        req = self._sync()
        texts_all, pids_all, cids_all = [], [], []
        for i in range(0, len(expanded), self.batch_size):
            chunk = expanded[i : i + self.batch_size]
            # `seed_per_request` is off by default because every measurement
            # before 2026-10-25 ran with one shared seed. Whether that shared
            # seed also makes the G samples of one prompt share a random stream
            # is the open question the diversity check answers; if it does, the
            # effective sample size is prompts, not responses.
            if self.seed_per_request:
                sp = [_params(base_seed * 1_000_003 + i + j) for j in range(len(chunk))]
            else:
                sp = _params(base_seed)
            outs = self.llm.generate([c["prompt"] for c in chunk], sp, lora_request=req, use_tqdm=False)
            for o in outs:
                cids_all.append(list(o.outputs[0].token_ids))
                pids_all.append(list(o.prompt_token_ids))
                texts_all.append(o.outputs[0].text)
        return Rollout(
            items=expanded,
            prompt_ids=pids_all,
            completion_ids=cids_all,
            completion_texts=texts_all,
            group_size=G,
            wall_ms=(time.perf_counter() - t0) * 1000.0,
            backend="vllm",
        )


def build_backend(name: str, cfg, policy, **kwargs):
    name = (name or "fake").lower()
    if name == "fake":
        return FakeBackend(cfg, policy, **kwargs)
    if name == "hf":
        return HFBackend(cfg, policy, **kwargs)
    if name == "vllm":
        return VLLMBackend(cfg, policy, **kwargs)
    raise ValueError(f"unknown rollout backend: {name!r}")
