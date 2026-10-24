"""Policy wrapper: base model + LoRA, tokenisation, and exact token log-probs.

Memory is the binding engineering constraint here, and it is dominated by the
vocabulary axis, not the weights: the LM head produces (B, T, 151936) logits, so
a micro-batch of 8 responses at ~650 tokens already costs ~1.6 GB in bf16 and
~3.2 GB once upcast to fp32 for an accurate log-sum-exp. `score_memory_bytes`
makes that explicit so a config can be rejected before a rented GPU is started
instead of OOM-ing 40 minutes in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class StubTokenizer:
    """Character-level stand-in used by the CPU dry-run.

    The dry-run has to exercise the *same* code path as production (forward,
    gather, log-softmax, backward, clipping, checkpointing) without downloading
    a 1.7B checkpoint, so the tokenizer and model are shrunk rather than skipped.
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "left"
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, text, add_special_tokens: bool = True):
        if isinstance(text, str):
            ids = [2 + (ord(ch) % (self.vocab_size - 2)) for ch in text]
            return {"input_ids": ids + [self.eos_token_id]}
        return {"input_ids": [self(t)["input_ids"] for t in text]}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        chars = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in (self.pad_token_id, self.eos_token_id):
                continue
            chars.append(chr(32 + (i % 95)))
        return "".join(chars)


class Policy:
    def __init__(self, cfg, device: str | None = None, *, build: bool = True):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.adapter_version = 0
        if build:
            self._build()

    # -- construction -------------------------------------------------------
    def _build(self) -> None:
        from peft import LoraConfig, get_peft_model

        if self.cfg.get("model.stub"):
            model = self._build_stub()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            name = self.cfg["model.name"]
            dtype = DTYPES[self.cfg["model.dtype"]]
            self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            kwargs = {
                "torch_dtype": DTYPES[self.cfg["model.dtype"]],
                "trust_remote_code": True,
                "attn_implementation": self.cfg["model.attn_implementation"],
            }
            if self.device == "cpu":
                kwargs["torch_dtype"] = torch.float32
            model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

        if self.cfg["model.gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        lora = self.cfg["model.lora"]
        lcfg = LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(model, lcfg).to(self.device)
        self.model.print_trainable_parameters()

    def _build_stub(self):
        """Tiny random transformer with a real Qwen-shaped attention layout."""
        from transformers import AutoConfig, AutoModelForCausalLM

        self.tokenizer = StubTokenizer()
        base = dict(
            vocab_size=len(self.tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=1024,
            tie_word_embeddings=False,
        )
        last_err = None
        for model_type in ("qwen3", "qwen2", "llama"):
            try:
                cfg = AutoConfig.for_model(model_type, **base)
                return AutoModelForCausalLM.from_config(cfg)
            except Exception as e:  # pragma: no cover - depends on transformers version
                last_err = e
                try:
                    cfg = AutoConfig.for_model(model_type, **{k: v for k, v in base.items() if k != "head_dim"})
                    return AutoModelForCausalLM.from_config(cfg)
                except Exception as e2:
                    last_err = e2
        raise RuntimeError(f"could not build a stub model: {last_err}")

    # -- introspection ------------------------------------------------------
    @property
    def n_params_total(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def n_params_trainable(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items() if "lora_" in k}

    def load_trainable_state_dict(self, sd: dict) -> None:
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected:
            raise ValueError(f"unexpected keys loading LoRA state: {unexpected[:5]}")

    def encode_prompts(self, prompts: list[str]) -> list[list[int]]:
        return [self.tokenizer(p, add_special_tokens=True)["input_ids"] for p in prompts]

    def score_memory_bytes(self, micro_batch: int, max_len: int, *, fp32: bool = True) -> int:
        r = 4 if fp32 else 2
        return micro_batch * max_len * len(self.tokenizer) * r

    # -- scoring ------------------------------------------------------------
    def score(
        self,
        prompt_ids: list[list[int]],
        completion_ids: list[list[int]],
        *,
        requires_grad: bool = False,
        fp32: bool = True,
    ) -> torch.Tensor:
        """Log-prob of each completion token under the current policy.

        Returns (B, Tc_max) float32; positions past a row's completion length are
        0 and must be masked by the caller.
        """
        B = len(prompt_ids)
        Tc = max(len(c) for c in completion_ids)
        Lmax = max(len(p) + len(c) for p, c in zip(prompt_ids, completion_ids))
        pad = self.tokenizer.pad_token_id

        input_ids = torch.full((B, Lmax), pad, dtype=torch.long, device=self.device)
        starts = []
        for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
            seq = p + c
            input_ids[i, Lmax - len(seq):] = torch.tensor(seq, dtype=torch.long, device=self.device)
            starts.append(Lmax - len(c))  # index of first completion token

        attn = torch.ones_like(input_ids)
        # left padding: mask the leading pads (pad id may equal eos, so mask by position)
        for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
            attn[i, : Lmax - (len(p) + len(c))] = 0

        ctx = torch.enable_grad() if requires_grad else torch.no_grad()
        with ctx:
            out = self.model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits  # (B, Lmax, V)
            if fp32 and logits.dtype != torch.float32:
                logits = logits.float()
            # log p(next token): gather the target logit, subtract logsumexp.
            # Avoids materialising a full (B, T, V) log_softmax buffer.
            targets = input_ids[:, 1:].unsqueeze(-1)                      # (B, Lmax-1, 1)
            tgt_logit = logits[:, :-1].gather(-1, targets).squeeze(-1)    # (B, Lmax-1)
            lse = torch.logsumexp(logits[:, :-1], dim=-1)                 # (B, Lmax-1)
            logp = tgt_logit - lse
            del logits, tgt_logit, lse

            # predictor index of completion token j in row i is starts_i + j - 1
            idx = torch.arange(Tc, device=self.device).unsqueeze(0) + \
                torch.tensor(starts, device=self.device).unsqueeze(1) - 1   # (B, Tc)
            last = logp.shape[1] - 1
            valid = idx <= last                    # rows shorter than Tc are padded out
            out_lp = logp.gather(1, idx.clamp(min=0, max=last))
            out_lp = torch.where(valid, out_lp, torch.zeros_like(out_lp))
        return out_lp

    def mask_for(self, completion_ids: list[list[int]]) -> torch.Tensor:
        Tc = max(len(c) for c in completion_ids)
        m = torch.zeros((len(completion_ids), Tc), dtype=torch.float32, device=self.device)
        for i, c in enumerate(completion_ids):
            m[i, : len(c)] = 1.0
        return m

    # -- adapter export (for the vLLM rollout backend) ----------------------
    def export_adapter(self, out_dir: str | Path) -> str:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.adapter_version += 1
        self.model.save_pretrained(str(out), selected_adapters=None)
        meta = {"version": self.adapter_version, "base": self.cfg["model.name"]}
        with open(out / "AUDIT_ADAPTER_META.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return str(out)

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()
