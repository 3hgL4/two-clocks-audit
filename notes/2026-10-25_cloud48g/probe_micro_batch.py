import sys, torch, gc
sys.path.insert(0, "src")
import config as C
from model import Policy

cfg = C.load_config("configs/cloud48g.yaml")
p = Policy(cfg, device="cuda")
inner = p.model.base_model.model
for path in ("gradient_checkpointing", "is_gradient_checkpointing", "_gradient_checkpointing_func"):
    print(f"[cfg] {path} =", getattr(inner, path, "MISSING"), "| on .model:",
          getattr(getattr(inner, "model", None), path, "MISSING"))
print("[cfg] model.training =", p.model.training)

Lp, Lc = 200, 384
for B in (8, 4, 2):
    pids = [list(range(1000, 1000+Lp))] * B
    cids = [list(range(2000, 2000+Lc))] * B
    p.model.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        lp = p.score(pids, cids, requires_grad=True)
        fwd_peak = torch.cuda.max_memory_allocated() / 2**30
        lp.sum().backward()
        tot_peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"[B={B}] fwd_peak {fwd_peak:.2f} GiB | fwd+bwd_peak {tot_peak:.2f} GiB | logits fp32 ~{B*584*151669*4/2**30:.2f}")
        del lp
    except torch.cuda.OutOfMemoryError as e:
        print(f"[B={B}] OOM at {torch.cuda.memory_allocated()/2**30:.2f} GiB")
