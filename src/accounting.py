"""Tri-currency cost accounting.

The paper states its budget in GPU-hours; a rented-GPU audit needs the other two
currencies as well, because they do not move together:

  responses  -- what the experiment grid is defined in (Br, mu, seeds)
  tokens     -- what actually costs money at constant cap (the paper never
                reports response length: 0 occurrences in the full text)
  FLOPs      -- what tells you whether two arms did the same amount of work

All three are recorded per rollout and per run, together with wall-clock and a
price snapshot, so a cost claim can always be recomputed instead of remembered.

FLOPs convention (Kaplan et al. / Hoffmann et al.):
  generation : 2 * N * new_tokens
  training   : 6 * N * trained_tokens      (fwd + bwd)
with N = active parameter count (base + LoRA). Forward-only scoring for the
old-policy log-probs is counted as 2 * N * tokens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RolloutCost:
    rollout_idx: int
    mu: int
    responses: int
    prompt_tokens: int
    completion_tokens: int
    trained_tokens: int
    flops_generation: float
    flops_scoring: float
    flops_update: float
    wall_ms_rollout: float
    wall_ms_update: float

    @property
    def flops_total(self) -> float:
        return self.flops_generation + self.flops_scoring + self.flops_update

    @property
    def wall_ms_total(self) -> float:
        return self.wall_ms_rollout + self.wall_ms_update


@dataclass
class CostMeter:
    n_params: int
    price_cny_per_hour: float
    gpu_type: str = "unknown"
    rollouts: list[RolloutCost] = field(default_factory=list)
    # totals carried over from a previous session (resume after a box shutdown);
    # without this the cumulative cost silently restarts at zero mid-run.
    base: dict = field(default_factory=dict)

    # -- recording ----------------------------------------------------------
    def add_rollout(
        self,
        *,
        rollout_idx: int,
        mu: int,
        n_responses: int,
        prompt_tokens: int,
        completion_tokens: int,
        trained_tokens: int,
        wall_ms_rollout: float,
        wall_ms_update: float,
    ) -> RolloutCost:
        n = self.n_params
        c = RolloutCost(
            rollout_idx=rollout_idx,
            mu=mu,
            responses=n_responses,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            trained_tokens=trained_tokens,
            flops_generation=2.0 * n * completion_tokens,
            flops_scoring=2.0 * n * (prompt_tokens + completion_tokens),
            flops_update=6.0 * n * trained_tokens,
            wall_ms_rollout=wall_ms_rollout,
            wall_ms_update=wall_ms_update,
        )
        self.rollouts.append(c)
        return c

    # -- aggregation --------------------------------------------------------
    def totals(self) -> dict:
        def s(attr: str) -> float:
            return float(sum(getattr(r, attr) for r in self.rollouts))

        wall_h = (s("wall_ms_rollout") + s("wall_ms_update")) / 3_600_000.0
        b = self.base or {}
        add = lambda k: float(b.get(k, 0.0))  # noqa: E731
        return {
            "n_rollouts": len(self.rollouts) + int(b.get("n_rollouts", 0)),
            "responses": int(s("responses") + add("responses")),
            "prompt_tokens": int(s("prompt_tokens") + add("prompt_tokens")),
            "completion_tokens": int(s("completion_tokens") + add("completion_tokens")),
            "trained_tokens": int(s("trained_tokens") + add("trained_tokens")),
            "flops_generation": s("flops_generation") + add("flops_generation"),
            "flops_scoring": s("flops_scoring") + add("flops_scoring"),
            "flops_update": s("flops_update") + add("flops_update"),
            "flops_total": s("flops_generation") + s("flops_scoring") + s("flops_update")
            + add("flops_total"),
            "wall_hours_measured": round(wall_h, 6),
            "gpu_hours_billed": round(wall_h + add("gpu_hours_billed"), 6),
            "cost_cny": round((wall_h + add("gpu_hours_billed")) * self.price_cny_per_hour, 4),
            "price_cny_per_hour": self.price_cny_per_hour,
            "gpu_type": self.gpu_type,
            "n_params": self.n_params,
            "resumed_from_totals": bool(b),
        }

    def as_rows(self) -> list[dict]:
        return [asdict(r) for r in self.rollouts]


def project_run_cost(
    n_params: int,
    total_responses: int,
    mean_completion_tokens: float,
    mean_prompt_tokens: float,
    price_cny_per_hour: float,
    measured_ms_per_response: float,
) -> dict:
    """Pre-flight estimate used before renting, and re-checked after 40 minutes.

    Everything is per-response, so the same function serves the a-priori budget
    (from a measured s/step) and the in-flight recomputation (from live logs).
    """
    wall_h = measured_ms_per_response * total_responses / 3_600_000.0
    flops = (
        2.0 * n_params * total_responses * mean_completion_tokens
        + 2.0 * n_params * total_responses * (mean_prompt_tokens + mean_completion_tokens)
        + 6.0 * n_params * total_responses * mean_completion_tokens
    )
    return {
        "total_responses": total_responses,
        "mean_completion_tokens": mean_completion_tokens,
        "wall_hours": round(wall_h, 4),
        "cost_cny": round(wall_h * price_cny_per_hour, 4),
        "flops_total": flops,
    }
