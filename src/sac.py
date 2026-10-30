"""SAC-mu: staleness-aware controller for the per-rollout update count.

Frozen design: jd-lab/discussions/2026-10-30_SAC-mu_实验方案v2_冻结稿.md (v2).
This module is pure Python (no torch) so the controller logic is unit-testable
without a GPU.

Fast loop (within a cycle): stop before optimizer step j if the ratio_p95
measured on step j's forward pass reaches U. Semantics note (off-by-one, pinned
against src/train.py): row mu_step=si records the ratio of the policy after si
updates vs the rollout policy -- it is measured DURING step si, BEFORE step si's
opt.step(). Row 0 is identically 1.0. The ratio after the final (mu-th) update
is never measured. Therefore a trigger on row si (si >= 1) means "skip update
si and break", the cycle executes si updates, and the trigger condition over a
fixed-mu historical cycle is exactly `max over rows si>=1 >= U` (any-hit).

Slow loop (between cycles): move mu_max one rung on the ladder per cycle --
triggered -> one rung down, clean -> one rung up. The ladder replaces v1's
additive/multiplicative AIMD (revision R-3 of the v2 doc): Br = 4096 = 2**12,
and every mu_max must divide Br so that b = Br/mu_max stays integral and the
micro/group invariants of plan_partition hold (any other value walks b into
fractions, e.g. mu=20 -> b=204.8). Calibration targets P(trigger at 16) = 0.5,
the stationary point of the rung walk at the best fixed arm (F4: mu16 = 0.517).

Calibration (frozen U): the 24 mu16_b256 cycles give per-cycle max ratio_p95
with the 12th/13th order statistics 1.074193 / 1.073432; U = 1.0738 yields
exactly 12/24 = 50%. Candidate, not validation: the data comes from fixed-mu
arms and is also the controller's own early operating regime.
"""

from __future__ import annotations

from dataclasses import dataclass

# Only values dividing Br (=4096 = 2**12) keep b integral; {4, 8, 16, 32} is
# the ladder inside [mu_min, mu_cap] of the v2 doc. mu=64 exists for the P4
# cost probe only (fixed arm, no controller).
LADDER = (4, 8, 16, 32)

START_RUNG = 16  # F4: best fixed arm; the controller starts where the audit ended


@dataclass(frozen=True)
class SacParams:
    """Frozen controller parameters (v2 doc section 2; do not tune post hoc)."""

    U: float                      # trigger threshold on ratio_p95 (frozen 1.0738)
    burn_in_cycles: int = 2       # cycles 0..1 run fixed mu=16, checks disabled
    slow_loop: bool = True        # False = fast loop only (ablation S)

    def __post_init__(self) -> None:
        if not (1.0 < float(self.U) <= 2.0):
            raise ValueError(f"SAC U must be in (1.0, 2.0], got {self.U}")
        if int(self.burn_in_cycles) < 0:
            raise ValueError("burn_in_cycles must be >= 0")


class SacController:
    """Rung-walk controller over LADDER; one instance per run."""

    def __init__(self, params: SacParams, br: int):
        if br <= 0:
            raise ValueError(f"br must be positive, got {br}")
        rungs = tuple(r for r in LADDER if br % r == 0)
        if START_RUNG not in rungs:
            raise ValueError(
                f"Br={br} has no ladder rung at {START_RUNG}: valid rungs {rungs}. "
                "The controller must start at the best fixed arm (F4)."
            )
        self.p = params
        self.rungs = rungs
        self._idx = rungs.index(START_RUNG)
        self._cycle = 0
        # (cycle_idx, mu_planned, executed_updates, triggered) per finished cycle
        self.history: list[tuple[int, int, int, bool]] = []

    @property
    def mu_max(self) -> int:
        return self.rungs[self._idx]

    @property
    def cycle_idx(self) -> int:
        return self._cycle

    def cycle_mu(self) -> int:
        """Planned split size for the current cycle (burn-in pins START_RUNG)."""
        if self._cycle < self.p.burn_in_cycles:
            return START_RUNG
        return self.mu_max

    def check(self, ratio_p95: float) -> bool:
        """Fast loop: fire once per cycle, on the step whose measured ratio_p95
        reaches U. Disabled during burn-in."""
        if self._cycle < self.p.burn_in_cycles:
            return False
        return float(ratio_p95) >= self.p.U

    def end_cycle(self, triggered: bool, executed: int) -> None:
        """Slow loop: one rung down on trigger, one up on a clean cycle.
        No-op rung move during burn-in (checks were disabled anyway)."""
        self.history.append((self._cycle, self.cycle_mu(), int(executed), bool(triggered)))
        if self._cycle < self.p.burn_in_cycles:
            self._cycle += 1
            return
        if self.p.slow_loop:
            if triggered:
                self._idx = max(0, self._idx - 1)
            else:
                self._idx = min(len(self.rungs) - 1, self._idx + 1)
        self._cycle += 1

    def state_dict(self) -> dict:
        return {
            "rung_idx": self._idx,
            "cycle": self._cycle,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict) -> None:
        self._idx = int(state["rung_idx"])
        self._cycle = int(state["cycle"])
        self.history = [tuple(h) for h in state.get("history", [])]
        if not (0 <= self._idx < len(self.rungs)):
            raise ValueError(f"restored rung_idx {self._idx} outside rungs {self.rungs}")
