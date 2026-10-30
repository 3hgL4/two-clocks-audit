"""SAC-mu controller, config validation, and tail-diagnostics tests (v2 doc).

Zero-dependency like the rest of the suite: `python tests/run_all.py`.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config as config_mod  # noqa: E402
import grpo  # noqa: E402
import sac as sac_mod  # noqa: E402


def _raises(exc, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------


def make_controller(**overrides) -> sac_mod.SacController:
    params = sac_mod.SacParams(U=1.0738, **overrides)
    return sac_mod.SacController(params, br=4096)


def test_ladder_only_divisors_of_br():
    c = make_controller()
    assert c.rungs == (4, 8, 16, 32)
    assert c.mu_max == 16


def test_br_without_start_rung_rejected():
    assert _raises(
        ValueError,
        sac_mod.SacController, sac_mod.SacParams(U=1.0738), 16388,
    )


def test_burn_in_pins_mu_and_disables_checks():
    c = make_controller(burn_in_cycles=2)
    assert c.cycle_mu() == 16
    assert c.check(1.09) is False          # checks disabled during burn-in
    c.end_cycle(triggered=False, executed=16)
    assert c.cycle_mu() == 16
    assert c.check(1.09) is False
    c.end_cycle(triggered=False, executed=16)
    assert c.cycle_mu() == 16              # armed, still at the start rung
    assert c.check(1.09) is True           # checks live from cycle 2


def test_trigger_threshold_is_frozen_value():
    c = make_controller()
    for _ in range(2):
        c.end_cycle(triggered=False, executed=16)
    assert c.check(1.0738) is True
    assert c.check(1.0737) is False
    assert c.check(1.073799) is False
    assert c.check(1.0738) is True


def test_slow_loop_moves_one_rung():
    c = make_controller()
    for _ in range(2):
        c.end_cycle(triggered=False, executed=16)
    c.end_cycle(triggered=True, executed=9)
    assert c.mu_max == 8                   # down one rung
    c.end_cycle(triggered=False, executed=8)
    assert c.mu_max == 16                  # up one rung


def test_slow_loop_floors_and_caps():
    c = make_controller()
    for _ in range(2):
        c.end_cycle(triggered=False, executed=16)
    for _ in range(8):                     # walk all the way down
        c.end_cycle(triggered=True, executed=4)
    assert c.mu_max == 4
    c.end_cycle(triggered=True, executed=4)
    assert c.mu_max == 4                   # floored
    for _ in range(8):                     # walk all the way up
        c.end_cycle(triggered=False, executed=32)
    assert c.mu_max == 32
    c.end_cycle(triggered=False, executed=32)
    assert c.mu_max == 32                  # capped


def test_slow_loop_disabled_is_ablation_s():
    c = make_controller(slow_loop=False)
    for _ in range(2):
        c.end_cycle(triggered=False, executed=16)
    c.end_cycle(triggered=True, executed=11)
    assert c.mu_max == 16                  # fast loop only: rung never moves
    c.end_cycle(triggered=False, executed=16)
    assert c.mu_max == 16


def test_history_records_every_cycle():
    c = make_controller(burn_in_cycles=1)
    c.end_cycle(triggered=False, executed=16)
    c.end_cycle(triggered=True, executed=9)
    assert c.history == [(0, 16, 16, False), (1, 16, 9, True)]


def test_state_dict_roundtrip():
    c = make_controller()
    for _ in range(2):                     # burn-in
        c.end_cycle(triggered=False, executed=16)
    c.end_cycle(triggered=True, executed=9)
    d = c.state_dict()
    c2 = make_controller()
    c2.load_state_dict(d)
    assert c2.mu_max == c.mu_max == 8
    assert c2.cycle_idx == c.cycle_idx == 3
    assert c2.history == c.history


def test_state_dict_rejects_corrupt_rung():
    c = make_controller()
    assert _raises(
        ValueError, c.load_state_dict,
        {"rung_idx": 99, "cycle": 0, "history": []},
    )


def test_params_validation():
    assert _raises(ValueError, sac_mod.SacParams, U=1.0)
    assert _raises(ValueError, sac_mod.SacParams, U=2.1)
    assert _raises(ValueError, sac_mod.SacParams, U=1.0738, burn_in_cycles=-1)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def _base_cfg() -> dict:
    return {
        "run": {"tag": "t", "seed": 0},
        "train": {
            "num_rollouts": 2,
            "responses_per_rollout": 4096,
            "micro_batch_responses": 4,
            "micro_group_responses": 8,
        },
    }


def test_config_sac_and_schedule_mutually_exclusive():
    raw = _base_cfg()
    raw["train"]["sac"] = {"enabled": True}
    raw["train"]["mu_schedule"] = [16, 16]
    assert _raises(ValueError, config_mod.Config(raw).validate)


def test_config_schedule_length_and_divisibility():
    raw = _base_cfg()
    raw["train"]["mu_schedule"] = [16]          # wrong length
    assert _raises(ValueError, config_mod.Config(raw).validate)
    raw["train"]["mu_schedule"] = [16, 12]      # 4096 % 12 != 0
    assert _raises(ValueError, config_mod.Config(raw).validate)


def test_config_schedule_valid_ok():
    raw = _base_cfg()
    raw["train"]["mu_schedule"] = [16, 8]
    config_mod.Config(raw).validate()           # must not raise


def test_config_sac_defaults_valid():
    raw = _base_cfg()
    raw["train"]["sac"] = {"enabled": True}
    config_mod.Config(raw).validate()           # frozen defaults must validate


def test_config_sac_U_out_of_measured_range_rejected():
    raw = _base_cfg()
    raw["train"]["sac"] = {"enabled": True, "U": 1.10}   # v1 value: 0/96 triggers
    assert _raises(ValueError, config_mod.Config(raw).validate)


# ---------------------------------------------------------------------------
# grpo tail diagnostics
# ---------------------------------------------------------------------------


def _loss_kwargs(ratio_vals: torch.Tensor) -> dict:
    """Build a grpo_loss call where valid-token ratios equal ratio_vals."""
    B, T = 1, ratio_vals.numel()
    old = torch.zeros(B, T)
    lp = old + ratio_vals.log().view(1, -1)
    mask = torch.ones(B, T)
    adv = torch.zeros(B, 1)
    return dict(logp=lp, old_logp=old, adv=adv, mask=mask)


def test_tail_stats_exact_values():
    ratios = torch.tensor([1.0, 1.01, 1.02, 1.2, 1.3, 0.9, 1.05, 1.04])
    _, stats = grpo.grpo_loss(eps_high=0.28, **_loss_kwargs(ratios))
    assert _approx(stats["ratio_max"], 1.3)
    assert _approx(stats["ratio_p99"], torch.quantile(ratios, 0.99).item())
    # unconditional share above 1+eps_high (=1.28), independent of advantage
    # sign: only 1.3 clears it (1.2 does not), so 1/8.
    assert _approx(stats["ratio_over_high_frac"], 1.0 / 8.0)


def test_tail_stats_zero_when_no_valid_tokens():
    B, T = 2, 4
    old = torch.zeros(B, T)
    lp = torch.zeros(B, T)
    mask = torch.zeros(B, T)
    _, stats = grpo.grpo_loss(lp, old, torch.zeros(B, 1), mask, eps_high=0.28)
    assert stats["ratio_max"] == 1.0
    assert stats["ratio_p99"] == 1.0
    assert stats["ratio_over_high_frac"] == 0.0
