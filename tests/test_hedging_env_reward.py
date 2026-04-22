import numpy as np
import pytest

from src.hedging_strategy.hedging_env import HedgingEnv


def _make_env(sigma=0.2, kappa=0.01):
    config = {
        "hedging_env": {"position_sign": -1.0, "transaction_cost": kappa},
        "simulation": {"maturity": 1.0, "gbm": {"sigma": sigma}},
        "derivative": {
            "strike": 100.0,
            "rf_rate": 0.0,
            "div_rate": 0.0,
            "option_type": "call",
        },
    }
    return HedgingEnv(config)


def test_no_rebalancing_means_zero_trade_cost():
    """If H_{i+1} == H_i, the trade cost at step i+1 must be exactly zero."""
    env = _make_env()
    path = np.array([100.0, 101.0, 102.0, 103.0])
    env.setup_env(path)
    env.set_initial_hedge(0.5)
    # First rebalance: H_1 = H_0 = 0.5, no trade.
    _, _, _, info1 = env.step(0.5)
    assert info1["trade_cost"] == pytest.approx(0.0, abs=1e-12)
    # Setup cost should be folded into R_1 via liquidation_cost==0 on this step.
    assert info1["liquidation_cost"] == pytest.approx(0.0, abs=1e-12)


def test_total_cost_matches_manual_computation():
    """Sum of rewards over an episode must equal the closed-form P&L."""
    env = _make_env(kappa=0.01)
    path = np.array([100.0, 102.0, 101.0, 103.0])
    env.setup_env(path)
    # Fixed policy: always hedge 50% of underlying.
    H = [0.5, 0.5, 0.5, 0.5]
    env.set_initial_hedge(H[0])
    total = 0.0
    for next_hedge in H[1:] + [0.5]:  # last step triggers liquidation
        _, r, done, _ = env.step(next_hedge)
        total += r
        if done:
            break
    # Manual: V_T - V_0 + sum of H·ΔS - all trade/setup/liquidation costs.
    V = env._precomputed_v
    pnl_asset = sum(H[i] * (path[i + 1] - path[i]) for i in range(len(path) - 1))
    pnl_deriv = V[-1] - V[0]
    setup = 0.01 * abs(path[0] * H[0])
    trades = sum(0.01 * path[i + 1] * abs(H[i + 1] - H[i]) for i in range(len(H) - 1))
    liquidation = 0.01 * path[-1] * abs(H[-1])
    expected = pnl_deriv + pnl_asset - setup - trades - liquidation
    assert total == pytest.approx(expected, abs=1e-9)


def test_trade_cost_is_current_rebalance_only():
    """Step i+1 trade cost = κ·S_{i+1}·|H_{i+1} - H_i|, not a stale rebalance."""
    env = _make_env(kappa=0.01)
    path = np.array([100.0, 110.0, 110.0])
    env.setup_env(path)
    env.set_initial_hedge(0.3)
    # Step 1: rebalance from 0.3 to 0.8 at S_1=110.
    _, _, _, info1 = env.step(0.8)
    expected_trade_1 = 0.01 * 110.0 * abs(0.8 - 0.3)
    assert info1["trade_cost"] == pytest.approx(expected_trade_1, abs=1e-12)
    # Step 2: no rebalance (H stays at 0.8), so trade_cost must be 0,
    # even though the previous step did trade. This is the bug fix.
    _, _, _, info2 = env.step(0.8)
    assert info2["trade_cost"] == pytest.approx(0.0, abs=1e-12)

