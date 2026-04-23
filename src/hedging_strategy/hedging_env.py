"""
Accounting P&L hedging environment — Cao et al. (2021), Section 3.1.

R_{i+1} = V_{i+1} - V_i + H_i(S_{i+1}-S_i) - κ|S_{i+1}(H_{i+1}-H_i)|
Initial cost: -κ|S_0 H_0|      (paper convention)
Final cost:   -κ|S_n H_n|

At the terminal step we force H_n = 0 so the transaction cost at t = T is
exactly the liquidation κ·S_n·|H_{n-1}|, matching the paper's "final reward
= -κ|S_n H_n|" without double-billing (a separate rebalance to H_n and then
a liquidation from H_n to 0).

State (dim 4) = [holding, log(S/K), TTM/T, σ_t/σ_ref]
"""
from __future__ import annotations
from typing import Any
import math
import numpy as np
from ..valuation.bs_valuation import BSValuation


class HedgingEnv:
    """Accounting-P&L hedging environment used for training and evaluation.

    Holds a single path at a time and exposes a ``gym``-style ``step``
    that returns the per-step reward, cost and next state. Forces
    liquidation at the terminal step so the final transaction cost is
    exactly ``kappa * S_n * |H_{n-1}|``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise pricing engine and per-run constants (kappa, sigma, maturity, strike)."""
        self.transac_cost = float(config["hedging_env"]["transaction_cost"])
        self.position_sign = float(config["hedging_env"]["position_sign"])
        self.valuation_sigma = float(config["simulation"]["gbm"]["sigma"])
        self.maturity = float(config["simulation"]["maturity"])
        self.valuation_engine = BSValuation(
            strike=config["derivative"]["strike"], maturity=self.maturity,
            rate=config["derivative"].get("rf_rate", 0.0),
            dividend=config["derivative"].get("div_rate", 0.0),
            option_type=config.get("derivative", {}).get("option_type", "call"))

    def setup_env(self, path_data):
        """Prepare the env for a new path and return the initial state.

        Accepts either a raw spot array or a ``{"S": ..., "sigma"/"variance": ...}``
        dict. Pre-computes the option value along the path and resets
        the time index and previous hedge.
        """
        if isinstance(path_data, dict):
            self.path_dict = {k: np.asarray(v, dtype=float) for k, v in path_data.items()}
            self.path_data = self.path_dict["S"]
        else:
            self.path_data = np.asarray(path_data, dtype=float)
            self.path_dict = {"S": self.path_data}
        if "sigma" in self.path_dict:
            self._vol_path = self.path_dict["sigma"]
        elif "variance" in self.path_dict:
            self._vol_path = np.sqrt(np.maximum(self.path_dict["variance"], 1e-10))
        else:
            self._vol_path = np.full_like(self.path_data, self.valuation_sigma)
        # Reference vol for state normalization: σ_0 of the process actually
        # used to generate this path. Decoupled from `valuation_sigma` (which
        # is tied to the BS pricing kernel) so that scale stays consistent
        # across processes even if σ_0 ≠ gbm.sigma.
        self.sigma_ref = float(self._vol_path[0])
        self.n_steps = len(self.path_data) - 1
        self.times = np.linspace(0.0, self.maturity, len(self.path_data))

        # Vectorized valuation over full path.
        p, _ = self.valuation_engine.price_and_delta(
            spot=self.path_data,
            t=self.times,
            sigma=self.valuation_sigma,
        )
        self._precomputed_v = self.position_sign * np.asarray(p, dtype=float)

        self.i = 0
        self.h_prev = 0.0
        return self._build_state(0, 0.0)

    def set_initial_hedge(self, H0: float) -> None:
        """Establish the initial hedge at t=0.

        Must be called once after setup_env() and before step().
        """
        self.h_prev = float(H0)

    def step(self, hedge: float):
        """Apply a rebalancing action, return (next_state, reward, done, info).

        Paper Section 3.1 Accounting P&L:
            R_{i+1} = V_{i+1} - V_i + H_i(S_{i+1} - S_i) - κ|S_{i+1}(H_{i+1} - H_i)|

        At the terminal step (i = n-1) we force H_n = 0, so the transaction
        cost is κ·S_n·|H_{n-1}|, which IS the paper's final liquidation cost.
        The caller is expected to pass hedge = 0.0 at the terminal step
        anyway (see orchestrator).
        """
        i = self.i
        done = (i == self.n_steps - 1)
        H_i = self.h_prev
        # Force liquidation at T: no policy decision at the terminal step.
        H_next = 0.0 if done else float(hedge)

        spot_t = float(self.path_data[i])
        spot_next = float(self.path_data[i + 1])
        V_i = float(self._precomputed_v[i])
        V_next = float(self._precomputed_v[i + 1])

        trade_cost = self.transac_cost * abs(spot_next * (H_next - H_i))
        reward = (V_next - V_i) + H_i * (spot_next - spot_t) - trade_cost
        # At T the trade IS the liquidation κ·S_n·|H_{n-1}|: expose it
        # separately for traceability, but do NOT subtract it a second time.
        liquidation_cost = trade_cost if done else 0.0

        self.i += 1
        self.h_prev = 0.0 if done else H_next
        next_state = self._build_state(self.i, self.h_prev)
        info = {
            "spot_t": spot_t, "spot_next": spot_next, "hedge": H_next,
            "trade_cost": trade_cost, "liquidation_cost": liquidation_cost,
            "reward": reward, "cost": -reward,
        }
        return next_state, reward, done, info

    def _build_state(self, step, hedge_pos):
        """Return the 4-dim state ``[holding, log(S/K), TTM/T, sigma/sigma_ref]`` at ``step``."""
        idx = step
        t = self.times[step]
        spot, vol = self.path_data[idx], self._vol_path[idx]
        ttm = max(self.maturity - t, 0.0)
        return np.asarray([
            hedge_pos,
            math.log(spot / self.valuation_engine.K),
            ttm / self.maturity if self.maturity > 0 else 0.0,
            vol / self.sigma_ref,
        ], dtype=np.float32)