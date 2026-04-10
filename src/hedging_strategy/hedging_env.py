from __future__ import annotations

from typing import Any

import numpy as np

from ..valuation.bs_valuation import BSValuation


class HedgingEnv:
    """
    Accounting P&L environment aligned with Cao et al. (2021).

    State (dim 4) = [holding, log-moneyness, normalized TTM, normalized vol]

    For GBM the volatility component is constant (= valuation sigma).
    For SABR / SVJ the volatility component is the current instantaneous
    volatility observed along the simulated path, giving the agent
    information it needs to adapt its hedge to the volatility regime.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.transac_cost = float(config["hedging_env"]["transaction_cost"])
        self.position_sign = float(config["hedging_env"]["position_sign"])
        self.derivative_type = config.get("derivative", {}).get("option_type", "call")

        self.valuation_sigma = float(config["simulation"]["gbm"]["sigma"])
        self.maturity = float(config["simulation"]["maturity"])

        self.valuation_engine = BSValuation(
            strike=config["derivative"]["strike"],
            maturity=self.maturity,
            rate=config["derivative"].get("rf_rate", 0.0),
            dividend=config["derivative"].get("div_rate", 0.0),
            option_type=self.derivative_type,
        )

    # ── public interface ────────────────────────────────────────────

    def setup_env(self, path_data: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(path_data, dict):
            self.path_dict = {k: np.asarray(v, dtype=float) for k, v in path_data.items()}
            self.path_data = self.path_dict["S"]
        else:
            self.path_data = np.asarray(path_data, dtype=float)
            self.path_dict = {"S": self.path_data}

        # Determine the volatility path for the state
        # SABR stores "sigma", SVJ stores "variance" (=> we take sqrt)
        if "sigma" in self.path_dict:
            self._vol_path = self.path_dict["sigma"]
        elif "variance" in self.path_dict:
            self._vol_path = np.sqrt(np.maximum(self.path_dict["variance"], 1e-10))
        else:
            # GBM: constant volatility
            self._vol_path = np.full_like(self.path_data, self.valuation_sigma)

        self.n_steps = len(self.path_data) - 1
        self.times = np.linspace(0.0, self.maturity, len(self.path_data))

        self.i = 0
        self.v_prev, _ = self._derivative_value(0)
        self.h_prev = 0.0

        self.episode_reward = 0.0
        self.episode_cost = 0.0

        return self._build_state(self.i, self.h_prev)

    def step(self, hedge: float) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        hedge = float(hedge)
        i = self.i
        spot_t = float(self.path_data[i])
        spot_next = float(self.path_data[i + 1])

        # Paper eq: trade cost uses S_{i+1}
        trade_cost = self.transac_cost * spot_next * abs(hedge - self.h_prev)
        v_next, _ = self._derivative_value(i + 1)
        reward = (v_next - self.v_prev) + hedge * (spot_next - spot_t) - trade_cost
        done = i == self.n_steps - 1

        liquidation_cost = 0.0
        if done:
            liquidation_cost = self.transac_cost * spot_next * abs(hedge)
            reward -= liquidation_cost

        self.episode_reward += reward
        self.episode_cost += -reward
        self.i += 1
        self.h_prev = 0.0 if done else hedge
        self.v_prev = v_next

        next_state = self._build_state(self.i, self.h_prev)
        info = {
            "spot_t": spot_t,
            "spot_next": spot_next,
            "hedge": hedge,
            "trade_cost": trade_cost,
            "liquidation_cost": liquidation_cost,
            "reward": reward,
            "cost": -reward,
            "episode_reward": self.episode_reward,
            "episode_cost": self.episode_cost,
        }
        return next_state, reward, done, info

    def option_price_t0(self) -> float:
        """BS price of the option at t=0 (used for normalising results)."""
        price, _ = self.valuation_engine.price_and_delta(
            spot=float(self.path_data[0]),
            t=0.0,
            sigma=self.valuation_sigma,
        )
        return abs(price)

    # ── private helpers ─────────────────────────────────────────────

    def _build_state(self, process_step: int, hedge_pos: float) -> np.ndarray:
        idx = min(process_step, len(self.path_data) - 1)
        t = self.times[min(process_step, len(self.times) - 1)]
        spot = self.path_data[idx]
        vol = self._vol_path[idx]
        ttm = max(self.maturity - t, 0.0)

        # Normalised components:
        #   holding:  already in [0, 1] (short call)
        #   spot:     log-moneyness  log(S/K)
        #   ttm:      fraction of total maturity  [0, 1]
        #   vol:      ratio to reference vol      σ_t / σ_ref  (≈1 under GBM)
        norm_spot = np.log(spot / self.valuation_engine.K)
        norm_ttm = ttm / self.maturity if self.maturity > 0 else 0.0
        norm_vol = vol / self.valuation_sigma

        return np.asarray([hedge_pos, norm_spot, norm_ttm, norm_vol], dtype=float)

    def _derivative_value(self, process_step: int) -> tuple[float, float]:
        price, delta = self.valuation_engine.price_and_delta(
            spot=float(self.path_data[process_step]),
            t=float(self.times[process_step]),
            sigma=self.valuation_sigma,
        )
        return self.position_sign * float(price), -self.position_sign * float(delta)
