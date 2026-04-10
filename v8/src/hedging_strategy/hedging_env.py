"""
Accounting P&L hedging environment — Cao et al. (2021), Section 3.1.

R_{i+1} = V_{i+1} - V_i + H_i(S_{i+1}-S_i) - κ|S_{i+1}(H_{i+1}-H_i)|
Initial cost: -κ|S_0 H_0|      (paper convention)
Final cost:   -κ|S_n H_n|

State (dim 4) = [holding, log(S/K), TTM/T, σ_t/σ_ref]
"""
from __future__ import annotations
from typing import Any
import numpy as np
from ..valuation.bs_valuation import BSValuation


class HedgingEnv:
    def __init__(self, config: dict[str, Any]) -> None:
        self.transac_cost = float(config["hedging_env"]["transaction_cost"])
        self.position_sign = float(config["hedging_env"]["position_sign"])
        self.derivative_type = config.get("derivative", {}).get("option_type", "call")
        self.valuation_sigma = float(config["simulation"]["gbm"]["sigma"])
        self.maturity = float(config["simulation"]["maturity"])
        self.valuation_engine = BSValuation(
            strike=config["derivative"]["strike"], maturity=self.maturity,
            rate=config["derivative"].get("rf_rate", 0.0),
            dividend=config["derivative"].get("div_rate", 0.0),
            option_type=self.derivative_type)

    def setup_env(self, path_data):
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
        self.n_steps = len(self.path_data) - 1
        self.times = np.linspace(0.0, self.maturity, len(self.path_data))
        self.i = 0
        self.v_prev, _ = self._derivative_value(0)
        self.h_prev = 0.0
        self.is_first_step = True
        self.episode_reward = 0.0
        self.episode_cost = 0.0
        return self._build_state(0, 0.0)

    def step(self, hedge: float):
        hedge = float(hedge)
        i = self.i
        spot_t = float(self.path_data[i])
        spot_next = float(self.path_data[i + 1])
        # Paper: initial setup cost uses S_0, subsequent use S_{i+1}
        if self.is_first_step:
            trade_cost = self.transac_cost * spot_t * abs(hedge - self.h_prev)
            self.is_first_step = False
        else:
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
        info = {"spot_t": spot_t, "spot_next": spot_next, "hedge": hedge,
                "trade_cost": trade_cost, "liquidation_cost": liquidation_cost,
                "reward": reward, "cost": -reward,
                "episode_reward": self.episode_reward, "episode_cost": self.episode_cost}
        return next_state, reward, done, info

    def option_price_t0(self) -> float:
        p, _ = self.valuation_engine.price_and_delta(
            spot=float(self.path_data[0]), t=0.0, sigma=self.valuation_sigma)
        return abs(p)

    def _build_state(self, step, hedge_pos):
        idx = min(step, len(self.path_data) - 1)
        t = self.times[min(step, len(self.times) - 1)]
        spot, vol = self.path_data[idx], self._vol_path[idx]
        ttm = max(self.maturity - t, 0.0)
        return np.asarray([hedge_pos,
                           np.log(spot / self.valuation_engine.K),
                           ttm / self.maturity if self.maturity > 0 else 0.0,
                           vol / self.valuation_sigma], dtype=float)

    def _derivative_value(self, step):
        p, d = self.valuation_engine.price_and_delta(
            spot=float(self.path_data[step]), t=float(self.times[step]),
            sigma=self.valuation_sigma)
        return self.position_sign * float(p), -self.position_sign * float(d)
