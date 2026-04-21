"""Raw hedging environment exposing delayed-reward ingredients (paper Section 3.1)."""
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
        self.K = float(config["derivative"]["strike"])
        self.valuation_engine = BSValuation(
            strike=self.K,
            maturity=self.maturity,
            rate=config["derivative"].get("rf_rate", 0.0),
            dividend=config["derivative"].get("div_rate", 0.0),
            option_type=self.derivative_type,
        )

    def setup_env(self, path_data):
        if isinstance(path_data, dict):
            paths = {k: np.asarray(v, dtype=float) for k, v in path_data.items()}
        else:
            paths = {"S": np.asarray(path_data, dtype=float)}

        self.path_data = paths["S"]
        self.n_steps = len(self.path_data) - 1
        self.times = np.linspace(0.0, self.maturity, len(self.path_data))

        # Precompute derivative values V_i along the path.
        p, _ = self.valuation_engine.price_and_delta(
            spot=self.path_data,
            t=self.times,
            sigma=self.valuation_sigma,
        )
        self._V = self.position_sign * np.asarray(p, dtype=float)
        self.i = 0
        self.H_prev = 0.0
        return self._build_state(0, self.H_prev)

    def apply_action(self, hedge: float):
        """Register H_i and return raw quantities; reward is assembled by the orchestrator."""
        H_new = float(hedge)
        i = self.i
        if i > self.n_steps:
            raise RuntimeError("apply_action called after terminal step")

        raw = {
            "S_i": float(self.path_data[i]),
            "V_i": float(self._V[i]),
            "H_prev": float(self.H_prev),
            "H_new": H_new,
        }

        self.H_prev = H_new
        if i < self.n_steps:
            self.i += 1
            next_state = self._build_state(self.i, self.H_prev)
        else:
            next_state = None

        return next_state, raw

    def _build_state(self, step, hedge_pos):
        """Return state as [holding, moneyness=S/K, ttm_norm=ttm/maturity].

        The holding stays in its native scale (fraction of underlying, typically
        in [0, 1.2]). The moneyness and normalized time-to-maturity are
        dimensionless and lie in comparable ranges, which is suitable for MLP
        inputs without further preprocessing.
        """
        idx = min(step, len(self.path_data) - 1)
        t = float(self.times[min(step, len(self.times) - 1)])
        spot = float(self.path_data[idx])
        ttm = float(np.maximum(self.maturity - t, 0.0))
        moneyness = spot / self.K
        ttm_norm = ttm / self.maturity if self.maturity > 0 else 0.0
        return np.asarray([
            hedge_pos,
            moneyness,
            ttm_norm,
        ], dtype=np.float32)

