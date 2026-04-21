"""
Bartlett delta under SABR — Bartlett (2006), Paper Section 5.

"Bartlett provides a better estimate of delta for the SABR model by
 considering both the impact of a change in S and the corresponding
 expected change in σ."

Δ_Bartlett = Δ_BS(σ_impl) + vega_BS · ρ ν / S
"""
from __future__ import annotations
import numpy as np
from ..valuation.sabr_valuation import bartlett_delta as _bartlett_delta


class BartlettDeltaBenchmark:
    def __init__(self, config: dict) -> None:
        self.position_sign = float(config["hedging_env"]["position_sign"])
        self.option_type = config.get("derivative", {}).get("option_type", "call")
        self.K = float(config["derivative"]["strike"])
        self.maturity = float(config["simulation"]["maturity"])
        self.r = float(config["derivative"].get("rf_rate", 0.0))
        self.q = float(config["derivative"].get("div_rate", 0.0))
        self.sigma_ref = float(config["simulation"]["gbm"]["sigma"])
        self.nu = float(config["simulation"]["sabr"]["nu"])
        self.rho = float(config["simulation"]["sabr"]["rho"])

    def __call__(self, state: np.ndarray, sigma_t: float | None = None) -> float:
        _, spot, ttm = state
        spot = float(spot)
        ttm = float(ttm)
        sigma_t = self.sigma_ref if sigma_t is None else float(sigma_t)

        if ttm <= 1e-14:
            if self.option_type == "call":
                return float(-self.position_sign * (1.0 if spot > self.K else 0.0))
            return float(-self.position_sign * (-1.0 if spot < self.K else 0.0))

        delta = _bartlett_delta(spot, self.K, ttm, self.r, self.q,
                                sigma_t, self.nu, self.rho, self.option_type)
        return float(-self.position_sign * delta)
