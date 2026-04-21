"""
Practitioner delta under SABR — Paper Section 5.

"A popular hedging procedure involves using a delta calculated by
 assuming the Black-Scholes model with σ set equal to the current
 implied volatility."

At each step, we compute the SABR implied vol from the current σ_t,
then compute the BS delta with that implied vol.
"""
from __future__ import annotations
import math
import numpy as np
from ..valuation.sabr_valuation import sabr_implied_vol, bs_delta_from_vol


class SABRPractitionerDeltaBenchmark:
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

        F = spot * math.exp((self.r - self.q) * ttm)
        sigma_impl = sabr_implied_vol(F, self.K, ttm, sigma_t, self.nu, self.rho)
        delta = bs_delta_from_vol(spot, self.K, ttm, self.r, self.q,
                                  sigma_impl, self.option_type)
        return float(-self.position_sign * delta)
