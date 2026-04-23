"""BS delta benchmark (constant σ) — used under GBM."""
from __future__ import annotations
import numpy as np
from ..valuation.bs_valuation import BSValuation


class BSDeltaBenchmark:
    """Black-Scholes delta-hedging benchmark with a constant volatility.

    Callable on the HedgingEnv state: returns the hedge position that
    offsets the BS delta of the option, scaled by the position sign.
    """

    def __init__(self, config: dict) -> None:
        self.position_sign = float(config["hedging_env"]["position_sign"])
        self.option_type = config.get("derivative", {}).get("option_type", "call")
        self.bs = BSValuation(
            strike=config["derivative"]["strike"],
            maturity=config["simulation"]["maturity"],
            rate=config["derivative"].get("rf_rate", 0.0),
            dividend=config["derivative"].get("div_rate", 0.0),
            option_type=self.option_type)
        self.sigma = float(config["simulation"]["gbm"]["sigma"])
        self.maturity = float(config["simulation"]["maturity"])

    def __call__(self, state: np.ndarray) -> float:
        """Return the hedge position from the current state (log-moneyness, normalized TTM)."""
        _, log_m, norm_ttm, _ = state
        spot = self.bs.K * float(np.exp(log_m))
        t = self.maturity * (1.0 - float(norm_ttm))
        _, delta = self.bs.price_and_delta(spot=spot, t=t, sigma=self.sigma)
        return float(-self.position_sign * delta)
