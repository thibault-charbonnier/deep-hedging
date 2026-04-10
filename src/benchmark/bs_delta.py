from __future__ import annotations

import numpy as np

from ..valuation.bs_valuation import BSValuation


class BSDeltaBenchmark:
    """
    Black-Scholes delta benchmark expressed as stock position.

    For a short option position, the hedge is the opposite of the option delta.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.position_sign = float(config["hedging_env"]["position_sign"])
        self.option_type = config.get("derivative", {}).get("option_type", "call")
        self.bs_valuation = BSValuation(
            strike=config["derivative"]["strike"],
            maturity=config["simulation"]["maturity"],
            rate=config["derivative"].get("rf_rate", 0.0),
            dividend=config["derivative"].get("div_rate", 0.0),
            option_type=self.option_type,
        )
        self.sigma = float(config["simulation"]["gbm"]["sigma"])
        self.maturity = float(config["simulation"]["maturity"])

    def __call__(self, state: np.ndarray) -> float:
        # State: [holding, log_moneyness, norm_ttm, norm_vol]
        _, log_moneyness, norm_ttm, _ = state
        spot = self.bs_valuation.K * float(np.exp(log_moneyness))
        t = self.maturity * (1.0 - float(norm_ttm))
        _, delta = self.bs_valuation.price_and_delta(
            spot=spot,
            t=t,
            sigma=self.sigma,
        )
        return float(-self.position_sign * delta)
