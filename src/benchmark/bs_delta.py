import numpy as np
from ..valuation import BSValuation


class BSDeltaBenchmark:
    """
    Classic deterministic Black-Scholes delta hedging strategy.
    """

    def __init__(self, config: dict) -> None:
        """_
        Parameters
        ----------
        config : dict
            The configuration dictionary containing the necessary parameters for the Black-Scholes valuation.
        """
        self.config = config
        self.bs_valuation = BSValuation(
            strike=self.config.get("derivative").get("strike"),
            maturity=config.get("simulation").get("maturity"),
            rate=config.get("derivative").get("rf_rate"),
            dividend=config.get("derivative").get("div_rate")
        )
        self.sigma = self.config.get("simulation").get("gbm").get("sigma")
        self.maturity = self.config.get("simulation").get("maturity")

    def __call__(self, state: np.ndarray) -> float:
        """
        Compute the hedge position according to BS delta strategy.

        Parameters
        ----------
        state : np.ndarray
            The state vector containing the current hedge position, spot price, and time to maturity.

        Returns
        -------
        float
            The hedge position to take according to the Black-Scholes delta hedging strategy.
        """
        _, spot, time_to_maturity = state
        _, delta = self.bs_valuation.price_and_delta(
            spot=spot, t=self.maturity - time_to_maturity,
            sigma=self.sigma
        )
        return delta