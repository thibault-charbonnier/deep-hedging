import math
from statistics import NormalDist


class BSValuation:
    """
    Minimal valuation engine to price and compute option's delta.
    """

    def __init__(
        self,
        strike: float,
        maturity: float,
        rate: float = 0.0,
        dividend: float = 0.0,
        option_type: str = "call",
    ) -> None:
        self.K = float(strike)
        self.T = float(maturity)
        self.r = float(rate)
        self.q = float(dividend)
        self.option_type = option_type.lower()

        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be either 'call' or 'put'.")

        self._normal = NormalDist()

    def price_and_delta(
        self,
        spot: float,
        t: float,
        sigma: float,
    ) -> tuple[float, float]:
        """
        Compute Black-Scholes price and delta at time t.

        Parameters
        ----------
        spot : float
            Current underlying spot S_t.
        t : float
            Current time.
        sigma : float
            Volatility used for valuation at time t.

        Returns
        -------
        tuple[float, float]
            (price, delta) of the given option.
        """
        S = float(spot)
        sigma = max(float(sigma), 1e-12)
        tau = max(self.T - float(t), 0.0)

        # Terminal case (discretization robust)
        if tau <= 1e-14:
            if self.option_type == "call":
                price = max(S - self.K, 0.0)
                delta = 1.0 if S > self.K else 0.0
            else:
                price = max(self.K - S, 0.0)
                delta = -1.0 if S < self.K else 0.0
            return price, delta

        sqrt_tau = math.sqrt(tau)

        d1 = (
            math.log(S / self.K)
            + (self.r - self.q + 0.5 * sigma * sigma) * tau
        ) / (sigma * sqrt_tau)

        d2 = d1 - sigma * sqrt_tau

        Nd1 = self._normal.cdf(d1)
        Nd2 = self._normal.cdf(d2)

        if self.option_type == "call":
            price = (
                S * math.exp(-self.q * tau) * Nd1
                - self.K * math.exp(-self.r * tau) * Nd2
            )
            delta = math.exp(-self.q * tau) * Nd1
        else:
            N_minus_d1 = self._normal.cdf(-d1)
            N_minus_d2 = self._normal.cdf(-d2)

            price = (
                self.K * math.exp(-self.r * tau) * N_minus_d2
                - S * math.exp(-self.q * tau) * N_minus_d1
            )
            delta = math.exp(-self.q * tau) * (Nd1 - 1.0)

        return price, delta