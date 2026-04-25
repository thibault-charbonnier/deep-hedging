import numpy as np
from scipy.stats import norm


class BSValuation:
    """Closed-form Black-Scholes pricer/delta for a European call or put."""

    def __init__(self, strike, maturity, rate=0.0, dividend=0.0, option_type="call"):
        self.K = float(strike)
        self.T = float(maturity)
        self.r = float(rate)
        self.q = float(dividend)
        self.option_type = option_type.lower()
        self.phi = 1.0 if self.option_type == "call" else -1.0

    def price_and_delta(self, spot, t, sigma):
        """Return ``(price, delta)`` at ``(spot, t, sigma)``"""
        S = np.asarray(spot, dtype=float)
        tau_raw = np.maximum(self.T - np.asarray(t, dtype=float), 0.0)
        sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
        phi = self.phi

        tau = np.maximum(tau_raw, 1e-14)
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * sigma_arr**2) * tau) / (sigma_arr * sqrt_tau)
        d2 = d1 - sigma_arr * sqrt_tau

        disc_q = np.exp(-self.q * tau)
        disc_r = np.exp(-self.r * tau)

        price = phi * (S * disc_q * norm.cdf(phi * d1) - self.K * disc_r * norm.cdf(phi * d2))
        delta = phi * disc_q * norm.cdf(phi * d1)

        payoff_sign = phi * (S - self.K)
        is_terminal = tau_raw <= 1e-14
        price = np.where(is_terminal, np.maximum(payoff_sign, 0.0), price)
        delta = np.where(is_terminal, phi * (payoff_sign > 0), delta)
        return price, delta


def option_price_t0(config: dict) -> float:
    """Return the absolute Black-Scholes price of the option at t=0 from a config dict."""
    maturity = float(config["simulation"]["maturity"])
    spot = float(config["simulation"]["S0"])
    sigma = float(config["simulation"]["gbm"]["sigma"])
    engine = BSValuation(
        strike=config["derivative"]["strike"],
        maturity=maturity,
        rate=config["derivative"].get("rf_rate", 0.0),
        dividend=config["derivative"].get("div_rate", 0.0),
        option_type=config.get("derivative", {}).get("option_type", "call"),
    )
    p, _ = engine.price_and_delta(spot=spot, t=0.0, sigma=sigma)
    return abs(float(p))
