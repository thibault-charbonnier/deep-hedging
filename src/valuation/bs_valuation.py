import numpy as np
from scipy.stats import norm


class BSValuation:
    def __init__(self, strike, maturity, rate=0.0, dividend=0.0, option_type="call"):
        self.K = float(strike)
        self.T = float(maturity)
        self.r = float(rate)
        self.q = float(dividend)
        self.option_type = option_type.lower()

    def price_and_delta(self, spot, t, sigma):
        # Vectorized implementation (works for scalars and arrays).
        S = np.asarray(spot, dtype=float)
        tau_raw = np.maximum(self.T - np.asarray(t, dtype=float), 0.0)
        sigma_arr = np.maximum(np.asarray(sigma, dtype=float), 1e-12)

        tau = np.maximum(tau_raw, 1e-14)
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * sigma_arr**2) * tau) / (sigma_arr * sqrt_tau)
        d2 = d1 - sigma_arr * sqrt_tau

        disc_q = np.exp(-self.q * tau)
        disc_r = np.exp(-self.r * tau)
        cdf_d1 = norm.cdf(d1)

        if self.option_type == "call":
            cdf_d2 = norm.cdf(d2)
            price = S * disc_q * cdf_d1 - self.K * disc_r * cdf_d2
            delta = disc_q * cdf_d1
            # Exact terminal payoff/delta where tau is truly zero.
            terminal_price = np.maximum(S - self.K, 0.0)
            terminal_delta = np.where(S > self.K, 1.0, 0.0)
        else:
            cdf_minus_d2 = norm.cdf(-d2)
            cdf_minus_d1 = norm.cdf(-d1)
            price = self.K * disc_r * cdf_minus_d2 - S * disc_q * cdf_minus_d1
            delta = disc_q * (cdf_d1 - 1.0)
            terminal_price = np.maximum(self.K - S, 0.0)
            terminal_delta = np.where(S < self.K, -1.0, 0.0)

        is_terminal = tau_raw <= 1e-14
        price = np.where(is_terminal, terminal_price, price)
        delta = np.where(is_terminal, terminal_delta, delta)
        return price, delta
