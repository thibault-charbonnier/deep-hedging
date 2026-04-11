import math
from statistics import NormalDist


class BSValuation:
    def __init__(self, strike, maturity, rate=0.0, dividend=0.0, option_type="call"):
        self.K = float(strike)
        self.T = float(maturity)
        self.r = float(rate)
        self.q = float(dividend)
        self.option_type = option_type.lower()
        self._N = NormalDist()

    def price_and_delta(self, spot, t, sigma):
        S, sigma = float(spot), max(float(sigma), 1e-12)
        tau = max(self.T - float(t), 0.0)
        if tau <= 1e-14:
            if self.option_type == "call":
                return max(S - self.K, 0.0), 1.0 if S > self.K else 0.0
            return max(self.K - S, 0.0), -1.0 if S < self.K else 0.0
        sqrt_tau = math.sqrt(tau)
        d1 = (math.log(S/self.K) + (self.r - self.q + 0.5*sigma*sigma)*tau) / (sigma*sqrt_tau)
        d2 = d1 - sigma*sqrt_tau
        if self.option_type == "call":
            price = S*math.exp(-self.q*tau)*self._N.cdf(d1) - self.K*math.exp(-self.r*tau)*self._N.cdf(d2)
            delta = math.exp(-self.q*tau)*self._N.cdf(d1)
        else:
            price = self.K*math.exp(-self.r*tau)*self._N.cdf(-d2) - S*math.exp(-self.q*tau)*self._N.cdf(-d1)
            delta = math.exp(-self.q*tau)*(self._N.cdf(d1) - 1.0)
        return price, delta
