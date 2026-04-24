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

from ..valuation.sabr_valuation import sabr_implied_vol, bs_delta_from_vol
from ._sabr_base import _SABRDeltaBase


class SABRPractitionerDeltaBenchmark(_SABRDeltaBase):
    """SABR practitioner delta benchmark — see module docstring."""

    def _compute_delta(self, spot: float, ttm: float, sigma_t: float) -> float:
        """Compute the SABR implied vol from (spot, ttm, sigma_t) and return the BS delta at it."""
        F = spot * math.exp((self.r - self.q) * ttm)
        sigma_impl = sabr_implied_vol(F, self.K, ttm, sigma_t, self.nu, self.rho)
        return bs_delta_from_vol(spot, self.K, ttm, self.r, self.q,
                                 sigma_impl, self.option_type)
