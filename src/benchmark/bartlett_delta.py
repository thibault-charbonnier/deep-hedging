"""
Bartlett delta under SABR — Bartlett (2006), Paper Section 5.

"Bartlett provides a better estimate of delta for the SABR model by
 considering both the impact of a change in S and the corresponding
 expected change in σ."

Δ_Bartlett = Δ_BS(σ_impl) + vega_BS · ρ ν / S
"""
from __future__ import annotations

from ..valuation.sabr_valuation import bartlett_delta as _bartlett_delta
from ._sabr_base import _SABRDeltaBase


class BartlettDeltaBenchmark(_SABRDeltaBase):
    """SABR Bartlett delta benchmark — see module docstring."""

    def _compute_delta(self, spot: float, ttm: float, sigma_t: float) -> float:
        """Call the closed-form Bartlett delta (BS delta + vega·ρ·ν/S correction)."""
        return _bartlett_delta(spot, self.K, ttm, self.r, self.q,
                               sigma_t, self.nu, self.rho, self.option_type)
