"""
Bartlett delta under SABR
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
