"""Shared base for SABR-style delta benchmarks."""
from __future__ import annotations

import numpy as np


class _SABRDeltaBase:
    """Common init, state decomposition and terminal payoff handling for SABR delta benchmarks.

    Subclasses implement ``_compute_delta(spot, ttm, sigma_t)`` for the
    non-terminal case; the terminal payoff delta (``phi`` at ITM, 0 at
    OTM) is handled here via the ``phi = +1`` (call) / ``-1`` (put) sign.
    """

    def __init__(self, config: dict) -> None:
        self.position_sign = float(config["hedging_env"]["position_sign"])
        self.option_type = config.get("derivative", {}).get("option_type", "call")
        self.phi = 1.0 if self.option_type == "call" else -1.0
        self.K = float(config["derivative"]["strike"])
        self.maturity = float(config["simulation"]["maturity"])
        self.r = float(config["derivative"].get("rf_rate", 0.0))
        self.q = float(config["derivative"].get("div_rate", 0.0))
        # σ_ref must match HedgingEnv.sigma_ref (= σ_0 of the process on path),
        # which under SABR is sabr.sigma0 — not gbm.sigma.
        self.sigma_ref = float(config["simulation"]["sabr"]["sigma0"])
        self.nu = float(config["simulation"]["sabr"]["nu"])
        self.rho = float(config["simulation"]["sabr"]["rho"])

    def _compute_delta(self, spot: float, ttm: float, sigma_t: float) -> float:
        """Return the option delta at ``(spot, ttm, sigma_t)`` (non-terminal case)."""
        raise NotImplementedError

    def __call__(self, state: np.ndarray) -> float:
        """Return the hedge position for ``state``, with terminal payoff fall-through."""
        _, log_m, norm_ttm, norm_vol = state
        spot = self.K * float(np.exp(log_m))
        ttm = self.maturity * float(norm_ttm)
        sigma_t = float(norm_vol) * self.sigma_ref

        if ttm <= 1e-14:
            # Terminal payoff delta: phi at ITM, 0 at OTM.
            delta = self.phi if self.phi * (spot - self.K) > 0 else 0.0
        else:
            delta = self._compute_delta(spot, ttm, sigma_t)
        return float(-self.position_sign * delta)
