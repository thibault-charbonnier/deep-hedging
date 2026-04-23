"""
SABR implied vol (Hagan et al. 2002) and Bartlett delta (2006).
Paper Section 5, β=1 case.
"""
import math
from statistics import NormalDist

_N = NormalDist()


def sabr_implied_vol(F, K, T, sigma0, nu, rho):
    """Hagan et al. (2002) implied vol for SABR β=1."""
    if T <= 1e-14 or sigma0 <= 1e-12:
        return sigma0
    B = 1.0 + (rho*nu*sigma0/4.0 + (2.0 - 3.0*rho**2)*nu**2/24.0) * T
    if abs(F - K) < 1e-8 * max(F, 1e-8):
        return sigma0 * B
    phi = (nu / sigma0) * math.log(F / K)
    disc = max(1.0 - 2.0*rho*phi + phi**2, 1e-12)
    chi = math.log((math.sqrt(disc) + phi - rho) / (1.0 - rho))
    if abs(chi) < 1e-12:
        return sigma0 * B
    return sigma0 * B * phi / chi


def bs_delta_from_vol(S, K, T, r, q, sigma_impl, option_type="call"):
    """BS delta given an implied vol."""
    if T <= 1e-14:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma_impl**2)*T) / (sigma_impl*sqrt_T)
    if option_type == "call":
        return math.exp(-q*T) * _N.cdf(d1)
    return math.exp(-q*T) * (_N.cdf(d1) - 1.0)


def bartlett_delta(S, K, T, r, q, sigma_t, nu, rho, option_type="call"):
    """
    Bartlett (2006) delta for SABR β=1.
    Δ_Bartlett = Δ_BS(σ_impl) + vega_BS · ρ ν / S
    """
    F = S * math.exp((r - q) * T)
    sigma_impl = sabr_implied_vol(F, K, T, sigma_t, nu, rho)
    if T <= 1e-14:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma_impl**2)*T) / (sigma_impl*sqrt_T)
    if option_type == "call":
        delta_bs = math.exp(-q*T) * _N.cdf(d1)
    else:
        delta_bs = math.exp(-q*T) * (_N.cdf(d1) - 1.0)
    vega = S * math.exp(-q*T) * _N.pdf(d1) * sqrt_T
    return delta_bs + vega * rho * nu / S
