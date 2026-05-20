"""Black-Scholes-Merton option pricing, Greeks, and implied volatility.

    from opengreeks.black_scholes_merton import (
        black_scholes_merton, implied_volatility,
        delta, gamma, vega, theta, rho,
    )

Signatures (with continuous dividend yield `q`):
    black_scholes_merton(flag, S, K, t, r, sigma, q) -> float
    implied_volatility(price, S, K, t, r, q, flag) -> float
    delta/gamma/vega/theta/rho(flag, S, K, t, r, sigma, q) -> float
"""
from ._opengreeks import (
    bsm_black_scholes_merton as black_scholes_merton,
    bsm_implied_volatility as implied_volatility,
    bsm_delta as delta,
    bsm_gamma as gamma,
    bsm_vega as vega,
    bsm_theta as theta,
    bsm_rho as rho,
    bsm_black_scholes_merton_array as black_scholes_merton_array,
    bsm_implied_volatility_array as implied_volatility_array,
    bsm_delta_array as delta_array,
    bsm_gamma_array as gamma_array,
    bsm_vega_array as vega_array,
    bsm_theta_array as theta_array,
    bsm_rho_array as rho_array,
)

__all__ = [
    "black_scholes_merton", "implied_volatility",
    "delta", "gamma", "vega", "theta", "rho",
    "black_scholes_merton_array", "implied_volatility_array",
    "delta_array", "gamma_array", "vega_array", "theta_array", "rho_array",
]
