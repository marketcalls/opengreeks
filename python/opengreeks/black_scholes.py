"""Black-Scholes (no-dividend) option pricing, Greeks, and implied volatility.

    from opengreeks.black_scholes import (
        black_scholes, implied_volatility,
        delta, gamma, vega, theta, rho,
    )

Signatures (no `q` parameter — equivalent to Black-Scholes-Merton with q=0):
    black_scholes(flag, S, K, t, r, sigma) -> float
    implied_volatility(price, S, K, t, r, flag) -> float
    delta/gamma/vega/theta/rho(flag, S, K, t, r, sigma) -> float
"""
from ._opengreeks import (
    bs_black_scholes as black_scholes,
    bs_implied_volatility as implied_volatility,
    bs_delta as delta,
    bs_gamma as gamma,
    bs_vega as vega,
    bs_theta as theta,
    bs_rho as rho,
    bs_black_scholes_array as black_scholes_array,
    bs_implied_volatility_array as implied_volatility_array,
    bs_delta_array as delta_array,
    bs_gamma_array as gamma_array,
    bs_vega_array as vega_array,
    bs_theta_array as theta_array,
    bs_rho_array as rho_array,
)

__all__ = [
    "black_scholes", "implied_volatility",
    "delta", "gamma", "vega", "theta", "rho",
    "black_scholes_array", "implied_volatility_array",
    "delta_array", "gamma_array", "vega_array", "theta_array", "rho_array",
]
