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
    # second / third order
    bsm_vanna as vanna,
    bsm_charm as charm,
    bsm_vomma as vomma,
    bsm_speed as speed,
    bsm_zomma as zomma,
    bsm_color as color,
    bsm_veta as veta,
    bsm_ultima as ultima,
    bsm_dual_delta as dual_delta,
    bsm_dual_gamma as dual_gamma,
    bsm_black_scholes_merton_array as black_scholes_merton_array,
    bsm_implied_volatility_array as implied_volatility_array,
    bsm_delta_array as delta_array,
    bsm_gamma_array as gamma_array,
    bsm_vega_array as vega_array,
    bsm_theta_array as theta_array,
    bsm_rho_array as rho_array,
    bsm_vanna_array as vanna_array,
    bsm_charm_array as charm_array,
    bsm_vomma_array as vomma_array,
    bsm_speed_array as speed_array,
    bsm_zomma_array as zomma_array,
    bsm_color_array as color_array,
    bsm_veta_array as veta_array,
    bsm_ultima_array as ultima_array,
    bsm_dual_delta_array as dual_delta_array,
    bsm_dual_gamma_array as dual_gamma_array,
)

__all__ = [
    "black_scholes_merton", "implied_volatility",
    "delta", "gamma", "vega", "theta", "rho",
    "vanna", "charm", "vomma", "speed", "zomma", "color",
    "veta", "ultima", "dual_delta", "dual_gamma",
    "black_scholes_merton_array", "implied_volatility_array",
    "delta_array", "gamma_array", "vega_array", "theta_array", "rho_array",
    "vanna_array", "charm_array", "vomma_array", "speed_array", "zomma_array",
    "color_array", "veta_array", "ultima_array", "dual_delta_array", "dual_gamma_array",
]
