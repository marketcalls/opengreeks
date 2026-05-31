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
    # second / third order
    bs_vanna as vanna,
    bs_charm as charm,
    bs_vomma as vomma,
    bs_speed as speed,
    bs_zomma as zomma,
    bs_color as color,
    bs_veta as veta,
    bs_ultima as ultima,
    bs_dual_delta as dual_delta,
    bs_dual_gamma as dual_gamma,
    bs_black_scholes_array as black_scholes_array,
    bs_implied_volatility_array as implied_volatility_array,
    bs_delta_array as delta_array,
    bs_gamma_array as gamma_array,
    bs_vega_array as vega_array,
    bs_theta_array as theta_array,
    bs_rho_array as rho_array,
    bs_vanna_array as vanna_array,
    bs_charm_array as charm_array,
    bs_vomma_array as vomma_array,
    bs_speed_array as speed_array,
    bs_zomma_array as zomma_array,
    bs_color_array as color_array,
    bs_veta_array as veta_array,
    bs_ultima_array as ultima_array,
    bs_dual_delta_array as dual_delta_array,
    bs_dual_gamma_array as dual_gamma_array,
)

__all__ = [
    "black_scholes", "implied_volatility",
    "delta", "gamma", "vega", "theta", "rho",
    "vanna", "charm", "vomma", "speed", "zomma", "color",
    "veta", "ultima", "dual_delta", "dual_gamma",
    "black_scholes_array", "implied_volatility_array",
    "delta_array", "gamma_array", "vega_array", "theta_array", "rho_array",
    "vanna_array", "charm_array", "vomma_array", "speed_array", "zomma_array",
    "color_array", "veta_array", "ultima_array", "dual_delta_array", "dual_gamma_array",
]
