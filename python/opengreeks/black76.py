"""Black-76 option pricing, Greeks, and implied volatility.

    from opengreeks.black76 import black, implied_volatility, delta, gamma, vega, theta, rho

Signatures:
    black(flag, F, K, t, r, sigma) -> float
    implied_volatility(price, F, K, r, t, flag) -> float
    delta/gamma/vega/theta/rho(flag, F, K, t, r, sigma) -> float

Plus NumPy batch variants (`*_array` suffix) for chain-wide computation:
    black_array(flag, F_arr, K_arr, t_arr, r_scalar, sigma_arr) -> ndarray

Argument convention: `flag` is `'c'` for call or `'p'` for put. Returned vega
is multiplied by 0.01 (price change per 1% IV move); theta is divided by 365
(per calendar day); rho is multiplied by 0.01.
"""
from ._opengreeks import (
    black76_black as black,
    black76_implied_volatility as implied_volatility,
    black76_delta as delta,
    black76_gamma as gamma,
    black76_vega as vega,
    black76_theta as theta,
    black76_rho as rho,
    # second / third order
    black76_vanna as vanna,
    black76_charm as charm,
    black76_vomma as vomma,
    black76_speed as speed,
    black76_zomma as zomma,
    black76_color as color,
    black76_veta as veta,
    black76_ultima as ultima,
    black76_dual_delta as dual_delta,
    black76_dual_gamma as dual_gamma,
    black76_black_array as black_array,
    black76_implied_volatility_array as implied_volatility_array,
    black76_delta_array as delta_array,
    black76_gamma_array as gamma_array,
    black76_vega_array as vega_array,
    black76_theta_array as theta_array,
    black76_rho_array as rho_array,
    black76_vanna_array as vanna_array,
    black76_charm_array as charm_array,
    black76_vomma_array as vomma_array,
    black76_speed_array as speed_array,
    black76_zomma_array as zomma_array,
    black76_color_array as color_array,
    black76_veta_array as veta_array,
    black76_ultima_array as ultima_array,
    black76_dual_delta_array as dual_delta_array,
    black76_dual_gamma_array as dual_gamma_array,
)

__all__ = [
    "black", "implied_volatility",
    "delta", "gamma", "vega", "theta", "rho",
    "vanna", "charm", "vomma", "speed", "zomma", "color",
    "veta", "ultima", "dual_delta", "dual_gamma",
    "black_array", "implied_volatility_array",
    "delta_array", "gamma_array", "vega_array", "theta_array", "rho_array",
    "vanna_array", "charm_array", "vomma_array", "speed_array", "zomma_array",
    "color_array", "veta_array", "ultima_array", "dual_delta_array", "dual_gamma_array",
]
