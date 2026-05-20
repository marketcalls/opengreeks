"""OpenGreeks — fast options pricing & Greeks. Rust core, Python API.

Submodules organize functions by pricing model:

    >>> from opengreeks.black76 import black, implied_volatility, delta
    >>> from opengreeks.black_scholes import black_scholes, implied_volatility
    >>> from opengreeks.black_scholes_merton import black_scholes_merton

Or namespaced:

    >>> import opengreeks
    >>> opengreeks.black76.black('c', 22000.0, 22000.0, 30/365, 0.07, 0.18)

Numerical conventions: vega × 0.01 (per 1% IV move), theta / 365 (per calendar
day), rho × 0.01 (per 1% rate move).
"""
from ._opengreeks import __version__
from . import black76, black_scholes, black_scholes_merton

__all__ = ["__version__", "black76", "black_scholes", "black_scholes_merton"]
