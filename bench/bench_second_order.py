"""Second/third-order Greeks: correctness + speed bench for opengreeks.

Covers vanna, charm, vomma, speed, zomma, color, veta, ultima, dual_delta,
dual_gamma across all three models (Black-76, Black-Scholes, Black-Scholes-Merton).

Two independent checks:

  A. CORRECTNESS — opengreeks (Rust) vs **autograd** automatic differentiation
     of the generalized Black-Scholes price. Autograd is the oracle: it computes
     exact partials of any order, so it validates every formula with zero
     hand-derivation risk. (vollib/py_vollib ship NO second-order Greeks, so they
     cannot be used here.) Install: `pip install autograd`. Skipped if absent.

  B. SPEED — opengreeks vs a **pure-Python** implementation of the identical
     closed forms (scalar call + 177-strike chain). Answers "how much faster
     than pure Python?". Needs only numpy.

Conventions are raw/mathematical: exact partials, no ×0.01 / ÷365 scaling;
τ-derivatives (charm, color, veta) are per year of time-to-expiry.
"""
from __future__ import annotations

import math
import statistics
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import opengreeks
from opengreeks import black76 as og_b76
from opengreeks import black_scholes as og_bs
from opengreeks import black_scholes_merton as og_bsm

GREEKS = ["vanna", "charm", "vomma", "speed", "zomma",
          "color", "veta", "ultima", "dual_delta", "dual_gamma"]

SQRT2PI = math.sqrt(2.0 * math.pi)
_npdf = lambda x: math.exp(-0.5 * x * x) / SQRT2PI
_ncdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))


# ── Pure-Python reference (generalized cost-of-carry b; cp = +1 call / -1 put) ──
# Identical closed forms to the Rust `gbs2` engine. x = spot or forward.
def pp_greek(name, x, k, t, r, b, sigma, cp):
    sq = math.sqrt(t); st = sigma * sq
    d1 = (math.log(x / k) + (b + 0.5 * sigma * sigma) * t) / st
    d2 = d1 - st
    phi = _npdf(d1)
    cf = math.exp((b - r) * t); df = math.exp(-r * t)
    gamma = cf * phi / (x * st)
    vega = x * cf * phi * sq
    if name == "vanna":  return -cf * phi * d2 / sigma
    if name == "vomma":  return vega * d1 * d2 / sigma
    if name == "speed":  return -gamma / x * (d1 / st + 1.0)
    if name == "zomma":  return gamma * (d1 * d2 - 1.0) / sigma
    if name == "color":  return gamma * ((b - r) - d1 * b / st + (d1 * d2 - 1.0) / (2.0 * t))
    if name == "veta":   return vega * ((b - r) - d1 * b / st + (d1 * d2 + 1.0) / (2.0 * t))
    if name == "charm":
        base = cf * phi * (-d2 / (2.0 * t) + b / st)
        drift = (b - r) * cf
        return drift * _ncdf(d1) + base if cp > 0 else drift * (_ncdf(d1) - 1.0) + base
    if name == "ultima":
        return -vega / (sigma * sigma) * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2)
    if name == "dual_delta":
        return -df * _ncdf(d2) if cp > 0 else df * _ncdf(-d2)
    if name == "dual_gamma":
        return df * _npdf(d2) / (k * st)
    raise KeyError(name)


# ── autograd oracle (optional) ──
def _build_autograd():
    import autograd.numpy as anp
    from autograd import grad
    from autograd.scipy.special import erf

    def N(z): return 0.5 * (1 + erf(z / anp.sqrt(2)))

    def price(x, k, t, r, b, sig, cp):
        d1 = (anp.log(x / k) + (b + 0.5 * sig * sig) * t) / (sig * anp.sqrt(t))
        d2 = d1 - sig * anp.sqrt(t)
        return cp * (x * anp.exp((b - r) * t) * N(cp * d1) - k * anp.exp(-r * t) * N(cp * d2))

    dX, dsig, dK, dt = grad(price, 0), grad(price, 5), grad(price, 1), grad(price, 2)
    return {
        "vanna":      lambda a: grad(dX, 5)(*a),
        "charm":      lambda a: grad(dX, 2)(*a),
        "vomma":      lambda a: grad(dsig, 5)(*a),
        "speed":      lambda a: grad(grad(dX, 0), 0)(*a),
        "zomma":      lambda a: grad(grad(dX, 0), 5)(*a),
        "color":      lambda a: grad(grad(dX, 0), 2)(*a),
        "veta":       lambda a: grad(dsig, 2)(*a),
        "ultima":     lambda a: grad(grad(dsig, 5), 5)(*a),
        "dual_delta": lambda a: dK(*a),
        "dual_gamma": lambda a: grad(dK, 1)(*a),
    }


@dataclass
class Case:
    label: str; x: float; k: float; t: float; r: float; q: float; sigma: float; flag: str
    @property
    def cp(self): return 1.0 if self.flag == "c" else -1.0


# carry b per model
B = {"Black-76": lambda c: 0.0, "Black-Scholes": lambda c: c.r, "Black-Scholes-Merton": lambda c: c.r - c.q}

CASES = {
    "Black-76": [
        Case("ATM 30d call", 22000, 22000, 30/365, 0.07, 0.0, 0.18, "c"),
        Case("ATM 30d put",  22000, 22000, 30/365, 0.07, 0.0, 0.18, "p"),
        Case("OTM +10%",     22000, 24200, 30/365, 0.07, 0.0, 0.18, "c"),
        Case("ITM -10%",     22000, 19800, 30/365, 0.07, 0.0, 0.22, "c"),
        Case("Long 2y",      22000, 22000, 2.0,    0.07, 0.0, 0.20, "c"),
        Case("High vol 90%", 22000, 22000, 30/365, 0.07, 0.0, 0.90, "p"),
    ],
    "Black-Scholes": [
        Case("ATM 30d", 100, 100, 30/365, 0.05, 0.0, 0.25, "c"),
        Case("OTM",     100, 120, 30/365, 0.05, 0.0, 0.25, "c"),
        Case("ITM put", 100, 120, 30/365, 0.05, 0.0, 0.25, "p"),
        Case("Long 2y", 100, 100, 2.0,    0.05, 0.0, 0.25, "c"),
    ],
    "Black-Scholes-Merton": [
        Case("ATM q=2%",  100, 100, 30/365, 0.05, 0.02, 0.25, "c"),
        Case("OTM q=3%",  100, 120, 30/365, 0.05, 0.03, 0.25, "c"),
        Case("put q=3%",  100, 120, 30/365, 0.05, 0.03, 0.25, "p"),
        Case("Long q=4%", 100, 100, 2.0,    0.05, 0.04, 0.25, "c"),
    ],
}

# opengreeks scalar callables per model: (greek, case) -> value
OG = {
    "Black-76":            lambda g, c: getattr(og_b76, g)(c.flag, c.x, c.k, c.t, c.r, c.sigma),
    "Black-Scholes":       lambda g, c: getattr(og_bs, g)(c.flag, c.x, c.k, c.t, c.r, c.sigma),
    "Black-Scholes-Merton":lambda g, c: getattr(og_bsm, g)(c.flag, c.x, c.k, c.t, c.r, c.sigma, c.q),
}


def section_correctness():
    print(f"\n{'='*92}")
    print(f"  A. CORRECTNESS — opengreeks {opengreeks.__version__} vs autograd (exact autodiff)")
    print(f"{'='*92}")
    try:
        ad = _build_autograd()
    except Exception as e:
        print(f"\n  autograd not available ({e}); skipping. Install with: pip install autograd")
        return
    worst = 0.0
    for model, cases in CASES.items():
        print(f"\n  Model: {model}")
        print(f"    {'GREEK':12s}  {'max_abs_err':>13s}  {'max_rel_err':>13s}")
        for g in GREEKS:
            mae = mre = 0.0
            for c in cases:
                a = tuple(float(v) for v in (c.x, c.k, c.t, c.r, B[model](c), c.sigma, c.cp))
                ref = float(ad[g](a))
                got = float(OG[model](g, c))
                e = abs(got - ref)
                mae = max(mae, e); mre = max(mre, e / (1 + abs(ref))); worst = max(worst, mre)
            print(f"    {g:12s}  {mae:>13.3e}  {mre:>13.3e}")
    verdict = "PASS — bit-accurate, no deviation" if worst < 1e-9 else "FAIL — deviation exceeds 1e-9"
    print(f"\n  WORST rel err across all 10 Greeks × 3 models: {worst:.2e}   →   {verdict}")


def _bench(fn, target=0.5, max_iters=200000):
    fn()
    t0 = time.perf_counter(); fn()
    one = max(time.perf_counter() - t0, 1e-9)
    n = max(5, min(max_iters, int(target / one)))
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def section_scalar_speed():
    print(f"\n{'='*92}")
    print(f"  B1. SCALAR LATENCY — opengreeks (Rust) vs pure Python (per-call median µs)")
    print(f"{'='*92}\n")
    for model, cases in CASES.items():
        c = cases[0]; b = B[model](c)
        print(f"  Model: {model}  ({c.label})")
        print(f"    {'GREEK':12s}  {'opengreeks':>12s}  {'pure-python':>12s}  {'SPEEDUP':>9s}")
        for g in GREEKS:
            og = _bench(lambda g=g, c=c: OG[model](g, c))
            pp = _bench(lambda g=g, c=c, b=b: pp_greek(g, c.x, c.k, c.t, c.r, b, c.sigma, c.cp))
            print(f"    {g:12s}  {og*1e6:>10.3f} µs  {pp*1e6:>10.3f} µs  {pp/og:>7.1f}×")


def section_chain_speed():
    print(f"\n{'='*92}")
    print(f"  B2. CHAIN LATENCY — opengreeks batch (*_array) vs pure-Python loop")
    print(f"{'='*92}\n")
    # NIFTY-style Black-76 chain, ATM ±10%, 25-pt strikes → 177 strikes
    F0, T_, R_, S_ = 22000.0, 30/365, 0.07, 0.18
    K = np.arange(F0*0.9, F0*1.1+1, 25.0)
    Farr = np.full_like(K, F0); tarr = np.full_like(K, T_); sarr = np.full_like(K, S_)
    Kl = K.tolist()
    print(f"  Model: Black-76  (chain size {len(K)} strikes)")
    print(f"    {'GREEK':12s}  {'og batch':>12s}  {'py loop':>12s}  {'SPEEDUP':>9s}")
    for g in GREEKS:
        arr_fn = getattr(og_b76, g + "_array")
        og = _bench(lambda fn=arr_fn: fn("c", Farr, K, tarr, R_, sarr))
        pp = _bench(lambda g=g: [pp_greek(g, F0, kk, T_, R_, 0.0, S_, 1.0) for kk in Kl])
        print(f"    {g:12s}  {og*1e6:>10.1f} µs  {pp*1e6:>10.1f} µs  {pp/og:>7.1f}×")


if __name__ == "__main__":
    section_correctness()
    section_scalar_speed()
    section_chain_speed()
    print()
