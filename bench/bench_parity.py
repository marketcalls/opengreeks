"""Quant-grade parity + performance bench: opengreeks vs (vollib | py_vollib).

Covers all three models: Black-76, Black-Scholes, Black-Scholes-Merton.

Auto-detects the installed baseline:
- venv with `vollib==1.0.7`  → baseline labeled "vollib 1.0.7"
- venv with `py_vollib==1.0.1` → baseline labeled "py_vollib 1.0.1"

Run in each venv; combine outputs into the dual-version RESULTS.md.
"""
from __future__ import annotations

import math
import statistics
import time
import warnings
from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError

warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import opengreeks
from opengreeks import black76 as og_b76
from opengreeks import black_scholes as og_bs
from opengreeks import black_scholes_merton as og_bsm


def _try_version(name):
    try: return version(name)
    except PackageNotFoundError: return None


if _try_version("vollib"):
    from vollib.black import black as v_b76_price
    from vollib.black.implied_volatility import implied_volatility as v_b76_iv
    from vollib.black.greeks.analytical import (
        delta as v_b76_delta, gamma as v_b76_gamma, vega as v_b76_vega,
        theta as v_b76_theta, rho as v_b76_rho,
    )
    from vollib.black_scholes import black_scholes as v_bs_price
    from vollib.black_scholes.implied_volatility import implied_volatility as v_bs_iv
    from vollib.black_scholes.greeks.analytical import (
        delta as v_bs_delta, gamma as v_bs_gamma, vega as v_bs_vega,
        theta as v_bs_theta, rho as v_bs_rho,
    )
    from vollib.black_scholes_merton import black_scholes_merton as v_bsm_price
    from vollib.black_scholes_merton.implied_volatility import implied_volatility as v_bsm_iv
    from vollib.black_scholes_merton.greeks.analytical import (
        delta as v_bsm_delta, gamma as v_bsm_gamma, vega as v_bsm_vega,
        theta as v_bsm_theta, rho as v_bsm_rho,
    )
    BASELINE = f"vollib {_try_version('vollib')}  (py_lets_be_rational {_try_version('py_lets_be_rational')})"
elif _try_version("py_vollib"):
    from py_vollib.black import black as v_b76_price
    from py_vollib.black.implied_volatility import implied_volatility as v_b76_iv
    from py_vollib.black.greeks.analytical import (
        delta as v_b76_delta, gamma as v_b76_gamma, vega as v_b76_vega,
        theta as v_b76_theta, rho as v_b76_rho,
    )
    from py_vollib.black_scholes import black_scholes as v_bs_price
    from py_vollib.black_scholes.implied_volatility import implied_volatility as v_bs_iv
    from py_vollib.black_scholes.greeks.analytical import (
        delta as v_bs_delta, gamma as v_bs_gamma, vega as v_bs_vega,
        theta as v_bs_theta, rho as v_bs_rho,
    )
    from py_vollib.black_scholes_merton import black_scholes_merton as v_bsm_price
    from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as v_bsm_iv
    from py_vollib.black_scholes_merton.greeks.analytical import (
        delta as v_bsm_delta, gamma as v_bsm_gamma, vega as v_bsm_vega,
        theta as v_bsm_theta, rho as v_bsm_rho,
    )
    BASELINE = f"py_vollib {_try_version('py_vollib')}  (py_lets_be_rational {_try_version('py_lets_be_rational')})"
else:
    raise SystemExit("Install either `vollib` or `py_vollib` to run the parity bench")


@dataclass
class B76Case:
    label: str; F: float; K: float; t: float; r: float; sigma: float; flag: str

@dataclass
class BSCase:
    label: str; S: float; K: float; t: float; r: float; sigma: float; flag: str

@dataclass
class BSMCase:
    label: str; S: float; K: float; t: float; r: float; sigma: float; q: float; flag: str


B76_CASES = [
    B76Case("ATM 30d call",       22000.0, 22000.0, 30/365, 0.07, 0.18, "c"),
    B76Case("ATM 30d put",        22000.0, 22000.0, 30/365, 0.07, 0.18, "p"),
    B76Case("Deep OTM (+15%)",    22000.0, 25300.0, 30/365, 0.07, 0.18, "c"),
    B76Case("Deep ITM (-15%)",    22000.0, 18700.0, 30/365, 0.07, 0.18, "c"),
    B76Case("Tiny T 1d ATM",      22000.0, 22000.0,  1/365, 0.07, 0.18, "c"),
    B76Case("Long T 2y ATM",      22000.0, 22000.0,    2.0, 0.07, 0.18, "c"),
    B76Case("High vol 90%",       22000.0, 22000.0, 30/365, 0.07, 0.90, "c"),
    B76Case("Low vol 5%",         22000.0, 22000.0, 30/365, 0.07, 0.05, "c"),
    B76Case("Small F (5.0)",      5.0,     5.0,     30/365, 0.07, 0.30, "c"),
    B76Case("Large F (1e5)",      1e5,     1e5,     30/365, 0.07, 0.20, "c"),
    B76Case("r=0",                22000.0, 22000.0, 30/365, 0.00, 0.18, "c"),
    B76Case("r=20%",              22000.0, 22000.0, 30/365, 0.20, 0.18, "c"),
]

BS_CASES = [
    BSCase("Doctest 1 (call)",    100.0, 90.0,  0.5,    0.01, 0.20, "c"),
    BSCase("Doctest 2 (put)",     100.0, 90.0,  0.5,    0.01, 0.20, "p"),
    BSCase("Hull 13.6 call",      42.0,  40.0,  0.5,    0.10, 0.20, "c"),
    BSCase("ATM 30d",             100.0, 100.0, 30/365, 0.05, 0.25, "c"),
    BSCase("Deep OTM 30d",        100.0, 130.0, 30/365, 0.05, 0.25, "c"),
    BSCase("Deep ITM 30d",        100.0, 70.0,  30/365, 0.05, 0.25, "c"),
    BSCase("Long T 2y",           100.0, 100.0, 2.0,    0.05, 0.25, "c"),
    BSCase("Small S (5)",         5.0,   5.0,   30/365, 0.05, 0.30, "c"),
    BSCase("Large S (1e5)",       1e5,   1e5,   30/365, 0.05, 0.20, "c"),
]

BSM_CASES = [
    BSMCase("Haug p4 put",        100.0, 95.0,  0.5,    0.10, 0.20, 0.05, "p"),
    BSMCase("Hull 17.1 (q=0)",    49.0,  50.0,  0.3846, 0.05, 0.20, 0.0,  "c"),
    BSMCase("ATM 30d q=2%",       100.0, 100.0, 30/365, 0.05, 0.25, 0.02, "c"),
    BSMCase("Deep OTM q=3%",      100.0, 130.0, 30/365, 0.05, 0.25, 0.03, "c"),
    BSMCase("Deep ITM q=3%",      100.0, 70.0,  30/365, 0.05, 0.25, 0.03, "c"),
    BSMCase("Long T 2y q=4%",     100.0, 100.0, 2.0,    0.05, 0.25, 0.04, "c"),
    BSMCase("High q=10%",         100.0, 100.0, 30/365, 0.05, 0.25, 0.10, "c"),
    BSMCase("Negative q=-2%",     100.0, 100.0, 30/365, 0.05, 0.25,-0.02, "c"),
]


def safe(fn, c):
    try: return fn(c)
    except Exception: return math.nan


def compare(model_name, cases, og_calls, v_calls):
    print(f"\n  Model: {model_name}")
    print(f"  {'CASE':25s}  " + "  ".join(f"{n:>9s}" for n in og_calls))
    print(f"  {'-'*25}  " + "  ".join(f"{'-'*9:>9s}" for _ in og_calls))
    summary = {n: [] for n in og_calls}
    for c in cases:
        row = [f"  {c.label:25s}"]
        for name in og_calls:
            a = safe(og_calls[name], c); b = safe(v_calls[name], c)
            if math.isnan(a) or math.isnan(b):
                row.append(f"  {'n/a':>9s}")
            else:
                err = abs(a - b)
                summary[name].append((err, b))
                row.append(f"  {err:>9.2e}")
        print("".join(row))
    print()
    print(f"  {'FUNCTION':9s}  {'max_abs':>11s}  {'mean_abs':>11s}  {'max_rel':>11s}  {'n':>3s}")
    for name in og_calls:
        errs = summary[name]
        if not errs: continue
        abs_e = [e for e, _ in errs]
        rel_e = [e / max(abs(r), 1e-300) for e, r in errs if r != 0]
        print(f"  {name:9s}  {max(abs_e):>11.3e}  {statistics.mean(abs_e):>11.3e}  "
              f"{(max(rel_e) if rel_e else 0):>11.3e}  {len(errs):>3d}")


def section_parity():
    print(f"\n{'='*100}")
    print(f"  A. PARITY  —  opengreeks {opengreeks.__version__}  vs  {BASELINE}")
    print(f"{'='*100}")

    compare("Black-76", B76_CASES,
        {
            "price": lambda c: og_b76.black(c.flag, c.F, c.K, c.t, c.r, c.sigma),
            "delta": lambda c: og_b76.delta(c.flag, c.F, c.K, c.t, c.r, c.sigma),
            "gamma": lambda c: og_b76.gamma(c.flag, c.F, c.K, c.t, c.r, c.sigma),
            "vega":  lambda c: og_b76.vega(c.flag,  c.F, c.K, c.t, c.r, c.sigma),
            "theta": lambda c: og_b76.theta(c.flag, c.F, c.K, c.t, c.r, c.sigma),
            "rho":   lambda c: og_b76.rho(c.flag,   c.F, c.K, c.t, c.r, c.sigma),
            "iv":    lambda c: og_b76.implied_volatility(v_b76_price(c.flag,c.F,c.K,c.t,c.r,c.sigma),c.F,c.K,c.r,c.t,c.flag),
        },
        {
            "price": lambda c: v_b76_price(c.flag, c.F, c.K, c.t, c.r, c.sigma),
            "delta": lambda c: v_b76_delta(c.flag, c.F, c.K, c.t, c.r, c.sigma),
            "gamma": lambda c: v_b76_gamma(c.flag, c.F, c.K, c.t, c.r, c.sigma),
            "vega":  lambda c: v_b76_vega(c.flag,  c.F, c.K, c.t, c.r, c.sigma),
            "theta": lambda c: v_b76_theta(c.flag, c.F, c.K, c.t, c.r, c.sigma),
            "rho":   lambda c: v_b76_rho(c.flag,   c.F, c.K, c.t, c.r, c.sigma),
            "iv":    lambda c: v_b76_iv(v_b76_price(c.flag,c.F,c.K,c.t,c.r,c.sigma),c.F,c.K,c.r,c.t,c.flag),
        })

    compare("Black-Scholes", BS_CASES,
        {
            "price": lambda c: og_bs.black_scholes(c.flag, c.S, c.K, c.t, c.r, c.sigma),
            "delta": lambda c: og_bs.delta(c.flag, c.S, c.K, c.t, c.r, c.sigma),
            "gamma": lambda c: og_bs.gamma(c.flag, c.S, c.K, c.t, c.r, c.sigma),
            "vega":  lambda c: og_bs.vega(c.flag,  c.S, c.K, c.t, c.r, c.sigma),
            "theta": lambda c: og_bs.theta(c.flag, c.S, c.K, c.t, c.r, c.sigma),
            "rho":   lambda c: og_bs.rho(c.flag,   c.S, c.K, c.t, c.r, c.sigma),
            "iv":    lambda c: og_bs.implied_volatility(v_bs_price(c.flag,c.S,c.K,c.t,c.r,c.sigma),c.S,c.K,c.t,c.r,c.flag),
        },
        {
            "price": lambda c: v_bs_price(c.flag, c.S, c.K, c.t, c.r, c.sigma),
            "delta": lambda c: v_bs_delta(c.flag, c.S, c.K, c.t, c.r, c.sigma),
            "gamma": lambda c: v_bs_gamma(c.flag, c.S, c.K, c.t, c.r, c.sigma),
            "vega":  lambda c: v_bs_vega(c.flag,  c.S, c.K, c.t, c.r, c.sigma),
            "theta": lambda c: v_bs_theta(c.flag, c.S, c.K, c.t, c.r, c.sigma),
            "rho":   lambda c: v_bs_rho(c.flag,   c.S, c.K, c.t, c.r, c.sigma),
            "iv":    lambda c: v_bs_iv(v_bs_price(c.flag,c.S,c.K,c.t,c.r,c.sigma),c.S,c.K,c.t,c.r,c.flag),
        })

    compare("Black-Scholes-Merton", BSM_CASES,
        {
            "price": lambda c: og_bsm.black_scholes_merton(c.flag, c.S, c.K, c.t, c.r, c.sigma, c.q),
            "delta": lambda c: og_bsm.delta(c.flag, c.S, c.K, c.t, c.r, c.sigma, c.q),
            "gamma": lambda c: og_bsm.gamma(c.flag, c.S, c.K, c.t, c.r, c.sigma, c.q),
            "vega":  lambda c: og_bsm.vega(c.flag,  c.S, c.K, c.t, c.r, c.sigma, c.q),
            "theta": lambda c: og_bsm.theta(c.flag, c.S, c.K, c.t, c.r, c.sigma, c.q),
            "rho":   lambda c: og_bsm.rho(c.flag,   c.S, c.K, c.t, c.r, c.sigma, c.q),
            "iv":    lambda c: og_bsm.implied_volatility(v_bsm_price(c.flag,c.S,c.K,c.t,c.r,c.sigma,c.q),c.S,c.K,c.t,c.r,c.q,c.flag),
        },
        {
            "price": lambda c: v_bsm_price(c.flag, c.S, c.K, c.t, c.r, c.sigma, c.q),
            "delta": lambda c: v_bsm_delta(c.flag, c.S, c.K, c.t, c.r, c.sigma, c.q),
            "gamma": lambda c: v_bsm_gamma(c.flag, c.S, c.K, c.t, c.r, c.sigma, c.q),
            "vega":  lambda c: v_bsm_vega(c.flag,  c.S, c.K, c.t, c.r, c.sigma, c.q),
            "theta": lambda c: v_bsm_theta(c.flag, c.S, c.K, c.t, c.r, c.sigma, c.q),
            "rho":   lambda c: v_bsm_rho(c.flag,   c.S, c.K, c.t, c.r, c.sigma, c.q),
            "iv":    lambda c: v_bsm_iv(v_bsm_price(c.flag,c.S,c.K,c.t,c.r,c.sigma,c.q),c.S,c.K,c.t,c.r,c.q,c.flag),
        })


def bench_scalar(fn, target=1.0, max_iters=20000):
    fn()
    t0 = time.perf_counter(); fn()
    one = max(time.perf_counter() - t0, 1e-9)
    n = max(5, min(max_iters, int(target / one)))
    times = []
    for _ in range(n):
        t0 = time.perf_counter(); fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def section_scalar_perf():
    print(f"\n{'='*100}")
    print(f"  B. SCALAR LATENCY (per-call median µs)  —  {BASELINE}")
    print(f"{'='*100}\n")

    def run(label, og, vol):
        print(f"  {label}")
        print(f"    {'FUNCTION':10s}  {'opengreeks':>12s}  {'baseline':>12s}  {'SPEEDUP':>10s}")
        for n, fa in og:
            ma = bench_scalar(fa); mb = bench_scalar(vol[n])
            print(f"    {n:10s}  {ma*1e6:>11.3f} µs  {mb*1e6:>11.3f} µs  {mb/ma:>9.1f}×")

    F,K,t,r,s,flag = 22000.0, 22000.0, 30/365, 0.07, 0.18, "c"
    p_b76 = v_b76_price(flag,F,K,t,r,s)
    run("Black-76", [
        ("price", lambda: og_b76.black(flag,F,K,t,r,s)),
        ("delta", lambda: og_b76.delta(flag,F,K,t,r,s)),
        ("vega",  lambda: og_b76.vega(flag,F,K,t,r,s)),
        ("theta", lambda: og_b76.theta(flag,F,K,t,r,s)),
        ("rho",   lambda: og_b76.rho(flag,F,K,t,r,s)),
        ("iv",    lambda: og_b76.implied_volatility(p_b76, F, K, r, t, flag)),
    ], {
        "price": lambda: v_b76_price(flag,F,K,t,r,s),
        "delta": lambda: v_b76_delta(flag,F,K,t,r,s),
        "vega":  lambda: v_b76_vega(flag,F,K,t,r,s),
        "theta": lambda: v_b76_theta(flag,F,K,t,r,s),
        "rho":   lambda: v_b76_rho(flag,F,K,t,r,s),
        "iv":    lambda: v_b76_iv(p_b76, F, K, r, t, flag),
    })

    S = 100.0; Kbs = 100.0; tbs = 30/365; rbs = 0.05; sbs = 0.25; flag = "c"
    p_bs = v_bs_price(flag,S,Kbs,tbs,rbs,sbs)
    run("Black-Scholes", [
        ("price", lambda: og_bs.black_scholes(flag,S,Kbs,tbs,rbs,sbs)),
        ("delta", lambda: og_bs.delta(flag,S,Kbs,tbs,rbs,sbs)),
        ("vega",  lambda: og_bs.vega(flag,S,Kbs,tbs,rbs,sbs)),
        ("theta", lambda: og_bs.theta(flag,S,Kbs,tbs,rbs,sbs)),
        ("rho",   lambda: og_bs.rho(flag,S,Kbs,tbs,rbs,sbs)),
        ("iv",    lambda: og_bs.implied_volatility(p_bs, S, Kbs, tbs, rbs, flag)),
    ], {
        "price": lambda: v_bs_price(flag,S,Kbs,tbs,rbs,sbs),
        "delta": lambda: v_bs_delta(flag,S,Kbs,tbs,rbs,sbs),
        "vega":  lambda: v_bs_vega(flag,S,Kbs,tbs,rbs,sbs),
        "theta": lambda: v_bs_theta(flag,S,Kbs,tbs,rbs,sbs),
        "rho":   lambda: v_bs_rho(flag,S,Kbs,tbs,rbs,sbs),
        "iv":    lambda: v_bs_iv(p_bs, S, Kbs, tbs, rbs, flag),
    })

    q = 0.02
    p_bsm = v_bsm_price(flag,S,Kbs,tbs,rbs,sbs,q)
    run("Black-Scholes-Merton", [
        ("price", lambda: og_bsm.black_scholes_merton(flag,S,Kbs,tbs,rbs,sbs,q)),
        ("delta", lambda: og_bsm.delta(flag,S,Kbs,tbs,rbs,sbs,q)),
        ("vega",  lambda: og_bsm.vega(flag,S,Kbs,tbs,rbs,sbs,q)),
        ("theta", lambda: og_bsm.theta(flag,S,Kbs,tbs,rbs,sbs,q)),
        ("rho",   lambda: og_bsm.rho(flag,S,Kbs,tbs,rbs,sbs,q)),
        ("iv",    lambda: og_bsm.implied_volatility(p_bsm, S, Kbs, tbs, rbs, q, flag)),
    ], {
        "price": lambda: v_bsm_price(flag,S,Kbs,tbs,rbs,sbs,q),
        "delta": lambda: v_bsm_delta(flag,S,Kbs,tbs,rbs,sbs,q),
        "vega":  lambda: v_bsm_vega(flag,S,Kbs,tbs,rbs,sbs,q),
        "theta": lambda: v_bsm_theta(flag,S,Kbs,tbs,rbs,sbs,q),
        "rho":   lambda: v_bsm_rho(flag,S,Kbs,tbs,rbs,sbs,q),
        "iv":    lambda: v_bsm_iv(p_bsm, S, Kbs, tbs, rbs, q, flag),
    })


def section_chain_perf():
    print(f"\n{'='*100}")
    print(f"  C. CHAIN-WIDE LATENCY (177 strikes, ATM ±10%)  —  {BASELINE}")
    print(f"{'='*100}\n")

    F0 = 22000.0; T_ = 30/365; R_ = 0.07; S_ = 0.18
    K_arr_b = np.arange(F0*0.9, F0*1.1+1, 25.0)
    F_arr = np.full_like(K_arr_b, F0)
    t_arr_b = np.full_like(K_arr_b, T_)
    s_arr_b = np.full_like(K_arr_b, S_)
    prices_b76 = np.array([v_b76_price("c", F0, K, T_, R_, S_) for K in K_arr_b])

    def chain_row(label, ogf, volf):
        ma = bench_scalar(ogf); mb = bench_scalar(volf)
        print(f"    {label:10s}  {ma*1e6:>11.1f} µs  {mb*1e6:>11.1f} µs  {mb/ma:>9.1f}×")

    print(f"  Model: Black-76  (chain size {len(K_arr_b)})")
    print(f"    {'FUNCTION':10s}  {'og batch':>12s}  {'baseline loop':>14s}  {'SPEEDUP':>10s}")
    chain_row("PRICE", lambda: og_b76.black_array("c", F_arr, K_arr_b, t_arr_b, R_, s_arr_b),
              lambda: [v_b76_price("c", F0, K, T_, R_, S_) for K in K_arr_b])
    chain_row("DELTA", lambda: og_b76.delta_array("c", F_arr, K_arr_b, t_arr_b, R_, s_arr_b),
              lambda: [v_b76_delta("c", F0, K, T_, R_, S_) for K in K_arr_b])
    chain_row("VEGA",  lambda: og_b76.vega_array("c", F_arr, K_arr_b, t_arr_b, R_, s_arr_b),
              lambda: [v_b76_vega("c", F0, K, T_, R_, S_) for K in K_arr_b])
    chain_row("IV",    lambda: og_b76.implied_volatility_array(prices_b76, F_arr, K_arr_b, R_, t_arr_b, "c"),
              lambda: [v_b76_iv(p, F0, K, R_, T_, "c") for p, K in zip(prices_b76, K_arr_b)])

    S0 = 100.0
    K_arr_s = np.arange(S0*0.9, S0*1.1+1, 0.25)
    S_arr = np.full_like(K_arr_s, S0)
    t_arr_s = np.full_like(K_arr_s, T_)
    s_arr_s = np.full_like(K_arr_s, S_)
    prices_bs = np.array([v_bs_price("c", S0, K, T_, R_, S_) for K in K_arr_s])

    print(f"\n  Model: Black-Scholes  (chain size {len(K_arr_s)})")
    print(f"    {'FUNCTION':10s}  {'og batch':>12s}  {'baseline loop':>14s}  {'SPEEDUP':>10s}")
    chain_row("PRICE", lambda: og_bs.black_scholes_array("c", S_arr, K_arr_s, t_arr_s, R_, s_arr_s),
              lambda: [v_bs_price("c", S0, K, T_, R_, S_) for K in K_arr_s])
    chain_row("DELTA", lambda: og_bs.delta_array("c", S_arr, K_arr_s, t_arr_s, R_, s_arr_s),
              lambda: [v_bs_delta("c", S0, K, T_, R_, S_) for K in K_arr_s])
    chain_row("VEGA",  lambda: og_bs.vega_array("c", S_arr, K_arr_s, t_arr_s, R_, s_arr_s),
              lambda: [v_bs_vega("c", S0, K, T_, R_, S_) for K in K_arr_s])
    chain_row("IV",    lambda: og_bs.implied_volatility_array(prices_bs, S_arr, K_arr_s, t_arr_s, R_, "c"),
              lambda: [v_bs_iv(p, S0, K, T_, R_, "c") for p, K in zip(prices_bs, K_arr_s)])

    q = 0.02
    prices_bsm = np.array([v_bsm_price("c", S0, K, T_, R_, S_, q) for K in K_arr_s])
    print(f"\n  Model: Black-Scholes-Merton  (chain size {len(K_arr_s)}, q={q})")
    print(f"    {'FUNCTION':10s}  {'og batch':>12s}  {'baseline loop':>14s}  {'SPEEDUP':>10s}")
    chain_row("PRICE", lambda: og_bsm.black_scholes_merton_array("c", S_arr, K_arr_s, t_arr_s, R_, s_arr_s, q),
              lambda: [v_bsm_price("c", S0, K, T_, R_, S_, q) for K in K_arr_s])
    chain_row("DELTA", lambda: og_bsm.delta_array("c", S_arr, K_arr_s, t_arr_s, R_, s_arr_s, q),
              lambda: [v_bsm_delta("c", S0, K, T_, R_, S_, q) for K in K_arr_s])
    chain_row("IV",    lambda: og_bsm.implied_volatility_array(prices_bsm, S_arr, K_arr_s, t_arr_s, R_, q, "c"),
              lambda: [v_bsm_iv(p, S0, K, T_, R_, q, "c") for p, K in zip(prices_bsm, K_arr_s)])


if __name__ == "__main__":
    section_parity()
    section_scalar_perf()
    section_chain_perf()
    print()
