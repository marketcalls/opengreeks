# OpenGreeks

### Fast options pricing & Greeks for Python — Rust core, drop-in for `vollib` / `py_vollib`.

```bash
pip install opengreeks
```

---

## Up to 183× faster than vollib. Bit-identical Greeks. One dependency.

OpenGreeks reimplements the Black-76, Black-Scholes, and Black-Scholes-Merton pricing paths in zero-dependency Rust, exposed through PyO3 with the **same function names and signatures** as `py_vollib` / `vollib`. Migration is a one-line import swap; the math is unchanged.

### Headline speedups vs `vollib==1.0.7` (latest canonical)

| Workload | vollib 1.0.7 | OpenGreeks | Speedup |
|---|---:|---:|---:|
| Black-76 chain — vega × 177 strikes | 795 µs | **4.3 µs** | **183×** |
| Black-76 chain — delta × 177 strikes | 695 µs | **6.3 µs** | **110×** |
| Black-76 chain — all-5 Greeks × 200 options | ~4.2 ms | **~45 µs** | **~94×** |
| Black-76 chain — price × 177 strikes | 812 µs | **9.4 µs** | **86×** |
| Implied volatility, single call | 28.75 µs | **0.50 µs** | **58×** |
| Black-76 IV × 177-strike chain | 4.37 ms | **0.26 ms** | **17×** |
| Black-76 theta (scalar) | 11.38 µs | 0.33 µs | 34× |
| Black-Scholes IV (scalar) | 18.88 µs | 0.50 µs | 38× |
| BSM IV (scalar) | 18.83 µs | 0.50 µs | 38× |

A full NIFTY option chain refresh (~200 options, all 5 Greeks + IV) drops from **~9 ms in vollib** to **~0.3 ms in OpenGreeks** — the difference between "saturates a core at 100 Hz" and "uses ~3% of one core."

### Parity that lets you trust the swap

**29 edge cases × 3 models × 7 functions each**, validated against `vollib==1.0.7`:

| Greek | max abs error vs vollib |
|---|---|
| **delta, gamma, vega, theta, rho** | **0.0e+00 — bit-for-bit identical** |
| price | 1.4e-12 (rel 1.7e-14, ~14 digits) |
| IV (well-conditioned) | 8.6e-10 |

Textbook anchors (Hull 13.6, Hull 17.1, Haug page 4) all pass. Full report: [`bench/RESULTS.md`](bench/RESULTS.md).

### One dependency, not seven

| Package | Runtime dependencies |
|---|---|
| `pip install vollib` | `py_lets_be_rational`, `cody_special`, `piecewise_rational`, `simplejson`, `numpy`, `pandas`, `scipy` |
| **`pip install opengreeks`** | **`numpy`** |

No pandas, no scipy, no Cython build hell, no `_testcapi` import errors on minimal Python distributions.

---

**Inspired by [`py_vollib`](https://github.com/vollib/py_vollib) and [`vollib`](https://pypi.org/project/vollib/)** by Gammon Capital LLC — same function names, same argument order, same numerical conventions. OpenGreeks reimplements the pricing math in Rust to deliver the speed and dependency wins above without changing any of the math.

| Model | OpenGreeks submodule | vollib equivalent |
|---|---|---|
| Black-76 (futures options) | `opengreeks.black76` | `vollib.black` |
| Black-Scholes (no dividends) | `opengreeks.black_scholes` | `vollib.black_scholes` |
| Black-Scholes-Merton (dividends) | `opengreeks.black_scholes_merton` | `vollib.black_scholes_merton` |

Function names, argument order, and numerical conventions (vega × 0.01, theta / 365, rho × 0.01) match vollib **exactly**.

---

## Quick start

### Black-76 (futures / NIFTY options)

```python
from opengreeks.black76 import black, implied_volatility, delta, gamma, vega, theta, rho

F, K, t, r, sigma = 22000.0, 22000.0, 30/365, 0.07, 0.18
price = black('c', F, K, t, r, sigma)        # 450.27
iv    = implied_volatility(price, F, K, r, t, 'c')   # 0.18
d     = delta('c', F, K, t, r, sigma)
g     = gamma('c', F, K, t, r, sigma)
v     = vega ('c', F, K, t, r, sigma)
th    = theta('c', F, K, t, r, sigma)
rh    = rho  ('c', F, K, t, r, sigma)
```

### Black-Scholes (equity, no dividend)

```python
from opengreeks.black_scholes import black_scholes, implied_volatility, delta

price = black_scholes('c', 100.0, 90.0, 0.5, 0.01, 0.20)  # 12.111581
```

### Black-Scholes-Merton (equity with continuous dividend yield)

```python
from opengreeks.black_scholes_merton import black_scholes_merton, implied_volatility, delta

price = black_scholes_merton('p', 100.0, 95.0, 0.5, 0.10, 0.20, 0.05)  # 2.4648 (Haug p.4)
```

### Chain-wide computation (NumPy batch)

For option-chain analytics, use the `*_array` variants — one PyO3 boundary crossing, internal tight loop:

```python
import numpy as np
from opengreeks import black76

K = np.linspace(20000.0, 24000.0, 200)
F = np.full_like(K, 22000.0)
t = np.full_like(K, 30/365)
s = np.full_like(K, 0.18)

prices = black76.black_array('c', F, K, t, 0.07, s)
ivs    = black76.implied_volatility_array(prices, F, K, 0.07, t, 'c')
deltas = black76.delta_array('c', F, K, t, 0.07, s)
```

---

## Migrating from py_vollib / vollib

The function signatures are byte-identical. Migration is a one-line import swap.

**Before:**
```python
from py_vollib.black.implied_volatility import implied_volatility as black_iv
from py_vollib.black.greeks.analytical import delta as black_delta, gamma as black_gamma
```

**After:**
```python
from opengreeks.black76 import implied_volatility as black_iv
from opengreeks.black76 import delta as black_delta, gamma as black_gamma
```

The aliases (`as black_iv` etc.) keep the rest of your code unchanged.

| Old import | New import |
|---|---|
| `from py_vollib.black import black` | `from opengreeks.black76 import black` |
| `from py_vollib.black.implied_volatility import implied_volatility` | `from opengreeks.black76 import implied_volatility` |
| `from py_vollib.black.greeks.analytical import delta, gamma, vega, theta, rho` | `from opengreeks.black76 import delta, gamma, vega, theta, rho` |
| `from py_vollib.black_scholes import black_scholes` | `from opengreeks.black_scholes import black_scholes` |
| `from py_vollib.black_scholes_merton import black_scholes_merton` | `from opengreeks.black_scholes_merton import black_scholes_merton` |

---

## Reproducing the benchmarks

Want to verify the speedups on your hardware before you commit? Two-command repro:

```bash
# Install the baseline + opengreeks itself
pip install 'vollib>=1.0.7' 'py_lets_be_rational>=1.0.1' numpy opengreeks

# Run the bench against your CPU
python -c "import urllib.request; exec(urllib.request.urlopen('https://raw.githubusercontent.com/marketcalls/opengreeks/main/bench/bench_parity.py').read())"
```

Or clone the repo and run [`bench/bench_parity.py`](bench/bench_parity.py) — prints a full parity + performance report in ~30 seconds. Headline numbers above are reproducible.

Full report including all 29 edge cases and chain-wide tables: [`bench/RESULTS.md`](bench/RESULTS.md).

---

## Project layout

```
OpenGreeks/                          # this monorepo
├── pyproject.toml                   # name = "opengreeks"   →   pip install opengreeks
├── Cargo.toml                       # Rust workspace
├── src/lib.rs                       # PyO3 cdylib  _opengreeks
├── python/opengreeks/
│   ├── __init__.py
│   ├── black76.py                   # opengreeks.black76 — vollib.black equivalent
│   ├── black_scholes.py             # opengreeks.black_scholes
│   └── black_scholes_merton.py      # opengreeks.black_scholes_merton
├── black76_rust/                    # pure-Rust Black-76 core (zero deps)
├── bsm_rust/                        # pure-Rust BSM (BS via q=0); depends on black76_rust
├── bench/
│   ├── bench_parity.py              # parity + performance bench
│   └── RESULTS.md                   # full report
└── .github/workflows/CI.yml         # cargo test + wheel matrix + PyPI publish
```

**Future pricing models** (Heston stochastic vol, SABR, American/Bermudan via tree, etc.) slot in as new `<model>_rust/` crates + `opengreeks.<model>` submodules without breaking the published API.

---

## Build from source

```bash
# Toolchain
rustup default stable
pip install maturin

# Develop install into a venv
cd OpenGreeks/
maturin develop --release

# Or build a wheel
maturin build --release
pip install target/wheels/opengreeks-*.whl
```

### Run tests

```bash
# Rust crates
cargo test --release

# Python parity + performance bench
pip install 'vollib>=1.0.7' 'py_lets_be_rational>=1.0.1' numpy
python bench/bench_parity.py
```

---

## Dependencies

| Layer | Dependencies |
|---|---|
| Rust core (`black76_rust`, `bsm_rust`) | None — std math only |
| PyO3 wrapper (build-time) | `pyo3`, `numpy` Rust crates |
| Python runtime | `numpy` only |

Compare to `vollib==1.0.7` which pulls in `py_lets_be_rational`, `cody_special`, `piecewise_rational`, `simplejson`, `numpy`, `pandas`, `scipy` (7 packages including pandas & scipy).

---

## Credits & inspiration

OpenGreeks is a Rust reimplementation that takes its **public API directly from [`py_vollib`](https://github.com/vollib/py_vollib) / [`vollib`](https://pypi.org/project/vollib/)** (© 2017 Gammon Capital LLC, MIT-licensed) — function names, argument orders, return-value conventions, and per-day theta / per-1% vega-rho scaling. The vollib code base is the canonical Python reference for these formulas and we recommend it as the parity oracle when validating any port.

Algorithmic credit:

- **Peter Jäckel** — "Let's Be Rational" (jaeckel.org). The IV inversion algorithm used by `py_lets_be_rational` and by future versions of OpenGreeks.
- **W. J. Cody** — "Rational Chebyshev approximations for the error function" (Math. Comp., 1969). Used for the normal CDF in both libraries.
- **A. R. Wichura** — "Algorithm AS 241" (Appl. Stat., 1988). Inverse normal CDF.
- **John Hull** and **Espen Haug** — textbook formula references; their Black-76 / BS / BSM examples are the validation anchors in `bench/bench_parity.py`.

## License

MIT — Copyright (c) 2026 Marketcalls / Rajandran R. See [LICENSE](LICENSE).
