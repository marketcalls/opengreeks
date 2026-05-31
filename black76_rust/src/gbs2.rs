//! Generalized Black-Scholes second- and third-order Greeks (cost-of-carry form).
//!
//! A single engine parameterized by the cost-of-carry `b` covers all three
//! shipped models:
//!   * Black-76 (options on futures): `b = 0`, `x = F`
//!   * Black-Scholes (no dividend):   `b = r`, `x = S`
//!   * Black-Scholes-Merton:          `b = r - q`, `x = S`
//!
//! Conventions are **raw / mathematical**: each function is the exact partial
//! derivative, with no industry scaling (no ×0.01, no ÷365). τ-derivatives
//! (charm, color, veta) are taken with respect to time-to-expiry in years.
//! `vega`/`vomma`/`veta`/`ultima` are per unit σ (i.e. per 1.00 = 100% vol),
//! not per 1%.
//!
//! Every formula here is validated to ~1e-12 against autograd automatic
//! differentiation across all three models (see `bench/`).

use crate::normal::{norm_cdf, norm_pdf};
use crate::OptionType;

/// Common intermediate quantities shared by the Greek formulas.
struct Ctx {
    d1: f64,
    d2: f64,
    phi: f64,   // φ(d1)
    cf: f64,    // e^{(b-r)t}  — carry-discount factor on the spot/forward term
    df: f64,    // e^{-rt}     — pure discount factor
    st: f64,    // σ·√t
    sqrt_t: f64,
    gamma: f64, // cf·φ / (x·σ√t)
    vega: f64,  // x·cf·φ·√t      (raw, per unit σ)
    x: f64,
    sigma: f64,
    t: f64,
    b: f64,
    r: f64,
}

#[inline]
fn ctx(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64) -> Ctx {
    let sqrt_t = t.sqrt();
    let st = sigma * sqrt_t;
    let d1 = ((x / k).ln() + (b + 0.5 * sigma * sigma) * t) / st;
    let d2 = d1 - st;
    let phi = norm_pdf(d1);
    let cf = ((b - r) * t).exp();
    let df = (-r * t).exp();
    let gamma = cf * phi / (x * st);
    let vega = x * cf * phi * sqrt_t;
    Ctx { d1, d2, phi, cf, df, st, sqrt_t, gamma, vega, x, sigma, t, b, r }
}

/// vanna = ∂²V/∂S∂σ = ∂delta/∂σ. Same for call and put.
pub fn vanna(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, _opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    -c.cf * c.phi * c.d2 / sigma
}

/// vomma (volga) = ∂²V/∂σ² = ∂vega/∂σ. Same for call and put.
pub fn vomma(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, _opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    c.vega * c.d1 * c.d2 / sigma
}

/// speed = ∂³V/∂S³ = ∂gamma/∂S. Same for call and put.
pub fn speed(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, _opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    -c.gamma / x * (c.d1 / c.st + 1.0)
}

/// zomma = ∂³V/∂S²∂σ = ∂gamma/∂σ. Same for call and put.
pub fn zomma(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, _opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    c.gamma * (c.d1 * c.d2 - 1.0) / sigma
}

/// color = ∂³V/∂S²∂τ = ∂gamma/∂τ (τ = time to expiry, years). Same call/put.
pub fn color(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, _opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    c.gamma * ((c.b - c.r) - c.d1 * c.b / c.st + (c.d1 * c.d2 - 1.0) / (2.0 * c.t))
}

/// veta = ∂²V/∂σ∂τ = ∂vega/∂τ (τ = time to expiry, years). Same call/put.
pub fn veta(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, _opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    c.vega * ((c.b - c.r) - c.d1 * c.b / c.st + (c.d1 * c.d2 + 1.0) / (2.0 * c.t))
}

/// charm = ∂²V/∂S∂τ = ∂delta/∂τ (τ = time to expiry, years).
pub fn charm(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    let dd1_dt = -c.d2 / (2.0 * c.t) + c.b / c.st;
    let base = c.cf * c.phi * dd1_dt;
    let drift = (c.b - c.r) * c.cf;
    match opt {
        OptionType::Call => drift * norm_cdf(c.d1) + base,
        OptionType::Put => drift * (norm_cdf(c.d1) - 1.0) + base,
    }
}

/// ultima = ∂³V/∂σ³ = ∂vomma/∂σ. Same for call and put.
pub fn ultima(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, _opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    let (d1, d2) = (c.d1, c.d2);
    -c.vega / (sigma * sigma) * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2)
}

/// dual_delta = ∂V/∂K.
pub fn dual_delta(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    match opt {
        OptionType::Call => -c.df * norm_cdf(c.d2),
        OptionType::Put => c.df * norm_cdf(-c.d2),
    }
}

/// dual_gamma = ∂²V/∂K². Same for call and put.
pub fn dual_gamma(x: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64, _opt: OptionType) -> f64 {
    let c = ctx(x, k, t, r, b, sigma);
    c.df * norm_pdf(c.d2) / (k * c.st)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    // Reference values from autograd autodiff of the generalized BSM price.
    // BSM case: x=S=100, K=100, t=0.5, r=0.07, q=0.03 (b=0.04), σ=0.30, call.
    #[test]
    fn bsm_second_order_anchors() {
        let (x, k, t, r, b, s, c) = (100.0, 100.0, 0.5, 0.07, 0.04, 0.30, OptionType::Call);
        assert!(approx_eq(vanna(x, k, t, r, b, s, c), 0.015131852409034252, 1e-9), "vanna {}", vanna(x, k, t, r, b, s, c));
        assert!(approx_eq(vomma(x, k, t, r, b, s, c), -0.2143679091279852, 1e-9), "vomma {}", vomma(x, k, t, r, b, s, c));
        assert!(approx_eq(dual_gamma(x, k, t, r, b, s, c), 0.01815822289084109, 1e-9), "dual_gamma {}", dual_gamma(x, k, t, r, b, s, c));
    }
}
