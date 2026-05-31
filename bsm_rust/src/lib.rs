//! Black-Scholes-Merton option pricing, Greeks, and IV.
//!
//! BS (no dividend) is BSM with `q = 0`. Numerical conventions: vega × 0.01
//! (per 1% IV move), theta / 365 (per calendar day), rho × 0.01.
//!
//! Internally we reduce to Black-76 via the forward `F = S · e^((r - q) t)`,
//! reusing `black76_rust`'s norm CDF and Newton IV solver. Greeks are derived
//! analytically in S-space (chain rule from F).

#![forbid(unsafe_code)]

use black76_rust::{
    implied_volatility as b76_iv, norm_cdf, norm_pdf, undiscounted_black_price,
    IvError, OptionType,
};

/// BSM d1 = [ln(S/K) + (r - q + σ²/2)·t] / (σ·√t).
#[inline]
pub fn d1(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
}

/// BSM d2 = d1 - σ·√t.
#[inline]
pub fn d2(s: f64, k: f64, t: f64, r: f64, q: f64, sigma: f64) -> f64 {
    d1(s, k, t, r, q, sigma) - sigma * t.sqrt()
}

/// BSM price `e^(-rT) · [F·N(d1) - K·N(d2)]` (call form; put via parity).
#[inline]
pub fn bsm_price(s: f64, k: f64, sigma: f64, t: f64, r: f64, q: f64, opt: OptionType) -> f64 {
    let f = s * ((r - q) * t).exp();
    (-r * t).exp() * undiscounted_black_price(f, k, sigma, t, opt)
}

/// BSM delta: `±e^(-qt)·N(±d1)`.
pub fn delta(s: f64, k: f64, sigma: f64, t: f64, r: f64, q: f64, opt: OptionType) -> f64 {
    let q_sign = opt.sign();
    let dd1 = d1(s, k, t, r, q, sigma);
    q_sign * (-q * t).exp() * norm_cdf(q_sign * dd1)
}

/// BSM gamma: `e^(-qt)·φ(d1) / (S·σ·√t)`. Same for call and put.
pub fn gamma(s: f64, k: f64, sigma: f64, t: f64, r: f64, q: f64, _opt: OptionType) -> f64 {
    let dd1 = d1(s, k, t, r, q, sigma);
    (-q * t).exp() * norm_pdf(dd1) / (s * sigma * t.sqrt())
}

/// BSM vega: `S·e^(-qt)·φ(d1)·√t · 0.01` (per 1% IV move, industry convention).
pub fn vega(s: f64, k: f64, sigma: f64, t: f64, r: f64, q: f64, _opt: OptionType) -> f64 {
    let dd1 = d1(s, k, t, r, q, sigma);
    s * (-q * t).exp() * norm_pdf(dd1) * t.sqrt() * 0.01
}

/// BSM theta, per calendar day (annual / 365).
pub fn theta(s: f64, k: f64, sigma: f64, t: f64, r: f64, q: f64, opt: OptionType) -> f64 {
    let dd1 = d1(s, k, t, r, q, sigma);
    let dd2 = d2(s, k, t, r, q, sigma);
    let e_qt = (-q * t).exp();
    let e_rt = (-r * t).exp();
    let first = s * e_qt * norm_pdf(dd1) * sigma / (2.0 * t.sqrt());
    match opt {
        OptionType::Call => {
            let second = -q * s * e_qt * norm_cdf(dd1);
            let third = r * k * e_rt * norm_cdf(dd2);
            -(first + second + third) / 365.0
        }
        OptionType::Put => {
            let second = -q * s * e_qt * norm_cdf(-dd1);
            let third = r * k * e_rt * norm_cdf(-dd2);
            (-first + second + third) / 365.0
        }
    }
}

/// BSM rho: `±t·K·e^(-rt)·N(±d2) · 0.01` (per 1% rate move).
pub fn rho(s: f64, k: f64, sigma: f64, t: f64, r: f64, q: f64, opt: OptionType) -> f64 {
    let q_sign = opt.sign();
    let dd2 = d2(s, k, t, r, q, sigma);
    q_sign * t * k * (-r * t).exp() * norm_cdf(q_sign * dd2) * 0.01
}

// ── Second- and third-order Greeks (BSM = generalized GBS with b = r - q) ──
// Raw / mathematical convention (exact partials, no scaling); τ-derivatives in
// years. BS (no dividend) is the q = 0 case. Each delegates to the shared
// `black76_rust::gbs2` engine with cost-of-carry b = r - q and the spot `s`.
macro_rules! bsm_higher {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        pub fn $name(s: f64, k: f64, sigma: f64, t: f64, r: f64, q: f64, opt: OptionType) -> f64 {
            black76_rust::gbs2::$name(s, k, t, r, r - q, sigma, opt)
        }
    };
}
bsm_higher!(/// BSM vanna: ∂²/∂S∂σ. Same for call and put.
    vanna);
bsm_higher!(/// BSM charm: ∂delta/∂τ (per year). Sign differs by option type.
    charm);
bsm_higher!(/// BSM vomma (volga): ∂²/∂σ². Same for call and put.
    vomma);
bsm_higher!(/// BSM speed: ∂gamma/∂S. Same for call and put.
    speed);
bsm_higher!(/// BSM zomma: ∂gamma/∂σ. Same for call and put.
    zomma);
bsm_higher!(/// BSM color: ∂gamma/∂τ (per year). Same for call and put.
    color);
bsm_higher!(/// BSM veta: ∂vega/∂τ (per year). Same for call and put.
    veta);
bsm_higher!(/// BSM ultima: ∂³/∂σ³. Same for call and put.
    ultima);
bsm_higher!(/// BSM dual_delta: ∂/∂K. Sign differs by option type.
    dual_delta);
bsm_higher!(/// BSM dual_gamma: ∂²/∂K². Same for call and put.
    dual_gamma);

/// BSM implied volatility. Inverts BSM price for σ via the F=S·e^((r-q)t) reduction
/// to Black-76, reusing `black76_rust`'s Newton/bisection solver.
pub fn implied_volatility(
    price: f64,
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    q: f64,
    opt: OptionType,
) -> Result<f64, IvError> {
    let f = s * ((r - q) * t).exp();
    b76_iv(price, f, k, t, r, opt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn bsm_haug_p4_anchor() {
        // S=100 K=95 q=.05 t=.5 r=.1 σ=.2 → put = 2.4648 (Haug, page 4)
        let p = bsm_price(100.0, 95.0, 0.2, 0.5, 0.1, 0.05, OptionType::Put);
        assert!(approx_eq(p, 2.4648, 1e-3), "got {}", p);
    }

    #[test]
    fn bs_equals_bsm_q0_anchor() {
        // Hull example 13.6: S=42, K=40, r=0.1, σ=0.2, t=0.5
        // BS call ≈ 4.7594 per Hull. BSM(q=0) should match.
        let p = bsm_price(42.0, 40.0, 0.2, 0.5, 0.1, 0.0, OptionType::Call);
        assert!(approx_eq(p, 4.759422392871542, 1e-9), "got {}", p);
    }

    #[test]
    fn bsm_delta_no_div_matches_hull() {
        // Hull 17.1: S=49 K=50 r=.05 t=.3846 q=0 σ=.2 → call delta ≈ 0.522
        let d = delta(49.0, 50.0, 0.2, 0.3846, 0.05, 0.0, OptionType::Call);
        assert!(approx_eq(d, 0.521601633972, 1e-9), "got {}", d);
    }

    #[test]
    fn bsm_iv_roundtrip() {
        let s = 100.0; let k = 100.0; let t = 0.5; let r = 0.05; let q = 0.02; let sigma = 0.25;
        let p = bsm_price(s, k, sigma, t, r, q, OptionType::Call);
        let iv = implied_volatility(p, s, k, t, r, q, OptionType::Call).unwrap();
        assert!(approx_eq(iv, sigma, 1e-10), "got {}", iv);
    }
}
