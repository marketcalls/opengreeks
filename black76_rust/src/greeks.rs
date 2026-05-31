//! Analytical Greeks for Black-76.
//!
//! Numerical conventions: vega × 0.01 (per 1% IV change), theta / 365
//! (per calendar day), rho = `-t · price · 0.01` (per 1% rate change,
//! same expression for call and put — the only `r` dependence in
//! Black-76 is the `e^(-rT)` discount factor).

use crate::normal::{norm_cdf, norm_pdf};
use crate::pricing::{black_price, d1, d2};
use crate::OptionType;

/// Black-76 delta: `± e^(-rT) · N(±d1)` (sign + for call, - for put).
pub fn delta(f: f64, k: f64, sigma: f64, t: f64, r: f64, opt: OptionType) -> f64 {
    let q = opt.sign();
    let dd1 = d1(f, k, t, sigma);
    q * (-r * t).exp() * norm_cdf(q * dd1)
}

/// Black-76 gamma: `e^(-rT) · φ(d1) / (F·σ·√T)`. Same for call and put.
pub fn gamma(f: f64, k: f64, sigma: f64, t: f64, r: f64, _opt: OptionType) -> f64 {
    let dd1 = d1(f, k, t, sigma);
    (-r * t).exp() * norm_pdf(dd1) / (f * sigma * t.sqrt())
}

/// Black-76 vega: `F · e^(-rT) · φ(d1) · √T · 0.01`. The `× 0.01` gives
/// price change per 1% IV change (industry convention).
pub fn vega(f: f64, k: f64, sigma: f64, t: f64, r: f64, _opt: OptionType) -> f64 {
    let dd1 = d1(f, k, t, sigma);
    f * (-r * t).exp() * norm_pdf(dd1) * t.sqrt() * 0.01
}

/// Black-76 theta, per calendar day (annual theta divided by 365).
pub fn theta(f: f64, k: f64, sigma: f64, t: f64, r: f64, opt: OptionType) -> f64 {
    let e_rt = (-r * t).exp();
    let two_sqrt_t = 2.0 * t.sqrt();
    let dd1 = d1(f, k, t, sigma);
    let dd2 = d2(f, k, t, sigma);
    let first = f * e_rt * norm_pdf(dd1) * sigma / two_sqrt_t;
    match opt {
        OptionType::Call => {
            let second = -r * f * e_rt * norm_cdf(dd1);
            let third = r * k * e_rt * norm_cdf(dd2);
            -(first + second + third) / 365.0
        }
        OptionType::Put => {
            let second = -r * f * e_rt * norm_cdf(-dd1);
            let third = r * k * e_rt * norm_cdf(-dd2);
            (-first + second + third) / 365.0
        }
    }
}

/// Black-76 rho: `-t · price · 0.01`. Same expression for call and put because
/// the only `r` dependence in Black-76 is through the `e^(-rT)` discount factor.
pub fn rho(f: f64, k: f64, sigma: f64, t: f64, r: f64, opt: OptionType) -> f64 {
    -t * black_price(f, k, sigma, t, r, opt) * 0.01
}

// ── Second- and third-order Greeks (Black-76 = generalized GBS with b = 0) ──
// Raw / mathematical convention (exact partials, no scaling); τ-derivatives in
// years. Each delegates to the shared `gbs2` engine with cost-of-carry b = 0
// and the forward `f` as the underlying.
macro_rules! b76_higher {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        pub fn $name(f: f64, k: f64, sigma: f64, t: f64, r: f64, opt: OptionType) -> f64 {
            crate::gbs2::$name(f, k, t, r, 0.0, sigma, opt)
        }
    };
}
b76_higher!(/// Black-76 vanna: ∂²/∂F∂σ. Same for call and put.
    vanna);
b76_higher!(/// Black-76 charm: ∂delta/∂τ (per year). Sign differs by option type.
    charm);
b76_higher!(/// Black-76 vomma (volga): ∂²/∂σ². Same for call and put.
    vomma);
b76_higher!(/// Black-76 speed: ∂gamma/∂F. Same for call and put.
    speed);
b76_higher!(/// Black-76 zomma: ∂gamma/∂σ. Same for call and put.
    zomma);
b76_higher!(/// Black-76 color: ∂gamma/∂τ (per year). Same for call and put.
    color);
b76_higher!(/// Black-76 veta: ∂vega/∂τ (per year). Same for call and put.
    veta);
b76_higher!(/// Black-76 ultima: ∂³/∂σ³. Same for call and put.
    ultima);
b76_higher!(/// Black-76 dual_delta: ∂/∂K. Sign differs by option type.
    dual_delta);
b76_higher!(/// Black-76 dual_gamma: ∂²/∂K². Same for call and put.
    dual_gamma);

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn greeks_anchor_reference_values() {
        // F=22000, K=22000, T=30/365, r=0.07, σ=0.18 reference values:
        // delta=0.5073649462, gamma=0.0003492669, vega=25.0094242996,
        // theta=-7.4164742252, rho=-0.3700845628
        let f = 22000.0;
        let k = 22000.0;
        let t = 30.0 / 365.0;
        let r = 0.07;
        let s = 0.18;
        let c = OptionType::Call;
        assert!(approx_eq(delta(f, k, s, t, r, c), 0.5073649462, 1e-9));
        assert!(approx_eq(gamma(f, k, s, t, r, c), 0.0003492669, 1e-9));
        assert!(approx_eq(vega(f, k, s, t, r, c),  25.0094242996, 1e-9));
        assert!(approx_eq(theta(f, k, s, t, r, c), -7.4164742252, 1e-9));
        assert!(approx_eq(rho(f, k, s, t, r, c),   -0.3700845628, 1e-9));
    }
}
