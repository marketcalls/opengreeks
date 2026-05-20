//! Black-76 pricing.
//!
//! `c(F,K,σ,T,r) = e^(-rT)·[F·N(d1) − K·N(d2)]`
//! `p(F,K,σ,T,r) = e^(-rT)·[K·N(-d2) − F·N(-d1)]`
//!
//! where `d1 = [ln(F/K) + σ²T/2] / (σ√T)` and `d2 = d1 - σ√T`.

use crate::normal::norm_cdf;
use crate::OptionType;

/// First Black-76 d-parameter: `d1 = [ln(F/K) + σ²T/2] / (σ√T)`.
#[inline]
pub fn d1(f: f64, k: f64, t: f64, sigma: f64) -> f64 {
    ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * t.sqrt())
}

/// Second Black-76 d-parameter: `d2 = d1 - σ√T`.
#[inline]
pub fn d2(f: f64, k: f64, t: f64, sigma: f64) -> f64 {
    d1(f, k, t, sigma) - sigma * t.sqrt()
}

/// Undiscounted Black-76 option value `F·N(d1) − K·N(d2)` (call form;
/// sign-flipped via put-call parity for puts). Useful inside the IV solver.
#[inline]
pub fn undiscounted_black_price(f: f64, k: f64, sigma: f64, t: f64, opt: OptionType) -> f64 {
    let q = opt.sign();
    let d1 = d1(f, k, t, sigma);
    let d2 = d2(f, k, t, sigma);
    q * (f * norm_cdf(q * d1) - k * norm_cdf(q * d2))
}

/// Discounted Black-76 option price `e^(-rT)·[F·N(d1) − K·N(d2)]` (call;
/// put via parity).
#[inline]
pub fn black_price(f: f64, k: f64, sigma: f64, t: f64, r: f64, opt: OptionType) -> f64 {
    (-r * t).exp() * undiscounted_black_price(f, k, sigma, t, opt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn atm_call_matches_known() {
        // F=22000, K=22000, T=30/365, r=0.07, σ=0.18 -> reference price 450.2695513559
        let p = black_price(22000.0, 22000.0, 0.18, 30.0 / 365.0, 0.07, OptionType::Call);
        assert!(approx_eq(p, 450.2695513559, 1e-9), "got {}", p);
    }

    #[test]
    fn put_call_parity() {
        // c - p = e^(-rT) (F - K) for Black-76
        let f = 22000.0;
        let k = 21500.0;
        let t = 30.0 / 365.0;
        let r = 0.07;
        let sigma = 0.22;
        let c = black_price(f, k, sigma, t, r, OptionType::Call);
        let p = black_price(f, k, sigma, t, r, OptionType::Put);
        let parity = (-r * t).exp() * (f - k);
        assert!(approx_eq(c - p, parity, 1e-12));
    }

    #[test]
    fn deep_otm_call_is_small_positive() {
        // F=22000, K=30000, T=30d, σ=0.18 -> very small but positive
        let p = black_price(22000.0, 30000.0, 0.18, 30.0 / 365.0, 0.07, OptionType::Call);
        assert!(p >= 0.0);
        assert!(p < 1.0, "expected sub-cent value, got {}", p);
    }
}
