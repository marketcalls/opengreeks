//! Implied volatility for Black-76.
//!
//! Strategy: Newton-Raphson with a Brenner-Subrahmanyam / Manaster-Koehler
//! hybrid initial guess, falling back to bisection on divergence. Typically
//! converges in 4-8 iterations to machine precision.
//!
//! This is a clean, correct, fast solver; not byte-for-byte identical to
//! Jaeckel's algorithm. A future milestone will port Jaeckel's "Let's Be
//! Rational" for an additional ~2× per-call speedup.

use crate::normal::norm_pdf;
use crate::pricing::{d1, undiscounted_black_price};
use crate::OptionType;

/// IV solver error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvError {
    /// Quoted price is below intrinsic value (no real IV exists).
    BelowIntrinsic,
    /// Quoted price exceeds the theoretical maximum (forward for call, K for put).
    AboveMaximum,
    /// Solver did not converge within iteration budget (should not occur for in-bounds prices).
    NotConverged,
}

const TWO_PI: f64 = 6.283185307179586476925286766559005768;
const MAX_NEWTON_ITERS: usize = 50;
const MAX_BISECTION_ITERS: usize = 100;
const PRICE_TOLERANCE_REL: f64 = 1e-12;
const SIGMA_LO: f64 = 1.0e-6;
const SIGMA_HI: f64 = 5.0;

/// Implied volatility from a quoted **discounted** option price.
///
/// `flag` semantics follow `OptionType`. Returns σ in absolute units (not %).
pub fn implied_volatility(
    discounted_price: f64,
    f: f64,
    k: f64,
    t: f64,
    r: f64,
    opt: OptionType,
) -> Result<f64, IvError> {
    // Work in undiscounted prices: undisc = e^(rT) · disc.
    let undisc_target = discounted_price * (r * t).exp();
    implied_volatility_undiscounted(undisc_target, f, k, t, opt)
}

/// IV from an undiscounted option price `F·N(d1) - K·N(d2)` (call form).
pub fn implied_volatility_undiscounted(
    undiscounted_price: f64,
    f: f64,
    k: f64,
    t: f64,
    opt: OptionType,
) -> Result<f64, IvError> {
    // Domain checks
    let intrinsic = match opt {
        OptionType::Call => (f - k).max(0.0),
        OptionType::Put => (k - f).max(0.0),
    };
    let upper = match opt {
        OptionType::Call => f,
        OptionType::Put => k,
    };
    if undiscounted_price < intrinsic - 1e-14 {
        return Err(IvError::BelowIntrinsic);
    }
    if undiscounted_price > upper + 1e-14 {
        return Err(IvError::AboveMaximum);
    }
    if undiscounted_price <= intrinsic {
        return Ok(SIGMA_LO);
    }

    let tol = PRICE_TOLERANCE_REL * (1.0 + undiscounted_price.abs());

    // Brenner-Subrahmanyam ATM-ish initial guess:
    //   σ₀ ≈ √(2π/T) · undisc_price / F   (call) or / K (put with K·N(-d2)-F·N(-d1))
    // Robust enough as a Newton seed for any non-edge case.
    let denom = match opt { OptionType::Call => f, OptionType::Put => k };
    let mut sigma = (TWO_PI / t).sqrt() * undiscounted_price / denom;
    if !sigma.is_finite() || sigma < SIGMA_LO { sigma = 0.20; }
    sigma = sigma.clamp(SIGMA_LO, SIGMA_HI);

    // Newton iterations on undiscounted price; vega here = F·√T·φ(d1) (no 0.01 scaling).
    // Convergence: stop when either price residual OR sigma step is tiny — the latter
    // matters for deep ITM where vega→0 makes price-tolerance unreliable.
    let mut prev_sigma = f64::NAN;
    for _ in 0..MAX_NEWTON_ITERS {
        let price = undiscounted_black_price(f, k, sigma, t, opt);
        let diff = price - undiscounted_price;
        if diff.abs() < tol {
            return Ok(sigma);
        }
        if !prev_sigma.is_nan() && (sigma - prev_sigma).abs() < 1e-12 * (1.0 + sigma) {
            return Ok(sigma);
        }
        let dd1 = d1(f, k, t, sigma);
        let vega = f * t.sqrt() * norm_pdf(dd1);
        if vega < 1e-15 {
            break;
        }
        let step = diff / vega;
        let next = sigma - step;
        if !next.is_finite() || next <= 0.0 || next > SIGMA_HI {
            break;
        }
        prev_sigma = sigma;
        sigma = next;
    }

    // Bisection fallback — guaranteed to converge for in-bounds prices.
    let mut lo = SIGMA_LO;
    let mut hi = SIGMA_HI;
    for _ in 0..MAX_BISECTION_ITERS {
        let mid = 0.5 * (lo + hi);
        let pm = undiscounted_black_price(f, k, mid, t, opt);
        if (pm - undiscounted_price).abs() < tol {
            return Ok(mid);
        }
        if pm < undiscounted_price { lo = mid; } else { hi = mid; }
        if hi - lo < 1e-14 {
            return Ok(0.5 * (lo + hi));
        }
    }
    Err(IvError::NotConverged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::black_price;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn iv_roundtrip_atm() {
        let f = 22000.0;
        let k = 22000.0;
        let t = 30.0 / 365.0;
        let r = 0.07;
        let sigma = 0.18;
        let opt = OptionType::Call;
        let p = black_price(f, k, sigma, t, r, opt);
        let iv = implied_volatility(p, f, k, t, r, opt).unwrap();
        assert!(approx_eq(iv, sigma, 1e-10), "got {}", iv);
    }

    #[test]
    fn iv_roundtrip_otm_chain() {
        let f = 22000.0;
        let t = 30.0 / 365.0;
        let r = 0.07;
        let sigma = 0.22;
        let opt = OptionType::Call;
        for k in (20000..=24000).step_by(100) {
            let k = k as f64;
            let p = black_price(f, k, sigma, t, r, opt);
            if p < 1e-10 { continue; }
            let iv = implied_volatility(p, f, k, t, r, opt).unwrap();
            assert!(approx_eq(iv, sigma, 1e-9), "K={} got iv={}", k, iv);
        }
    }

    #[test]
    fn iv_below_intrinsic_errors() {
        let res = implied_volatility(0.0, 22000.0, 20000.0, 30.0 / 365.0, 0.0, OptionType::Call);
        assert_eq!(res, Err(IvError::BelowIntrinsic));
    }
}
