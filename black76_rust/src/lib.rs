//! Black-76 option pricing, analytical Greeks, and implied volatility.
//!
//! Zero external dependencies. All math via `f64` std methods + a
//! hand-ported Cody erfc rational-Chebyshev approximation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod erf;
mod normal;
mod pricing;
mod greeks;
mod iv;

pub use normal::{norm_cdf, norm_pdf, inverse_norm_cdf};
pub use pricing::{black_price, undiscounted_black_price, d1, d2};
pub use greeks::{delta, gamma, vega, theta, rho};
pub use iv::{implied_volatility, IvError};

/// Option type. `Call` ↔ `is_call = true`, `Put` ↔ `is_call = false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    /// Call option (right to buy).
    Call,
    /// Put option (right to sell).
    Put,
}

impl OptionType {
    /// Returns +1.0 for Call, -1.0 for Put. Matches Jaeckel's `q` convention.
    #[inline]
    pub fn sign(self) -> f64 {
        match self { OptionType::Call => 1.0, OptionType::Put => -1.0 }
    }

    /// Parse from a single-character flag: `'c'`/`'C'` → Call, `'p'`/`'P'` → Put.
    pub fn from_flag(flag: char) -> Option<Self> {
        match flag {
            'c' | 'C' => Some(OptionType::Call),
            'p' | 'P' => Some(OptionType::Put),
            _ => None,
        }
    }
}
