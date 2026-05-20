//! Standard normal CDF, PDF, and inverse CDF.
//!
//! `norm_cdf` uses Cody's erfc plus the Abramowitz-Stegun (26.2.12)
//! asymptotic expansion for very negative arguments, matching
//! `py_lets_be_rational.normaldistribution.norm_cdf`.
//!
//! `inverse_norm_cdf` is Algorithm AS 241 (Wichura 1988) — Φ⁻¹ accurate
//! to ~1 part in 10¹⁶.

use crate::erf::erfc;

const ONE_OVER_SQRT_TWO_PI: f64 = 0.3989422804014326779399460599343818684758586311649;
const ONE_OVER_SQRT_TWO: f64 = 0.7071067811865475244008443621048490392848359376887;

const ASYMPTOTIC_THRESHOLD_HIGH: f64 = -10.0;

/// Standard normal probability density function `φ(x) = (1/√(2π))·exp(-x²/2)`.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    ONE_OVER_SQRT_TWO_PI * (-0.5 * x * x).exp()
}

/// Standard normal cumulative distribution function `Φ(z)`.
///
/// For `z ≤ -10` uses an asymptotic expansion; otherwise uses `0.5·erfc(-z/√2)`.
/// Matches `py_lets_be_rational.normaldistribution.norm_cdf` to f64 precision.
pub fn norm_cdf(z: f64) -> f64 {
    let dbl_epsilon = f64::EPSILON;
    let asymptotic_threshold_low = -1.0 / dbl_epsilon.sqrt();

    if z <= ASYMPTOTIC_THRESHOLD_HIGH {
        // Asymptotic expansion: Φ(z) = φ(z)/|z| · [1 - 1/z² + 3/z⁴ - 15/z⁶ + ...]
        let mut sum = 1.0;
        if z >= asymptotic_threshold_low {
            let zsqr = z * z;
            let mut i: f64 = 1.0;
            let mut g: f64 = 1.0;
            let dbl_max = f64::MAX;
            let mut a: f64 = dbl_max;

            let mut lasta;
            // First term unrolled (matches the Python implementation's first do-while iteration).
            lasta = a;
            let mut x = (4.0 * i - 3.0) / zsqr;
            let mut y = x * ((4.0 * i - 1.0) / zsqr);
            a = g * (x - y);
            sum -= a;
            g *= y;
            i += 1.0;
            a = a.abs();

            while lasta > a && a >= sum.abs() * dbl_epsilon {
                lasta = a;
                x = (4.0 * i - 3.0) / zsqr;
                y = x * ((4.0 * i - 1.0) / zsqr);
                a = g * (x - y);
                sum -= a;
                g *= y;
                i += 1.0;
                a = a.abs();
            }
        }
        return -norm_pdf(z) * sum / z;
    }
    0.5 * erfc(-z * ONE_OVER_SQRT_TWO)
}

// AS 241 coefficients
const AS241_SPLIT1: f64 = 0.425;
const AS241_SPLIT2: f64 = 5.0;
const AS241_CONST1: f64 = 0.180625;
const AS241_CONST2: f64 = 1.6;

const AS241_A: [f64; 8] = [
    3.3871328727963666080,
    1.3314166789178437745e2,
    1.9715909503065514427e3,
    1.3731693765509461125e4,
    4.5921953931549871457e4,
    6.7265770927008700853e4,
    3.3430575583588128105e4,
    2.5090809287301226727e3,
];
const AS241_B: [f64; 7] = [
    4.2313330701600911252e1,
    6.8718700749205790830e2,
    5.3941960214247511077e3,
    2.1213794301586595867e4,
    3.9307895800092710610e4,
    2.8729085735721942674e4,
    5.2264952788528545610e3,
];
const AS241_C: [f64; 8] = [
    1.42343711074968357734,
    4.63033784615654529590,
    5.76949722146069140550,
    3.64784832476320460504,
    1.27045825245236838258,
    2.41780725177450611770e-1,
    2.27238449892691845833e-2,
    7.74545014278341407640e-4,
];
const AS241_D: [f64; 7] = [
    2.05319162663775882187,
    1.67638483018380384940,
    6.89767334985100004550e-1,
    1.48103976427480074590e-1,
    1.51986665636164571966e-2,
    5.47593808499534494600e-4,
    1.05075007164441684324e-9,
];
const AS241_E: [f64; 8] = [
    6.65790464350110377720,
    5.46378491116411436990,
    1.78482653991729133580,
    2.96560571828504891230e-1,
    2.65321895265761230930e-2,
    1.24266094738807843860e-3,
    2.71155556874348757815e-5,
    2.01033439929228813265e-7,
];
const AS241_F: [f64; 7] = [
    5.99832206555887937690e-1,
    1.36929880922735805310e-1,
    1.48753612908506148525e-2,
    7.86869131145613259100e-4,
    1.84631831751005468180e-5,
    1.42151175831644588870e-7,
    2.04426310338993978564e-15,
];

/// Inverse standard normal CDF (Wichura AS 241). Accurate to ~1 part in 10¹⁶.
pub fn inverse_norm_cdf(u: f64) -> f64 {
    if u <= 0.0 { return u.ln(); }
    if u >= 1.0 { return (1.0 - u).ln(); }

    let q = u - 0.5;
    if q.abs() <= AS241_SPLIT1 {
        let r = AS241_CONST1 - q * q;
        let num = ((((((AS241_A[7] * r + AS241_A[6]) * r + AS241_A[5]) * r + AS241_A[4]) * r + AS241_A[3]) * r + AS241_A[2]) * r + AS241_A[1]) * r + AS241_A[0];
        let den = ((((((AS241_B[6] * r + AS241_B[5]) * r + AS241_B[4]) * r + AS241_B[3]) * r + AS241_B[2]) * r + AS241_B[1]) * r + AS241_B[0]) * r + 1.0;
        return q * num / den;
    }

    let r0 = if q < 0.0 { u } else { 1.0 - u };
    let mut r = (-(r0.ln())).sqrt();
    let ret = if r < AS241_SPLIT2 {
        r -= AS241_CONST2;
        let num = ((((((AS241_C[7] * r + AS241_C[6]) * r + AS241_C[5]) * r + AS241_C[4]) * r + AS241_C[3]) * r + AS241_C[2]) * r + AS241_C[1]) * r + AS241_C[0];
        let den = ((((((AS241_D[6] * r + AS241_D[5]) * r + AS241_D[4]) * r + AS241_D[3]) * r + AS241_D[2]) * r + AS241_D[1]) * r + AS241_D[0]) * r + 1.0;
        num / den
    } else {
        r -= AS241_SPLIT2;
        let num = ((((((AS241_E[7] * r + AS241_E[6]) * r + AS241_E[5]) * r + AS241_E[4]) * r + AS241_E[3]) * r + AS241_E[2]) * r + AS241_E[1]) * r + AS241_E[0];
        let den = ((((((AS241_F[6] * r + AS241_F[5]) * r + AS241_F[4]) * r + AS241_F[3]) * r + AS241_F[2]) * r + AS241_F[1]) * r + AS241_F[0]) * r + 1.0;
        num / den
    };
    if q < 0.0 { -ret } else { ret }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn norm_cdf_anchors() {
        assert!(approx_eq(norm_cdf(0.0), 0.5, 1e-15));
        assert!(approx_eq(norm_cdf(1.0), 0.8413447460685429, 1e-14));
        assert!(approx_eq(norm_cdf(-1.0), 0.15865525393145707, 1e-14));
        assert!(approx_eq(norm_cdf(1.96), 0.9750021048517795, 1e-14));
        assert!(approx_eq(norm_cdf(3.0), 0.9986501019683699, 1e-14));
        assert!(approx_eq(norm_cdf(-3.0), 0.0013498980316301035, 1e-14));
    }

    #[test]
    fn norm_cdf_deep_tails() {
        // Asymptotic path triggers at z <= -10
        assert!(norm_cdf(-12.0) > 0.0);
        assert!(norm_cdf(-12.0) < 1e-32);
        assert!(norm_cdf(-20.0) > 0.0);
    }

    #[test]
    fn norm_pdf_anchors() {
        assert!(approx_eq(norm_pdf(0.0), 0.3989422804014327, 1e-15));
        assert!(approx_eq(norm_pdf(1.0), 0.24197072451914337, 1e-15));
    }

    #[test]
    fn inverse_norm_cdf_roundtrip() {
        for u in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999] {
            let z = inverse_norm_cdf(u);
            assert!(approx_eq(norm_cdf(z), u, 1e-12), "u={}", u);
        }
    }
}
