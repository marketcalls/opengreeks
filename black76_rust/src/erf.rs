//! W. J. Cody (1969) rational-Chebyshev erf / erfc / erfcx.
//!
//! Reference: W. J. Cody, "Rational Chebyshev approximations for the error
//! function," Math. Comp. 23 (1969), pp. 631–638. Accurate to ~18 significant
//! decimal digits in IEEE 754 f64.

const A: [f64; 5] = [
    3.1611237438705656,
    113.864154151050156,
    377.485237685302021,
    3209.37758913846947,
    0.185777706184603153,
];
const B: [f64; 4] = [
    23.6012909523441209,
    244.024637934444173,
    1282.61652607737228,
    2844.23683343917062,
];
const C: [f64; 9] = [
    0.564188496988670089,
    8.88314979438837594,
    66.1191906371416295,
    298.635138197400131,
    881.95222124176909,
    1712.04761263407058,
    2051.07837782607147,
    1230.33935479799725,
    2.15311535474403846e-8,
];
const D: [f64; 8] = [
    15.7449261107098347,
    117.693950891312499,
    537.181101862009858,
    1621.38957456669019,
    3290.79923573345963,
    4362.61909014324716,
    3439.36767414372164,
    1230.33935480374942,
];
const P: [f64; 6] = [
    0.305326634961232344,
    0.360344899949804439,
    0.125781726111229246,
    0.0160837851487422766,
    6.58749161529837803e-4,
    0.0163153871373020978,
];
const Q: [f64; 5] = [
    2.56852019228982242,
    1.87295284992346047,
    0.527905102951428412,
    0.0605183413124413191,
    0.00233520497626869185,
];

const SQRPI: f64 = 0.56418958354775628695;
const THRESH: f64 = 0.46875;
const SIXTEEN: f64 = 16.0;

const XINF: f64 = 1.79e308;
const XNEG: f64 = -26.628;
const XSMALL: f64 = 1.11e-16;
const XBIG: f64 = 26.543;
const XHUGE: f64 = 6.71e7;
const XMAX: f64 = 2.53e307;

#[inline]
fn d_int(x: f64) -> f64 {
    if x > 0.0 { x.floor() } else { -(-x).floor() }
}

/// CALERF: shared routine for erf (jint=0), erfc (jint=1), erfcx (jint=2).
fn calerf(x: f64, jint: u8) -> f64 {
    let y = x.abs();

    let mut result: f64;

    if y <= THRESH {
        // |x| <= 0.46875 — evaluate erf via rational on x^2
        let ysq = if y > XSMALL { y * y } else { 0.0 };
        let mut xnum = A[4] * ysq;
        let mut xden = ysq;
        for i in 0..3 {
            xnum = (xnum + A[i]) * ysq;
            xden = (xden + B[i]) * ysq;
        }
        result = x * (xnum + A[3]) / (xden + B[3]);
        if jint != 0 {
            result = 1.0 - result;
        }
        if jint == 2 {
            result *= ysq.exp();
        }
        return result;
    } else if y <= 4.0 {
        // 0.46875 < |x| <= 4 — rational in y for erfc
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        result = (xnum + C[7]) / (xden + D[7]);
        if jint != 2 {
            let ysq = d_int(y * SIXTEEN) / SIXTEEN;
            let del = (y - ysq) * (y + ysq);
            result *= (-ysq * ysq).exp() * (-del).exp();
        }
    } else {
        // |x| > 4 — asymptotic expansion for erfc
        result = 0.0;
        if y >= XBIG {
            if jint != 2 || y >= XMAX {
                return fix_up_for_negative(jint, result, x);
            }
            if y >= XHUGE {
                result = SQRPI / y;
                return fix_up_for_negative(jint, result, x);
            }
        }
        let ysq = 1.0 / (y * y);
        let mut xnum = P[5] * ysq;
        let mut xden = ysq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        result = ysq * (xnum + P[4]) / (xden + Q[4]);
        result = (SQRPI - result) / y;
        if jint != 2 {
            let ysq2 = d_int(y * SIXTEEN) / SIXTEEN;
            let del = (y - ysq2) * (y + ysq2);
            result *= (-ysq2 * ysq2).exp() * (-del).exp();
        }
    }

    fix_up_for_negative(jint, result, x)
}

#[inline]
fn fix_up_for_negative(jint: u8, mut result: f64, x: f64) -> f64 {
    match jint {
        0 => {
            result = (0.5 - result) + 0.5;
            if x < 0.0 { result = -result; }
            result
        }
        1 => {
            if x < 0.0 { result = 2.0 - result; }
            result
        }
        _ => {
            // jint == 2 (erfcx)
            if x < 0.0 {
                if x < XNEG {
                    result = XINF;
                } else {
                    let ysq = d_int(x * SIXTEEN) / SIXTEEN;
                    let del = (x - ysq) * (x + ysq);
                    let y = (ysq * ysq).exp() * del.exp();
                    result = (y + y) - result;
                }
            }
            result
        }
    }
}

/// Error function `erf(x) = 2/√π · ∫₀ˣ e^(-t²) dt`.
/// Exposed for completeness — internally we only call `erfc` (from `norm_cdf`).
#[allow(dead_code)]
#[inline]
pub fn erf(x: f64) -> f64 { calerf(x, 0) }

/// Complementary error function `erfc(x) = 1 - erf(x)`.
#[inline]
pub fn erfc(x: f64) -> f64 { calerf(x, 1) }

/// Scaled complementary error function `erfcx(x) = e^(x²) · erfc(x)`.
/// Exposed for completeness — not used internally.
#[allow(dead_code)]
#[inline]
pub fn erfcx(x: f64) -> f64 { calerf(x, 2) }

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn erf_known_values() {
        // From standard tables / Wolfram.
        assert!(approx_eq(erf(0.0), 0.0, 1e-15));
        assert!(approx_eq(erf(1.0), 0.8427007929497149, 1e-14));
        assert!(approx_eq(erf(0.5), 0.5204998778130465, 1e-14));
        assert!(approx_eq(erf(-1.0), -0.8427007929497149, 1e-14));
        assert!(approx_eq(erf(3.0), 0.9999779095030014, 1e-14));
    }

    #[test]
    fn erfc_known_values() {
        assert!(approx_eq(erfc(0.0), 1.0, 1e-15));
        assert!(approx_eq(erfc(1.0), 0.15729920705028513, 1e-14));
        assert!(approx_eq(erfc(2.0), 0.004677734981047266, 1e-14));
        assert!(approx_eq(erfc(-1.0), 1.8427007929497148, 1e-14));
    }

    #[test]
    fn erf_erfc_complementary() {
        for x in [-3.0, -1.5, -0.3, 0.0, 0.3, 1.5, 3.0] {
            assert!(approx_eq(erf(x) + erfc(x), 1.0, 1e-14), "x={}", x);
        }
    }
}
