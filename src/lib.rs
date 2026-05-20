//! PyO3 bindings for OpenGreeks. Single cdylib `_opengreeks` exposing all model
//! functions with model-prefixed names. The Python layer (`opengreeks.black76`
//! etc.) re-exports them under clean per-model names.

use black76_rust::{
    delta as rs_delta, gamma as rs_gamma, implied_volatility as rs_iv,
    black_price as rs_price, rho as rs_rho, theta as rs_theta, vega as rs_vega,
    IvError, OptionType,
};
use bsm_rust as bsm;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::prelude::*;

#[inline]
fn parse_flag(flag: &str) -> PyResult<OptionType> {
    let c = flag.chars().next().ok_or_else(|| PyValueError::new_err("flag must be 'c' or 'p'"))?;
    OptionType::from_flag(c).ok_or_else(|| PyValueError::new_err(format!(
        "flag must be 'c' or 'p', got {:?}", flag
    )))
}

#[inline]
fn iv_err(e: IvError) -> PyErr {
    match e {
        IvError::BelowIntrinsic => PyValueError::new_err("price is below intrinsic value"),
        IvError::AboveMaximum   => PyValueError::new_err("price exceeds theoretical maximum"),
        IvError::NotConverged   => PyRuntimeError::new_err("IV solver failed to converge"),
    }
}

// =============================================================================
// Black-76 — scalar
// =============================================================================

#[pyfunction]
#[pyo3(signature = (flag, F, K, t, r, sigma))]
#[allow(non_snake_case)]
fn black76_black(flag: &str, F: f64, K: f64, t: f64, r: f64, sigma: f64) -> PyResult<f64> {
    let opt = parse_flag(flag)?;
    Ok(rs_price(F, K, sigma, t, r, opt))
}

#[pyfunction]
#[pyo3(signature = (price, F, K, r, t, flag))]
#[allow(non_snake_case)]
fn black76_implied_volatility(price: f64, F: f64, K: f64, r: f64, t: f64, flag: &str) -> PyResult<f64> {
    let opt = parse_flag(flag)?;
    rs_iv(price, F, K, t, r, opt).map_err(iv_err)
}

macro_rules! greek_scalar {
    ($name:ident, $core:ident) => {
        #[pyfunction]
        #[pyo3(signature = (flag, F, K, t, r, sigma))]
        #[allow(non_snake_case)]
        fn $name(flag: &str, F: f64, K: f64, t: f64, r: f64, sigma: f64) -> PyResult<f64> {
            let opt = parse_flag(flag)?;
            Ok($core(F, K, sigma, t, r, opt))
        }
    };
}
greek_scalar!(black76_delta, rs_delta);
greek_scalar!(black76_gamma, rs_gamma);
greek_scalar!(black76_vega,  rs_vega);
greek_scalar!(black76_theta, rs_theta);
greek_scalar!(black76_rho,   rs_rho);

// =============================================================================
// Black-76 — batch / NumPy
// =============================================================================

fn run_chain<F>(
    py: Python<'_>,
    flag: &str,
    forwards: PyReadonlyArray1<f64>,
    strikes: PyReadonlyArray1<f64>,
    ts: PyReadonlyArray1<f64>,
    r: f64,
    sigmas: PyReadonlyArray1<f64>,
    f: F,
) -> PyResult<Py<PyArray1<f64>>>
where
    F: Fn(f64, f64, f64, f64, f64, OptionType) -> f64,
{
    let opt = parse_flag(flag)?;
    let fs = forwards.as_slice()?;
    let ks = strikes.as_slice()?;
    let tt = ts.as_slice()?;
    let ss = sigmas.as_slice()?;
    let n = fs.len();
    if !(ks.len() == n && tt.len() == n && ss.len() == n) {
        return Err(PyValueError::new_err("F, K, t, sigma arrays must have equal length"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f(fs[i], ks[i], ss[i], tt[i], r, opt));
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (flag, F, K, t, r, sigma))]
#[allow(non_snake_case)]
fn black76_black_array<'py>(
    py: Python<'py>,
    flag: &str,
    F: PyReadonlyArray1<'py, f64>,
    K: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    r: f64,
    sigma: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    run_chain(py, flag, F, K, t, r, sigma, rs_price)
}

macro_rules! greek_array {
    ($name:ident, $core:ident) => {
        #[pyfunction]
        #[pyo3(signature = (flag, F, K, t, r, sigma))]
        #[allow(non_snake_case)]
        fn $name<'py>(
            py: Python<'py>,
            flag: &str,
            F: PyReadonlyArray1<'py, f64>,
            K: PyReadonlyArray1<'py, f64>,
            t: PyReadonlyArray1<'py, f64>,
            r: f64,
            sigma: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Py<PyArray1<f64>>> {
            run_chain(py, flag, F, K, t, r, sigma, $core)
        }
    };
}
greek_array!(black76_delta_array, rs_delta);
greek_array!(black76_gamma_array, rs_gamma);
greek_array!(black76_vega_array,  rs_vega);
greek_array!(black76_theta_array, rs_theta);
greek_array!(black76_rho_array,   rs_rho);

#[pyfunction]
#[pyo3(signature = (price, F, K, r, t, flag))]
#[allow(non_snake_case)]
fn black76_implied_volatility_array<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    F: PyReadonlyArray1<'py, f64>,
    K: PyReadonlyArray1<'py, f64>,
    r: f64,
    t: PyReadonlyArray1<'py, f64>,
    flag: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let opt = parse_flag(flag)?;
    let prices = price.as_slice()?;
    let fs = F.as_slice()?;
    let ks = K.as_slice()?;
    let tt = t.as_slice()?;
    let n = prices.len();
    if !(fs.len() == n && ks.len() == n && tt.len() == n) {
        return Err(PyValueError::new_err("price, F, K, t arrays must have equal length"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match rs_iv(prices[i], fs[i], ks[i], tt[i], r, opt) {
            Ok(s) => s,
            Err(_) => f64::NAN,
        });
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

// =============================================================================
// Black-Scholes-Merton — scalar (with continuous dividend yield `q`)
// =============================================================================

#[pyfunction]
#[pyo3(signature = (flag, S, K, t, r, sigma, q))]
#[allow(non_snake_case)]
fn bsm_black_scholes_merton(flag: &str, S: f64, K: f64, t: f64, r: f64, sigma: f64, q: f64) -> PyResult<f64> {
    let opt = parse_flag(flag)?;
    Ok(bsm::bsm_price(S, K, sigma, t, r, q, opt))
}

#[pyfunction]
#[pyo3(signature = (price, S, K, t, r, q, flag))]
#[allow(non_snake_case)]
fn bsm_implied_volatility(price: f64, S: f64, K: f64, t: f64, r: f64, q: f64, flag: &str) -> PyResult<f64> {
    let opt = parse_flag(flag)?;
    bsm::implied_volatility(price, S, K, t, r, q, opt).map_err(iv_err)
}

macro_rules! bsm_greek_scalar {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (flag, S, K, t, r, sigma, q))]
        #[allow(non_snake_case)]
        fn $name(flag: &str, S: f64, K: f64, t: f64, r: f64, sigma: f64, q: f64) -> PyResult<f64> {
            let opt = parse_flag(flag)?;
            Ok($core(S, K, sigma, t, r, q, opt))
        }
    };
}
bsm_greek_scalar!(bsm_delta, bsm::delta);
bsm_greek_scalar!(bsm_gamma, bsm::gamma);
bsm_greek_scalar!(bsm_vega,  bsm::vega);
bsm_greek_scalar!(bsm_theta, bsm::theta);
bsm_greek_scalar!(bsm_rho,   bsm::rho);

// =============================================================================
// Black-Scholes — BSM with q=0 (signatures without `q`)
// =============================================================================

#[pyfunction]
#[pyo3(signature = (flag, S, K, t, r, sigma))]
#[allow(non_snake_case)]
fn bs_black_scholes(flag: &str, S: f64, K: f64, t: f64, r: f64, sigma: f64) -> PyResult<f64> {
    let opt = parse_flag(flag)?;
    Ok(bsm::bsm_price(S, K, sigma, t, r, 0.0, opt))
}

#[pyfunction]
#[pyo3(signature = (price, S, K, t, r, flag))]
#[allow(non_snake_case)]
fn bs_implied_volatility(price: f64, S: f64, K: f64, t: f64, r: f64, flag: &str) -> PyResult<f64> {
    let opt = parse_flag(flag)?;
    bsm::implied_volatility(price, S, K, t, r, 0.0, opt).map_err(iv_err)
}

macro_rules! bs_greek_scalar {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (flag, S, K, t, r, sigma))]
        #[allow(non_snake_case)]
        fn $name(flag: &str, S: f64, K: f64, t: f64, r: f64, sigma: f64) -> PyResult<f64> {
            let opt = parse_flag(flag)?;
            Ok($core(S, K, sigma, t, r, 0.0, opt))
        }
    };
}
bs_greek_scalar!(bs_delta, bsm::delta);
bs_greek_scalar!(bs_gamma, bsm::gamma);
bs_greek_scalar!(bs_vega,  bsm::vega);
bs_greek_scalar!(bs_theta, bsm::theta);
bs_greek_scalar!(bs_rho,   bsm::rho);

// =============================================================================
// BSM / BS — batch
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn run_bsm_chain<F>(
    py: Python<'_>,
    flag: &str,
    spots: PyReadonlyArray1<f64>,
    strikes: PyReadonlyArray1<f64>,
    ts: PyReadonlyArray1<f64>,
    r: f64,
    sigmas: PyReadonlyArray1<f64>,
    q: f64,
    f: F,
) -> PyResult<Py<PyArray1<f64>>>
where
    F: Fn(f64, f64, f64, f64, f64, f64, OptionType) -> f64,
{
    let opt = parse_flag(flag)?;
    let ss = spots.as_slice()?;
    let ks = strikes.as_slice()?;
    let tt = ts.as_slice()?;
    let sg = sigmas.as_slice()?;
    let n = ss.len();
    if !(ks.len() == n && tt.len() == n && sg.len() == n) {
        return Err(PyValueError::new_err("S, K, t, sigma arrays must have equal length"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(f(ss[i], ks[i], sg[i], tt[i], r, q, opt));
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (flag, S, K, t, r, sigma, q))]
#[allow(non_snake_case)]
fn bsm_black_scholes_merton_array<'py>(
    py: Python<'py>, flag: &str,
    S: PyReadonlyArray1<'py, f64>, K: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>, r: f64, sigma: PyReadonlyArray1<'py, f64>, q: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    run_bsm_chain(py, flag, S, K, t, r, sigma, q, bsm::bsm_price)
}

macro_rules! bsm_greek_array {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (flag, S, K, t, r, sigma, q))]
        #[allow(non_snake_case)]
        fn $name<'py>(
            py: Python<'py>, flag: &str,
            S: PyReadonlyArray1<'py, f64>, K: PyReadonlyArray1<'py, f64>,
            t: PyReadonlyArray1<'py, f64>, r: f64, sigma: PyReadonlyArray1<'py, f64>, q: f64,
        ) -> PyResult<Py<PyArray1<f64>>> {
            run_bsm_chain(py, flag, S, K, t, r, sigma, q, $core)
        }
    };
}
bsm_greek_array!(bsm_delta_array, bsm::delta);
bsm_greek_array!(bsm_gamma_array, bsm::gamma);
bsm_greek_array!(bsm_vega_array,  bsm::vega);
bsm_greek_array!(bsm_theta_array, bsm::theta);
bsm_greek_array!(bsm_rho_array,   bsm::rho);

#[pyfunction]
#[pyo3(signature = (price, S, K, t, r, q, flag))]
#[allow(non_snake_case)]
fn bsm_implied_volatility_array<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    S: PyReadonlyArray1<'py, f64>, K: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>, r: f64, q: f64, flag: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let opt = parse_flag(flag)?;
    let prices = price.as_slice()?;
    let ss = S.as_slice()?;
    let ks = K.as_slice()?;
    let tt = t.as_slice()?;
    let n = prices.len();
    if !(ss.len() == n && ks.len() == n && tt.len() == n) {
        return Err(PyValueError::new_err("price, S, K, t arrays must have equal length"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match bsm::implied_volatility(prices[i], ss[i], ks[i], tt[i], r, q, opt) {
            Ok(s) => s,
            Err(_) => f64::NAN,
        });
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

// BS batch — same as BSM batch but q=0
#[pyfunction]
#[pyo3(signature = (flag, S, K, t, r, sigma))]
#[allow(non_snake_case)]
fn bs_black_scholes_array<'py>(
    py: Python<'py>, flag: &str,
    S: PyReadonlyArray1<'py, f64>, K: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>, r: f64, sigma: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    run_bsm_chain(py, flag, S, K, t, r, sigma, 0.0, bsm::bsm_price)
}

macro_rules! bs_greek_array {
    ($name:ident, $core:path) => {
        #[pyfunction]
        #[pyo3(signature = (flag, S, K, t, r, sigma))]
        #[allow(non_snake_case)]
        fn $name<'py>(
            py: Python<'py>, flag: &str,
            S: PyReadonlyArray1<'py, f64>, K: PyReadonlyArray1<'py, f64>,
            t: PyReadonlyArray1<'py, f64>, r: f64, sigma: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Py<PyArray1<f64>>> {
            run_bsm_chain(py, flag, S, K, t, r, sigma, 0.0, $core)
        }
    };
}
bs_greek_array!(bs_delta_array, bsm::delta);
bs_greek_array!(bs_gamma_array, bsm::gamma);
bs_greek_array!(bs_vega_array,  bsm::vega);
bs_greek_array!(bs_theta_array, bsm::theta);
bs_greek_array!(bs_rho_array,   bsm::rho);

#[pyfunction]
#[pyo3(signature = (price, S, K, t, r, flag))]
#[allow(non_snake_case)]
fn bs_implied_volatility_array<'py>(
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    S: PyReadonlyArray1<'py, f64>, K: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>, r: f64, flag: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let opt = parse_flag(flag)?;
    let prices = price.as_slice()?;
    let ss = S.as_slice()?;
    let ks = K.as_slice()?;
    let tt = t.as_slice()?;
    let n = prices.len();
    if !(ss.len() == n && ks.len() == n && tt.len() == n) {
        return Err(PyValueError::new_err("price, S, K, t arrays must have equal length"));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(match bsm::implied_volatility(prices[i], ss[i], ks[i], tt[i], r, 0.0, opt) {
            Ok(s) => s,
            Err(_) => f64::NAN,
        });
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

// =============================================================================
// Module registration
// =============================================================================

#[pymodule]
fn _opengreeks(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Black-76 — scalar
    m.add_function(wrap_pyfunction!(black76_black, m)?)?;
    m.add_function(wrap_pyfunction!(black76_implied_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(black76_delta, m)?)?;
    m.add_function(wrap_pyfunction!(black76_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(black76_vega, m)?)?;
    m.add_function(wrap_pyfunction!(black76_theta, m)?)?;
    m.add_function(wrap_pyfunction!(black76_rho, m)?)?;
    // Black-76 — batch
    m.add_function(wrap_pyfunction!(black76_black_array, m)?)?;
    m.add_function(wrap_pyfunction!(black76_implied_volatility_array, m)?)?;
    m.add_function(wrap_pyfunction!(black76_delta_array, m)?)?;
    m.add_function(wrap_pyfunction!(black76_gamma_array, m)?)?;
    m.add_function(wrap_pyfunction!(black76_vega_array, m)?)?;
    m.add_function(wrap_pyfunction!(black76_theta_array, m)?)?;
    m.add_function(wrap_pyfunction!(black76_rho_array, m)?)?;

    // BSM — scalar
    m.add_function(wrap_pyfunction!(bsm_black_scholes_merton, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_implied_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_delta, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_vega, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_theta, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_rho, m)?)?;
    // BSM — batch
    m.add_function(wrap_pyfunction!(bsm_black_scholes_merton_array, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_implied_volatility_array, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_delta_array, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_gamma_array, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_vega_array, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_theta_array, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_rho_array, m)?)?;
    // BS — scalar (BSM with q=0)
    m.add_function(wrap_pyfunction!(bs_black_scholes, m)?)?;
    m.add_function(wrap_pyfunction!(bs_implied_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(bs_delta, m)?)?;
    m.add_function(wrap_pyfunction!(bs_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(bs_vega, m)?)?;
    m.add_function(wrap_pyfunction!(bs_theta, m)?)?;
    m.add_function(wrap_pyfunction!(bs_rho, m)?)?;
    // BS — batch
    m.add_function(wrap_pyfunction!(bs_black_scholes_array, m)?)?;
    m.add_function(wrap_pyfunction!(bs_implied_volatility_array, m)?)?;
    m.add_function(wrap_pyfunction!(bs_delta_array, m)?)?;
    m.add_function(wrap_pyfunction!(bs_gamma_array, m)?)?;
    m.add_function(wrap_pyfunction!(bs_vega_array, m)?)?;
    m.add_function(wrap_pyfunction!(bs_theta_array, m)?)?;
    m.add_function(wrap_pyfunction!(bs_rho_array, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
