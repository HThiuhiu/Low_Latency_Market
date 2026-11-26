//! Compiled hot paths for qforecast. Every function mirrors a pure-Python reference that stays in the package
//! (tests/test_rust_parity.py compares them, bit for bit where the operations allow):
//!   rolling_mean     <- qforecast.swing.features._rolling_mean_py
//!   rolling_extreme  <- qforecast.swing.signals._roll with numpy.nanmax / numpy.nanmin
//!   cs_rank_gauss    <- qforecast.swing.features._cs_rank_gauss_py
//! The Python package uses them through qforecast/accel.py when the wheel is installed; QF_NO_RUST=1 turns them off.

// pyo3 0.22 macros expand `?` on PyErr into a same-type conversion
#![allow(clippy::useless_conversion)]

mod rank;
mod rolling;

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;

/// Per row: ranks of the masked finite values -> normal scores (van der Waerden).
#[pyfunction]
#[pyo3(signature = (x, mask, min_names=5, ties=false))]
fn cs_rank_gauss<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    min_names: usize,
    ties: bool,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let (xa, ma) = (x.as_array(), mask.as_array());
    if xa.dim() != ma.dim() {
        return Err(pyo3::exceptions::PyValueError::new_err("x and mask must have the same shape"));
    }
    let out = py.allow_threads(|| rank::cs_rank_gauss(xa, ma, min_names, ties));
    Ok(out.into_pyarray_bound(py))
}

/// Trailing NaN-aware mean over `w` rows; NaN when fewer than min_frac * w finite values.
#[pyfunction]
#[pyo3(signature = (x, w, min_frac=0.8))]
fn rolling_mean<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w: usize,
    min_frac: f64,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let xa = x.as_array();
    let out = py.allow_threads(|| rolling::rolling_mean(xa, w, min_frac));
    Ok(out.into_pyarray_bound(py))
}

/// Trailing NaN-aware max (is_max=True) or min over `w` rows.
#[pyfunction]
fn rolling_extreme<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    w: usize,
    is_max: bool,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let xa = x.as_array();
    let out = py.allow_threads(|| rolling::rolling_extreme(xa, w, is_max));
    Ok(out.into_pyarray_bound(py))
}

/// Inverse standard normal CDF (Wichura AS241), exposed for tests.
#[pyfunction]
fn ndtri(p: f64) -> f64 {
    rank::ndtri(p)
}

#[pymodule]
fn qforecast_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cs_rank_gauss, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_extreme, m)?)?;
    m.add_function(wrap_pyfunction!(ndtri, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
