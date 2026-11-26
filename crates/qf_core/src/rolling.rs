//! Trailing-window statistics. Each function reproduces its numpy reference (same operations in the same order),
//! checked by tests/test_rust_parity.py:
//!   rolling_mean     <- qforecast.swing.features._rolling_mean_py (cumsum difference, NaN-aware)
//!   rolling_extreme  <- numpy.nanmax / numpy.nanmin over a trailing sliding_window_view

use numpy::ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use std::collections::VecDeque;

/// Trailing mean over `w` rows (axis 0): s = cumsum(v)[k] - cumsum(v)[k-w], n likewise; NaN when n < min_frac*w.
/// Rows are swept once, all columns at a time (contiguous memory), keeping the full running sums like numpy does.
pub fn rolling_mean(x: ArrayView2<'_, f64>, w: usize, min_frac: f64) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut cs = vec![0.0f64; t * n]; // cumulative sums of finite values
    let mut cn = vec![0i64; t * n]; // cumulative counts of finite values
    for k in 0..t {
        for j in 0..n {
            let v = x[[k, j]];
            let (fv, ok) = if v.is_finite() { (v, 1) } else { (0.0, 0) };
            let i = k * n + j;
            if k == 0 {
                cs[i] = fv;
                cn[i] = ok;
            } else {
                cs[i] = cs[i - n] + fv;
                cn[i] = cn[i - n] + ok;
            }
        }
    }
    let thr = min_frac * w as f64;
    let mut out = vec![f64::NAN; t * n];
    out.par_chunks_mut(n.max(1)).enumerate().for_each(|(k, o)| {
        for (j, oj) in o.iter_mut().enumerate() {
            let i = k * n + j;
            let (s, c) = if k >= w && w > 0 { (cs[i] - cs[i - w * n], cn[i] - cn[i - w * n]) } else { (cs[i], cn[i]) };
            let c = c as f64;
            *oj = if c >= thr { s / c.max(1.0) } else { f64::NAN };
        }
    });
    Array2::from_shape_vec((t, n), out).expect("shape")
}

/// Trailing max (`is_max`) or min over `w` rows, ignoring NaN; NaN for the first w-1 rows and for all-NaN
/// windows (numpy nanmax / nanmin over sliding_window_view). Monotonic deque: O(T) per column, columns in parallel.
pub fn rolling_extreme(x: ArrayView2<'_, f64>, w: usize, is_max: bool) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = Array2::from_elem((t, n), f64::NAN);
    if w == 0 || t < w {
        return out;
    }
    let cols: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|j| {
            let mut col = vec![f64::NAN; t];
            let mut dq: VecDeque<usize> = VecDeque::new(); // row indices, values monotone (decreasing for max)
            for k in 0..t {
                let v = x[[k, j]];
                if !v.is_nan() {
                    while let Some(&b) = dq.back() {
                        let bv = x[[b, j]];
                        if (is_max && bv <= v) || (!is_max && bv >= v) {
                            dq.pop_back();
                        } else {
                            break;
                        }
                    }
                    dq.push_back(k);
                }
                while let Some(&f) = dq.front() {
                    if f + w <= k {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
                if k + 1 >= w {
                    if let Some(&f) = dq.front() {
                        col[k] = x[[f, j]];
                    }
                }
            }
            col
        })
        .collect();
    for (j, col) in cols.into_iter().enumerate() {
        out.index_axis_mut(Axis(1), j).assign(&numpy::ndarray::Array1::from(col));
    }
    out
}
