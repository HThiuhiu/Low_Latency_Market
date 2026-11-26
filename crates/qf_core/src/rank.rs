//! Cross-sectional rank -> normal scores. Mirrors qforecast.swing.features._cs_rank_gauss_py.

use numpy::ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

/// Inverse standard normal CDF, Wichura (1988) algorithm AS241 (PPND16), ~1e-16 relative accuracy.
#[allow(clippy::excessive_precision)] // published AS241 coefficients, kept verbatim
pub fn ndtri(p: f64) -> f64 {
    if p.is_nan() || p <= 0.0 || p >= 1.0 {
        return if p == 0.0 {
            f64::NEG_INFINITY
        } else if p == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q
            * (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
                + 45921.953931549871457)
                * r
                + 13731.693765509461125)
                * r
                + 1971.5909503065514427)
                * r
                + 133.14166789178437745)
                * r
                + 3.387132872796366608)
            / (((((((5226.495278852545925 * r + 28729.085735721942674) * r + 39307.89580009271061) * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0);
    }
    let mut r = if q < 0.0 { p } else { 1.0 - p };
    r = (-r.ln()).sqrt();
    let val = if r <= 5.0 {
        let r = r - 1.6;
        (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
            + 1.27045825245236838258)
            * r
            + 3.64784832476320460504)
            * r
            + 5.7694972214606914055)
            * r
            + 4.6303378461565452959)
            * r
            + 1.42343711074968357734)
            / (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r
                + 0.14810397642748007459)
                * r
                + 0.68976733498510000455)
                * r
                + 1.6763848301838038494)
                * r
                + 2.05319162663775882187)
                * r
                + 1.0)
    } else {
        let r = r - 5.0;
        (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
            + 0.026532189526576123093)
            * r
            + 0.29656057182850489123)
            * r
            + 1.7848265399172913358)
            * r
            + 5.4637849111641143699)
            * r
            + 6.6579046435011037772)
            / (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
                + 7.868691311456132591e-4)
                * r
                + 0.0148753612908506148525)
                * r
                + 0.13692988092273580531)
                * r
                + 0.59983220655588793769)
                * r
                + 1.0)
    };
    if q < 0.0 {
        -val
    } else {
        val
    }
}

fn row(x: &[f64], m: &[bool], min_names: usize, ties: bool, out: &mut [f64]) {
    let idx: Vec<usize> = (0..x.len()).filter(|&j| m[j]).collect();
    if idx.is_empty() {
        return;
    }
    for &j in &idx {
        out[j] = 0.0; // members: neutral unless ranked
    }
    let ok: Vec<usize> = idx.iter().copied().filter(|&j| x[j].is_finite()).collect();
    let n = ok.len();
    if n < min_names {
        return;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[ok[a]].total_cmp(&x[ok[b]])); // stable, like numpy mergesort
    let nf = n as f64;
    if ties {
        // average rank (1-based) of each group of equal values, like scipy.stats.rankdata
        let mut i = 0;
        while i < n {
            let mut e = i;
            while e + 1 < n && x[ok[order[e + 1]]] == x[ok[order[i]]] {
                e += 1;
            }
            let avg = (i + e) as f64 / 2.0 + 1.0;
            for &o in &order[i..=e] {
                out[ok[o]] = ndtri((avg - 0.5) / nf);
            }
            i = e + 1;
        }
    } else {
        for (pos, &o) in order.iter().enumerate() {
            out[ok[o]] = ndtri((pos as f64 + 0.5) / nf);
        }
    }
}

pub fn cs_rank_gauss(x: ArrayView2<'_, f64>, mask: ArrayView2<'_, bool>, min_names: usize, ties: bool) -> Array2<f64> {
    let (t, n) = x.dim();
    let mut out = vec![f64::NAN; t * n];
    if n > 0 {
        out.par_chunks_mut(n).enumerate().for_each(|(k, o)| {
            let (xv, mv) = (x.index_axis(Axis(0), k), mask.index_axis(Axis(0), k));
            // C-contiguous rows are read in place; other layouts fall back to a row copy
            let (xb, mb);
            let xr: &[f64] = match xv.as_slice() {
                Some(s) => s,
                None => {
                    xb = xv.to_vec();
                    &xb
                }
            };
            let mr: &[bool] = match mv.as_slice() {
                Some(s) => s,
                None => {
                    mb = mv.to_vec();
                    &mb
                }
            };
            row(xr, mr, min_names, ties, o);
        });
    }
    Array2::from_shape_vec((t, n), out).expect("shape")
}
