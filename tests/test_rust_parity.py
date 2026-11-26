# ruff: noqa: E501
"""The compiled hot paths (crates/qf_core) must give the same numbers as the Python references.
Skipped when the optional qforecast_core wheel is not installed (the CI "rust" job builds and installs it)."""

import numpy as np
import pytest

rs = pytest.importorskip("qforecast_core")


@pytest.mark.parametrize(("T", "N"), [(1, 1), (40, 3), (500, 41)])
@pytest.mark.parametrize("w", [1, 7, 28, 90])
def test_rolling_mean_bit_identical(T, N, w):
    from qforecast.swing.features import _rolling_mean_py

    rng = np.random.default_rng(T * 1000 + N + w)
    x = rng.normal(0, 1, (T, N))
    x[rng.random(x.shape) < 0.15] = np.nan
    x[: T // 3, -1] = np.nan  # a coin listed later
    for min_frac in (0.5, 0.8, 1.0):
        a = rs.rolling_mean(x, w, min_frac)
        b = _rolling_mean_py(x, w, min_frac)
        assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize(("T", "N"), [(1, 1), (40, 3), (500, 41)])
@pytest.mark.parametrize("w", [1, 7, 28, 90])
def test_rolling_extreme_bit_identical(T, N, w):
    from qforecast.swing.signals import _roll_py

    rng = np.random.default_rng(T + N * 7 + w)
    x = rng.normal(0, 1, (T, N))
    x[rng.random(x.shape) < 0.2] = np.nan
    x[: min(T, 2 * w), 0] = np.nan  # an all-NaN window
    assert np.array_equal(rs.rolling_extreme(x, w, True), _roll_py(np.nanmax, x, w), equal_nan=True)
    assert np.array_equal(rs.rolling_extreme(x, w, False), _roll_py(np.nanmin, x, w), equal_nan=True)


def test_rank_gauss_matches_python():
    from qforecast.swing.features import _cs_rank_gauss_py

    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 25))
    x[:, 3] = x[:, 4]  # ties keep the column order
    x[rng.random(x.shape) < 0.1] = np.nan
    mask = rng.random(x.shape) > 0.2
    mask[:5] = False
    np.testing.assert_allclose(rs.cs_rank_gauss(x, mask, 5, False), _cs_rank_gauss_py(x, mask, 5),
                               rtol=1e-12, atol=1e-14, equal_nan=True)


def test_ndtri_matches_scipy():
    from scipy.special import ndtri

    p = np.concatenate([np.linspace(1e-12, 1e-3, 50), np.linspace(1e-3, 1 - 1e-3, 500), 1 - np.linspace(1e-12, 1e-3, 50)])
    np.testing.assert_allclose([rs.ndtri(v) for v in p], ndtri(p), rtol=1e-13, atol=1e-13)


def test_strategy_context_is_the_same_with_and_without_rust(monkeypatch):
    """End to end: the signals, ranks, vol and beta the strategy uses do not depend on the backend."""
    from qforecast.swing.strategy import Context
    from tests.synth import make_cube

    cube = make_cube(T=260, seed=11)
    fast = Context.build(cube)
    monkeypatch.setenv("QF_NO_RUST", "1")
    ref = Context.build(cube)
    for name in ("Z", "vol", "beta", "f7"):
        np.testing.assert_allclose(getattr(fast, name), getattr(ref, name), rtol=1e-12, atol=1e-14, equal_nan=True, err_msg=name)
