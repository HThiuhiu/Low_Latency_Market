import numpy as np
import pytest

from qforecast.swing.features import Trailing, cs_rank_gauss, rolling_beta, rolling_mean
from qforecast.swing.signals import SPECS, compute
from tests.synth import cut, make_cube


def test_rolling_mean_matches_naive_and_needs_enough_values():
    x = np.arange(10.0)[:, None]
    x[3] = np.nan
    m = rolling_mean(x, 4, min_frac=0.75)
    assert m[5, 0] == pytest.approx(np.nanmean(x[2:6]))
    assert np.isnan(rolling_mean(x, 4, min_frac=1.0)[5, 0])  # one value missing in the window


def test_rolling_beta_recovers_a_known_beta():
    rng = np.random.default_rng(0)
    rb = rng.normal(0, 0.02, 400)
    r = np.c_[0.5 * rb, 2.0 * rb + rng.normal(0, 1e-4, 400)]
    b = rolling_beta(r, rb, 90)
    assert b[-1, 0] == pytest.approx(0.5, abs=1e-9) and b[-1, 1] == pytest.approx(2.0, abs=0.01)


def test_rank_gauss_is_symmetric_and_masks():
    x = np.array([[3.0, 1.0, 2.0, np.nan, 5.0, 4.0, 9.0]])
    mask = np.array([[True, True, True, True, True, True, False]])
    z = cs_rank_gauss(x, mask)
    assert np.isnan(z[0, 6])  # not a member
    assert z[0, 3] == 0.0  # member without a value: neutral
    assert z[0, 4] > z[0, 5] > z[0, 0] > z[0, 2] > z[0, 1]
    assert z[0, :6].sum() == pytest.approx(0.0, abs=1e-12)


def test_every_signal_is_point_in_time():
    cube = make_cube(T=260)
    full = compute(cube, Trailing(cube), cube.member)
    k = 200
    past = cut(cube, k)
    part = compute(past, Trailing(past), past.member)
    assert set(full) == {s.name for s in SPECS}
    for name in full:
        a, b = full[name][: k + 1], part[name]
        assert np.array_equal(np.isnan(a), np.isnan(b)), name
        assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=1e-12), name
