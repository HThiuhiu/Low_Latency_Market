# ruff: noqa: E501
"""C5 leg rules: 5 long / 8 short, skip-and-replace of shorts with negative funding, a thin leg keeps its cap."""

from pathlib import Path

import numpy as np

from qforecast.swing.book import fixed_cap_weights, leg_weights, tranche_targets
from qforecast.swing.strategy import Context, SwingSpec, book_targets, score, staggered_target_row
from tests.synth import cut, make_cube

C5 = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "c5.json"


def _setup(n=40, seed=0):
    rng = np.random.default_rng(seed)
    score_ = rng.normal(size=(3, n))
    mask = np.ones((3, n), bool)
    vol = rng.uniform(0.02, 0.08, size=(3, n))
    return score_, mask, vol, np.array([1])


def test_defaults_unchanged():
    s, mask, vol, rb = _setup()
    a = tranche_targets(s, mask, vol, rb, "LS", n=6)
    b = tranche_targets(s, mask, vol, rb, "LS", n=6, n_short=6, skip_short=None, fixed_cap=False)
    c = tranche_targets(s, mask, vol, rb, "LS", n=6, fixed_cap=True)  # >= 4 names: same weights
    assert np.array_equal(a, b) and np.allclose(a, c, atol=1e-15)
    assert (a[1] > 0).sum() == 6 and (a[1] < 0).sum() == 6


def test_asymmetric_legs():
    s, mask, vol, rb = _setup()
    w = tranche_targets(s, mask, vol, rb, "LS", n=5, n_short=8)[1]
    order = np.argsort(s[1], kind="mergesort")
    assert set(np.flatnonzero(w > 0)) == set(order[-5:]) and set(np.flatnonzero(w < 0)) == set(order[:8])
    assert abs(w[w > 0].sum() - 0.5) < 1e-12 and abs(w[w < 0].sum() + 0.5) < 1e-12
    assert np.abs(w).max() <= 0.125 + 1e-12


def test_skip_and_replace():
    s, mask, vol, rb = _setup()
    order = np.argsort(s[1], kind="mergesort")
    skip = np.zeros_like(mask)
    skip[1, order[[0, 3]]] = True  # the worst-ranked and the 4th worst may not be shorted
    w = tranche_targets(s, mask, vol, rb, "LS", n=5, n_short=8, skip_short=skip)[1]
    assert set(np.flatnonzero(w < 0)) == set(order[[1, 2, 4, 5, 6, 7, 8, 9]])  # next-ranked names take the two places
    assert set(np.flatnonzero(w > 0)) == set(order[-5:])


def test_thin_leg_keeps_cap():
    s, mask, vol, rb = _setup()
    order = np.argsort(s[1], kind="mergesort")
    skip = np.zeros_like(mask)
    skip[1, order[:16]] = True
    skip[1, order[[2, 9]]] = False  # only two eligible shorts in the search window
    w = tranche_targets(s, mask, vol, rb, "LS", n=5, n_short=8, skip_short=skip, fixed_cap=True)[1]
    assert set(np.flatnonzero(w < 0)) == set(order[[2, 9]])
    assert np.allclose(w[w < 0], -0.125)  # 25 % of a leg each, not 50 %
    relaxed = tranche_targets(s, mask, vol, rb, "LS", n=5, n_short=8, skip_short=skip)[1]
    assert abs(relaxed[relaxed < 0].sum() + 0.5) < 1e-12  # without the fixed cap the leg would concentrate


def test_fixed_cap_weights():
    assert np.allclose(fixed_cap_weights(np.array([1.0, 2.0, 3.0])), 0.25)
    y = np.arange(1.0, 9.0)
    assert np.allclose(fixed_cap_weights(y), leg_weights(y))


def test_c5_spec_loads():
    spec = SwingSpec.load(C5)
    assert (spec.n, spec.n_short, spec.hedge, spec.short_stop) == (5, 8, "beta", 0.4)
    assert spec.skip_neg_funding and spec.thin_leg_btc and len(spec.weights) == 6


def test_c5_never_shorts_negative_funding_and_hedges_beta():
    spec = SwingSpec.load(C5)
    ctx = Context.build(make_cube(seed=3))
    k = 260
    w = book_targets(ctx, spec, np.array([k]))[k]
    shorts = np.flatnonzero(w < 0)
    shorts = shorts[shorts != ctx.cube.bench]
    assert len(shorts) and (ctx.f7[k, shorts] >= 0).all()
    assert abs(np.clip(w, 0, None) @ ctx.beta[k] + np.clip(w, None, 0) @ ctx.beta[k]) < 1e-9


def test_c5_targets_are_point_in_time():
    spec = SwingSpec.load(C5)
    cube = make_cube(seed=4)
    for k in (200, 250):
        a, b = Context.build(cube), Context.build(cut(cube, k))
        assert np.allclose(np.nan_to_num(score(a, spec)[k], nan=-9), np.nan_to_num(score(b, spec)[k], nan=-9))
        wa = staggered_target_row(a, spec, k)
        assert np.abs(wa).sum() > 0.5 and np.allclose(wa, staggered_target_row(b, spec, k), atol=1e-12)
