import numpy as np

from qforecast.trainer.dataset import chronological_split, eligible_indices, fit_normaliser, make_targets

IV = 60_000


def test_targets_are_future_log_returns():
    close = np.array([100.0, 101.0, 102.0, 99.0, 100.0])
    y = make_targets(close, [1, 2])
    assert np.isclose(y[0, 0], np.log(101 / 100))
    assert np.isclose(y[1, 1], np.log(99 / 101))
    assert np.isnan(y[-1]).all() and np.isnan(y[-2, 1])


def test_eligible_indices_skip_gaps_and_warmup():
    n, L, H = 400, 32, 10
    t = np.arange(n, dtype=np.int64) * IV
    t[200:] += IV  # one missing bar between 199 and 200
    valid = np.ones(n, bool)
    valid[:50] = False
    idx = eligible_indices(t, valid, IV, L, H)
    assert idx.min() == 50 + L - 1
    assert idx.max() == n - 1 - H
    for i in idx:  # no window+label span crosses the gap
        lo, hi = i - L + 1, i + H
        assert not (lo < 200 <= hi)
    assert 189 in idx and 190 not in idx and 231 in idx and 230 not in idx


def test_split_is_chronological_with_embargo():
    idx = np.arange(100, 10_000)
    H = 60
    sp = chronological_split(idx, embargo=H)
    assert sp.train.max() + H < sp.val.min()
    assert sp.val.max() + H < sp.test.min()
    assert len(np.intersect1d(sp.train, sp.test)) == 0


def test_normaliser_uses_training_rows_only():
    feats = np.zeros((1_000, 2))
    feats[600:] = 1e6  # "future" regime must not leak into the stats
    mean, std = fit_normaliser(feats, np.arange(100, 500), lookback=64)
    assert np.allclose(mean, 0) and np.all(std < 1e-6)
