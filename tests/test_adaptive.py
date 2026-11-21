# ruff: noqa: E501
"""Adaptive weights: the batched engine equals the numpy book, window scores are trailing only, and the daily
weights of row t do not change when every row after t is removed from the cube."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from qforecast.adaptive import engine as E
from qforecast.adaptive import search as S
from qforecast.adaptive import weights as AW
from qforecast.swing.strategy import Context, SwingSpec, run_staggered, staggered_target_row
from tests.synth import cut, make_cube

C5 = SwingSpec.load(Path(__file__).resolve().parents[1] / "artifacts" / "models" / "c5.json")


def _day(ms):
    return datetime.fromtimestamp(int(ms) / 1000, UTC).strftime("%Y-%m-%d")


@pytest.fixture(scope="module")
def ctx():
    return Context.build(make_cube(T=300, seed=5))


def test_engine_matches_the_numpy_staggered_book(ctx):
    kA, kS = 130, 290
    p = E.Prep(ctx, C5, kA, kS)
    rng = np.random.default_rng(0)
    W = np.vstack([AW.base_weights(C5), S.nrm(rng.normal(size=(2, 6)))])
    P, _ = E.run_batch(p, W, torch.float64, "cpu")
    for i, w in enumerate(W):
        spec = SwingSpec(**{**C5.to_dict(), "weights": dict(zip(E.SIGNALS, w, strict=True))})
        ref = run_staggered(ctx, spec, kA, kS, 7e-4).pnl[kA:kS]
        assert np.abs(P[i].numpy() - ref).max() < 1e-12


def test_window_scores_are_trailing_sums(ctx):
    rng = np.random.default_rng(1)
    Pg = torch.as_tensor(rng.normal(0, 0.01, (4, 80)))
    L = 10
    ret, sr = S.window_scores(Pg, L, "ret"), S.window_scores(Pg, L, "sr")
    x = Pg.numpy()
    t = L + 7 + 5  # column t scores the window t-L .. t-1
    win = x[:, t - L:t]
    assert np.allclose(ret[:, t - L - 7].numpy(), win.sum(1))
    assert np.allclose(sr[:, t - L - 7].numpy(), win.mean(1) / win.std(1))


def test_pool_is_normalised_and_starts_with_the_base():
    base = AW.base_weights(C5)
    W = S.pool(base, n=50)
    assert W.shape == (2 + 12 + 100, 6) and np.allclose(np.abs(W).sum(1), 1.0)
    assert np.allclose(W[0], base) and np.array_equal(np.sign(W[14:64]), np.tile(np.sign(base), (50, 1)))


def test_daily_weights_are_point_in_time_and_reproducible(ctx):
    k = 280
    day = _day(ctx.cube.t[k])
    a, _ = AW.compute(ctx, C5, day, day, n_pool=60)
    b, _ = AW.compute(Context.build(cut(make_cube(T=300, seed=5), k)), C5, day, day, n_pool=60)
    c, _ = AW.compute(ctx, C5, day, day, n_pool=60)
    assert np.allclose(np.abs(a[day]).sum(), 1.0)
    assert np.allclose(a[day], b[day], atol=1e-6)  # rows after k never move the weights of row k
    assert np.array_equal(a[day], c[day])  # same seed rule, same answer


def test_dyn_score_uses_each_sub_books_own_day(ctx, tmp_path):
    k = 280
    rng = np.random.default_rng(2)
    by_day = {_day(ctx.cube.t[q]): dict(zip(E.SIGNALS, S.nrm(rng.normal(size=6))[0], strict=True)) for q in range(k - 6, k + 1)}
    s, w_k = AW.dyn_score(ctx, C5, k, by_day)
    assert w_k == by_day[_day(ctx.cube.t[k])]
    for q in range(k - 6, k + 1):
        own = SwingSpec(**{**C5.to_dict(), "weights": by_day[_day(ctx.cube.t[q])]})
        from qforecast.swing.strategy import score

        assert np.allclose(np.nan_to_num(s[q], nan=-9), np.nan_to_num(score(ctx, own)[q], nan=-9))
    w = staggered_target_row(ctx, C5, k, s=s)  # sub-books with different weights may net out: gross <= 1
    assert np.isfinite(w).all() and 0.1 < np.abs(w).sum() <= 1.0 + 1e-9
    out = AW.save_days({d: np.array(list(v.values())) for d, v in by_day.items()}, {}, "a", "b", tmp_path / "w.json")
    AW.save_days({d: np.zeros(6) for d in by_day}, {}, "a", "b", tmp_path / "w.json")  # a frozen day is never overwritten
    assert AW.load_cache(tmp_path / "w.json")["weights"] == out["weights"]
