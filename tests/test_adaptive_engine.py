# ruff: noqa: E501
"""The batched torch engine gives the same daily P&L as the reference book, and the fast path keeps the
point-in-time and reproducibility guarantees of the daily weights."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from qforecast.adaptive import search as S
from qforecast.adaptive import weights as AW
from qforecast.adaptive.engine import TorchBook
from qforecast.swing.strategy import Context, SwingSpec
from tests.synth import cut, make_cube

C5 = SwingSpec.load(Path(__file__).resolve().parents[1] / "artifacts" / "models" / "c5.json")


def _day(ms):
    return datetime.fromtimestamp(int(ms) / 1000, UTC).strftime("%Y-%m-%d")


@pytest.fixture(scope="module")
def ctx():
    return Context.build(make_cube(T=300, seed=5))


def test_engine_matches_the_reference_book(ctx):
    kA, kS = 130, 290
    rng = np.random.default_rng(0)
    W = np.vstack([AW.base_weights(C5), S.nrm(rng.normal(size=(3, 6)))])
    fast = TorchBook(ctx, C5, kA, kS, dt=torch.float64).evaluate(W).numpy()
    ref = S.NumpyBook(ctx, C5, kA, kS).evaluate(W).numpy()
    assert np.abs(fast - ref).max() < 1e-12
    sub = TorchBook(ctx, C5, kA, kS, dt=torch.float64).evaluate(W, 200, 260).numpy()  # a window inside the book
    assert np.abs(sub - S.NumpyBook(ctx, C5, kA, kS).evaluate(W, 200, 260).numpy()).max() < 1e-12


def test_fast_daily_weights_are_point_in_time_and_reproducible(ctx):
    k = 280
    day = _day(ctx.cube.t[k])
    a, _ = AW.compute(ctx, C5, day, day, n_pool=60)
    b, _ = AW.compute(Context.build(cut(make_cube(T=300, seed=5), k)), C5, day, day, n_pool=60)
    c, _ = AW.compute(ctx, C5, day, day, n_pool=60)
    assert np.allclose(np.abs(a[day]).sum(), 1.0)
    assert np.allclose(a[day], b[day], atol=1e-6)
    assert np.array_equal(a[day], c[day])
