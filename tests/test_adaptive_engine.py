# ruff: noqa: E501
"""The batched torch engine gives the same daily P&L as the reference book."""

from pathlib import Path

import numpy as np
import pytest
import torch

from qforecast.adaptive import search as S
from qforecast.adaptive import weights as AW
from qforecast.adaptive.engine import TorchBook
from qforecast.swing.strategy import Context, SwingSpec
from tests.synth import make_cube

C5 = SwingSpec.load(Path(__file__).resolve().parents[1] / "artifacts" / "models" / "c5.json")


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
