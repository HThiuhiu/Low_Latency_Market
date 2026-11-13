# ruff: noqa: E501
"""Daily cube: a trading symbol whose REST call fails must stop the build, never fall back silently to the
monthly archives (they have no current month, so the coin would drop out of that day's ranking)."""

import asyncio
import time

import numpy as np
import orjson
import pytest

from qforecast.swing import data as D
from qforecast.swing.cube import BENCH, Cube, alive_member, select_universe

DAY = D.DAY


class Store:
    def __init__(self, root, rest_ok=True):
        self.root, self.rest_ok, self.rest_calls = root, rest_ok, 0

    async def _get(self, url, params=None):
        self.rest_calls += 1
        if not self.rest_ok:
            return None  # what PerpStore._get returns on 400 / 403 / 404
        today = (int(time.time() * 1000) // DAY) * DAY
        t = max(params["startTime"], today - 5 * DAY)
        return orjson.dumps([[t + i * DAY, 1, 1, 1, 1, 1, t + i * DAY + DAY - 1, 1, 1, 1, 1] for i in range((today - t) // DAY + 1)])

    async def _zip_csv(self, path, ncols):  # archives stop at last month: no current-month bars
        return np.array([[0.0] * 11])


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def nosleep(_):
        return None

    monkeypatch.setattr(D.asyncio, "sleep", nosleep)


def _now_end():
    return (int(time.time() * 1000) // DAY) * DAY


def test_trading_symbol_rest_failure_raises_and_caches_nothing(tmp_path):
    st = Store(tmp_path, rest_ok=False)
    with pytest.raises(D.IncompleteData):
        asyncio.run(D.daily_klines(st, "XYZUSDT", 0, _now_end(), listed_since=0))
    assert st.rest_calls == 3  # retried
    assert not (tmp_path / "d1k" / "XYZUSDT.npy").exists()


def test_delisted_symbol_uses_archives(tmp_path):
    a = asyncio.run(D.daily_klines(Store(tmp_path, rest_ok=False), "OLDUSDT", 0, _now_end(), listed_since=None))
    assert len(a) == 1


def test_trading_symbol_has_yesterday_and_drops_today(tmp_path):
    end = _now_end()
    a = asyncio.run(D.daily_klines(Store(tmp_path), "BTCUSDT", end - 10 * DAY, end, listed_since=0))
    assert int(a[:, 0].max()) == end - DAY  # yesterday's final bar kept, today's unfinished bar dropped
    assert (tmp_path / "d1k" / "BTCUSDT.npy").exists()


def test_stale_cache_without_yesterday_is_refetched(tmp_path):
    end = _now_end()
    p = tmp_path / "d1k" / "BTCUSDT.npy"
    p.parent.mkdir(parents=True)
    np.save(p, np.array([[float(end - 3 * DAY)] + [1.0] * 10]))  # written today but missing yesterday
    st = Store(tmp_path)
    a = asyncio.run(D.daily_klines(st, "BTCUSDT", end - 10 * DAY, end, listed_since=0))
    assert st.rest_calls == 1 and int(a[:, 0].max()) == end - DAY


def test_funding_rows_paid_and_last():
    t = np.arange(0, 5 * DAY, DAY)
    fr = np.array([[8 * 3600e3, 0.001], [16 * 3600e3, 0.002], [DAY, 0.003], [3 * DAY + 1, -0.004]])
    paid, last = D.funding_rows(fr, t)
    # events in (t_k, t_k + 1d] are paid over row k; an event exactly at t_1 belongs to row 0
    assert np.allclose(paid, [0.006, 0.0, 0.0, -0.004, 0.0])
    assert np.isnan(last[0]) or last[0] == 0.003
    assert last[1] == 0.003 and last[3] == -0.004 and last[4] == -0.004


def test_universe_ranks_only_with_past_rows():
    m0 = 1_700_000_000_000 // DAY * DAY
    days = np.arange(m0 - 120 * DAY, m0 + 40 * DAY, DAY, dtype=float)
    a_vol = np.where(days < m0, 100.0, 1.0)  # A big before the month, tiny after
    b_vol = np.where(days < m0, 10.0, 1e9)  # B huge only AFTER the ranking day
    d1 = {"AUSDT": np.c_[days, a_vol, np.ones_like(days)], "BUSDT": np.c_[days, b_vol, np.ones_like(days)],
          BENCH: np.c_[days, np.full_like(days, 1e12), np.ones_like(days)]}
    uni = select_universe(d1, m0, m0 + DAY, top_n=1)
    first = min(uni)
    assert uni[first] == ["AUSDT"]  # the future volume of B does not leak into the ranking; BTC never ranked


def test_cube_roundtrip_and_alive_member(tmp_path):
    T, N = 6, 3
    z = np.ones((T, N))
    qv = z.copy()
    qv[2:, 2] = 0.0  # third coin stops trading at row 2
    cube = Cube(t=np.arange(T) * DAY, syms=[BENCH, "AUSDT", "BUSDT"], o=z, h=z, lo=z, c=z, qv=qv, tbq=z * 0.5,
                funding_paid=z * 0, funding_last=z * 0, member=np.ones((T, N), bool))
    cube.save(tmp_path / "c.npz")
    back = Cube.load(tmp_path / "c.npz")
    assert back.syms == cube.syms and back.bench == 0 and np.array_equal(back.qv, qv)
    alive = alive_member(back, 1)
    assert alive[1, 2] and not alive[3, 2]
    r = back.ret_next()
    assert np.all(r[:-2] == 0.0) and np.isnan(r[-1]).all()
