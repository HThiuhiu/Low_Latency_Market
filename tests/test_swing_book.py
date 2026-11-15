# ruff: noqa: E501
import numpy as np
import pytest

from qforecast.swing import book as B
from qforecast.swing.backtest import run as backtest
from qforecast.swing.strategy import Context, SwingSpec, run_book, run_staggered, score, staggered_target_row
from tests.synth import cut, make_cube

SPEC = SwingSpec(mode="LS", H=7, every=7, n=5, hedge="beta", short_stop=0.4,
                 weights={"vol_28d": -1, "beta_90d": -1, "max_7d": -1, "funding_7d": -3, "near_high_90d": 3, "mom_28d": 3})


def test_leg_weights_cap_and_sum():
    w = B.leg_weights(np.array([10.0, 1, 1, 1, 1, 1]), cap=0.25)
    assert abs(w.sum() - 1) < 1e-12 and w.max() <= 0.25 + 1e-9


def test_tranches_ls_dollar_neutral_and_overlap_mean():
    T, N = 30, 20
    rng = np.random.default_rng(0)
    score_ = rng.standard_normal((T, N))
    mask = np.ones((T, N), bool)
    vol = np.full((T, N), 0.05)
    rebal = np.arange(0, T, 7)
    tr = B.tranche_targets(score_, mask, vol, rebal, "LS", n=4)
    for k in rebal:
        assert abs(tr[k].sum()) < 1e-12 and abs(np.abs(tr[k]).sum() - 1) < 1e-12
    w = B.overlap(tr, rebal, 14)
    assert np.allclose(w[rebal[2]], (tr[rebal[1]] + tr[rebal[2]]) / 2)


def test_simulate_trades_only_on_rebalance_and_drifts():
    T, N = 10, 2
    tgt = np.zeros((T, N))
    tgt[0] = [1.0, 0.0]
    ret = np.zeros((T, N))
    ret[:, 0] = 0.01
    res = B.simulate(tgt, np.array([0]), np.arange(T - 1), ret, np.zeros((T, N)), 1e-3)
    assert res.turnover[0] == 1.0 and np.all(res.turnover[1:T - 1] == 0)
    assert res.pnl[1] > 0.01  # the long drifted up, so it earns on a larger weight


def test_short_stop_closes_squeezed_short():
    T, N = 6, 2
    tgt = np.zeros((T, N))
    tgt[0] = [0.5, -0.5]
    close = np.ones((T, N))
    close[2:, 1] = 1.6  # the short rallies 60 %
    ret = np.zeros((T, N))
    ret[1, 1] = 0.6
    res = B.simulate(tgt, np.array([0]), np.arange(T - 1), ret, np.zeros((T, N)), 0.0, short_stop=0.4, close=close)
    assert res.turnover[2] > 0 and res.exposure[3] == 0.5  # bought back at row 2, only the long remains


def test_funding_is_paid_by_longs_and_earned_by_shorts():
    T, N = 4, 2
    tgt = np.zeros((T, N))
    tgt[0] = [0.5, -0.5]
    fund = np.zeros((T, N))
    fund[0] = [0.001, 0.001]
    res = B.simulate(tgt, np.array([0]), np.arange(T - 1), np.zeros((T, N)), fund, 0.0)
    assert res.funding[0] == pytest.approx(0.0)
    fund[1] = [0.002, 0.0]
    res = B.simulate(tgt, np.array([0]), np.arange(T - 1), np.zeros((T, N)), fund, 0.0)
    assert res.funding[1] == pytest.approx(-0.001)


def test_beta_hedge_balances_leg_betas_and_keeps_gross():
    ctx = Context.build(make_cube())
    k = 250
    w = staggered_target_row(ctx, SPEC, k)
    assert abs(np.abs(w).sum() - 1.0) < 1e-9
    tr = B.tranche_targets(score(ctx, SPEC), ctx.mask, ctx.vol, np.array([k]), "LS", n=5)
    from qforecast.swing.strategy import book_targets

    h = book_targets(ctx, SPEC, np.array([k]))[k]
    assert abs(np.clip(h, 0, None) @ ctx.beta[k] + np.clip(h, None, 0) @ ctx.beta[k]) < 1e-9
    assert abs(np.abs(h).sum() - np.abs(tr[k]).sum()) < 1e-9


def test_targets_are_point_in_time():
    cube = make_cube()
    k = 240
    a, b = Context.build(cube), Context.build(cut(cube, k))
    assert np.allclose(np.nan_to_num(score(a, SPEC)[k], nan=-9), np.nan_to_num(score(b, SPEC)[k], nan=-9))
    assert np.allclose(staggered_target_row(a, SPEC, k), staggered_target_row(b, SPEC, k), atol=1e-12)


def test_staggered_is_the_mean_of_the_weekday_sub_books():
    ctx = Context.build(make_cube())
    k0, k1 = 130, 300
    st = run_staggered(ctx, SPEC, k0, k1, 7e-4)
    subs = [run_book(ctx, SPEC, k0, k1, 7e-4, phase=ph).pnl for ph in range(7)]
    assert np.allclose(st.pnl[k0:k1], np.mean(subs, axis=0)[k0:k1])
    assert np.isfinite(st.pnl[k0:k1]).all()


def test_exchange_style_backtest_runs_and_charges_costs():
    ctx = Context.build(make_cube())
    res = backtest(ctx, SPEC, capital=1000.0, gross=1.0)
    s = res["summary"]
    assert s["fees_usdt"] > 0 and s["fills"] > 50 and res["period"]["days"] > 150
    assert abs(s["end_equity"] - res["equity"][-1]["equity"]) < 1e-9
    assert 0.8 < s["avg_gross"] < 1.2
