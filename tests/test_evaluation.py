import numpy as np
import pytest

from qforecast import blockcv, evaluation


def test_metrics_on_known_series():
    d = np.array([0.01, -0.02, 0.03, 0.0, -0.01])
    m = evaluation.metrics(d)
    assert m["maxdd_pct"] == pytest.approx(-2.0)
    assert m["usd_per_10k"] == pytest.approx(100.0)
    assert evaluation.cvar(d, 0.2) == pytest.approx(-0.02)


def test_bootstrap_finds_a_real_edge_and_not_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, 2000)
    better = base + 0.002  # same noise, higher mean: a clear edge
    same = base.copy()
    out = evaluation.boot_sharpe_diff({"better": better, "same": same}, base, block=10, draws=400, seed=0)
    assert out["better"]["p_one_sided"] < 0.01
    assert out["same"]["p_one_sided"] > 0.05


def test_stationary_bootstrap_indices_in_range():
    idx = evaluation.stationary_idx(50, 5.0, 20, np.random.default_rng(0))
    assert idx.shape == (20, 50) and idx.min() >= 0 and idx.max() < 50


def test_blocks_purge_every_boundary_but_the_first():
    b = blockcv.blocks(0, 600)
    assert len(b) == 6 and b[0][0] == 0 and b[1][0] == 100 + 7


def test_block_report_needs_five_positive_blocks():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 1200)
    steady = base + 0.001
    lucky = base.copy()
    lucky[:200] += 0.02  # all the gain in one block: positive on average, not stable
    assert blockcv.block_report(base, steady, 0, 1200)["pass"]
    rep = blockcv.block_report(base, lucky, 0, 1200)
    assert rep["mean_diff_ann_pct"] > 0 and not rep["pass"]


def test_cpcv_reports_every_group():
    def fit_and_run(tr, te):
        return np.zeros(len(te)), np.full(len(te), 0.001)

    out = blockcv.cpcv(600, fit_and_run)
    assert out["positive_groups"] == "6/6" and len(out["path_gain_ann_pct"]) == 5


def test_newey_west_t_shrinks_with_autocorrelation():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 3000)
    ar = np.zeros(3000)
    for i in range(1, 3000):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    assert abs(blockcv.nw_t(ar + 0.05)) < abs((ar + 0.05).mean() / (ar.std() / np.sqrt(3000)))
