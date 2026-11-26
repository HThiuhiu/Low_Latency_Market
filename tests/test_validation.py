import math

import numpy as np

from qforecast import validation as V


def test_purged_walk_forward_gaps_and_order():
    sp = V.purged_walk_forward(n=5000, label_len=24, n_folds=4, fold_len=500, val_len=300, embargo=10)
    assert len(sp) == 4
    for s in sp:
        gap = 24 + 10
        assert s.train.max() + gap < s.val.min()
        assert s.val.max() + gap < s.test.min()
        assert s.test_start == s.test.min()
    assert sp[-1].test.max() == 4999
    tests = np.concatenate([s.test for s in sp])
    assert len(np.unique(tests)) == len(tests)  # folds do not overlap


def test_walk_forward_respects_holdout_end():
    sp = V.purged_walk_forward(n=5000, label_len=5, n_folds=2, fold_len=200, val_len=100, end=4000)
    assert max(s.test.max() for s in sp) == 3999


def test_cpcv_paths_cover_every_group_once():
    splits, paths = V.cpcv_splits(n=600, n_groups=6, k_test=2, label_len=5, embargo=3)
    assert len(splits) == 15 and len(paths) == 5
    for p in paths:
        assert sorted(g for _, g in p) == list(range(6))
    for train, test, combo in splits:
        assert not set(train) & set(test)
        edges = np.linspace(0, 600, 7).astype(int)
        for g in combo:  # purge before and embargo after each test group
            a, b = edges[g], edges[g + 1]
            assert not set(range(max(0, a - 5), a)) & set(train)
            assert not set(range(b, min(600, b + 3))) & set(train)


def test_cross_sectional_ic_perfect_and_ties():
    s = np.tile(np.arange(10.0), (3, 1))
    assert np.allclose(V.cross_sectional_ic(s, s * 2 + 1), 1.0)
    assert np.allclose(V.cross_sectional_ic(s, -s), -1.0)
    tie = np.array([[1, 1, 2, 3, 4, 5.0]])
    assert V.cross_sectional_ic(tie, tie)[0] > 0.99
    few = np.full((1, 10), np.nan)
    few[0, :3] = [1, 2, 3]
    assert np.isnan(V.cross_sectional_ic(few, few)[0])


def test_ic_stats_uses_non_overlapping_periods():
    # overlapping 24-period labels: IC is a 24-period moving average of independent noise
    e = np.random.default_rng(0).normal(0, 0.2, 24 * 400 + 23)
    ic = 0.02 + np.convolve(e, np.ones(24) / 24, mode="valid")
    a, b = V.ic_stats(ic, step=1), V.ic_stats(ic, step=24)
    assert b["n"] == 400 and a["n"] == len(ic)
    assert abs(a["ic_t"]) > 2.5 * abs(b["ic_t"])  # naive t inflated by ~sqrt(24)


def test_psr_and_dsr_penalise_more_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 5000)
    assert V.psr(r) > 0.9
    d1, d100 = V.dsr(r, 1), V.dsr(r, 100)
    assert d1["sr0"] == 0.0 and d100["sr0"] > 0
    assert d100["dsr"] < d1["dsr"]
    noise = rng.normal(0, 0.01, 5000)
    assert V.dsr(noise, 50)["dsr"] < 0.5


def test_expected_max_sharpe_grows_with_trials():
    v = 1 / 1000
    assert V.expected_max_sharpe(10, v) < V.expected_max_sharpe(100, v) < V.expected_max_sharpe(1000, v)


def test_pbo_noise_vs_real_edge():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, (2000, 20))
    p_noise = V.pbo_cscv(noise, n_blocks=10)["pbo"]
    assert 0.2 < p_noise < 0.8
    edge = noise.copy()
    edge[:, 0] += 0.3  # one configuration really is better everywhere
    assert V.pbo_cscv(edge, n_blocks=10)["pbo"] < 0.05


def test_periods_for_t():
    n = V.periods_for_t(0.03, 10)
    assert math.isclose(n, (3 / 3 / 0.03) ** 2, rel_tol=1e-9)


def test_registry_counts_distinct_configs(tmp_path):
    reg = V.TrialRegistry(tmp_path / "t.jsonl")
    reg.log("fam", {"a": 1}, {"x": 1})
    reg.log("fam", {"a": 1}, {"x": 2})  # same config re-run: N does not grow
    reg.log("fam", {"a": 2}, {"x": 3})
    reg.log("other", {"a": 9}, {})
    reg.log("fam", {"pick": "x"}, {}, note="HOLDOUT")
    assert reg.count("fam") == 3
    assert len(reg.holdout_opened("fam")) == 1


def test_holdout_index():
    t = np.arange(0, 100 * 86_400_000, 3_600_000)
    k = V.Holdout(10).end_index(t)
    assert t[k] > t[-1] - 10 * 86_400_000 >= t[k - 1]
    assert V.Holdout(0).end_index(t) == len(t)
