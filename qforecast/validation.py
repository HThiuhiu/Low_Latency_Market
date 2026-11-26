"""Validation toolkit: leak-free splits, multiple-testing-aware statistics, trial registry.

Why each piece exists:

* ``purged_walk_forward`` / ``cpcv_splits`` - financial labels overlap in time, so plain k-fold
  leaks the future. Samples whose label window touches the test block are purged from the
  training set and an embargo is added after it (López de Prado, AFML ch. 7).
* ``ic_stats`` - rank IC with a t-stat on NON-overlapping periods only; overlapping H-bar labels
  inflate a naive t-stat by about sqrt(H).
* ``psr`` / ``dsr`` - probabilistic and deflated Sharpe ratio (Bailey & López de Prado 2012,
  2014): is the Sharpe still significant after skew, fat tails and N tries?
* ``pbo_cscv`` - probability of backtest overfitting via combinatorially symmetric CV
  (Bailey, Borwein, López de Prado & Zhu 2017): how often the in-sample winner ends up in the
  bottom half out of sample.
* ``TrialRegistry`` - every configuration ever evaluated is appended to a JSONL file, so N
  (the number of tries that the deflated Sharpe must pay for) is counted, not remembered.
* ``Holdout`` - the last stretch of data is locked; opening it is logged in the registry.

Everything here is plain numpy/scipy and transport-agnostic.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- splits
@dataclass(frozen=True)
class Split:
    train: np.ndarray  # integer indices into the sample axis
    val: np.ndarray
    test: np.ndarray
    test_start: int  # sample-axis position where the test block starts (for labelling)


def purged_walk_forward(n: int, label_len: int, n_folds: int, fold_len: int, val_len: int,
                        embargo: int = 0, min_train: int = 1, end: int | None = None) -> list[Split]:
    """Expanding-window walk-forward on a time axis of ``n`` equally spaced samples.

    Sample i's label uses samples (i, i + label_len]. Layout per fold (oldest fold first)::

        [ train ........ ] gap [ val ] gap [ test ]
        gap = label_len + embargo  -> no train/val label overlaps the next block

    ``end`` (exclusive) lets the caller keep a locked holdout after the last fold.
    """
    end = n if end is None else end
    gap = label_len + embargo
    out = []
    for k in reversed(range(n_folds)):
        t1 = end - k * fold_len
        t0 = t1 - fold_len
        v1 = t0 - gap
        v0 = v1 - val_len
        tr1 = v0 - gap
        if tr1 < min_train or v0 < 0:
            continue
        out.append(Split(np.arange(0, tr1), np.arange(v0, v1), np.arange(t0, t1), t0))
    return out


def cpcv_splits(n: int, n_groups: int, k_test: int, label_len: int,
                embargo: int = 0) -> tuple[list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]],
                                           list[list[tuple[int, int]]]]:
    """Combinatorial purged CV (AFML ch. 12).

    The axis is cut into ``n_groups`` contiguous groups; every combination of ``k_test`` groups
    is a test set, the rest (minus purge + embargo around each test group) is training.

    Returns (splits, paths): splits = [(train_idx, test_idx, test_groups)], and
    paths[j] = [(split_number, group)] giving, for every group, which split's forecast forms
    backtest path j. There are k_test / n_groups * C(n_groups, k_test) paths.
    """
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [(edges[g], edges[g + 1]) for g in range(n_groups)]
    combos = list(itertools.combinations(range(n_groups), k_test))
    splits = []
    for combo in combos:
        test = np.concatenate([np.arange(*groups[g]) for g in combo])
        keep = np.ones(n, dtype=bool)
        for g in combo:
            a, b = groups[g]
            keep[max(0, a - label_len):min(n, b + embargo)] = False  # purge before, embargo after
        splits.append((np.flatnonzero(keep), test, combo))
    n_paths = k_test * len(combos) // n_groups
    seen = {g: 0 for g in range(n_groups)}
    paths: list[list[tuple[int, int]]] = [[] for _ in range(n_paths)]
    for s, combo in enumerate(combos):
        for g in combo:
            paths[seen[g]].append((s, g))
            seen[g] += 1
    for p in paths:
        p.sort(key=lambda x: x[1])
    return splits, paths


# --------------------------------------------------------------------------- IC
def cross_sectional_ic(score: np.ndarray, target: np.ndarray, min_names: int = 5) -> np.ndarray:
    """Per-row Spearman rank correlation of (T, N) arrays, NaN-aware and vectorised. Returns
    (T,) with NaN rows where fewer than ``min_names`` assets have both values."""
    from scipy.stats import rankdata

    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=float)
    ok = np.isfinite(score) & np.isfinite(target)
    if score.ndim == 1:
        score, target, ok = score[None], target[None], ok[None]
    a = rankdata(np.where(ok, score, np.nan), axis=1, nan_policy="omit")
    b = rankdata(np.where(ok, target, np.nan), axis=1, nan_policy="omit")
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.nansum(a, 1) / n
        mb = np.nansum(b, 1) / n
        da, db = a - ma[:, None], b - mb[:, None]
        num = np.nansum(da * db, 1)
        den = np.sqrt(np.nansum(da * da, 1) * np.nansum(db * db, 1))
        out = num / den
    out[(n < min_names) | ~np.isfinite(out)] = np.nan
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 0-based."""
    from scipy.stats import rankdata

    return rankdata(x) - 1.0


def ic_stats(ic: np.ndarray, step: int = 1) -> dict:
    """Mean IC and its t-stat using every ``step``-th period (non-overlapping labels)."""
    x = ic[np.isfinite(ic)]
    sub = ic[::max(1, step)]
    sub = sub[np.isfinite(sub)]
    if len(sub) < 3:
        return {"ic": float(np.mean(x)) if len(x) else float("nan"), "ic_t": float("nan"), "n": len(sub)}
    return {"ic": float(np.mean(x)), "ic_t": float(sub.mean() / (sub.std(ddof=1) / math.sqrt(len(sub)))),
            "ic_ir": float(sub.mean() / sub.std(ddof=1)) if sub.std(ddof=1) > 0 else 0.0,
            "hit": float((sub > 0).mean()), "n": len(sub)}


def periods_for_t(ic: float, n_names: int, t_target: float = 3.0) -> float:
    """Independent periods needed for a mean IC to reach ``t_target`` when the per-period IC
    noise is ~1/sqrt(n_names - 1) (pure-noise standard error of a rank correlation)."""
    sd = 1.0 / math.sqrt(max(n_names - 1, 1))
    return (t_target * sd / ic) ** 2 if ic else float("inf")


# --------------------------------------------------------------------------- Sharpe tests
def _moments(r: np.ndarray) -> tuple[float, float, float, int]:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    sd = r.std(ddof=1) if n > 1 else 0.0
    if n < 3 or sd == 0:
        return 0.0, 0.0, 3.0, n
    z = (r - r.mean()) / sd
    return float(r.mean() / sd), float((z ** 3).mean()), float((z ** 4).mean()), n


def psr(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe ratio: P(true per-period SR > sr_benchmark). Kurtosis is the raw
    (non-excess) fourth moment, 3 for a normal distribution."""
    from scipy.stats import norm

    sr, skew, kurt, n = _moments(returns)
    if n < 3:
        return float("nan")
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(max(denom, 1e-12))))


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """E[max per-period SR] of ``n_trials`` independent skill-less strategies whose estimated
    SRs have variance ``sr_var`` (False Strategy Theorem, Bailey & López de Prado 2014)."""
    from scipy.stats import norm

    if n_trials <= 1:
        return 0.0
    return math.sqrt(sr_var) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                                + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * math.e)))


def dsr(returns: np.ndarray, n_trials: int, trial_srs: np.ndarray | None = None) -> dict:
    """Deflated Sharpe ratio of the selected strategy.

    ``trial_srs``: per-period SRs of every configuration tried (same frequency as ``returns``).
    Without them the variance of a skill-less SR estimate, ~1/T, is used.
    """
    sr, _, _, n = _moments(returns)
    if trial_srs is not None and len(trial_srs) > 1:
        var = float(np.var(np.asarray(trial_srs, dtype=float), ddof=1))
    else:
        var = 1.0 / max(n, 1)
    sr0 = expected_max_sharpe(n_trials, var)
    return {"sr": sr, "sr0": sr0, "dsr": psr(returns, sr0), "n_trials": int(n_trials), "n_obs": n}


def pbo_cscv(perf: np.ndarray, n_blocks: int = 16, metric: str = "sharpe") -> dict:
    """Probability of backtest overfitting.

    ``perf``: (T, N) matrix, row = period, column = configuration (e.g. hourly net PnL of each
    config over the same out-of-sample periods). Rows are cut into ``n_blocks`` contiguous
    blocks; for every half/half combination the in-sample best column is ranked out of sample.
    PBO = share of combinations where that rank is at or below the median (logit <= 0).
    """
    perf = np.asarray(perf, dtype=float)
    T, N = perf.shape
    if N < 2 or T < n_blocks * 2:
        return {"pbo": float("nan"), "n_combos": 0, "logits": []}
    perf = np.nan_to_num(perf)
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    s1 = np.stack([perf[a:b].sum(0) for a, b in zip(edges[:-1], edges[1:], strict=True)])
    s2 = np.stack([(perf[a:b] ** 2).sum(0) for a, b in zip(edges[:-1], edges[1:], strict=True)])
    cnt = np.diff(edges).astype(float)

    def score(idx):
        n = cnt[list(idx)].sum()
        m = s1[list(idx)].sum(0) / n
        if metric == "mean":
            return m
        v = s2[list(idx)].sum(0) / n - m * m
        return m / np.sqrt(np.maximum(v, 1e-18))

    logits, degr = [], []
    all_b = set(range(n_blocks))
    for is_b in itertools.combinations(range(n_blocks), n_blocks // 2):
        if 0 not in is_b:  # each split and its mirror are the same experiment; count once
            continue
        oos_b = tuple(sorted(all_b - set(is_b)))
        for a, b in ((is_b, oos_b), (oos_b, is_b)):
            ins, oos = score(a), score(b)
            best = int(np.argmax(ins))
            rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1) + 1
            w = rank / (N + 1)
            logits.append(math.log(w / (1 - w)))
            degr.append((float(ins[best]), float(oos[best])))
    lg = np.array(logits)
    d = np.array(degr)
    slope = float(np.polyfit(d[:, 0], d[:, 1], 1)[0]) if len(d) > 2 and d[:, 0].std() > 0 else float("nan")
    return {"pbo": float((lg <= 0).mean()), "n_combos": len(lg), "logit_median": float(np.median(lg)),
            "is_oos_slope": slope, "oos_best_mean": float(d[:, 1].mean())}


# --------------------------------------------------------------------------- registry
def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


class TrialRegistry:
    """Append-only JSONL log of every evaluated configuration.

    One line per trial: {ts, family, hash, config, metrics, note}. ``count(family)`` is the N
    used by the deflated Sharpe ratio; re-running the same config does not increase N.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _rows(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def log(self, family: str, config: dict, metrics: dict, note: str = "") -> str:
        h = config_hash(config)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "family": family, "hash": h,
               "config": config, "metrics": metrics, "note": note}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")
        return h

    def count(self, family: str) -> int:
        return len({r["hash"] for r in self._rows() if r.get("family") == family})

    def trials(self, family: str) -> list[dict]:
        return [r for r in self._rows() if r.get("family") == family]

    def holdout_opened(self, family: str) -> list[dict]:
        return [r for r in self.trials(family) if r.get("note") == "HOLDOUT"]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


@dataclass(frozen=True)
class Holdout:
    """The last ``days`` of the sample are locked. ``end_index`` = first locked sample."""

    days: int

    def end_index(self, times_ms: np.ndarray) -> int:
        if self.days <= 0 or len(times_ms) == 0:
            return len(times_ms)
        cut = times_ms[-1] - self.days * 86_400_000
        return int(np.searchsorted(times_ms, cut, side="right"))


# --------------------------------------------------------------------------- PnL summary
def pnl_summary(pnl: np.ndarray, periods_per_year: float) -> dict:
    x = np.asarray(pnl, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {"n": len(x)}
    eq = np.cumsum(x)
    dd = eq - np.maximum.accumulate(eq)
    sd = x.std(ddof=1)
    return {"n": len(x), "mean_bps": float(x.mean() * 1e4),
            "sharpe": float(x.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0,
            "t": float(x.mean() / (sd / math.sqrt(len(x)))) if sd > 0 else 0.0,
            "total_pct": float(x.sum() * 100), "maxdd_pct": float(dd.min() * 100),
            "ann_ret_pct": float(x.mean() * periods_per_year * 100),
            "ann_vol_pct": float(sd * math.sqrt(periods_per_year) * 100)}
