# ruff: noqa: E501
"""Adaptive signal weights: every day, the weight set with the best mean / std of daily P&L over the last 30 days,
searched over the whole weight space (covering pool + local refinement, qforecast.adaptive.search).

    python -m qforecast.adaptive.weights --from 2026-11-20 --to 2026-11-25               # C5

Every weight set runs through the reference book (qforecast.adaptive.search.NumpyBook), one after the other: exact,
but slow, so the covering pool stays small.

Timing (no look-ahead): the weights of signal row t are traded at the open of t + 1. They use the P&L of columns
t-31 .. t-2 only: column t-1 earns open[t] -> open[t + 1], which is not known yet at that open. A day already in
the cache is never recomputed: the weights a book traded on are frozen.
The search is random (perturbations of the refinement): the generator is seeded with the day number of ``--from``,
so a run is reproducible; report ranges over several seeds, not one run.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from qforecast.adaptive import search as S

CACHE = Path("data/adaptive/weights.json")
WINDOW, WARM_DAYS = 30, 50


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def _day(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, UTC).strftime("%Y-%m-%d")


def base_weights(spec, signals=S.SIGNALS) -> np.ndarray:
    return S.nrm([float(spec.weights.get(s, 0.0)) for s in signals])[0]


def compute(ctx, spec, start: str, end: str, seed: int | None = None, window: int = WINDOW, warm_days: int = WARM_DAYS,
            n_pool: int = S.NPOOL, top: int = 16, kids: int = 31) -> tuple[dict[str, np.ndarray], dict]:
    """{day: (F,) weights} for signal-row days start .. end (inclusive), and the refinement summary.
    ``n_pool``: draws per half of the covering pool. ``top`` / ``kids``: refinement size."""
    d0 = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    warm = (d0 - timedelta(days=warm_days)).strftime("%Y-%m-%d")
    end_excl = (datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
    t = ctx.cube.t
    p = S.NumpyBook(ctx, spec, int(np.searchsorted(t, _ms(warm))), int(np.searchsorted(t, _ms(end_excl))))
    rowday = [_day(v) for v in p.t]
    assert rowday[-1] == end, f"cube has no row {end} (last {rowday[-1]})"
    rows = [i for i in range(p.D) if rowday[i] >= start]
    base = base_weights(spec, p.signals)
    Wp = S.pool(base, n_pool)
    Pg = S.run_pool(p, Wp, bs=1024)
    rng = np.random.default_rng(int(d0.timestamp()) // 86400 if seed is None else seed)
    rw, gap = S.refine(p, Wp, S.window_scores(Pg, window, "ret"), S.window_scores(Pg, window, "sr"), window, rng, base,
                       first=rows[0] - 1, top=top, kids=kids)
    return {rowday[i]: S.nrm(rw["sr"][i - 1])[0] for i in rows}, gap


def load_cache(path: Path = CACHE) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"weights": {}, "runs": []}


def save_days(out: dict[str, np.ndarray], gap: dict, start: str, end: str, path: Path = CACHE, seconds: float = 0.0,
              signals=S.SIGNALS) -> dict:
    cache = load_cache(path)
    for d, w in out.items():
        cache["weights"].setdefault(d, dict(zip(signals, [round(float(x), 5) for x in w], strict=True)))  # never overwrite a frozen day
    cache.setdefault("runs", []).append({"from": start, "to": end, "at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
                                         "seconds": round(seconds), "refine_gap": gap})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    tmp.replace(path)
    return cache


def dyn_score(ctx, spec, k: int, by_day: dict[str, dict]) -> tuple[np.ndarray, dict]:
    """(T, N) score where the last rebalance row of each staggered sub-book (rows k - every + 1 .. k) is scored with
    that day's own weights (a sub-book keeps the weights of the day it re-picked); other rows keep the spec weights
    (they do not enter the target at row k). Also returns the weights of row k."""
    from qforecast.swing.strategy import score, weight_vector

    out = score(ctx, spec).copy()
    Z = np.nan_to_num(ctx.Z)
    rows = list(range(max(0, k - spec.every + 1), k + 1))
    for q in rows:
        w = weight_vector(ctx, by_day[_day(ctx.cube.t[q])])
        out[q] = np.where(ctx.mask[q], Z[q] @ w, np.nan)
    return out, by_day[_day(ctx.cube.t[k])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True, help="last signal row day (inclusive)")
    ap.add_argument("--spec", default="artifacts/models/c5.json")
    ap.add_argument("--cube", default="data/um/cube_daily_top40.npz")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    from qforecast.swing.cube import Cube
    from qforecast.swing.strategy import Context, SwingSpec

    t0 = time.time()
    ctx = Context.build(Cube.load(Path(a.cube)))
    out, gap = compute(ctx, SwingSpec.load(a.spec), a.start, a.end, a.seed)
    save_days(out, gap, a.start, a.end, Path(a.cache), time.time() - t0)
    for d, w in out.items():
        print(d, " ".join(f"{s}={x:+.3f}" for s, x in zip(S.SIGNALS, w, strict=True)))
    print(f"ok {a.start}..{a.end} {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
