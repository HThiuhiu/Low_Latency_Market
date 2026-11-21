# Adaptive signal weights

Code: `qforecast/adaptive/` (`engine.py`, `search.py`, `weights.py`). Tests: `tests/test_adaptive.py`.

C5 ranks coins with 6 signals and fixed weights. The adaptive variant asks a different question every day:
which weight set would have done best over the last 30 days? It then trades the next day with that set.

## Algorithm

| Stage | What it does |
|---|---|
| Batched engine | `engine.run_batch` runs the full C5 book (tranches, beta hedge, thin leg, squeeze guard, costs, funding, 7 sub-books) for a batch of weight sets at once in torch. A test checks it against the numpy book to 1e-12 |
| Covering pool | About 120 000 weight sets: Dirichlet draws that keep the base signs, draws with free signs, the base set, the equal-sign set and each single signal. The pool runs once over the period |
| Window scores | Cumulative sums give each set's trailing 30-day return and mean / std for every day in one pass |
| Local refinement | Each day the 16 best pool sets by each objective are perturbed in 2 rounds (noise 0.06, then 0.02; about 1 500 candidates). Every candidate is re-run on that day's window |
| Daily weights | The set with the best trailing mean / std, normalised to an L1 norm of 1 |

## Timing and leakage

| Rule | Detail |
|---|---|
| Decision | The weights of signal row t are traded at the open of t + 1 |
| Data used | The P&L of columns t-31 .. t-2 only. Column t-1 earns open[t] → open[t+1], which is unknown at that open |
| Test | `test_daily_weights_are_point_in_time_and_reproducible` recomputes the weights of row t on a cube cut at row t. They must be identical |
| Sub-books | Each staggered sub-book keeps the weights of the day it re-picked (`weights.dyn_score`) |
| Frozen days | A day already in the cache is never recomputed: the weights a book traded on stay fixed |
| Randomness | The generator is seeded with the day number of the first day of a run, so a run is reproducible. Judge the method by the spread over several seeds, never by one run |

## Cost

| Range | Hardware | Time |
|---|---|---|
| One day | Laptop CPU, 4 threads | 5-10 min |
| Years | GPU (`QF_DEV=cuda`) | Required |

## What we found

In walk-forward tests over 2021-2025, the adaptive weights did not beat the fixed C5 weights. The winning set
changes from window to window, and the best set of the last 30 days predicts little about the next one. The
module is published as a research tool. In the paper bot it runs beside C5 as a monitored second book, not as a
replacement. To test a change to it, use `qforecast.blockcv` (gain in at least 5 of 6 time blocks) and then a
sealed forward window (`qforecast.vault`).
