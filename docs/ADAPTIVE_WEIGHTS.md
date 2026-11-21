# Adaptive signal weights

Code: `qforecast/adaptive/` (`search.py`, `weights.py`). Tests: `tests/test_adaptive.py`.

C5 ranks coins with 6 signals and fixed weights. The adaptive variant asks a different question every day:
which weight set would have done best over the last 30 days? It then trades the next day with that set.

## Algorithm

| Stage | What it does |
|---|---|
| Evaluator | `search.NumpyBook` runs each weight set through the full C5 book (`run_staggered`: tranches, beta hedge, thin leg, squeeze guard, costs, funding, 7 sub-books), one set after the other |
| Covering pool | Dirichlet draws that keep the base signs, draws with free signs, the base set, the equal-sign set and each single signal. The pool runs once over the period |
| Window scores | Cumulative sums give each set's trailing 30-day return and mean / std for every day in one pass |
| Local refinement | Each day the best pool sets by each objective are perturbed in two rounds, the second with smaller steps. Every candidate is re-run on that day's window |
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

## Limits of this first version

Every weight set is a full book run in Python, one after the other. The search is exact but slow, so the covering
pool stays small and covers the weight space only coarsely. Making the evaluation fast is the next step.
