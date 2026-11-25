# Adaptive signal weights

Code: `qforecast/adaptive/` (`engine.py`, `search.py`, `weights.py`). Tests: `tests/test_adaptive.py`,
`tests/test_adaptive_engine.py`.

C5 ranks coins with 6 signals and fixed weights. The adaptive variant asks a different question every day:
which weight set would have done best over the last 30 days? It then trades the next day with that set.

## Algorithm

| Stage | What it does |
|---|---|
| Evaluator | `engine.TorchBook` runs the full C5 book (tranches, beta hedge, thin leg, squeeze guard, costs, funding, 7 sub-books) for a whole batch of weight sets at once. `search.NumpyBook` is the reference: one `run_staggered` call per set |
| Covering pool | Dirichlet draws that keep the base signs, draws with free signs, the base set, the equal-sign set and each single signal. The pool runs once over the period |
| Window scores | Cumulative sums give each set's trailing 30-day return and mean / std for every day in one pass |
| Local refinement | Each day the best pool sets by each objective are perturbed in two rounds, the second with smaller steps. Every candidate is re-run on that day's window |
| Daily weights | The set with the best trailing mean / std, normalised to an L1 norm of 1 |

## Making it fast

The first version ran every weight set through the Python book, one after the other. The search was exact but
slow, so the pool could only cover the weight space coarsely. The work on speed:

| Change | Effect |
|---|---|
| Weight sets become a batch dimension | One pass over the rows ranks the coins for every set with one matrix product and one stable sort. Tranches are built with scatter operations, and the 7 sub-books of every set live in one tensor |
| The pool runs once, windows come from cumulative sums | Every day's trailing score is a difference of two running sums. No window is re-simulated for the pool |
| Only the refinement re-runs a window | Each day re-simulates just its small candidate set on its own window |
| float32 for the search, float64 for the check | The search runs in float32 on a CPU or a GPU (`QF_DEV=cuda`). The parity test runs in float64 |
| Chunked batches | The pool is evaluated in chunks, so memory stays bounded on a laptop |
| The reference stays | `NumpyBook` is kept, and `test_engine_matches_the_reference_book` checks that both evaluators give the same daily P&L |

With the batched engine the pool is large enough to cover the whole weight space every day, and long ranges run on
a GPU.

## Timing and leakage

| Rule | Detail |
|---|---|
| Decision | The weights of signal row t are traded at the open of t + 1 |
| Data used | The P&L of columns t-31 .. t-2 only. Column t-1 earns open[t] → open[t+1], which is unknown at that open |
| Test | The weights of row t are recomputed on a cube cut at row t, with both evaluators. They must be identical |
| Sub-books | Each staggered sub-book keeps the weights of the day it re-picked (`weights.dyn_score`) |
| Frozen days | A day already in the cache is never recomputed: the weights a book traded on stay fixed |
| Randomness | The generator is seeded with the day number of the first day of a run, so a run is reproducible. Judge the method by the spread over several seeds, never by one run |

## What we found

In walk-forward tests over 2021-2025, the adaptive weights did not beat the fixed C5 weights. The winning set
changes from window to window, and the best set of the last 30 days predicts little about the next one. The
module is published as a research tool. In the paper bot it runs beside C5 as a monitored second book, not as a
replacement. To test a change to it, use `qforecast.blockcv` (gain in at least 5 of 6 time blocks) and then a
sealed forward window (`qforecast.vault`).
