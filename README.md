**English** | [Tiếng Việt](README.vi.md)

# From a market-data pipeline to an adaptive trading algorithm

This repository follows one path in five chapters. It starts with filtering and computing on raw market data,
moves to building and testing models on that data, and ends with a trading algorithm that adapts over time and a
fast engine that lets it search well. Each chapter is a branch with a pull request on top of the previous one.

> Paper trading only. Nothing here sends real orders. This is not financial advice.

## The story

### 1. Filtering and computing on market data: a data pipeline

The first question was how to take raw exchange data, clean it, and compute on it without wasting time.

- **Real-time pipeline** (the original code of this repository): live Binance market data, incremental features,
  ring buffers and Redis Streams microservices. It is measured and tuned for CPU and I/O, see
  [docs/LOW_LATENCY_PIPELINE.md](docs/LOW_LATENCY_PIPELINE.md).
- **Daily point-in-time cube** (`step-01/data-pipeline`): every crypto USDT perp ever listed, delisted coins kept,
  dead coins filtered out, a monthly top-40 universe ranked only on past data, funding aligned to days.
- **Signals computed from it** (`step-02/signal-features`): 17 daily signals from the literature, ranked across
  coins every day.

### 2. Building a model on the data

- **First model** (in the pipeline): a CNN-BiLSTM-attention forecaster on minute bars, served on ONNX Runtime.
- **Cross-sectional book model** (`step-03/book-model`): the signals become a daily score. The score becomes a
  long/short book with inverse-volatility legs, a beta hedge, a squeeze guard, costs and funding. An
  exchange-style backtest replays it on a USDT account.

### 3. Testing the models

The minute model had no edge once costs were paid. Before trusting the next model, the project needed tools that
can say no.

- **Research toolkit** (`step-04/research-toolkit`): purged walk-forward and combinatorial purged CV, deflated
  Sharpe, probability of backtest overfitting, paired bootstrap, a 6-block history screen and a sealed vault for
  clean tests.
- **C5** (`step-05/strategy-c5`): the book that came out of this testing. A 6-signal score, a narrow long leg and a
  wide short leg, shorts with negative funding skipped, beta hedged. See [docs/C5.md](docs/C5.md).

### 4. An algorithm that adapts over time

C5 keeps its signal weights fixed. Markets change, so the next question was: what if the weights are re-chosen
every day?

- **Adaptive weights** (`step-06/adaptive-weights`): every day, search the weight space for the set with the best
  trailing 30-day mean / std, refine it locally, and trade the next day with it. No look-ahead: a test recomputes
  every day on a cube that ends that day.
- **Deployment** (`step-07/paper-dashboard`): a paper bot that trades C5 and the adaptive book side by side on live
  order books, replays the days the machine was off, and shows both on a local dashboard.

In walk-forward tests the adaptive weights have not beaten the fixed ones yet. They run as a monitored research
book. See [docs/ADAPTIVE_WEIGHTS.md](docs/ADAPTIVE_WEIGHTS.md).

### 5. Making the adaptive algorithm fast

The first adaptive version ran every candidate weight set through the book one after the other. It was exact, but
the search had to stay small, so it covered the weight space only coarsely.

- **Batched engine** (`step-08/fast-adaptive-engine`): the weight set becomes a tensor dimension. One pass over the
  days ranks the coins for every set at once and keeps all sub-books in one tensor. It runs on a CPU or a GPU.
- **Less recomputation**: the pool runs once over the period, trailing window scores come from cumulative sums,
  and only the small refinement set is re-simulated each day.
- **Checked against the reference**: the slow book is kept as the reference, and a test checks that both give the
  same daily P&L.

Now the search covers the whole weight space every day, and long ranges run on a GPU.

## Map of the code

| Chapter | Module | What it does |
|---|---|---|
| 1 | `qforecast/services/`, `core/`, `storage/` | Real-time ingest, incremental features, Redis Streams / in-memory bus |
| 1 | `qforecast/exchange/vision.py`, `qforecast/swing/cube.py`, `data.py` | Point-in-time perp history and the daily cube |
| 1 | `qforecast/swing/features.py`, `signals.py` | Rolling statistics, beta, cross-sectional ranks, signal library |
| 2 | `qforecast/model/`, `trainer/` | CNN-BiLSTM-attention forecaster, ONNX export |
| 2 | `qforecast/swing/book.py`, `strategy.py`, `paperbook.py`, `backtest.py` | Book model, strategy spec, staggered sub-books, exchange-style backtest |
| 3 | `qforecast/validation.py`, `evaluation.py`, `blockcv.py`, `vault.py` | Leak-free validation and clean-test vault |
| 3 | `artifacts/models/c5.json` | The C5 spec |
| 4 | `qforecast/adaptive/search.py`, `weights.py` | Weight-space search and daily adaptive weights |
| 4 | `qforecast/swing/account.py`, `paper.py`, `dashboard.py` | Virtual account, paper bot, local dashboard |
| 5 | `qforecast/adaptive/engine.py` | Batched torch engine for many weight sets at once |

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m qforecast.swing.data                                      # build the daily cube
python -m qforecast.swing.backtest --start 2025-01-01 --capital 1000 # exchange-style backtest of C5
python -m qforecast.swing.paper --plan                              # today's C5 picks and orders (no trade)
python -m qforecast.swing.dashboard                                 # paper books + daily scheduler
python -m qforecast.adaptive.weights --from 2026-11-20 --to 2026-11-25   # adaptive weights for a range of days
pytest -q
```

## Build log

| Chapter | Dates | Branch |
|---|---|---|
| 1. Data pipeline | 2026-11-10 → 11-12 | `step-01/data-pipeline`, `step-02/signal-features` |
| 2. Model on the data | 2026-11-13 → 11-15 | `step-03/book-model` |
| 3. Testing the models | 2026-11-16 → 11-19 | `step-04/research-toolkit`, `step-05/strategy-c5` |
| 4. Adaptive algorithm | 2026-11-20 → 11-22 | `step-06/adaptive-weights`, `step-07/paper-dashboard` |
| 5. Speed | 2026-11-23 → 11-25 | `step-08/fast-adaptive-engine` |

## Rules the code follows

| Rule | Where it is enforced |
|---|---|
| Decide at the close of day k, trade at the open of k + 1 | `qforecast/swing/cube.py` (one timing convention for every module) |
| No survivorship: delisted coins stay in past rankings | `select_universe` ranks with rows strictly before each month |
| No look-ahead | Tests cut the cube at row k. Signals, scores, targets and adaptive weights of row k must not change |
| No rebalance-day luck | Results are the mean of the weekday sub-books |
| Costs are always on | Cost per side + real funding. The paper bot walks the real order book |
| History can only reject | A variant must beat its base in 5 of 6 purged time blocks. Only unseen data (paper, sealed vault) can confirm it |

The C5 weights and leg sizes were chosen on 2021-2026 data, so a backtest on that history is in-sample. Read it as
a test that the code works, not as a forecast of returns.

## Tests

`pytest -q` runs offline on synthetic data. It covers:

- the data guards and the point-in-time universe;
- look-ahead in signals, targets and adaptive weights;
- the book maths and the C5 leg rules;
- the batched engine against the reference book;
- a paper step with replayed days, through the dashboard API.
