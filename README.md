**English** | [Tiếng Việt](README.vi.md)

# Crypto trading simulator and research toolkit

This project is a daily, cross-sectional trading simulator for Binance USDT-M perpetuals. It contains:

- research tools that make it hard to fool yourself;
- **C5**, a long/short swing strategy;
- an **adaptive signal-weight optimiser**;
- a paper-trading bot with a small local dashboard.

The project began as a low-latency pipeline for minute bars, kept in
[docs/LOW_LATENCY_PIPELINE.md](docs/LOW_LATENCY_PIPELINE.md). The minute model had no edge after costs. The work
then moved to daily books, and to testing ideas properly before trusting them.

> Paper trading only. Nothing here sends real orders. This is not financial advice.

## What is inside

| Part | Module | What it does |
|---|---|---|
| Data | `qforecast/exchange/vision.py`, `qforecast/swing/data.py` | Downloads every crypto USDT perp ever listed (delisted coins included) from data.binance.vision and REST. Builds a daily cube of the monthly top-40 by volume |
| Signals | `qforecast/swing/signals.py`, `features.py` | 17 daily signals from the literature (momentum, trend, funding crowding, low volatility, MAX, size, beta, taker flow). Each is ranked across coins every day |
| Simulator | `qforecast/swing/book.py`, `strategy.py` | Long/short tranches with inverse-vol legs, a beta hedge, a squeeze guard, costs and funding. Seven staggered weekly sub-books |
| Exchange-style backtest | `qforecast/swing/backtest.py`, `paperbook.py` | The same targets replayed on a USDT account: next-open fills, minimum orders, per-coin P&L |
| Research tools | `qforecast/validation.py`, `evaluation.py`, `blockcv.py`, `vault.py` | Purged walk-forward, CPCV, deflated Sharpe, PBO, paired bootstrap, a 6-block history screen, a sealed vault for clean tests |
| C5 strategy | `artifacts/models/c5.json`, [docs/C5.md](docs/C5.md) | 6-signal score, 5 long / 8 short, shorts with negative funding skipped, beta hedged |
| Adaptive weights | `qforecast/adaptive/`, [docs/ADAPTIVE_WEIGHTS.md](docs/ADAPTIVE_WEIGHTS.md) | A batched torch engine runs about 120 000 weight sets at once. Each day it picks the best set of the last 30 days, then refines it locally |
| Paper bot + dashboard | `qforecast/swing/paper.py`, `account.py`, `dashboard.py` | Virtual accounts on live order books. Missed days are replayed. A local page at 127.0.0.1:8765 |

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m qforecast.swing.data                                      # daily cube, ~10-20 min the first time
python -m qforecast.swing.backtest --start 2025-01-01 --capital 1000 # exchange-style backtest of C5
python -m qforecast.swing.paper --plan                              # today's C5 picks and orders (no trade)
python -m qforecast.swing.dashboard                                 # paper books + daily scheduler
python -m qforecast.adaptive.weights --from 2026-11-20 --to 2026-11-25   # adaptive weights (CPU: minutes per run)
pytest -q
```

## How it was built

Each step is a branch with a pull request on top of the previous one.

| Date | Step | Branch |
|---|---|---|
| 2026-11-10 → 11-12 | Research tools first: purged CV, CPCV, PSR / DSR, PBO, bootstrap, block screen, vault | `step-01/validation-toolkit` |
| 2026-11-13 | Point-in-time data: every perp ever listed, monthly top-40, daily cube | `step-02/daily-data-cube` |
| 2026-11-14 | Signal library with the sign each paper predicts, no look-ahead test | `step-03/signal-library` |
| 2026-11-15 → 11-17 | Book simulator, strategy spec, staggered sub-books, exchange-style backtest | `step-04/book-simulator` |
| 2026-11-18 → 11-19 | C5: asymmetric legs, skip shorts with negative funding, thin-leg cap | `step-05/strategy-c5` |
| 2026-11-20 → 11-21 | Adaptive weights: batched engine, covering pool + refinement, daily cache | `step-06/adaptive-weights` |
| 2026-11-22 → 11-25 | Paper bot on live order books, dashboard, docs | `step-07/paper-dashboard` |

## Rules the code follows

| Rule | Where it is enforced |
|---|---|
| Decide at the close of day k, trade at the open of k + 1 | `qforecast/swing/cube.py` (one timing convention for every module) |
| No survivorship: delisted coins stay in past rankings | `select_universe` ranks with rows strictly before each month |
| No look-ahead | Tests cut the cube at row k. Signals, scores, targets and adaptive weights of row k must not change |
| No rebalance-day luck | Results are the mean of 7 weekday sub-books |
| Costs are always on | 7 bps per side + real funding. The paper bot walks the real order book |
| History can only reject | A variant must beat its base in 5 of 6 purged time blocks. Only unseen data (paper, sealed vault) can confirm it |

The C5 weights and leg sizes were chosen on 2021-2026 data, so a backtest on that history is in-sample. Read it as
a test that the code works, not as a forecast of returns.

## Tests

`pytest -q` runs offline on synthetic data. It covers the data guards, the point-in-time universe, signal and
target look-ahead, the book maths, the C5 leg rules, the torch engine against the numpy book (to 1e-12), adaptive
weight leakage and reproducibility, and a paper step with replayed days through the dashboard API.

## Earlier work

The original low-latency pipeline is still in the repository: incremental features, CNN-BiLSTM-attention on ONNX
Runtime, Redis Streams microservices and CPU / I/O benchmarks. See
[docs/LOW_LATENCY_PIPELINE.md](docs/LOW_LATENCY_PIPELINE.md) and [RESULTS_BENCHMARK.md](RESULTS_BENCHMARK.md).
