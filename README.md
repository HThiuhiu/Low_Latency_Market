**English** | [Tiếng Việt](README.vi.md)

# Adaptive Crypto Trading: from raw data to a fast adaptive algorithm

[![ci](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml/badge.svg)](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml)

A personal research project on Binance USDT-M perpetuals. It covers a real-time data pipeline, a cross-sectional
long/short strategy (**C5**), a validation toolkit built so that I would not fool myself, and an **adaptive
signal-weight algorithm** running on a batched torch engine. Everything runs as paper trading.

## Highlights

- **Point-in-time data.** Delisted coins are kept, dead coins are filtered out, and the universe is ranked only on
  past data. Tests prove that no signal sees the future.
- **Exchange-realistic simulation.** Costs, funding, minimum orders, a beta hedge, a short-squeeze guard, and
  staggered sub-books so a result does not depend on its start day.
- **Honest validation.** Purged and combinatorial CV, deflated Sharpe, PBO, a multi-block history screen, and a
  sealed vault for clean tests.
- **Adaptive algorithm.** Signal weights are re-optimised every day over a trailing window, with strict
  no-look-ahead timing.
- **Performance engineering.** The adaptive search runs on a batched torch engine (CPU / GPU). A slower reference
  implementation is kept, and a test checks that both give the same P&L.

## How it evolved

Each step answered a need left open by the previous one.

| Step | Question | What I built |
|---|---|---|
| 1 | How do I clean market data and compute on it fast? | Real-time pipeline, point-in-time daily cube, signal library |
| 2 | Can a model on this data make decisions? | Minute-bar CNN-BiLSTM forecaster, then a long/short book model with an exchange-style backtest |
| 3 | How do I tell an edge from luck? | Validation toolkit, then the C5 strategy built under its rules |
| 4 | Can the strategy adapt as the market changes? | Daily adaptive signal weights, deployed on a paper bot with a dashboard |
| 5 | Can the adaptive search be fast enough to be thorough? | Batched torch engine, running sums instead of re-simulation, GPU support |

Each step is a branch, from `step-01/data-pipeline` to `step-08/fast-adaptive-engine`, merged into main in order.

## Architecture

| Layer | Modules |
|---|---|
| Real-time pipeline | `qforecast/services/`, `core/`, `storage/`, `model/` |
| Data | `qforecast/exchange/vision.py`, `qforecast/swing/cube.py`, `data.py` |
| Signals | `qforecast/swing/features.py`, `signals.py` |
| Strategy and simulation | `qforecast/swing/book.py`, `strategy.py`, `backtest.py`, `artifacts/models/c5.json` |
| Validation | `qforecast/validation.py`, `evaluation.py`, `blockcv.py`, `vault.py` |
| Adaptive algorithm | `qforecast/adaptive/search.py`, `weights.py`, `engine.py` |
| Paper trading | `qforecast/swing/account.py`, `paper.py`, `dashboard.py` |

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m qforecast.swing.data              # build the daily cube
python -m qforecast.swing.backtest          # backtest C5
python -m qforecast.swing.paper --plan      # today's picks (no trade)
python -m qforecast.swing.dashboard         # paper books on http://127.0.0.1:8765
pytest -q                                   # offline tests on synthetic data
```

## Status

- C5 and the adaptive book run on paper, the only clean test left.
- In walk-forward tests the adaptive weights have not beaten the fixed ones yet. They are kept as a research book.
- The C5 weights were chosen on the same history the backtest uses, so a backtest shows that the code works, not
  what to expect.

## Docs

[C5 strategy](docs/C5.md) · [Adaptive weights](docs/ADAPTIVE_WEIGHTS.md) ·
[Low-latency pipeline](docs/LOW_LATENCY_PIPELINE.md) · [Benchmarks](RESULTS_BENCHMARK.md)

> Paper trading only. No real orders. Not financial advice.
