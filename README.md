**English** | [Tiếng Việt](README.vi.md)

# qforecast: a crypto research stack that tries to prove its own strategies wrong

[![ci](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml/badge.svg)](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![mode](https://img.shields.io/badge/trading-paper%20only-lightgrey)

A solo, end-to-end system for Binance USDT-M perpetual futures (crypto futures with no expiry date). It covers live
market data, a toolkit that guards against backtest overfitting, and a daily long/short strategy running on paper.
Every idea is held to one rule: keep it only if it survives tests built to reject it. The minute-bar deep-learning
model did not. One daily strategy was kept, and it runs on paper until data it has never seen confirms or rejects
it. Built with Python asyncio, Redis Streams, PyTorch, a custom Triton GPU kernel, Docker and CI.

- **Real-time pipeline:** Binance WebSocket streams, with missed bars backfilled on every reconnect, features
  updated in place as each bar arrives, and ONNX Runtime inference.
- **Validation toolkit:** guards against using future data, ignoring delisted coins, and keeping the luckiest of
  many variants.
- **Daily long/short strategy, C5** (the frozen spec in [`artifacts/models/c5.json`](artifacts/models/c5.json)): a
  market-neutral portfolio, long some coins and short others, adjusted every day. It ranks the most liquid
  perpetuals on published factors: low volatility, low beta (moves less than BTC), no recent spike, low funding
  (cheap to hold), closeness to recent highs, and taker buying. It buys the top, shorts the bottom and hedges out
  BTC.

**Status:** C5 has run as a paper portfolio on live Binance prices since November 2026, with no real orders. That
is far too short to tell skill from luck; only more forward data can.

## Where to look

| If you hire for | Open first |
|---|---|
| Quant research | [`validation.py`](qforecast/validation.py), [`vault.py`](qforecast/vault.py), [`blockcv.py`](qforecast/blockcv.py), [C5 strategy](docs/C5.md) |
| ML / GPU | [`scan_triton.py`](qforecast/model/scan_triton.py), [`adaptive/engine.py`](qforecast/adaptive/engine.py), [adaptive weights](docs/ADAPTIVE_WEIGHTS.md) |
| Backend / data | [`services/`](qforecast/services/), [`swing/paper.py`](qforecast/swing/paper.py), [`exchange/vision.py`](qforecast/exchange/vision.py), [pipeline write-up](docs/LOW_LATENCY_PIPELINE.md) |

## Engineering highlights

Live data flows ingestor → signal → analytics → gateway, each an asyncio service on Redis Streams that can swap to
an in-memory bus in tests. The daily strategy runs as a scheduled job on the same exchange and data code.

| What | Problem it solves | Code and tests |
|---|---|---|
| **Event-driven services:** Redis Streams or an in-memory bus, Docker Compose, Prometheus | One crashed stage or dropped socket must not stop the pipeline or lose bars. A supervisor restarts it with backoff, and the gateway exposes every stage's metrics to Prometheus | [`services/all.py`](qforecast/services/all.py), [`test_bus.py`](tests/test_bus.py) |
| **CPU and I/O overhead removed:** incremental features, ring buffers, ONNX Runtime, thread placement | Overhead, not maths, dominated the naive pipeline | [write-up](docs/LOW_LATENCY_PIPELINE.md); rerun: `python -m benchmarks.run_all` |
| **Custom GPU kernel:** a fused Triton kernel for the core recurrence of an optional state-space (Mamba-style) layer | A sequential scan wastes the GPU. Kept as GPU engineering after the model was dropped; tested on GPU against a reference that passes gradcheck | [`scan_triton.py`](qforecast/model/scan_triton.py), [`test_scan.py`](tests/test_scan.py) |
| **Batched simulation engine:** the strategy for a whole batch of signal-weight sets in one pass (PyTorch, CPU or GPU) | A thorough daily weight search is too slow one set at a time. The slow version stays as the parity check | [`engine.py`](qforecast/adaptive/engine.py), [`test_adaptive_engine.py`](tests/test_adaptive_engine.py) |
| **Unattended paper bot:** a scheduled daily job with saved state; every day missed while the machine was off is replayed at that day's open | A laptop bot must survive sleep and restarts without silently skipping trades | [`paper.py`](qforecast/swing/paper.py), [`test_swing_paper.py`](tests/test_swing_paper.py) |

## What was tested, and the verdict

| Idea | How it was tested | Verdict |
|---|---|---|
| Minute-bar CNN-BiLSTM forecaster | Held-out periods with trading costs ([details](docs/LOW_LATENCY_PIPELINE.md#limitations)) | Dropped: no edge after costs |
| Daily long/short strategy (C5) | Point-in-time data, seven staggered start days, corrections for the number of variants tried | Kept as the candidate; on paper (see limitations) |
| [Adaptive weights](docs/ADAPTIVE_WEIGHTS.md) (re-optimised daily) | Walk-forward, point-in-time | Did not beat fixed weights; kept on paper for comparison |

The method behind these verdicts:

| Backtest pitfall | How it is handled | Test |
|---|---|---|
| Survivorship: today's symbol list holds only survivors | Every perp ever listed is downloaded; the universe is ranked only on data before each month | [`test_swing_data.py`](tests/test_swing_data.py) |
| Look-ahead: using information not yet known | One timing rule everywhere: decide at the close, trade at the next open | [`test_swing_signals.py`](tests/test_swing_signals.py), [`test_c5.py`](tests/test_c5.py), [`test_adaptive.py`](tests/test_adaptive.py) |
| Ignoring costs and funding | Fees, real funding, and paper fills that walk the order book for the full order size | [`test_swing_book.py`](tests/test_swing_book.py), [`test_swing_account.py`](tests/test_swing_account.py) |
| Luck of the start date | The mean of seven sub-portfolios that rebalance on different weekdays | [`test_swing_book.py`](tests/test_swing_book.py) |
| Keeping the luckiest of many variants | Deflated Sharpe and the probability of backtest overfitting (PBO) discount a result by the number of variants tried; a winner must also win in most time blocks | [`test_validation.py`](tests/test_validation.py), [`test_evaluation.py`](tests/test_evaluation.py) |
| Treating history as confirmation | History can only reject; confirmation comes from sealed or forward data | [`test_vault.py`](tests/test_vault.py) |

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-dev.txt

# Verify the four core claims offline, on a synthetic market (tests/synth.py): no signal sees the future;
# the fast engine matches the reference; the overfitting test tells noise from an edge; sealed data cannot be swapped.
pytest -q tests/test_swing_signals.py::test_every_signal_is_point_in_time \
          tests/test_adaptive_engine.py::test_engine_matches_the_reference_book \
          tests/test_validation.py::test_pbo_noise_vs_real_edge \
          tests/test_vault.py::test_vault_rejects_changed_cube

pytest -q                                   # full suite, offline
python -m qforecast.swing.data              # download the history and build the daily dataset
python -m qforecast.swing.backtest          # C5 backtest on a USDT account: real order sizes, fees, funding
python -m qforecast.swing.paper --plan      # today's picks and orders, nothing is traded
python -m qforecast.swing.dashboard         # paper portfolios on http://127.0.0.1:8765
```

No market data is stored in the repository. It comes from Binance's public endpoints and archives, with no API key.
On every push and pull request, CI runs `ruff`, `pytest`, a benchmark smoke run and a Docker build. The GPU kernel
test is skipped on machines without an NVIDIA GPU.

## Key decisions and trade-offs

- **History can only reject.** Every design choice was tested on past data, but only data recorded after the design
  was frozen can confirm one.
- **Dropped the minute model instead of tuning it.** It showed no edge after trading costs, and more tuning on the
  same data would mostly have fitted noise.
- **Simulated taker fills (market orders).** This is the conservative choice. Cheaper maker fills are possible but
  unproven until measured.
- **Kept the slow reference after writing the fast engine.** A parity test against it is cheaper than one silent
  bug in a search that runs every day.

## Limitations and next steps

| Limitation | Next step |
|---|---|
| C5's signal weights were chosen on the same history the backtest uses, so backtests prove that the code works, not what to expect | Score the sealed forward window: data recorded after the design was frozen, hash-locked so it cannot be swapped |
| Only taker fills are simulated | Measure maker fills (resting limit orders, lower fee) with small live orders before any real capital |
| Adaptive weights are unproven | Keep them on paper beside C5, and switch only if forward data favours them |
| Results are daily-close and single-exchange; the squeeze guard checks daily closes only | Model intraday squeezes and liquidation before any real capital |

<details>
<summary><b>How the project evolved</b>: one branch per step, readable in order</summary>

The real-time pipeline and the minute-bar model came first and are on the default branch. Each step after that is
a self-contained branch that answers a gap the previous one left.

| Branch | What it adds |
|---|---|
| `step-01/data-pipeline` | Full perp history with delisted coins, a monthly universe, the daily dataset |
| `step-02/signal-features` | Signals from published research, ranked across coins every day |
| `step-03/book-model` | Long/short portfolio with the market hedge, fees and funding; backtest on a USDT account |
| `step-04/research-toolkit` | Purged and combinatorial cross-validation, deflated Sharpe, PBO, block screen, sealed vault |
| `step-05/strategy-c5` | C5, built under those rules |
| `step-06/adaptive-weights` | Adaptive weights, with a point-in-time test |
| `step-07/paper-dashboard` | Paper bot on the live order book, replay of missed days, local dashboard |
| `step-08/fast-adaptive-engine` | Batched simulation engine with a parity test against the reference |

</details>

<details>
<summary><b>Repository layout</b></summary>

The repository is named after its first phase, the low-latency pipeline. The package is `qforecast`.

```text
qforecast/   services/ core/ storage/ model/ trainer/ sim/   real-time pipeline and the minute-bar model
             agent/                                          LLM advisor whose proposals fixed risk rules can veto (paper only)
             exchange/ swing/ adaptive/                      daily data, C5, paper bot, adaptive weights
             validation.py evaluation.py blockcv.py vault.py validation toolkit
tests/  benchmarks/  deploy/  docs/  research/  ModelTrade/ (the original prototype)
```

Benchmark write-up: [RESULTS_BENCHMARK.md](RESULTS_BENCHMARK.md).

</details>

> Research and paper trading only. No real orders are sent. Not financial advice.
