**English** | [Tiếng Việt](README.vi.md)

# qforecast: a crypto research stack that tries to prove its own strategies wrong

[![ci](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml/badge.svg)](https://github.com/HThiuhiu/Low_Latency_Market/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B%20%28CI%203.12%29-blue)
![mode](https://img.shields.io/badge/trading-paper%20only-lightgrey)

A solo, end-to-end research and trading system for Binance crypto markets. It has three parts: asyncio services
on Redis Streams for live spot data, a toolkit that guards against backtest overfitting, and a market-neutral daily
strategy on USDT-M perpetual futures (crypto futures with no expiry date) that runs unattended on paper. It is built
with Python, PyTorch, a custom Triton GPU kernel, Docker and CI. Every idea must survive tests built to reject it.
The minute-bar deep-learning model did not, and was dropped. One daily strategy was kept, and it is waiting for
data it has never seen.

- **Daily long/short strategy** (code name C5, spec in [`artifacts/models/c5.json`](artifacts/models/c5.json)).
  Each day it ranks the most liquid perpetuals on published factors such as calm prices, low sensitivity to BTC
  and low holding cost. It buys the top, shorts the bottom, and sizes the two sides so that BTC moves cancel out.
- **Status:** a paper portfolio on live Binance prices since November 2026, with no real orders. That is far too
  short to tell skill from luck.

## Where to look

| If you hire for | Open first |
|---|---|
| Quant research | [`validation.py`](qforecast/validation.py), [`vault.py`](qforecast/vault.py), [`blockcv.py`](qforecast/blockcv.py), [C5 strategy](docs/C5.md) |
| ML / GPU | [`scan_triton.py`](qforecast/model/scan_triton.py), [`adaptive/engine.py`](qforecast/adaptive/engine.py), [adaptive weights](docs/ADAPTIVE_WEIGHTS.md) |
| Backend / data | [`services/`](qforecast/services/), [`swing/paper.py`](qforecast/swing/paper.py), [`exchange/vision.py`](qforecast/exchange/vision.py), [pipeline write-up](docs/LOW_LATENCY_PIPELINE.md) |

## Engineering highlights

Live spot data flows ingestor → signal → analytics → gateway. Each stage is an asyncio service on Redis Streams
that can swap to an in-memory bus in tests. The daily strategy runs as a scheduled job.

| What | Problem it solves | Code and tests |
|---|---|---|
| **Event-driven services**, Docker Compose, Prometheus | A crashed stage or a dropped socket must not stop the pipeline or lose bars. The supervisor restarts a stage with backoff, the ingestor backfills missed bars on reconnect, and the gateway exposes every stage's metrics | [`services/all.py`](qforecast/services/all.py), [`ingestor.py`](qforecast/services/ingestor.py); [`test_supervisor.py`](tests/test_supervisor.py), [`test_bus.py`](tests/test_bus.py) |
| **CPU and I/O overhead removed:** incremental features, ring buffers, ONNX Runtime, thread placement | Overhead, not maths, dominated the naive pipeline | [write-up](docs/LOW_LATENCY_PIPELINE.md); rerun: `python -m benchmarks.run_all` |
| **Custom GPU kernel:** fused Triton kernel for the recurrence of an optional state-space (Mamba-style) layer | A sequential scan wastes the GPU. It is tested on GPU against a reference that passes gradcheck, and kept after the model was dropped | [`scan_triton.py`](qforecast/model/scan_triton.py), [`test_scan.py`](tests/test_scan.py) |
| **Batched simulation engine:** the strategy for a whole batch of signal-weight sets in one pass (CPU or GPU) | A thorough daily weight search is too slow one set at a time. The slow version stays as the parity check | [`engine.py`](qforecast/adaptive/engine.py), [`test_adaptive_engine.py`](tests/test_adaptive_engine.py) |
| **Unattended paper bot:** saved state, and every day missed while the machine was off is replayed at that day's open | A laptop bot must survive sleep and restarts without silently skipping trades | [`paper.py`](qforecast/swing/paper.py), [`test_swing_paper.py`](tests/test_swing_paper.py) |
| **Known-answer tests:** statistics checked on synthetic data where the right answer is known (pure noise vs a planted edge, a sealed file that was changed) | Statistics code can be wrong and still look right | [`test_validation.py`](tests/test_validation.py), [`test_evaluation.py`](tests/test_evaluation.py), [`test_vault.py`](tests/test_vault.py) |

## What was tested, and the verdict

| Idea | How it was tested | Verdict |
|---|---|---|
| Minute-bar CNN-BiLSTM forecaster | Held-out periods with trading costs ([details](docs/LOW_LATENCY_PIPELINE.md#limitations)) | Dropped: no edge after costs |
| Daily long/short strategy (C5) | Point-in-time data, seven staggered start days, corrections for the number of variants tried | Kept as the candidate; on paper |
| [Adaptive weights](docs/ADAPTIVE_WEIGHTS.md) (re-optimised daily) | Walk-forward, point-in-time | Did not beat fixed weights; kept on paper for comparison |

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

# The four core claims, offline on a synthetic market: no signal sees the future, the fast engine matches the
# reference, the overfitting test tells noise from an edge, and sealed data cannot be swapped.
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
CI runs `ruff`, `pytest`, a benchmark smoke run and a Docker build on every push; the GPU test needs an NVIDIA GPU.

## Key decisions and trade-offs

- **Dropped the minute model instead of tuning it.** It showed no edge after trading costs, and more tuning on the
  same data would mostly have fitted noise.
- **Simulated taker fills (market orders).** This is the conservative choice; cheaper maker fills stay unproven
  until measured.
- **Kept the slow reference after writing the fast engine.** A parity test is cheaper than one silent bug in a
  search that runs every day.

## Limitations and next steps

| Limitation | Next step |
|---|---|
| C5's weights were chosen on the same history the backtest uses, so backtests prove that the code works, not what to expect | Score the sealed forward window: data recorded after the design was frozen, hash-locked so it cannot be swapped |
| Only taker fills are simulated | Measure maker fills (resting limit orders, lower fee) with small live orders before any real capital |
| Adaptive weights are unproven | Keep them on paper beside C5; switch only if forward data favours them |
| Daily closes and one exchange only | Model intraday squeezes and liquidation before any real capital |

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
qforecast/   services/ core/ storage/ model/ trainer/ sim/ agent/   real-time spot pipeline, the minute-bar model,
                                                                    and an LLM advisor limited to veto power (paper only)
             exchange/ swing/ adaptive/                             daily perp data, C5, paper bot, adaptive weights
             validation.py evaluation.py blockcv.py vault.py        validation toolkit
tests/  benchmarks/  deploy/  docs/  research/  ModelTrade/ (the original prototype)
```

Benchmark write-up: [RESULTS_BENCHMARK.md](RESULTS_BENCHMARK.md).

</details>

> Research and paper trading only. No real orders are sent. Not financial advice.
