**English** | [Tiếng Việt](README.vi.md)

# From raw market data to an adaptive trading algorithm

This project was not planned up front. It grew one question at a time. Each step began with a need the previous
step had left open. I learned what was required to meet it, built that, and the answer raised the next question.

> Paper trading only. Nothing here sends real orders. This is not financial advice.

## The path at a glance

| Step | The need | What I built | What it opened |
|---|---|---|---|
| 1 | Clean raw exchange data and compute on it fast | A real-time pipeline, then a point-in-time daily data cube and a signal library | Data I can trust, ready for a model |
| 2 | Turn that data into decisions | A minute-bar forecaster, then a long/short book model with an exchange-style backtest | A model, and the question of whether it is any good |
| 3 | Tell a real edge from luck | A validation toolkit, then the C5 strategy built under its rules | A strategy with fixed signal weights |
| 4 | Let the strategy follow a changing market | Daily re-optimised signal weights, deployed on a paper bot with a dashboard | An algorithm that works but searches too little |
| 5 | Make the adaptive search fast enough to be thorough | A batched engine on CPU / GPU that evaluates many weight sets at once | A search that covers the whole weight space every day |

---

## Step 1 · Can I clean market data and compute on it fast?

**The idea.** Before any trading, I wanted to understand data: how to take what an exchange sends, filter it, and
compute on it without wasting time.

**What I tried.** A real-time pipeline for live Binance data: incremental features, ring buffers, and
microservices over Redis Streams. I measured where CPU and I/O time went and removed the waste. The write-up is in
[docs/LOW_LATENCY_PIPELINE.md](docs/LOW_LATENCY_PIPELINE.md).

**What I learned.** Fast code on the wrong data is useless. Three traps showed up:

- coins that were delisted vanish from today's symbol lists, so any history built from those lists only contains
  survivors;
- a halted coin can keep a frozen price with no volume for weeks;
- funding must be charged to the day the position is actually held.

**What I built to meet it.**

- `qforecast/exchange/vision.py`: every USDT perp ever listed, delisted coins included, from the public archives.
- `qforecast/swing/cube.py`, `data.py`: a daily cube with one timing rule for everything (decide at the close,
  trade at the next open). The universe is a monthly top-40 ranked only on data before that month. Dead coins are
  filtered out. The build stops if a listed coin is missing, instead of silently dropping it.
- `qforecast/swing/features.py`, `signals.py`: 17 daily signals from published research, ranked across coins
  every day. A test recomputes every signal on a cube cut at that day, to prove nothing leaks from the future.

**The next need.** Clean data is only a means. Can it drive decisions?

## Step 2 · Can a model on this data make decisions?

**The idea.** Build a model on the data and let it choose positions.

**What I tried.** First, a CNN-BiLSTM-attention network that forecasts one coin on minute bars, served on ONNX
Runtime. Then a different kind of model: instead of forecasting one coin, rank all coins against each other every
day and hold the best and the worst.

**What I learned.** A forecast is not a trade. Costs, funding, position sizes and market direction decide the
result as much as the signal does. To judge a model I needed a simulator that trades like an exchange.

**What I built to meet it.**

- `qforecast/swing/book.py`: long the top of the ranking, short the bottom, legs sized by inverse volatility with
  a cap per coin, positions held for a week, costs and funding every day, and a guard that buys back a short that
  rallies too far.
- `qforecast/swing/strategy.py`: one JSON spec turns into book targets for every caller. It adds a beta hedge so
  the book earns the spread between coins, not the market direction. The book is split into weekday sub-books so
  a result never depends on the day it happened to start.
- `qforecast/swing/backtest.py`, `paperbook.py`: the same targets replayed on a USDT account, with next-open
  fills, minimum order sizes and per-coin P&L.

**The next need.** The backtest now produces numbers. How do I know they are not luck?

## Step 3 · How do I tell a real edge from luck?

**The idea.** Test every model honestly before believing it.

**What I tried.** I ran the minute-bar forecaster through real costs. Its edge disappeared. A daily book can look
good for the wrong reasons too: overlapping labels, many tries, one lucky period.

**What I learned.** Most ideas fail, and a good backtest is not proof. History can only reject an idea. Only data
that the design never saw can confirm one.

**What I built to meet it.**

- `qforecast/validation.py`: purged walk-forward and combinatorial purged CV, rank IC with honest t-stats, a
  Sharpe ratio deflated for the number of tries, and the probability of backtest overfitting.
- `qforecast/evaluation.py`, `blockcv.py`: a paired bootstrap, and a screen where a variant must beat its base in
  most separate time blocks, not just on average.
- `qforecast/vault.py`: sealed data that a test may open once, and only after its plan is written down.
- **C5** (`artifacts/models/c5.json`, [docs/C5.md](docs/C5.md)): the strategy that came out of this testing. Each
  rule answers one risk the tests exposed:
  - a wider short leg than long leg, so one short that rallies hurts less;
  - shorts with negative funding are skipped, because that short side is already crowded;
  - staggered sub-books and a beta hedge remove timing luck and market direction.

**The next need.** C5 keeps its signal weights fixed. Markets change. Should the weights change too?

## Step 4 · Can the strategy adapt as the market changes?

**The idea.** Every day, re-choose the signal weights from what worked over the last weeks, and trade with them
the next day.

**What I tried.** A daily search of the weight space. A spread-out pool of weight sets is scored on the trailing
window. The best ones are refined locally, and the winner trades the next day. The timing is strict: a day's
weights use only P&L that was already known at the next open. A test recomputes each day on a cube that ends that
day.

**What I learned.** The method works and is honest, but two limits showed up.

- In walk-forward tests the adaptive weights have not beaten the fixed ones yet. They run as a monitored research
  book, not as a replacement ([docs/ADAPTIVE_WEIGHTS.md](docs/ADAPTIVE_WEIGHTS.md)).
- The first version evaluated every candidate weight set one after the other, through the full Python book. It
  was exact but slow, so it could look at only a small part of the weight space.

**What I built to meet it.**

- `qforecast/adaptive/search.py`, `weights.py`: the search, the daily weights, and a cache. A day's weights never
  change after that day is traded.
- `qforecast/swing/account.py`, `paper.py`, `dashboard.py`: a paper bot that runs C5 and the adaptive book side by
  side on live order books, with a virtual account and no API key. Days missed while the computer was off are
  replayed. A local dashboard shows both books.

**The next need.** The adaptive search is only as good as how much of the space it can look at. It has to be fast.

## Step 5 · Can the adaptive search be fast enough to be thorough?

**The idea.** Apply what Step 1 taught about speed: measure first, then remove the waste. This time the target was
the adaptive search.

**What I tried.**

| Change | Why |
|---|---|
| Weight sets become a tensor dimension | One pass over the days ranks the coins for every set at once, with one matrix product and one stable sort. All sub-books live in one tensor |
| The pool runs once; window scores come from running sums | No trailing window is re-simulated for the pool |
| Only the refinement re-runs a window | Each day re-simulates just its small candidate set |
| float32 for the search, float64 for the check; chunked batches | Fast on a GPU, bounded memory on a laptop, exact where it is verified |
| The slow version stays as the reference | A test checks that both give the same daily P&L |

**What I learned.** Speed changed the method itself, not just its run time. The search now covers the whole weight
space every day, and long periods run on a GPU. The answers stay identical to the slow reference.

**What I built.** `qforecast/adaptive/engine.py`, with the search switched to it by default
(`--backend numpy` keeps the reference).

---

## Where it stands

- C5 and the adaptive book trade on paper. That is the only clean test left, because all past data was used to
  design them.
- The adaptive weights are a research tool until unseen data shows they help.
- Every rule above is enforced by a test: point-in-time data, no look-ahead, costs always on, history only rejects.

## Map of the code

| Step | Module | Role |
|---|---|---|
| 1 | `qforecast/services/`, `core/`, `storage/` | Real-time ingest, incremental features, Redis Streams / in-memory bus |
| 1 | `qforecast/exchange/vision.py`, `qforecast/swing/cube.py`, `data.py` | Point-in-time perp history and the daily cube |
| 1 | `qforecast/swing/features.py`, `signals.py` | Rolling statistics, beta, cross-sectional ranks, signal library |
| 2 | `qforecast/model/`, `trainer/` | CNN-BiLSTM-attention forecaster, ONNX export |
| 2 | `qforecast/swing/book.py`, `strategy.py`, `paperbook.py`, `backtest.py` | Book model, strategy spec, sub-books, exchange-style backtest |
| 3 | `qforecast/validation.py`, `evaluation.py`, `blockcv.py`, `vault.py` | Honest testing and the clean-test vault |
| 3 | `artifacts/models/c5.json` | The C5 spec |
| 4 | `qforecast/adaptive/search.py`, `weights.py` | Weight-space search and daily adaptive weights |
| 4 | `qforecast/swing/account.py`, `paper.py`, `dashboard.py` | Virtual account, paper bot, local dashboard |
| 5 | `qforecast/adaptive/engine.py` | Batched engine for many weight sets at once |

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

python -m qforecast.swing.data                                      # build the daily cube
python -m qforecast.swing.backtest --start 2025-01-01 --capital 1000 # exchange-style backtest of C5
python -m qforecast.swing.paper --plan                              # today's C5 picks and orders (no trade)
python -m qforecast.swing.dashboard                                 # paper books + daily scheduler
python -m qforecast.adaptive.weights --from 2026-11-20 --to 2026-11-25   # adaptive weights for a range of days
pytest -q                                                           # offline, synthetic data
```

The C5 weights were chosen on the same history the cube covers. A backtest on it shows that the code works, not
what to expect.

## Build log

Each step is a branch with a pull request on top of the previous one.

| Step | Dates | Branches |
|---|---|---|
| 1. Data | 2026-11-10 → 11-12 | `step-01/data-pipeline`, `step-02/signal-features` |
| 2. Model | 2026-11-13 → 11-15 | `step-03/book-model` |
| 3. Testing | 2026-11-16 → 11-19 | `step-04/research-toolkit`, `step-05/strategy-c5` |
| 4. Adaptive | 2026-11-20 → 11-22 | `step-06/adaptive-weights`, `step-07/paper-dashboard` |
| 5. Speed | 2026-11-23 → 11-26 | `step-08/fast-adaptive-engine` |
