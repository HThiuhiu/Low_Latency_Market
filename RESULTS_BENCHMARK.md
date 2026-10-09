**English** | [Tiếng Việt](RESULTS_BENCHMARK.vi.md)

# Benchmark results

All numbers come from one machine and, unless marked otherwise, from one session on
**2026-10-09 (16:00–16:20)**:
Intel Core i7-13700H laptop (6 P-cores with Hyper-Threading = logical CPUs 0–11, 8 E-cores =
logical CPUs 12–19), Windows 11, Python 3.13.

Raw, auto-generated reports:
- [benchmarks/RESULTS.md](benchmarks/RESULTS.md): component micro-benchmarks;
- [benchmarks/BEFORE_AFTER.md](benchmarks/BEFORE_AFTER.md): prototype vs tuned pipeline,
  interleaved;
- [benchmarks/HETERO.md](benchmarks/HETERO.md): bulk-work scheduling across P/E-cores.

---

## 0. How to read these numbers

- **p50** is the median; **p99** is the latency that 99% of calls beat. **Speed-up** is
  `before ÷ after` of the p50.
- **Absolute timings on a laptop drift 2–3× between sessions** (thermal state, power
  management, OS scheduling). For example, the pandas feature recompute measured 5.5 ms,
  6.9 ms and 18.5 ms in three sessions. **Ratios are stable**, so every comparison below is
  a ratio measured within one session.
- The end-to-end comparison (§1) runs both versions **interleaved, bar by bar, in one
  process**, so both see identical machine conditions. Speed-ups carry **95% bootstrap
  confidence intervals**.
- ✅ = measured · 📐 = derived from sizes, not timed.

---

## 1. End-to-end: prototype vs tuned pipeline ✅

**Workload:** the per-bar hot path (decode frame → features → normalise → inference →
serialise), 300 real ETHUSDT bars. Each bar runs through both versions. Source:
[BEFORE_AFTER.md](benchmarks/BEFORE_AFTER.md).

| CPU policy | Before (prototype approach) | After (tuned) | Speed-up (95% CI) |
|---|---:|---:|---:|
| All CPUs | 56.6 ms | 1.07 ms | **52.7×** (49.0–56.0) |
| P-cores only | 35.2 ms | 0.60 ms | **58.3×** (55.5–60.8) |

Per stage (all CPUs):

| Stage | Before | After | Speed-up |
|---|---:|---:|---:|
| Frame decode | 32.0 µs | 22.4 µs | 1.4× |
| Features | 19,744 µs | 40.9 µs | **483×** |
| Normalise + model input | 152 µs | 63.6 µs | 2.4× |
| Model inference | 35,608 µs | 902 µs | **39.5×** |
| Serialise result | 106 µs | 33.5 µs | 3.2× |

**Where the time goes:**
- **Before:** features 35–44%, PyTorch inference 56–63%, everything else under 1%.
- **After:** inference 81–84%, everything else 16–19%.

Earlier sessions measured the same comparison at 57× (55–70) and 46× (45–54): same order
of magnitude, different absolute speeds.

Per order-book update (decode + serialise for the bus): 4.7 µs → 1.1 µs (**4.3×**).
Single-thread throughput went from 213k to 909k msg/s, and messages shrank from 132 B
(JSON) to 60 B (msgpack).

---

## 2. Where unoptimised PyTorch training and inference lose time ✅

### 2.1 Inference: the same model through different runtimes

Batch 1, input (1, 128, 14), 158k parameters. Source: [RESULTS.md](benchmarks/RESULTS.md).

| Runtime | p50 | vs PyTorch default |
|---|---:|---:|
| PyTorch eager, 12 threads (default style) | 4,224 µs | 1.0× |
| PyTorch eager, 1 thread | 665 µs | 6.4× |
| TorchScript (frozen), 1 thread | 535 µs | 7.9× |
| ONNX Runtime, 12 threads | 293 µs | 14.4× |
| **ONNX Runtime, 1 thread** | **181 µs** | **23.4×** |

Two conclusions:
- Running the same arithmetic in 181 µs means **more than 95% of the 4.2 ms eager call is
  overhead**: Python dispatch per operator, and thread-pool wake-up and synchronisation for
  tiny matrix multiplications.
- **One thread beats twelve** for batch-1 inference, in both PyTorch and ONNX Runtime.

The architecture also matters. A 2-layer BiLSTM over 64 steps took 472 µs on ONNX Runtime
with 1 thread; a 1-layer BiLSTM over 32 steps, behind two stride-2 convolutions, takes
181 µs (**2.6×**), at equal forecast quality.

### 2.2 Training: thread count

One training step of the LSTM model (batch 512, forward + backward + AdamW), 3 repeats
each. Reproduce with `python -m benchmarks.bench_train_threads`.

| Threads | ms per step, median (min–max) | Samples/s | vs torch default |
|---:|---:|---:|---:|
| 1 | 274 (267–313) | 1,867 | 0.61× |
| 2 | 267 (162–335) | 1,920 | 0.63× |
| 4 | 151 (115–221) | 3,396 | 1.12× |
| 6 | 146 (132–238) | 3,499 | 1.15× |
| 8 | 206 (199–216) | 2,489 | 0.82× |
| 12 | 171 (171–173) | 2,991 | 0.98× |
| **14 (torch default)** | **168** (157–169) | 3,041 | 1.00× |
| 20 | 158 (157–161) | 3,246 | 1.07× |

What this shows:
- **14 threads buy only 1.6× over one thread**, a parallel efficiency of about 12%. Most of
  the extra cores' capacity goes to synchronisation and to the serial LSTM recurrence.
- **More threads is not monotonic**: 8 threads are slower than 4 or 6.

A profile of the same step (`python -m benchmarks.profile_train`) puts about 55% of the
time in the oneDNN LSTM forward and backward kernels, and about 1% in data loading.

### 2.3 Training: architecture and data pipeline

| | Before | After | Change |
|---|---:|---:|---:|
| Time per training sample (2-layer BiLSTM/64 steps → 1-layer/32 steps), from the training logs | ~2.3 ms | ~0.24 ms | ~9× |
| Memory for 90k training windows (materialised N×128×14 array → index-gathered batches) 📐 | ~645 MB | ~7 MB | ~90× |

---

## 3. Component micro-benchmarks ✅

Source: [RESULTS.md](benchmarks/RESULTS.md).

### 3.1 Features per new bar

| Variant | p50 | Speed-up |
|---|---:|---:|
| pandas recompute over 500 bars (prototype) | 5,519 µs | 1× |
| numba, still recomputing 500 bars | 113 µs | 49× (from **compilation**) |
| **numba incremental, one new bar** | **3.1 µs** | **1,780×** (plus the **algorithm**) |

The incremental kernel is shared by training and serving. Tests check it bit for bit
against the offline path and against an independent pandas implementation.

### 3.2 Wire format: encode + decode of one message

| Message | json | orjson | **msgspec msgpack** | Size, json → msgpack |
|---|---:|---:|---:|---:|
| Kline | 6.3 µs | 1.1 µs | **0.6 µs** (10.5×) | 263 → 104 B |
| BookTicker | 4.5 µs | 0.7 µs | **0.4 µs** (11.2×) | 137 → 64 B |

### 3.3 WebSocket frame decoding

From `orjson → dict → float(str) → struct` to typed `msgspec` decoding:

| Frame | Before | After | Speed-up |
|---|---:|---:|---:|
| bookTicker | 1.4 µs | 0.8 µs | 1.75× |
| aggTrade | 1.3 µs | 1.0 µs | 1.3× |
| kline | 2.1 µs | 1.7 µs | 1.24× |

### 3.4 Bar-history storage

130,228 bars × 11 columns:

| Format | Size | Write | Read | Lossless |
|---|---:|---:|---:|:---:|
| `.npy` float64 (before) | 11.46 MB | 5 ms | 5.1 ms | ✓ |
| `.npy` memory-mapped | 11.46 MB | 5 ms | 0.45 ms (lazy) | ✓ |
| Parquet + zstd | 8.09 MB | 105 ms | 14.5 ms | ✓ |
| QCOL, 1 thread | 2.88 MB | 347 ms | 17.5 ms | ✓ |
| **QCOL, parallel column decode** | **2.88 MB** (4.0× smaller) | 347 ms | 10.6 ms | ✓ |

QCOL stores each column as `float64 → exact scaled int64 → optional delta → byte-shuffle
→ zstd`. Reads are about 2× slower than raw `.npy`; that is the deliberate price of a
4× smaller file.

### 3.5 Tick capture (`recorder` service)

Real recorded Binance stream, 7,273 messages:

| Metric | Value |
|---|---:|
| Append a message on the event loop, p50 / p99 | 0.4 / 0.5 µs |
| Compress + write on the I/O thread | 132 MB/s ≈ 1.8M msg/s |
| Read back + decompress | 2.6M msg/s |
| On disk per message | **16.2 B** (4.4× smaller than msgpack, ~8× smaller than JSON) |
| On disk per order-book update | **12.7 B** vs 146 B as raw JSON (11.5×) |

---

## 4. CPU scheduling on a hybrid CPU ✅

### 4.1 P-core vs E-core, one core at a time

| Task | P-core | E-core | E-core slower by |
|---|---:|---:|---:|
| `json.dumps`, single-threaded Python | 1.43–1.69 µs | 4.0–4.5 µs | ~2.7–3× |
| ONNX inference, hot loop | 204 µs | 463 µs | 2.3× |
| ONNX inference right after a 1 ms sleep | 426 µs | 2,013 µs | 4.7× |

(These were measured in an earlier session on 2026-10-08/09.)

Under full load, one *logical* CPU of a P-core shares its physical core with its
Hyper-Threading sibling:
- per logical CPU, a P-core is only **1.00–1.48×** an E-core (0.86–1.83× across sessions);
- per physical core, it is **2.0–3.0×**.

### 4.2 Bulk work: pinned + dynamic queue vs Windows scheduling

**Workload:** 20 worker processes, one per logical CPU. Each run executes a fixed batch:
- **inference:** 132k model windows;
- **ticks:** 13.8M Binance frames.

The metric is the **makespan**: the time from releasing all workers until the last one
finishes. The full 11-strategy table is in [HETERO.md](benchmarks/HETERO.md). The rows
that matter:

| Strategy | inference | ticks | Idle share |
|---|---:|---:|---:|
| All cores, equal split, Windows scheduling (baseline) | 2.81 s | 2.88 s | 9–12% |
| All cores, **dynamic queue**, Windows scheduling | 2.58 s (1.09×) | 2.68 s (1.08×) | 1% |
| All cores, dynamic queue, Windows + EcoQoS off | 2.94 s (0.96×) | 2.52 s (1.14×) | 1% |
| **Pinned**, equal split | 3.30 s (**0.85×**) | 3.26 s (**0.88×**) | 25–27% |
| Pinned, speed-proportional split | 2.57 s (1.09×) | 2.94 s (0.98×) | 9–15% |
| **Pinned + dynamic queue** | 2.61 s (1.08×) | 2.64 s (1.09×) | 1–2% |
| Pinned + dynamic queue + EcoQoS off | 2.44 s (1.15×) | 2.73 s (1.05×) | 1% |
| P-cores only, dynamic queue | 3.40 s (0.83×) | 4.06 s (0.71×) | 1% |

**Pinned + dynamic queue vs Windows scheduling + dynamic queue, across sessions:**

| Session | Windows behaviour | inference | ticks |
|---|---|---:|---:|
| A (earlier report version) | **EcoQoS parked the workers on E-cores** (P-cores 12–13% busy) | 8.54 s → 2.61 s (**3.27×**) | 5.98 s → 2.30 s (**2.60×**) |
| B | used every core | 2.84 s → 2.83 s (1.00×) | 3.94 s → 3.57 s (1.10×) |
| C (this session, [HETERO.md](benchmarks/HETERO.md)) | used every core | 2.58 s → 2.61 s (0.99×) | 2.68 s → 2.64 s (1.02×) |

**Dynamic queue vs equal split, both pinned**, across sessions A–C: **1.07–1.46×** faster.

**Findings:**
1. **A dynamic queue is the scheduler to use.** It needs no calibration, keeps idle time
   at 1–2%, and is the best or tied for best in every session.
2. **Pinned + dynamic vs Windows + dynamic:**
   - when Windows uses every core, the two are **equivalent** (0.92–1.21× across sessions
     and workloads, mostly within noise);
   - when EcoQoS parks background workers on E-cores, pinning wins by **2.6–3.3×**,
     because an affinity mask forces the work onto the P-cores.

   Pinning is therefore the safer default.
3. **Pinning with an equal split is the worst option.** P-core workers finish early and
   idle 25–27% of the time waiting for E-core workers; it is even worse than letting
   Windows move work around.
4. **Speed-proportional splits are unreliable**: 0.99–1.28× vs a pinned equal split across
   sessions, because speeds calibrated before the run drift during it (Hyper-Threading,
   thermal state).
5. **Do not switch the E-cores off for bulk work.** P-cores only is 0.71–0.83× of the
   baseline.

Each strategy has 3–6 runs per session, and the min–max ranges of neighbouring rows often
overlap. Treat differences under about 10% as trends.

### 4.3 The EcoQoS finding ✅

Windows 11 **EcoQoS** (power throttling) can tag background processes as efficiency-class,
and the hybrid scheduler then keeps them on E-cores while P-cores idle. Diagnostic session
(2026-10-09): inference, 20 unpinned processes, 4 consecutive runs per fresh process group.

| Configuration | Makespan of the 4 runs | P-core busy |
|---|---|---:|
| Default, group 1 | 3.40 · 6.56 · 6.46 · 6.09 s | 11–27% |
| Default, group 2 | 7.30 · 7.63 · 7.87 · 9.00 s | 15–20% |
| EcoQoS off, group 1 | 2.90 · 2.84 · 2.81 · 2.94 s | 97–100% |
| EcoQoS off, group 2 | 3.31 · 3.17 · 2.96 · 3.02 s | 97–100% |

The effect **depends on machine state**: it did not occur in sessions B and C. Rejected
explanations, kept for the record:
- "processes pinned and then unpinned stay slow";
- core parking (busy-waiting to keep cores awake did not help);
- process priority (ABOVE_NORMAL and HIGH did not help).

The services now opt out of EcoQoS at start-up (`qforecast/core/cpu.py`, `QF_HIGH_QOS=1`
by default).

### 4.4 Real-time pipeline: CPU policy

Replay of 181,500 messages through the real `signal` and `analytics` services on one event
loop:

| Session | All CPUs | P-cores only | Bar → forecast p50, all → P-cores |
|---|---:|---:|---:|
| 2026-10-08 | 18.4k msg/s | 60.0k msg/s (**3.3×**) | 1,901 → 655 µs |
| 2026-10-09 morning | 77.4k msg/s | 102.9k msg/s (1.3×) | 426 → 393 µs |
| 2026-10-09 (this session) | 76.2k msg/s | 77.0k msg/s (1.0×) | 508 → 524 µs |

Restricting the real-time service to P-cores removes a **run-to-run risk** rather than
adding constant speed: when the OS places the event loop on E-cores the penalty reaches
3×, and when it does not there is no difference.

---

## 5. Not measured

| Item | Why not |
|---|---|
| Redis hop between services (distributed mode) | Docker was not running on the dev machine; the Redis bus is tested with `fakeredis` only |
| Network latency from the exchange | exchange/host clock offset makes one-way latency unreliable |
| Linux tuning (`isolcpus`, `SCHED_FIFO`, performance governor) | Windows-only development machine |
| Live inference when the core is cold (1 bar per minute) | observed at ~0.7–0.9 ms p50 vs 0.2–0.3 ms in hot loops; not benchmarked systematically |
