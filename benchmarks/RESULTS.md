# Benchmark results

Machine: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel · 20 logical CPUs · Windows 11 · Python 3.13.2  
Generated: 2026-10-09 16:03 — reproduce with `QF_BENCH_PCORES=0-11 python -m benchmarks.run_all`.

Timings are wall-clock `perf_counter_ns` per call; p50/p99 over thousands of runs after warm-up, GC disabled during the timed loop. Laptop CPU: expect ±20% run-to-run noise. Micro-benchmarks run on logical CPUs 0-11 (P-cores of this hybrid CPU); the pipeline replay compares all CPUs, P-cores only and one pinned P-core.

## Feature engine (per new bar)

Per-bar feature latency: legacy pandas recompute vs incremental numba kernel.

The original project recomputed every indicator with pandas over the whole history
each time it needed a prediction. The live engine instead advances an O(window)
numba kernel by one bar.

| Variant | p50 (µs) | p99 (µs) | Speed-up (p50) |
|---|---:|---:|---:|
| Legacy: pandas add_features over 500 bars | 5,518.65 | 6,831.08 | 1.0x |
| pandas, same 14 features over 500 bars | 4,981.85 | 7,812.85 | 1.1x |
| numba batch recompute over 500 bars | 113.40 | 277.91 | 48.7x |
| numba incremental update (1 bar, live path) | 3.10 | 3.90 | 1780.2x |

## Model inference (batch 1)

Batch-1 inference latency: PyTorch eager vs TorchScript vs ONNX Runtime.

Model: 158,309 params, input (1, 128, 14); ONNX vs PyTorch max |diff| = 1.2e-07

| Variant | p50 (µs) | p99 (µs) | Speed-up (p50) |
|---|---:|---:|---:|
| ONNX RT, 1 thread — 2-layer BiLSTM / 64 steps | 471.70 | 908.43 | 9.0x |
| PyTorch eager (12 threads) | 4,223.75 | 5,367.50 | 1.0x |
| PyTorch eager (1 threads) | 665.05 | 1,316.40 | 6.4x |
| TorchScript frozen (1 thread) | 535.35 | 1,056.14 | 7.9x |
| ONNX Runtime (12 threads) | 293.00 | 575.13 | 14.4x |
| ONNX Runtime (1 threads) | 180.80 | 384.89 | 23.4x |

## Message serialization (encode + decode)

Wire format: encode + decode round trip of one market-data message.

**BookTicker** — payload bytes: json 137, orjson 124, msgpack (array-like struct) 64

| Variant | p50 (µs) | p99 (µs) | Speed-up (p50) |
|---|---:|---:|---:|
| json (stdlib) dict | 4.50 | 6.90 | 1.0x |
| orjson dict | 0.70 | 0.90 | 6.4x |
| msgspec msgpack typed struct | 0.40 | 0.50 | 11.2x |

**Kline** — payload bytes: json 263, orjson 240, msgpack (array-like struct) 104

| Variant | p50 (µs) | p99 (µs) | Speed-up (p50) |
|---|---:|---:|---:|
| json (stdlib) dict | 6.30 | 10.40 | 1.0x |
| orjson dict | 1.10 | 1.30 | 5.7x |
| msgspec msgpack typed struct | 0.60 | 0.70 | 10.5x |

## I/O: decoding, storage, tick capture

I/O paths: exchange frame decoding, bar-history storage formats, tick-log capture.

### Frame decoding (WebSocket JSON → typed struct)

| Variant | p50 (µs) | p99 (µs) | Speed-up (p50) |
|---|---:|---:|---:|
| bookTicker — orjson → dict → struct (before) | 1.40 | 1.80 | — |
| bookTicker — msgspec typed decode (now) | 0.80 | 0.90 | — |
| aggTrade — orjson → dict → struct (before) | 1.30 | 1.80 | — |
| aggTrade — msgspec typed decode (now) | 1.00 | 1.20 | — |
| kline (closed) — orjson → dict → struct (before) | 2.10 | 2.70 | — |
| kline (closed) — msgspec typed decode (now) | 1.70 | 2.00 | — |

### Bar-history storage

Data: real Binance ETHUSDT 1m history (130,228 bars), 11 columns, 11.46 MB in memory (float64). Read = file → (N, 11) float64 array, OS file cache warm.

| Format | File size | vs .npy | Write (ms) | Read (ms) | Lossless |
|---|---:|---:|---:|---:|:---:|
| .npy float64 (before) | 11.46 MB | 1.0× smaller | 5 | 5.09 | yes |
| .npy memory-mapped (lazy, data paged in on access) | 11.46 MB | 1.0× smaller | 5 | 0.45 | yes |
| Parquet + zstd (pyarrow) | 8.09 MB | 1.4× smaller | 105 | 14.54 | yes |
| QCOL, 1 thread | 2.88 MB | 4.0× smaller | 347 | 17.51 | yes |
| **QCOL, parallel column decode (now)** | 2.88 MB | 4.0× smaller | 347 | 10.63 | yes |

### Tick-log capture (recorder)

Data: real recorded Binance stream (7,273 messages).

| Metric | Value |
|---|---:|
| Append one message to the block buffer (event loop) p50 / p99 | 0.40 / 0.50 µs |
| Compress + write 256 KB blocks (I/O thread) | 132 MB/s · 1,792,218 msgs/s |
| Read back + decompress | 2,584,669 msgs/s |
| Payload (msgpack) per message | 70.9 B |
| On disk per message | **16.2 B** (4.4× smaller than msgpack, ~8× smaller than a JSON book frame) |

## End-to-end replay (signal + analytics services, in-memory bus)

End-to-end replay through the real service code over the in-memory bus.

A synthetic market (bars + ~100 book updates and ~20 trades per bar) is pushed
through the *same* ``signal.run`` and ``analytics.run`` coroutines used in
production; latencies come from the services' own histograms.

- **all CPUs**: 181,500 messages (1500 bars, 100 book updates + 20 trades per bar) in 2.38s → **76,192 msgs/s** on one event loop, 1310 forecasts, 0 dropped.
- **P-cores only**: 181,500 messages (1500 bars, 100 book updates + 20 trades per bar) in 2.36s → **76,972 msgs/s** on one event loop, 1310 forecasts, 0 dropped.
- **pinned to CPU 0**: 181,500 messages (1500 bars, 100 book updates + 20 trades per bar) in 2.95s → **61,606 msgs/s** on one event loop, 1310 forecasts, 0 dropped.

| Stage | all CPUs p50 / p99 (µs) | P-cores only p50 / p99 (µs) | pinned to CPU 0 p50 / p99 (µs) |
|---|---:|---:|---:|
| Feature update (per bar) | 10.75 / 43.01 | 10.75 / 45.06 | 11.78 / 59.39 |
| ONNX inference | 278.53 / 753.66 | 278.53 / 589.82 | 294.91 / 1,572.86 |
| Bar published → forecast published | 507.9 / 1,310.72 | 524.29 / 1,179.65 | 557.06 / 2,883.58 |
| Analytics state update (per msg) | 0.3 / 7.17 | 0.42 / 6.91 | 0.42 / 7.17 |
| Analytics snapshot | 31.74 / 147.46 | 32.77 / 98.3 | 32.77 / 229.38 |
