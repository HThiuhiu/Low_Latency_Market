# Before vs after tuning, measured in the same run

Machine: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel · 20 logical CPUs · Windows 11 · Python 3.13.2  
Run: 2026-10-09 16:04 · Data: real ETHUSDT 1-minute bars (Binance cache), 5,808 order-book updates · Reproduce: `QF_BENCH_PCORES=0-11 python -m benchmarks.bench_before_after`

## What is compared

| Stage | Before (prototype approach) | After (tuned) |
|---|---|---|
| Frame decode | `orjson` → dict → `float(str)` → struct | `msgspec` decodes straight into a typed struct |
| Features | pandas recomputes 14 indicators over 500 bars | incremental numba kernel, new bar only |
| Normalise + model input | MinMax over the window, new tensor | z-score into a ring buffer, zero-copy view |
| Model | PyTorch eager, default threads, 2-layer BiLSTM / 64 steps | ONNX Runtime, 1 thread, 1-layer BiLSTM / 32 steps |
| Serialise result | `json.dumps` of a dict | msgpack struct |
| Bar history on disk | `.npy` float64 | QCOL (columnar, compressed, lossless) |
| Tick capture | raw JSON frames | block-compressed tick log |

Method:
- 300 consecutive bars; **both versions process every bar**, alternating which one goes first, so neither always runs second.
- Both versions see the same machine state (thermal, clock frequency), so the **speed-up ratio is trustworthy** even though absolute laptop timings drift between runs.
- Speed-ups are ratios of medians (p50) with a **95% bootstrap confidence interval** (2,000 resamples).
- Model weights are random: latency depends on the architecture, not on weight values.

## Per bar (hot path), all CPUs (20 logical CPUs)

| Stage | Before p50 | After p50 | Before p99 | After p99 | Speed-up (95% CI) |
|---|---:|---:|---:|---:|---:|
| Frame decode | 32.0 µs | 22.4 µs | 115 µs | 63.7 µs | **1.4×** (1.3–1.6) |
| Features | 19,744 µs | 40.9 µs | 36,768 µs | 157 µs | **483.3×** (438.7–524.4) |
| Normalise + model input | 152 µs | 63.6 µs | 411 µs | 208 µs | **2.4×** (2.2–2.6) |
| Model inference | 35,608 µs | 902 µs | 53,216 µs | 1,749 µs | **39.5×** (36.9–42.2) |
| Serialise result | 106 µs | 33.5 µs | 313 µs | 144 µs | **3.2×** (2.9–3.5) |
| **Total per bar** | 56,560 µs | 1,073 µs | 84,295 µs | 2,117 µs | **52.7×** (49.0–56.0) |

Where the time goes, before: frame decode 0% · features 35% · normalise + model input 0% · model inference 63% · serialise result 0%  
Where the time goes, after: frame decode 2% · features 4% · normalise + model input 6% · model inference 84% · serialise result 3%

## Per bar (hot path), P-cores only (12 logical CPUs)

| Stage | Before p50 | After p50 | Before p99 | After p99 | Speed-up (95% CI) |
|---|---:|---:|---:|---:|---:|
| Frame decode | 24.8 µs | 17.4 µs | 142 µs | 64.0 µs | **1.4×** (1.3–1.5) |
| Features | 15,341 µs | 30.9 µs | 24,662 µs | 125 µs | **496.5×** (467.9–526.0) |
| Normalise + model input | 132 µs | 40.6 µs | 472 µs | 147 µs | **3.2×** (3.1–3.4) |
| Model inference | 19,798 µs | 488 µs | 27,297 µs | 1,205 µs | **40.6×** (38.9–42.3) |
| Serialise result | 72.7 µs | 20.6 µs | 224 µs | 75.1 µs | **3.5×** (3.2–3.8) |
| **Total per bar** | 35,218 µs | 604 µs | 51,059 µs | 1,370 µs | **58.3×** (55.5–60.8) |

Where the time goes, before: frame decode 0% · features 44% · normalise + model input 0% · model inference 56% · serialise result 0%  
Where the time goes, after: frame decode 3% · features 5% · normalise + model input 7% · model inference 81% · serialise result 3%

## Per order-book update (decode + serialise for the bus)

| | Before | After | Speed-up (95% CI) |
|---|---:|---:|---:|
| Latency p50 | 4.7 µs | 1.1 µs | **4.3×** (4.3–4.4) |
| Latency p99 | 33.4 µs | 7.8 µs | |
| Throughput (1 thread) | 212,766 msg/s | 909,091 msg/s | |
| Message size on the bus | 132 B (JSON) | 60 B (msgpack) | 2.2× smaller |

## Storage

| | Before | After | Change |
|---|---:|---:|---:|
| Bar history file (130,228 bars) | 11.46 MB (.npy) | 2.88 MB (QCOL) | 4.0× smaller |
| Read bar history | 10.3 ms | 20.8 ms | 0.49× (deliberate trade: slower read, smaller file) |
| Tick capture, per order-book update | 146 B (JSON frame) | 12.7 B (tick log) | 11.5× smaller |

## Summary

| CPU policy | Before, per bar | After, per bar | Speed-up |
|---|---:|---:|---:|
| all CPUs | 56,560 µs | 1,073 µs | **53×** |
| P-cores only | 35,218 µs | 604 µs | **58×** |
| before on all CPUs → after on P-cores | 56,560 µs | 604 µs | **94×** |

## Reading the results

- **Features gain less here than in the micro-benchmark** ([RESULTS.md](RESULTS.md)). The versions are interleaved, so the tuned path runs right after pandas and PyTorch have evicted its data from cache; the micro-benchmark measures a hot loop. These numbers are closer to live operation, where each bar is also interleaved with other work.
- **The speed-up differs between "all CPUs" and "P-cores only" mostly because the *before* version changes**: multi-threaded PyTorch eager benefits from fewer threads, while the single-threaded ONNX Runtime path barely moves.
- **After tuning, inference dominates the remaining time.** That is the next target (keeping the core warm between bars, a smaller model, or O(1) streaming inference).
- **Slower history reads are a deliberate trade**: a few milliseconds at trainer start-up for a ~4× smaller file.
- **Not included:** network latency from the exchange and the Redis hop of the distributed mode (not measurable on this machine).
