# Scheduling bulk work across P-cores and E-cores

Machine: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel · 20 logical CPUs · Windows 11 · Python 3.13.2  
Run: 2026-10-09 16:05 · P-cores: CPU 0–11, E-cores: CPU 12–19 · Reproduce: `python -m benchmarks.bench_hetero`

Question: for a large batch of work, is **turning every core on and splitting the work evenly** worse than a dispatcher that accounts for **P-cores and E-cores running at different speeds**? And how does **pinning + a dynamic queue** compare with **letting Windows schedule**?

## Design

- **20 worker processes**, one per logical CPU (no GIL contention). Process start-up is never timed.
- All participating workers are **released together** by a shared event. The result is the **makespan**: from release until the **last** worker finishes, i.e. the time to finish the whole batch.
- **Same total work** for every strategy.
- Four configuration groups (Windows scheduling / pinned, EcoQoS default / off), each on **fresh processes** every round. 3 rounds, strategy order rotated. Median (min–max) is reported.
- **Idle share** = fraction of worker time spent finished and waiting for the slowest worker.
- **CPU busy P / E** = mean utilisation of the P-core and E-core groups during the run (sampled with `psutil`): it shows whether the OS actually used the P-cores.
- **Speed calibration** is measured with **every worker running at once**, so it includes Hyper-Threading contention (two logical CPUs share one physical P-core).

Two workloads:
- **inference**: ONNX Runtime, 1 thread, each job is one batch of 16 windows (SIMD/AVX2-heavy, like scoring a model over a backtest);
- **ticks**: decode 1,000 Binance frames (msgspec) + market-analytics updates (Python-heavy).

## The 11 strategies

Numbers 1–11 identify the strategies (rows of the result tables). Each one combines three choices.

**How work is split.** Example: 10,000 jobs, 20 workers (12 on P-core logical CPUs, 8 on E-cores), and one P-core worker is 1.4× faster than one E-core worker:
- **Equal split:** every worker gets 10,000 / 20 = 500 jobs regardless of its speed. P-core workers finish early and wait for the E-core workers, so the whole batch waits for the slowest worker.
- **Speed-proportional split:** measure each core's speed first, then assign work in proportion (about 568 jobs per P-core worker, 398 per E-core worker) so everyone finishes together.
- **Dynamic queue:** nothing is assigned up front. Jobs sit in a shared queue and each worker grabs the next chunk when it finishes, so faster workers automatically do more.

**Who places processes on cores:** *Windows scheduling* (the OS may move a process between cores at any time) or *pinned* (each process is fixed to one logical CPU with an affinity mask).

**EcoQoS:** a Windows 11 mechanism that can park background processes on E-cores (see the finding at the end); "EcoQoS off" means each worker opts out with `SetProcessInformation`.

| # | Core placement | EcoQoS off | Work split |
|---:|---|:---:|---|
| 1 | Windows | no | equal ← **baseline**: "turn every core on and split evenly" |
| 2 | Windows | no | dynamic queue |
| 3 | Windows | **yes** | equal ← "every core on", with Windows guaranteed to use the P-cores |
| 4 | Windows | **yes** | dynamic queue |
| 5 | pinned | no | equal |
| 6 | pinned | no | speed-proportional |
| 7 | pinned | no | dynamic queue |
| 8 | pinned | **yes** | equal |
| 9 | pinned | **yes** | speed-proportional |
| 10 | pinned | **yes** | dynamic queue |
| 11 | pinned, **P-cores only** | no | dynamic queue ← reference: what if the E-cores are switched off |

**How to read it:** row 3 vs rows 4, 9 and 10 answers "does smart dispatching beat turning everything on and splitting evenly?". Rows 2/4 vs 7/10 compare **pinned + dynamic queue** with **Windows scheduling + dynamic queue**. Row 1 vs row 3 isolates the EcoQoS effect.

## Workload: inference

Total work: 8,255 batches (= 132,080 windows); the dynamic queue hands out 10 at a time.

**Speed calibration (all workers busy, averaged over rounds):**

| Unit | Throughput | vs one E-core |
|---|---:|---:|
| 1 logical CPU of a P-core (shared with its HT sibling) | 211.0 batches/s | 1.48× |
| 1 physical P-core (= 2 logical CPUs) | 421.9 batches/s | 2.96× |
| 1 E-core (1 logical CPU = 1 core) | 142.7 batches/s | 1.00× |

| Strategy | Runs | Makespan, median (min–max) | Throughput | vs row 1 | vs row 3 | Idle share | CPU busy P / E | Work done P / E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. **Baseline:** all cores, equal split, Windows scheduling (default) | 6 | 2.81 s (2.73–3.48) | 46,963 windows/s | **1.00×** | **1.07×** | 12% | 99% / 99% | — (OS decides) |
| 2. All cores, dynamic queue, Windows scheduling (default) | 6 | 2.58 s (2.49–3.04) | 51,293 windows/s | **1.09×** | **1.17×** | 1% | 100% / 100% | — (OS decides) |
| 3. All cores, equal split, Windows scheduling + **EcoQoS off** | 6 | 3.00 s (2.59–3.58) | 44,013 windows/s | **0.94×** | **1.00×** | 11% | 99% / 100% | — (OS decides) |
| 4. All cores, dynamic queue, Windows scheduling + **EcoQoS off** | 6 | 2.94 s (2.56–3.63) | 44,864 windows/s | **0.96×** | **1.02×** | 1% | 100% / 100% | — (OS decides) |
| 5. Pinned one process per CPU, equal split | 3 | 3.30 s (3.17–3.86) | 40,060 windows/s | **0.85×** | **0.91×** | 27% | 74% / 100% | 60% / 40% |
| 6. Pinned, **speed-proportional split** | 3 | 2.57 s (2.38–2.77) | 51,416 windows/s | **1.09×** | **1.17×** | 9% | 98% / 100% | 69% / 31% |
| 7. Pinned, **dynamic queue** | 3 | 2.61 s (2.58–3.11) | 50,688 windows/s | **1.08×** | **1.15×** | 2% | 100% / 100% | 70% / 30% |
| 8. Pinned, equal split + EcoQoS off | 3 | 3.16 s (3.05–3.31) | 41,792 windows/s | **0.89×** | **0.95×** | 26% | 69% / 100% | 60% / 40% |
| 9. Pinned, **speed-proportional** + EcoQoS off | 3 | 2.87 s (2.66–3.25) | 45,967 windows/s | **0.98×** | **1.04×** | 13% | 95% / 100% | 68% / 32% |
| 10. Pinned, **dynamic queue** + EcoQoS off | 3 | 2.44 s (2.31–2.49) | 54,174 windows/s | **1.15×** | **1.23×** | 1% | 100% / 100% | 72% / 28% |
| 11. P-cores only, dynamic queue (reference) | 3 | 3.40 s (3.12–3.40) | 38,857 windows/s | **0.83×** | **0.88×** | 1% | 100% / 68% | 100% / 0% |

**Every Windows-scheduled run** (rows 1–4):

| Row | EcoQoS off | Makespan | CPU busy P / E |
|---|:---:|---:|---:|
| 1 | no | 2.80 s | 99% / 100% |
| 1 | no | 3.48 s | 99% / 100% |
| 1 | no | 2.84 s | 100% / 100% |
| 1 | no | 2.82 s | 99% / 99% |
| 1 | no | 2.73 s | 96% / 98% |
| 1 | no | 2.77 s | 96% / 97% |
| 2 | no | 3.04 s | 100% / 100% |
| 2 | no | 2.62 s | 100% / 100% |
| 2 | no | 2.54 s | 100% / 100% |
| 2 | no | 2.49 s | 100% / 100% |
| 2 | no | 2.60 s | 100% / 100% |
| 2 | no | 2.55 s | 100% / 100% |
| 3 | yes | 3.58 s | 100% / 100% |
| 3 | yes | 3.33 s | 99% / 99% |
| 3 | yes | 2.67 s | 100% / 100% |
| 3 | yes | 2.59 s | 99% / 100% |
| 3 | yes | 2.61 s | 98% / 94% |
| 3 | yes | 3.38 s | 100% / 100% |
| 4 | yes | 3.63 s | 100% / 100% |
| 4 | yes | 2.93 s | 100% / 100% |
| 4 | yes | 2.56 s | 100% / 100% |
| 4 | yes | 2.56 s | 100% / 100% |
| 4 | yes | 2.95 s | 100% / 100% |
| 4 | yes | 3.05 s | 100% / 100% |

## Workload: ticks

Total work: 13,827 blocks of 1,000 frames (= 13,827,000 frames); the dynamic queue hands out 17 at a time.

**Speed calibration (all workers busy, averaged over rounds):**

| Unit | Throughput | vs one E-core |
|---|---:|---:|
| 1 logical CPU of a P-core (shared with its HT sibling) | 320.5 blocks of 1,000 frames/s | 1.00× |
| 1 physical P-core (= 2 logical CPUs) | 641.0 blocks of 1,000 frames/s | 2.00× |
| 1 E-core (1 logical CPU = 1 core) | 320.5 blocks of 1,000 frames/s | 1.00× |

| Strategy | Runs | Makespan, median (min–max) | Throughput | vs row 1 | vs row 3 | Idle share | CPU busy P / E | Work done P / E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. **Baseline:** all cores, equal split, Windows scheduling (default) | 6 | 2.88 s (2.58–3.81) | 4,798,619 frames/s | **1.00×** | **1.01×** | 9% | 100% / 100% | — (OS decides) |
| 2. All cores, dynamic queue, Windows scheduling (default) | 6 | 2.68 s (2.51–3.27) | 5,161,954 frames/s | **1.08×** | **1.08×** | 1% | 100% / 100% | — (OS decides) |
| 3. All cores, equal split, Windows scheduling + **EcoQoS off** | 6 | 2.90 s (2.42–3.54) | 4,761,881 frames/s | **0.99×** | **1.00×** | 9% | 100% / 100% | — (OS decides) |
| 4. All cores, dynamic queue, Windows scheduling + **EcoQoS off** | 6 | 2.52 s (2.39–3.25) | 5,478,833 frames/s | **1.14×** | **1.15×** | 1% | 100% / 100% | — (OS decides) |
| 5. Pinned one process per CPU, equal split | 3 | 3.26 s (3.07–3.38) | 4,236,031 frames/s | **0.88×** | **0.89×** | 25% | 90% / 96% | 60% / 40% |
| 6. Pinned, **speed-proportional split** | 3 | 2.94 s (2.92–3.28) | 4,706,062 frames/s | **0.98×** | **0.99×** | 15% | 97% / 100% | 61% / 39% |
| 7. Pinned, **dynamic queue** | 3 | 2.64 s (2.44–2.92) | 5,245,600 frames/s | **1.09×** | **1.10×** | 1% | 100% / 100% | 60% / 40% |
| 8. Pinned, equal split + EcoQoS off | 3 | 3.26 s (2.88–3.45) | 4,247,503 frames/s | **0.89×** | **0.89×** | 23% | 84% / 84% | 60% / 40% |
| 9. Pinned, **speed-proportional** + EcoQoS off | 3 | 2.63 s (2.61–3.04) | 5,254,499 frames/s | **1.10×** | **1.10×** | 9% | 98% / 100% | 58% / 42% |
| 10. Pinned, **dynamic queue** + EcoQoS off | 3 | 2.73 s (2.46–3.14) | 5,059,420 frames/s | **1.05×** | **1.06×** | 1% | 100% / 100% | 58% / 42% |
| 11. P-cores only, dynamic queue (reference) | 3 | 4.06 s (3.95–4.36) | 3,406,375 frames/s | **0.71×** | **0.72×** | 1% | 100% / 55% | 100% / 0% |

**Every Windows-scheduled run** (rows 1–4):

| Row | EcoQoS off | Makespan | CPU busy P / E |
|---|:---:|---:|---:|
| 1 | no | 2.87 s | 100% / 100% |
| 1 | no | 2.89 s | 100% / 100% |
| 1 | no | 2.87 s | 98% / 97% |
| 1 | no | 2.58 s | 100% / 100% |
| 1 | no | 3.81 s | 98% / 99% |
| 1 | no | 3.21 s | 99% / 100% |
| 2 | no | 2.64 s | 100% / 100% |
| 2 | no | 2.51 s | 100% / 100% |
| 2 | no | 3.16 s | 100% / 100% |
| 2 | no | 2.60 s | 100% / 100% |
| 2 | no | 3.27 s | 100% / 100% |
| 2 | no | 2.71 s | 100% / 100% |
| 3 | yes | 2.42 s | 100% / 100% |
| 3 | yes | 3.19 s | 100% / 100% |
| 3 | yes | 2.68 s | 100% / 100% |
| 3 | yes | 2.75 s | 100% / 100% |
| 3 | yes | 3.06 s | 100% / 100% |
| 3 | yes | 3.54 s | 95% / 94% |
| 4 | yes | 2.49 s | 100% / 100% |
| 4 | yes | 2.55 s | 100% / 100% |
| 4 | yes | 2.50 s | 100% / 100% |
| 4 | yes | 2.39 s | 100% / 100% |
| 4 | yes | 3.14 s | 100% / 100% |
| 4 | yes | 3.25 s | 100% / 100% |

## Conclusions (computed from this run)

- **inference** (one P-core logical CPU = 1.48× one E-core under full load):
  - **EcoQoS off** alone (row 1 → 3): **0.94×**; median P-core busy 100% (default) → 100% (EcoQoS off).
  - Once Windows uses every core (vs row 3): Windows + dynamic queue 1.02×, pinned + equal 0.95×, pinned + speed-proportional 1.04×, pinned + dynamic queue 1.23×.
  - **Pinned + dynamic vs Windows + dynamic:** 0.99× (default QoS), 1.21× (EcoQoS off).
  - Fastest: 10. Pinned, dynamic queue + EcoQoS off, **1.15×** vs row 1.
- **ticks** (one P-core logical CPU = 1.00× one E-core under full load):
  - **EcoQoS off** alone (row 1 → 3): **0.99×**; median P-core busy 100% (default) → 100% (EcoQoS off).
  - Once Windows uses every core (vs row 3): Windows + dynamic queue 1.15×, pinned + equal 0.89×, pinned + speed-proportional 1.10×, pinned + dynamic queue 1.06×.
  - **Pinned + dynamic vs Windows + dynamic:** 1.02× (default QoS), 0.92× (EcoQoS off).
  - Fastest: 4. All cores, dynamic queue, Windows scheduling + EcoQoS off, **1.14×** vs row 1.

## Reading the results

- **Equal splits create stragglers.** When cores run at different speeds, fast workers finish early and wait for the slowest one; the *idle share* column shows the wasted capacity.
- **Speed-proportional splits** need a calibration step and drift when speeds change mid-run (thermal state, other processes, short calibration windows).
- **A dynamic queue** needs no prior knowledge: whoever finishes takes the next chunk, so it adapts automatically. The cost is one synchronisation per chunk (a shared lock and counter) and a tail of at most one chunk.
- **Windows scheduling** (unpinned) can rebalance by moving processes onto idle cores, but only if it is *allowed* to use the P-cores (see the EcoQoS finding).
- **Hyper-Threading:** with every logical CPU busy, the two threads of a P-core share one physical core, so *per logical CPU* a P-core can be only as fast as, or slower than, an E-core (especially for Python code). Per *physical core* the P-core is clearly faster.
- **P-cores only:** for throughput work, switching the E-cores off is usually slower: they are weaker but still extra capacity.
- **Different from the latency problem:** a real-time thread that must be fast *every time* should avoid E-cores; a batch that must finish *as a whole* should use **every core** with a sensible dispatcher.

## Finding: Windows 11 EcoQoS can park background processes on E-cores

**EcoQoS** (power throttling) lets Windows 11 tag background processes as efficiency-class; the hybrid-aware scheduler then prefers E-cores for them, even when P-cores are idle. A process can opt out with `SetProcessInformation(ProcessPowerThrottling, ControlMask=EXECUTION_SPEED, StateMask=0)`.

**It did not happen in this run:** even under default scheduling the P-cores were 100–100% busy (rows 1–2), so rows 1 and 3 are close. The effect **depends on machine state** (most likely which window has focus, since EcoQoS targets background processes).

**Separate diagnostic run (2026-10-09, inference, 20 unpinned processes, 4 consecutive runs per fresh process group):**

| Configuration | Makespan of the 4 runs | P-core busy |
|---|---|---:|
| Default, group 1 | 3.40 · 6.56 · 6.46 · 6.09 s | 11–27% |
| Default, group 2 | 7.30 · 7.63 · 7.87 · 9.00 s | 15–20% |
| EcoQoS off, group 1 | 2.90 · 2.84 · 2.81 · 2.94 s | 97–100% |
| EcoQoS off, group 2 | 3.31 · 3.17 · 2.96 · 3.02 s | 97–100% |

How the cause was found (kept for the record):
1. *"Processes that were pinned and then unpinned stay slow"*: **wrong**, fresh processes were slow too.
2. *"Core parking"*: **wrong**, keeping every core awake with a busy-wait did not help.
3. *"Process priority"*: ABOVE_NORMAL and HIGH priority classes did **not** help.
4. *"EcoQoS"*: the first attempt seemed to have no effect, but the API call itself had failed (ctypes truncated the 64-bit process handle). Called correctly, the P-cores were used immediately. **Confirmed**, whenever the effect occurs.

Applied to the project: every service calls `disable_power_throttling()` at start-up on Windows (`QF_HIGH_QOS`, on by default; `qforecast/core/cpu.py`). It is harmless when the effect does not occur and keeps the services on the P-cores when it does. Linux has no EcoQoS.
