# ruff: noqa: E501  (markdown report rows are long by nature)
"""High-throughput work on a hybrid CPU: naive "all cores, equal split" vs speed-aware scheduling.

    QF_BENCH_PCORES=0-11 python -m benchmarks.bench_hetero      # writes HETERO.md

Question: with P-cores and E-cores enabled, does dispatching work according to each
core's speed beat simply turning everything on and splitting the work evenly?

Setup
* One persistent worker *process* per logical CPU (no GIL contention), reused by every
  strategy so process start-up is never timed. A shared start event releases all
  participants at once; the result is the makespan (start -> last worker done).
* Two workloads, because the P/E speed ratio depends on the kind of work:
    inference  ONNX Runtime, 1 thread, batch 16 windows (SIMD-heavy, like a backtest)
    ticks      Binance frame decode (msgspec) + market-analytics updates (Python-heavy)
* Speed calibration is measured with *all* workers running at once, so it includes
  Hyper-Threading contention between the two logical CPUs of a P-core.

Strategies (same total work), each configuration on fresh processes every round:
  1-2   Windows scheduling (unpinned), default QoS: equal split / dynamic queue   (1 = baseline)
  3-4   Windows scheduling, EcoQoS disabled: equal split / dynamic queue
  5-7   pinned one worker per CPU: equal / speed-proportional / dynamic queue
  8-10  pinned + EcoQoS disabled: equal / speed-proportional / dynamic queue
  11    P-cores only, dynamic queue (reference: what the E-cores add)

Finding: by default Windows 11 EcoQoS keeps these background worker processes on the
E-cores (P-cores 10-30% busy); opting out with SetProcessInformation fixes it.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEGEND = """## The 11 strategies

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

**How to read it:** row 3 vs rows 4, 9 and 10 answers "does smart dispatching beat turning everything on and splitting evenly?". Rows 2/4 vs 7/10 compare **pinned + dynamic queue** with **Windows scheduling + dynamic queue**. Row 1 vs row 3 isolates the EcoQoS effect."""

BATCH = 16
OS_REPEATS = 2  # OS-scheduled strategies are sampled twice per round
TARGET_SECONDS = 2.0  # total work is sized so the best strategy takes about this long
ROUNDS = 3


# --------------------------------------------------------------------------- workloads
def _build_workload(name: str, model_path: str, seed: int):
    import numpy as np

    if name == "inference":
        from qforecast.model.predictor import make_session

        sess = make_session(Path(model_path), threads=1)
        rng = np.random.default_rng(seed)
        pool = rng.normal(size=(8, BATCH, 128, 14)).astype(np.float32)

        def run(i: int) -> None:
            sess.run(None, {"x": pool[i % 8]})

        return run

    if name == "ticks":
        import orjson

        from qforecast.core.analysis import MarketAnalyzer
        from qforecast.exchange.binance import parse_message
        from qforecast.schemas import BookTicker, Trade

        rng = np.random.default_rng(seed)
        frames, px = [], 2_430.0
        for j in range(1_000):
            px += float(rng.normal(0, 0.05))
            if j % 5:
                d = {"u": 80_000_000_000 + j, "s": "ETHUSDT", "b": f"{px:.8f}",
                     "B": f"{rng.uniform(0, 20):.8f}", "a": f"{px + 0.01:.8f}", "A": f"{rng.uniform(0, 20):.8f}"}
                frames.append(orjson.dumps({"stream": "ethusdt@bookTicker", "data": d}))
            else:
                d = {"e": "aggTrade", "E": 1_791_476_760_000 + j, "s": "ETHUSDT", "a": j, "p": f"{px:.8f}",
                     "q": f"{rng.uniform(0, 2):.8f}", "f": j, "l": j, "T": 1_791_476_760_000 + j, "m": bool(j % 2)}
                frames.append(orjson.dumps({"stream": "ethusdt@aggTrade", "data": d}))
        an = MarketAnalyzer("ETHUSDT")
        handlers = {BookTicker: an.on_book, Trade: an.on_trade}

        def run(i: int) -> None:  # one item = 1,000 frames
            for j, f in enumerate(frames):
                m = parse_message(f, 0)
                handlers[type(m)](m)
                if j % 250 == 249:
                    an.snapshot()

        return run

    raise ValueError(name)


def _worker(wid: int, cpu: int, conn, start_evt, counter, lock, model_path: str) -> None:
    import psutil

    proc = psutil.Process()
    all_cpus = list(range(os.cpu_count()))
    workloads: dict = {}
    conn.send(("up", wid))
    while True:
        cmd = conn.recv()
        if cmd["op"] == "stop":
            return
        if cmd.get("highqos"):
            from qforecast.core.cpu import disable_power_throttling

            disable_power_throttling()
        if cmd.get("priority"):
            proc.nice(getattr(psutil, cmd["priority"]))
        proc.cpu_affinity([cpu] if cmd["pin"] else all_cpus)
        if cmd["workload"] not in workloads:
            workloads[cmd["workload"]] = _build_workload(cmd["workload"], model_path, wid)
        run = workloads[cmd["workload"]]
        for i in range(2):  # warm caches / lazy init on this core
            run(i)
        conn.send(("ready", wid))
        if cmd.get("spin"):  # keep this core awake (busy-wait) until the start signal
            while not start_evt.is_set():
                pass
        else:
            start_evt.wait()
        t0 = time.perf_counter_ns()
        done = 0
        if cmd["mode"] == "static":
            for i in range(cmd["n"]):
                run(i)
            done = cmd["n"]
        else:  # dynamic: fetch-and-add on a shared counter, chunk by chunk
            k, total = cmd["chunk"], cmd["total"]
            while True:
                with lock:
                    s = counter.value
                    if s >= total:
                        break
                    counter.value = s + k
                for i in range(s, min(s + k, total)):
                    run(i)
                    done += 1
        conn.send(("done", wid, t0, time.perf_counter_ns(), done))


# --------------------------------------------------------------------------- orchestration
class Pool:
    def __init__(self, model_path: str):
        ctx = mp.get_context("spawn")
        self.n = os.cpu_count()
        self.start = ctx.Event()
        self.counter = ctx.Value("q", 0, lock=False)
        self.lock = ctx.Lock()
        self.conns, self.procs = [], []
        for cpu in range(self.n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(cpu, cpu, b, self.start, self.counter, self.lock, model_path),
                            daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        for c in self.conns:
            assert c.recv()[0] == "up"

    def run(self, workload: str, participants: list[int], pin: bool, mode: str, shares=None,
            total: int = 0, chunk: int = 1, highqos: bool = False, spin: bool = False,
            priority: str | None = None) -> dict:
        self.start.clear()
        self.counter.value = 0
        for w in participants:
            cmd = {"op": "run", "workload": workload, "pin": pin, "mode": mode, "highqos": highqos, "spin": spin,
                   "priority": priority}
            if mode == "static":
                cmd["n"] = shares[w]
            else:
                cmd.update(total=total, chunk=chunk)
            self.conns[w].send(cmd)
        for w in participants:
            assert self.conns[w].recv()[0] == "ready"
        import threading

        import psutil

        samples: list[list[float]] = []
        stop = threading.Event()

        def sampler() -> None:  # per-CPU utilisation while the run is in flight
            psutil.cpu_percent(percpu=True)
            while not stop.wait(0.25):
                samples.append(psutil.cpu_percent(percpu=True))

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        t_start = time.perf_counter_ns()
        self.start.set()
        res = {}
        for w in participants:
            _, wid, t0, t1, done = self.conns[w].recv()
            res[wid] = (t0, t1, done)
        stop.set()
        th.join()
        makespan = (max(t1 for _, t1, _ in res.values()) - t_start) / 1e9
        busy = {w: (t1 - t0) / 1e9 for w, (t0, t1, _) in res.items()}
        mid = samples[1:-1] or samples or [[0.0] * self.n]
        util = [statistics.mean(s[c] for s in mid) for c in range(self.n)]
        return {"makespan": makespan, "busy": busy, "done": {w: d for w, (_, _, d) in res.items()},
                "util": util}

    def close(self) -> None:
        for c in self.conns:
            c.send({"op": "stop"})
        for p in self.procs:
            p.join(timeout=5)


def split(total: int, weights: dict[int, float]) -> dict[int, int]:
    """Largest-remainder apportionment of `total` items by weight."""
    s = sum(weights.values())
    raw = {w: total * v / s for w, v in weights.items()}
    out = {w: int(x) for w, x in raw.items()}
    for w in sorted(raw, key=lambda w: raw[w] - out[w], reverse=True)[: total - sum(out.values())]:
        out[w] += 1
    return out


def make_model(path: Path) -> None:
    import torch

    from qforecast.core.features import N_FEATURES
    from qforecast.model.net import build_model

    torch.manual_seed(0)
    net = build_model("lstm", N_FEATURES, 5).eval()
    torch.onnx.export(net, (torch.zeros(BATCH, 128, N_FEATURES),), str(path), input_names=["x"],
                      output_names=["y"], opset_version=17, dynamo=False)


# --------------------------------------------------------------------------- main
def calibrate(pool: Pool, workload: str, workers: list[int]) -> dict[int, float]:
    """Per-worker speed with *every* worker pinned and busy at once (includes HT contention)."""
    probe = 40 if workload == "inference" else 6
    rates = {w: 0.0 for w in workers}
    for _ in range(2):
        r = pool.run(workload, workers, True, "static", shares={w: probe for w in workers})
        for w in workers:
            rates[w] += probe / r["busy"][w] / 2
    return rates


def rotate(items: list, k: int) -> list:
    k %= len(items)
    return items[k:] + items[:k]


def main() -> None:
    import warnings
    from collections import defaultdict

    warnings.filterwarnings("ignore")
    from benchmarks.common import machine, parse_cpus

    n = os.cpu_count()
    allw = list(range(n))
    pcores = parse_cpus(os.environ.get("QF_BENCH_PCORES", "0-11"))
    ecores = [c for c in allw if c not in pcores]
    tmp = Path(tempfile.mkdtemp())
    model = str(tmp / "m.onnx")
    make_model(Path(model))

    R = {
        1: "1. **Baseline:** all cores, equal split, Windows scheduling (default)",
        2: "2. All cores, dynamic queue, Windows scheduling (default)",
        3: "3. All cores, equal split, Windows scheduling + **EcoQoS off**",
        4: "4. All cores, dynamic queue, Windows scheduling + **EcoQoS off**",
        5: "5. Pinned one process per CPU, equal split",
        6: "6. Pinned, **speed-proportional split**",
        7: "7. Pinned, **dynamic queue**",
        8: "8. Pinned, equal split + EcoQoS off",
        9: "9. Pinned, **speed-proportional** + EcoQoS off",
        10: "10. Pinned, **dynamic queue** + EcoQoS off",
        11: "11. P-cores only, dynamic queue (reference)",
    }
    OS_ROWS, PINNED_ROWS = (1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11)

    lines = ["# Scheduling bulk work across P-cores and E-cores", "",
             f"Machine: {machine()}  ",
             f"Run: {time.strftime('%Y-%m-%d %H:%M')} · P-cores: CPU {pcores[0]}–{pcores[-1]}, "
             f"E-cores: CPU {ecores[0]}–{ecores[-1]} · Reproduce: `python -m benchmarks.bench_hetero`", "",
             "Question: for a large batch of work, is **turning every core on and splitting the work "
             "evenly** worse than a dispatcher that accounts for **P-cores and E-cores running at "
             "different speeds**? And how does **pinning + a dynamic queue** compare with **letting "
             "Windows schedule**?", "",
             "## Design", "",
             f"- **{n} worker processes**, one per logical CPU (no GIL contention). Process start-up "
             "is never timed.",
             "- All participating workers are **released together** by a shared event. The result is "
             "the **makespan**: from release until the **last** worker finishes, i.e. the time to "
             "finish the whole batch.",
             "- **Same total work** for every strategy.",
             "- Four configuration groups (Windows scheduling / pinned, EcoQoS default / off), each on "
             f"**fresh processes** every round. {ROUNDS} rounds, strategy order rotated. Median "
             "(min–max) is reported.",
             "- **Idle share** = fraction of worker time spent finished and waiting for the slowest "
             "worker.",
             "- **CPU busy P / E** = mean utilisation of the P-core and E-core groups during the run "
             "(sampled with `psutil`): it shows whether the OS actually used the P-cores.",
             "- **Speed calibration** is measured with **every worker running at once**, so it "
             "includes Hyper-Threading contention (two logical CPUs share one physical P-core).",
             "", "Two workloads:",
             f"- **inference**: ONNX Runtime, 1 thread, each job is one batch of {BATCH} windows "
             "(SIMD/AVX2-heavy, like scoring a model over a backtest);",
             "- **ticks**: decode 1,000 Binance frames (msgspec) + market-analytics updates "
             "(Python-heavy).",
             ]
    lines += ["", LEGEND]
    summary = []
    for workload in ("inference", "ticks"):
        print(f"[{workload}] sizing", flush=True)
        pool = Pool(model)
        rates0 = calibrate(pool, workload, allw)
        pool.close()
        total = int(sum(rates0.values()) * TARGET_SECONDS)
        chunk = max(1, total // (n * 40))  # ~40 chunks per worker
        equal = split(total, {w: 1 for w in allw})
        results: dict[int, list] = defaultdict(list)
        calib = []

        def run(pool, row, parts, pin, mode, shares, hq,
                _res=results, _wl=workload, _total=total, _chunk=chunk):
            _res[row].append(pool.run(_wl, parts, pin, mode, shares=shares, total=_total,
                                      chunk=_chunk, highqos=hq))

        for rnd in range(ROUNDS):
            print(f"[{workload}] round {rnd + 1}/{ROUNDS}", flush=True)
            for hq, rows in ((False, (1, 2)), (True, (3, 4))):  # Windows scheduling
                pool = Pool(model)
                for _ in range(OS_REPEATS):
                    for row in rotate(list(rows), rnd):
                        run(pool, row, allw, False, "static" if row in (1, 3) else "dynamic", equal, hq)
                pool.close()
            for hq, (r_eq, r_prop, r_dyn) in ((False, (5, 6, 7)), (True, (8, 9, 10))):  # pinned
                pool = Pool(model)
                if hq:  # opt out before calibrating so shares reflect high-QoS speeds
                    pool.run(workload, allw, True, "static", shares={w: 1 for w in allw}, highqos=True)
                rates = calibrate(pool, workload, allw)
                calib.append(rates)
                prop = split(total, rates)
                jobs = [(r_eq, allw, "static", equal), (r_prop, allw, "static", prop), (r_dyn, allw, "dynamic", None)]
                if not hq:
                    jobs.append((11, pcores, "dynamic", None))
                for row, parts, mode, shares in rotate(jobs, rnd):
                    run(pool, row, parts, True, mode, shares, hq)
                pool.close()

        p_rate = statistics.mean(statistics.mean(c[w] for w in pcores) for c in calib)
        e_rate = statistics.mean(statistics.mean(c[w] for w in ecores) for c in calib)
        unit = "batches" if workload == "inference" else "blocks of 1,000 frames"
        what = "windows" if workload == "inference" else "frames"
        per_item = BATCH if workload == "inference" else 1_000
        med = {r: statistics.median(x["makespan"] for x in v) for r, v in results.items()}
        lines += ["", f"## Workload: {workload}", "",
                  f"Total work: {total:,} {unit} (= {total * per_item:,} {what}); the dynamic queue hands "
                  f"out {chunk} at a time.", "",
                  "**Speed calibration (all workers busy, averaged over rounds):**", "",
                  "| Unit | Throughput | vs one E-core |", "|---|---:|---:|",
                  f"| 1 logical CPU of a P-core (shared with its HT sibling) | {p_rate:,.1f} {unit}/s | {p_rate / e_rate:.2f}× |",
                  f"| 1 physical P-core (= 2 logical CPUs) | {2 * p_rate:,.1f} {unit}/s | {2 * p_rate / e_rate:.2f}× |",
                  f"| 1 E-core (1 logical CPU = 1 core) | {e_rate:,.1f} {unit}/s | 1.00× |", "",
                  "| Strategy | Runs | Makespan, median (min–max) | Throughput | vs row 1 | vs row 3 | Idle share | CPU busy P / E | Work done P / E |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in OS_ROWS + PINNED_ROWS:
            runs = results[row]
            parts = pcores if row == 11 else allw
            ms = [x["makespan"] for x in runs]
            idle = statistics.median(1 - sum(x["busy"].values()) / (len(parts) * x["makespan"]) for x in runs)
            up = statistics.median(statistics.mean(x["util"][c] for c in pcores) for x in runs)
            ue = statistics.median(statistics.mean(x["util"][c] for c in ecores) for x in runs)
            if row in PINNED_ROWS:
                ps = statistics.median(sum(x["done"].get(w, 0) for w in pcores) / total * 100 for x in runs)
                share = f"{ps:.0f}% / {100 - ps:.0f}%"
            else:
                share = "— (OS decides)"
            lines.append(f"| {R[row]} | {len(runs)} | {med[row]:.2f} s ({min(ms):.2f}–{max(ms):.2f}) | "
                         f"{total * per_item / med[row]:,.0f} {what}/s | **{med[1] / med[row]:.2f}×** | "
                         f"**{med[3] / med[row]:.2f}×** | {idle * 100:.0f}% | {up:.0f}% / {ue:.0f}% | {share} |")
        lines += ["", "**Every Windows-scheduled run** (rows 1–4):", "",
                  "| Row | EcoQoS off | Makespan | CPU busy P / E |", "|---|:---:|---:|---:|"]
        for row in OS_ROWS:
            for x in results[row]:
                up = statistics.mean(x["util"][c] for c in pcores)
                ue = statistics.mean(x["util"][c] for c in ecores)
                lines.append(f"| {row} | {'yes' if row in (3, 4) else 'no'} | {x['makespan']:.2f} s | {up:.0f}% / {ue:.0f}% |")
        up_def = statistics.median(statistics.mean(x["util"][c] for c in pcores) for r in (1, 2) for x in results[r])
        up_hq = statistics.median(statistics.mean(x["util"][c] for c in pcores) for r in (3, 4) for x in results[r])
        summary.append((workload, p_rate / e_rate, med, up_def, up_hq))

    lines += ["", "## Conclusions (computed from this run)", ""]
    for workload, ratio, med, up_def, up_hq in summary:
        best = min((r for r in R if r != 11), key=lambda r: med[r])
        lines.append(
            f"- **{workload}** (one P-core logical CPU = {ratio:.2f}× one E-core under full load):\n"
            f"  - **EcoQoS off** alone (row 1 → 3): **{med[1] / med[3]:.2f}×**; median P-core busy "
            f"{up_def:.0f}% (default) → {up_hq:.0f}% (EcoQoS off).\n"
            f"  - Once Windows uses every core (vs row 3): Windows + dynamic queue {med[3] / med[4]:.2f}×, "
            f"pinned + equal {med[3] / med[8]:.2f}×, pinned + speed-proportional {med[3] / med[9]:.2f}×, "
            f"pinned + dynamic queue {med[3] / med[10]:.2f}×.\n"
            f"  - **Pinned + dynamic vs Windows + dynamic:** {med[2] / med[7]:.2f}× (default QoS), "
            f"{med[4] / med[10]:.2f}× (EcoQoS off).\n"
            f"  - Fastest: {R[best].replace('**', '')}, **{med[1] / med[best]:.2f}×** vs row 1.")
    lines += ["", "## Reading the results", "",
              "- **Equal splits create stragglers.** When cores run at different speeds, fast workers "
              "finish early and wait for the slowest one; the *idle share* column shows the wasted "
              "capacity.",
              "- **Speed-proportional splits** need a calibration step and drift when speeds change "
              "mid-run (thermal state, other processes, short calibration windows).",
              "- **A dynamic queue** needs no prior knowledge: whoever finishes takes the next chunk, so "
              "it adapts automatically. The cost is one synchronisation per chunk (a shared lock and "
              "counter) and a tail of at most one chunk.",
              "- **Windows scheduling** (unpinned) can rebalance by moving processes onto idle cores, "
              "but only if it is *allowed* to use the P-cores (see the EcoQoS finding).",
              "- **Hyper-Threading:** with every logical CPU busy, the two threads of a P-core share one "
              "physical core, so *per logical CPU* a P-core can be only as fast as, or slower than, an "
              "E-core (especially for Python code). Per *physical core* the P-core is clearly faster.",
              "- **P-cores only:** for throughput work, switching the E-cores off is usually slower: "
              "they are weaker but still extra capacity.",
              "- **Different from the latency problem:** a real-time thread that must be fast *every "
              "time* should avoid E-cores; a batch that must finish *as a whole* should use **every "
              "core** with a sensible dispatcher.",
              ]
    up_def_max = max(x[3] for x in summary)
    observed = up_def_max < 60  # P-cores mostly idle under default scheduling in this run
    lines += ["", "## Finding: Windows 11 EcoQoS can park background processes on E-cores", "",
              "**EcoQoS** (power throttling) lets Windows 11 tag background processes as "
              "efficiency-class; the hybrid-aware scheduler then prefers E-cores for them, even when "
              "P-cores are idle. A process can opt out with `SetProcessInformation(ProcessPowerThrottling, "
              "ControlMask=EXECUTION_SPEED, StateMask=0)`.",
              "",
              (f"**It happened in this run:** under default scheduling the P-cores were only about "
               f"{up_def_max:.0f}% busy (rows 1–2), while with EcoQoS off they were fully used (rows 3–4)."
               if observed else
               f"**It did not happen in this run:** even under default scheduling the P-cores were "
               f"{min(x[3] for x in summary):.0f}–{up_def_max:.0f}% busy (rows 1–2), so rows 1 and 3 are "
               "close. The effect **depends on machine state** (most likely which window has focus, "
               "since EcoQoS targets background processes)."),
              "",
              "**Separate diagnostic run (2026-10-09, inference, 20 unpinned processes, 4 consecutive "
              "runs per fresh process group):**",
              "",
              "| Configuration | Makespan of the 4 runs | P-core busy |",
              "|---|---|---:|",
              "| Default, group 1 | 3.40 · 6.56 · 6.46 · 6.09 s | 11–27% |",
              "| Default, group 2 | 7.30 · 7.63 · 7.87 · 9.00 s | 15–20% |",
              "| EcoQoS off, group 1 | 2.90 · 2.84 · 2.81 · 2.94 s | 97–100% |",
              "| EcoQoS off, group 2 | 3.31 · 3.17 · 2.96 · 3.02 s | 97–100% |",
              "",
              "How the cause was found (kept for the record):",
              "1. *\"Processes that were pinned and then unpinned stay slow\"*: **wrong**, fresh "
              "processes were slow too.",
              "2. *\"Core parking\"*: **wrong**, keeping every core awake with a busy-wait did not help.",
              "3. *\"Process priority\"*: ABOVE_NORMAL and HIGH priority classes did **not** help.",
              "4. *\"EcoQoS\"*: the first attempt seemed to have no effect, but the API call itself had "
              "failed (ctypes truncated the 64-bit process handle). Called correctly, the P-cores were "
              "used immediately. **Confirmed**, whenever the effect occurs.",
              "",
              "Applied to the project: every service calls `disable_power_throttling()` at start-up on "
              "Windows (`QF_HIGH_QOS`, on by default; `qforecast/core/cpu.py`). It is harmless when the "
              "effect does not occur and keeps the services on the P-cores when it does. Linux has no "
              "EcoQoS."]
    out = Path(__file__).with_name("HETERO.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
