"""Run every benchmark and write benchmarks/RESULTS.md.

    python -m benchmarks.run_all
    QF_BENCH_PCORES=0-11 python -m benchmarks.run_all   # hybrid CPUs: list the P-cores

On hybrid (P/E-core) CPUs set QF_BENCH_PCORES: micro-benchmarks then run on the
P-cores only, and the pipeline replay compares all CPUs vs P-cores only vs one
pinned P-core to show the effect of CPU affinity.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from benchmarks import (  # noqa: E402
    bench_features,
    bench_inference,
    bench_io,
    bench_pipeline,
    bench_serialization,
)
from benchmarks.common import machine, parse_cpus  # noqa: E402

SECTIONS = [
    ("Feature engine (per new bar)", bench_features),
    ("Model inference (batch 1)", bench_inference),
    ("Message serialization (encode + decode)", bench_serialization),
    ("I/O: decoding, storage, tick capture", bench_io),
    ("End-to-end replay (signal + analytics services, in-memory bus)", bench_pipeline),
]


def main() -> None:
    pcores = parse_cpus(os.environ.get("QF_BENCH_PCORES", ""))
    note = ""
    if pcores:
        import psutil

        psutil.Process().cpu_affinity(pcores)
        os.environ.setdefault("QF_BENCH_PIN", str(pcores[0]))
        note = (f" Micro-benchmarks run on logical CPUs {pcores[0]}-{pcores[-1]} (P-cores of this "
                f"hybrid CPU); the pipeline replay compares all CPUs, P-cores only and one pinned P-core.")
    parts = [
        "# Benchmark results",
        "",
        f"Machine: {machine()}  ",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')} — reproduce with "
        f"`{'QF_BENCH_PCORES=' + os.environ['QF_BENCH_PCORES'] + ' ' if pcores else ''}"
        f"python -m benchmarks.run_all`.",
        "",
        "Timings are wall-clock `perf_counter_ns` per call; p50/p99 over thousands of runs after "
        "warm-up, GC disabled during the timed loop. Laptop CPU: expect ±20% run-to-run noise." + note,
    ]
    for title, mod in SECTIONS:
        print(f"running: {title}", flush=True)
        parts += ["", f"## {title}", "", (mod.__doc__ or "").strip(), "", mod.run()]
    out = Path(__file__).with_name("RESULTS.md")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
