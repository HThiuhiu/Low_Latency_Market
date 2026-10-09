"""I/O paths: exchange frame decoding, bar-history storage formats, tick-log capture."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.common import table, timeit  # noqa: E402
from qforecast.exchange.binance import parse_message  # noqa: E402
from qforecast.schemas import BookTicker, Kline, Trade, encode  # noqa: E402
from qforecast.storage import columnar  # noqa: E402
from qforecast.storage.ticklog import TickLogWriter, read_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FRAMES = {
    "bookTicker": b'{"stream":"ethusdt@bookTicker","data":{"u":81859439942,"s":"ETHUSDT",'
                  b'"b":"2433.88000000","B":"7.87880000","a":"2433.89000000","A":"10.30320000"}}',
    "aggTrade": b'{"stream":"ethusdt@aggTrade","data":{"e":"aggTrade","E":1791476760012,'
                b'"s":"ETHUSDT","a":2876150101,"p":"2433.88000000","q":"0.41230000",'
                b'"f":3912345671,"l":3912345673,"T":1791476760010,"m":true,"M":true}}',
    "kline (closed)": b'{"stream":"ethusdt@kline_1m","data":{"e":"kline","E":1791476760012,'
                      b'"s":"ETHUSDT","k":{"t":1791476700000,"T":1791476759999,"s":"ETHUSDT",'
                      b'"i":"1m","f":1,"L":2,"o":"2430.10000000","c":"2430.53000000",'
                      b'"h":"2431.20000000","l":"2429.80000000","v":"512.30000000","n":6421,'
                      b'"x":true,"q":"1245000.10000000","V":"260.10000000","Q":"632000.1","B":"0"}}}',
}


def legacy_parse(raw: bytes, recv_ns: int):
    """The previous implementation: orjson -> dict -> float(str) per field -> struct."""
    msg = orjson.loads(raw)
    stream, d = msg["stream"], msg["data"]
    if stream.endswith("@bookTicker"):
        return BookTicker(d["s"], d["u"], float(d["b"]), float(d["B"]), float(d["a"]),
                          float(d["A"]), recv_ns)
    if stream.endswith("@aggTrade"):
        return Trade(d["s"], float(d["p"]), float(d["q"]), d["m"], d["T"], d["E"], recv_ns)
    k = d["k"]
    return Kline(k["s"], k["t"], k["T"], float(k["o"]), float(k["h"]), float(k["l"]),
                 float(k["c"]), float(k["v"]), k["n"], float(k["V"]), d["E"], recv_ns)


def bench_decode() -> str:
    rows = []
    for name, frame in FRAMES.items():
        assert legacy_parse(frame, 1) == parse_message(frame, 1)
        rows.append((f"{name} — orjson → dict → struct (before)",
                     timeit(lambda f=frame: legacy_parse(f, 1), n=100_000, warmup=10_000)))
        rows.append((f"{name} — msgspec typed decode (now)",
                     timeit(lambda f=frame: parse_message(f, 1), n=100_000, warmup=10_000)))
    return table(rows)


def _bars() -> tuple[np.ndarray, str]:
    for p in (ROOT / "data" / "ETHUSDT_1m.qcol", ROOT / "data" / "ETHUSDT_1m.npy"):
        if p.exists():
            raw = columnar.load(p) if p.suffix == ".qcol" else np.load(p)
            return np.ascontiguousarray(raw), f"real Binance ETHUSDT 1m history ({len(raw):,} bars)"
    from tests.conftest import synthetic_bars

    b = synthetic_bars(100_000)
    return np.round(np.c_[np.arange(len(b)) * 60_000, b, b[:, :4]], 2), "synthetic bars"


def _t(fn, reps: int = 15) -> float:
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def bench_storage() -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw, source = _bars()
    out = [f"Data: {source}, {raw.shape[1]} columns, {raw.nbytes / 1e6:.2f} MB in memory "
           "(float64). Read = file → (N, 11) float64 array, OS file cache warm.", "",
           "| Format | File size | vs .npy | Write (ms) | Read (ms) | Lossless |",
           "|---|---:|---:|---:|---:|:---:|"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        npy = d / "a.npy"
        w = _t(lambda: np.save(npy, raw), 5)
        base = npy.stat().st_size
        rows = [(".npy float64 (before)", base, w, _t(lambda: np.load(npy)),
                 np.array_equal(np.load(npy), raw))]
        rows.append((".npy memory-mapped (lazy, data paged in on access)", base, w,
                     _t(lambda: np.load(npy, mmap_mode="r")), True))
        pqf = d / "a.parquet"
        tbl = pa.table({f"c{i}": raw[:, i] for i in range(raw.shape[1])})
        w = _t(lambda: pq.write_table(tbl, pqf, compression="zstd"), 5)

        def read_pq():
            t = pq.read_table(pqf)
            return np.column_stack([t.column(i).to_numpy() for i in range(t.num_columns)])

        rows.append(("Parquet + zstd (pyarrow)", pqf.stat().st_size, w, _t(read_pq),
                     np.array_equal(read_pq(), raw)))
        qf = d / "a.qcol"
        w = _t(lambda: columnar.save(qf, raw), 3)
        rows.append(("QCOL, 1 thread", qf.stat().st_size, w,
                     _t(lambda: columnar.load(qf, threads=1)),
                     np.array_equal(columnar.load(qf), raw)))
        rows.append(("**QCOL, parallel column decode (now)**", qf.stat().st_size, w,
                     _t(lambda: columnar.load(qf)), True))
    for name, size, w, r, ok in rows:
        out.append(f"| {name} | {size / 1e6:.2f} MB | {base / size:.1f}× smaller | {w:,.0f} | "
                   f"{r:,.2f} | {'yes' if ok else 'NO'} |")
    return "\n".join(out)


def _tick_payloads() -> tuple[list[tuple[str, bytes]], str]:
    tick_dir = ROOT / "data" / "ticks" / "ethusdt"
    if tick_dir.exists() and any(tick_dir.glob("*.qtlg")):
        msgs = list(read_dir(tick_dir))
        if len(msgs) > 1_000:
            return msgs, f"real recorded Binance stream ({len(msgs):,} messages)"
    rng = np.random.default_rng(0)
    msgs, px = [], 2_430.0
    for i in range(20_000):
        px += float(rng.normal(0, 0.05))
        qty = round(float(rng.uniform(0, 20)), 4)
        if i % 5:
            msg = BookTicker("ETHUSDT", 80_000_000_000 + i, round(px, 2), qty, round(px + 0.01, 2), qty / 2)
            msgs.append(("book", encode(msg)))
        else:
            msg = Trade("ETHUSDT", round(px, 2), qty / 10, bool(i % 2), 1_791_476_760_000 + i * 20)
            msgs.append(("trade", encode(msg)))
    return msgs, "synthetic book/trade stream"


def bench_ticklog() -> str:
    msgs, source = _tick_payloads()
    topics = sorted({t for t, _ in msgs})
    payload = sum(len(p) for _, p in msgs)
    with tempfile.TemporaryDirectory() as d:
        w = TickLogWriter(Path(d), topics)
        it = iter(msgs * 3)
        r_append = timeit(lambda: w.append(*next(it)), n=len(msgs) * 2, warmup=len(msgs) // 2)
        w2 = TickLogWriter(Path(d) / "b", topics)  # 256 KB blocks, cut at frame boundaries
        blocks = []
        for t, p in msgs:
            if w2.append(t, p):
                blocks.append(w2.take_block())
        blocks.append(w2.take_block())
        raw_len = sum(len(b) for b in blocks)
        t0 = time.perf_counter()
        for b in blocks:  # what the recorder's I/O thread does
            w2.write_block(b)
        w2.close()
        dt = time.perf_counter() - t0
        disk = w2.written_bytes
        t0 = time.perf_counter()
        n_read = sum(1 for _ in read_dir(Path(d) / "b"))
        rd = time.perf_counter() - t0
        assert n_read == len(msgs)
    json_bytes = 137  # Binance bookTicker JSON frame payload, for scale (bench_serialization)
    return "\n".join([
        f"Data: {source}.", "",
        "| Metric | Value |", "|---|---:|",
        f"| Append one message to the block buffer (event loop) p50 / p99 | "
        f"{r_append['p50_us']:.2f} / {r_append['p99_us']:.2f} µs |",
        f"| Compress + write 256 KB blocks (I/O thread) | {raw_len / dt / 1e6:,.0f} MB/s · "
        f"{len(msgs) / dt:,.0f} msgs/s |",
        f"| Read back + decompress | {len(msgs) / rd:,.0f} msgs/s |",
        f"| Payload (msgpack) per message | {payload / len(msgs):.1f} B |",
        f"| On disk per message | **{disk / len(msgs):.1f} B** ({payload / disk:.1f}× smaller than "
        f"msgpack, ~{json_bytes / (disk / len(msgs)):.0f}× smaller than a JSON book frame) |",
    ])


def run() -> str:
    return "\n\n".join([
        "### Frame decoding (WebSocket JSON → typed struct)", bench_decode(),
        "### Bar-history storage", bench_storage(),
        "### Tick-log capture (recorder)", bench_ticklog(),
    ])


if __name__ == "__main__":
    print(run())
