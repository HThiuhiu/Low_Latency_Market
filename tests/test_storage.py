"""Storage formats: QCOL must be lossless; tick logs must round-trip and survive crashes."""

import asyncio
import os
from dataclasses import replace

import numpy as np
import pytest

from qforecast.storage import columnar
from qforecast.storage.ticklog import TickLogWriter, read_dir, read_ticklog


def test_qcol_is_lossless_and_compact(tmp_path):
    rng = np.random.default_rng(0)
    n = 5_000
    t = 1_700_000_000_000 + np.arange(n) * 60_000
    price = np.round(2_000 + np.cumsum(rng.normal(0, 1, n)), 2)  # 2-decimal prices
    vol = np.round(rng.uniform(0, 1_000, n), 4)
    weird = rng.normal(size=n)  # not decimal: must fall back to raw float64
    weird[::7] = np.nan
    m = np.c_[t, price, vol, rng.integers(0, 10_000, n), weird]
    size = columnar.save(tmp_path / "x.qcol", m)
    back = columnar.load(tmp_path / "x.qcol")
    np.testing.assert_array_equal(back, m)  # NaN-aware, bit-exact
    encs = [c["enc"] for c in columnar.describe(tmp_path / "x.qcol")]
    assert encs[-1] == "raw_f64" and encs[1] in ("int", "delta_int")
    assert size < m.nbytes / 2


def test_qcol_serial_and_parallel_decode_agree(tmp_path):
    m = np.round(np.random.default_rng(1).uniform(1, 100, (1_000, 6)), 3)
    columnar.save(tmp_path / "x.qcol", m)
    np.testing.assert_array_equal(columnar.load(tmp_path / "x.qcol", threads=1),
                                  columnar.load(tmp_path / "x.qcol", threads=4))


def test_ticklog_roundtrip_in_order(tmp_path):
    w = TickLogWriter(tmp_path, ["a", "b"], block_bytes=100)  # tiny blocks: many flushes
    sent = [("a" if i % 3 else "b", bytes([i % 256]) * (i % 50)) for i in range(500)]
    for topic, payload in sent:
        if w.append(topic, payload):
            w.write_block(w.take_block())
    w.close()
    assert list(read_dir(tmp_path)) == sent
    assert w.blocks > 10


def test_ticklog_survives_torn_tail(tmp_path):
    w = TickLogWriter(tmp_path, ["a"], block_bytes=64)
    for i in range(100):
        if w.append("a", bytes([i])):
            w.write_block(w.take_block())
    w.close()
    path = w.path
    full = list(read_ticklog(path))
    with open(path, "r+b") as f:  # simulate a crash in the middle of the last block
        f.truncate(os.path.getsize(path) - 5)
    partial = list(read_ticklog(path))
    assert 0 < len(partial) < len(full) and partial == full[:len(partial)]


async def test_recorder_service_captures_bus_traffic(tmp_path):
    from qforecast.bus import MemoryBus
    from qforecast.config import Settings
    from qforecast.core.metrics import Registry
    from qforecast.services import recorder

    s = replace(Settings(), data_dir=tmp_path, record_flush_ms=20, record_block_kb=1)
    bus = MemoryBus()
    task = asyncio.create_task(recorder.run(s, bus, Registry("recorder")))
    await asyncio.sleep(0.05)
    sent = []
    for i in range(300):
        topic = (s.t_book, s.t_trade, s.t_kline)[i % 3]
        payload = f"msg-{i}".encode()
        sent.append((topic, payload))
        await bus.publish(topic, payload)
        if i % 50 == 0:
            await asyncio.sleep(0)
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(read_dir(tmp_path / "ticks" / s.sym)) == sent
