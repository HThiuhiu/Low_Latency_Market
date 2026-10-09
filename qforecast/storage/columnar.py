"""QCOL: a tiny lossless columnar format for bar history.

Exchange data is decimal with a fixed number of digits ("2430.53", "663.85210000"),
so a float64 column is really a scaled integer column. Each column is stored as

    float64 --(x 10^k, exact)--> int64 --(delta)--> zstd

Prices move a few ticks per bar, so deltas are tiny integers and compress far better
than raw IEEE-754 bytes. ``save`` verifies the round trip bit for bit and falls back to
raw float64 for any column that is not exactly representable, so the format is
lossless by construction. See benchmarks/bench_io.py for size / speed vs .npy and
Parquet.

Layout:  b"QCOL1" | u32 header_len | JSON header | column blobs (zstd frames)
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import zstandard as zstd

MAGIC = b"QCOL1"
MAX_SCALE = 10  # decimal digits tried per column


def _shuffle(a: np.ndarray) -> bytes:
    """Byte-transpose 8-byte values: all low bytes, then all next bytes, ... Small
    integers have long runs of 0x00 / 0xFF in their high bytes, which zstd crushes."""
    return np.ascontiguousarray(a.view(np.uint8).reshape(-1, 8).T).tobytes()


def _unshuffle(data: bytes, dtype) -> np.ndarray:
    return np.ascontiguousarray(np.frombuffer(data, np.uint8).reshape(8, -1).T).view(dtype).ravel()


def _scaled_ints(col: np.ndarray) -> tuple[int, np.ndarray] | None:
    if not np.isfinite(col).all():
        return None
    for k in range(MAX_SCALE + 1):
        scaled = np.rint(col * 10.0**k)
        if np.abs(scaled).max(initial=0) >= 2**53:  # beyond exact float64 integers
            return None
        ints = scaled.astype(np.int64)
        if np.array_equal(ints / 10.0**k if k else ints.astype(np.float64), col):
            return k, ints
    return None


def _encode_column(col: np.ndarray, level: int) -> tuple[dict, bytes]:
    cctx = zstd.ZstdCompressor(level=level)
    candidates = [({"enc": "raw_f64"}, cctx.compress(_shuffle(np.ascontiguousarray(col))))]
    found = _scaled_ints(col)
    if found is not None:
        k, ints = found
        candidates.append(({"enc": "int", "scale": k}, cctx.compress(_shuffle(ints))))
        delta = np.diff(ints, prepend=np.int64(0))
        candidates.append(({"enc": "delta_int", "scale": k}, cctx.compress(_shuffle(delta))))
    return min(candidates, key=lambda c: len(c[1]))  # smallest encoding wins, per column


def _decode_column(meta: dict, blob: bytes, n: int) -> np.ndarray:
    data = zstd.ZstdDecompressor().decompress(blob, max_output_size=n * 8)
    if meta["enc"] == "raw_f64":
        return _unshuffle(data, np.float64)
    ints = _unshuffle(data, np.int64)
    if meta["enc"] == "delta_int":
        ints = np.cumsum(ints)
    k = meta["scale"]
    return ints / 10.0**k if k else ints.astype(np.float64)


def save(path: Path, matrix: np.ndarray, names: list[str] | None = None, level: int = 9) -> int:
    """Write a 2-D float64 matrix column-wise; returns the file size in bytes."""
    matrix = np.asarray(matrix, dtype=np.float64)
    n, m = matrix.shape
    names = names or [f"c{i}" for i in range(m)]
    cols, blobs = [], []
    for j in range(m):
        meta, blob = _encode_column(matrix[:, j], level)
        meta.update(name=names[j], nbytes=len(blob))
        cols.append(meta)
        blobs.append(blob)
    header = json.dumps({"rows": n, "columns": cols}).encode()
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(header)) + header)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)  # atomic: readers never see a half-written file
    return path.stat().st_size


def load(path: Path, threads: int | None = None) -> np.ndarray:
    """Read a QCOL file into an (N, M) float64 array.

    Columns are independent zstd frames, so they are decoded in parallel threads
    (zstd and the numpy kernels release the GIL). The result is column-major
    (Fortran order): every column is one contiguous block, which is how it is decoded
    and how column-wise consumers (feature code) read it.
    """
    buf = memoryview(Path(path).read_bytes())
    if bytes(buf[:5]) != MAGIC:
        raise ValueError(f"{path} is not a QCOL file")
    (hlen,) = struct.unpack_from("<I", buf, 5)
    header = json.loads(bytes(buf[9:9 + hlen]))
    n, cols = header["rows"], header["columns"]
    out_t = np.empty((len(cols), n), dtype=np.float64)  # row j of out_t == column j
    spans, off = [], 9 + hlen
    for meta in cols:
        spans.append((off, off + meta["nbytes"]))
        off += meta["nbytes"]

    def work(j: int) -> None:
        a, b = spans[j]
        out_t[j] = _decode_column(cols[j], buf[a:b], n)

    workers = threads if threads is not None else min(len(cols), os.cpu_count() or 1, 8)
    if workers <= 1:
        for j in range(len(cols)):
            work(j)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, range(len(cols))))
    return out_t.T


def describe(path: Path) -> list[dict]:
    buf = Path(path).read_bytes()
    (hlen,) = struct.unpack_from("<I", buf, 5)
    return json.loads(buf[9:9 + hlen])["columns"]
