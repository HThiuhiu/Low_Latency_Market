"""Append-only, block-compressed tick log (market-data capture for replay/backtests).

File layout (one file per symbol per hour, rotated by the writer):

    b"QTLG1" | u32 header_len | JSON header {"topics": [...], "created": ...}
    block*   = u32 compressed_len | u32 raw_len | zstd frame
    raw block = frame* ; frame = u8 topic_id | u16 payload_len | payload (msgpack)

Design choices
* Messages are appended to an in-memory bytearray (O(1), no syscall per message);
  a block is compressed and written only when it reaches ``block_bytes`` or on a timer.
* Compression + write run on a single background thread (zstd releases the GIL), so
  disk I/O never blocks the event loop; one worker keeps blocks in order.
* Every block is an independent zstd frame: a crash loses at most the block being
  written, and the reader stops cleanly at a torn tail.
* Payloads are the exact bytes published on the bus, so a replay is bit-identical.
"""

from __future__ import annotations

import json
import os
import struct
import time
from collections.abc import Iterator
from pathlib import Path

import zstandard as zstd

MAGIC = b"QTLG1"
_BLOCK_HDR = struct.Struct("<II")
_FRAME_HDR = struct.Struct("<BH")


class TickLogWriter:
    def __init__(self, directory: Path, topics: list[str], block_bytes: int = 256 * 1024,
                 level: int = 3, rotate_s: int = 3600, prefix: str = "ticks"):
        if len(topics) > 255:
            raise ValueError("at most 255 topics")
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.topics = list(topics)
        self.topic_id = {t: i for i, t in enumerate(self.topics)}
        self.block_bytes = block_bytes
        self.rotate_s = rotate_s
        self.prefix = prefix
        self._buf = bytearray()
        self._cctx = zstd.ZstdCompressor(level=level)
        self._file = None
        self._file_slot = -1
        self.path: Path | None = None
        # stats
        self.messages = 0
        self.raw_bytes = 0
        self.written_bytes = 0
        self.blocks = 0

    # ---- event-loop side (cheap) -------------------------------------------
    def append(self, topic: str, payload: bytes) -> bool:
        """Buffer one message. Returns True when the block is full and should be flushed."""
        buf = self._buf
        buf += _FRAME_HDR.pack(self.topic_id[topic], len(payload))
        buf += payload
        self.messages += 1
        return len(buf) >= self.block_bytes

    def take_block(self) -> bytes | None:
        """Hand the current buffer to the writer thread and start a new one."""
        if not self._buf:
            return None
        raw, self._buf = bytes(self._buf), bytearray()
        return raw

    # ---- writer-thread side (compress + syscalls) ---------------------------
    def write_block(self, raw: bytes) -> int:
        self._maybe_rotate()
        comp = self._cctx.compress(raw)
        self._file.write(_BLOCK_HDR.pack(len(comp), len(raw)))
        self._file.write(comp)
        self._file.flush()  # hand to the OS page cache; fsync is left to the OS
        self.raw_bytes += len(raw)
        self.written_bytes += len(comp) + _BLOCK_HDR.size
        self.blocks += 1
        return len(comp)

    def _maybe_rotate(self) -> None:
        slot = int(time.time()) // self.rotate_s
        if slot == self._file_slot and self._file is not None:
            return
        if self._file is not None:
            self._file.close()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(slot * self.rotate_s))
        self.path = self.dir / f"{self.prefix}_{stamp}.qtlg"
        new = not self.path.exists()
        self._file = open(self.path, "ab", buffering=1 << 20)
        if new:
            header = json.dumps({"topics": self.topics, "created": time.time()}).encode()
            self._file.write(MAGIC + struct.pack("<I", len(header)) + header)
        self._file_slot = slot

    def close(self) -> None:
        raw = self.take_block()
        if raw:
            self.write_block(raw)
        if self._file is not None:
            self._file.close()
            self._file = None


def read_ticklog(path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (topic, payload) in write order; stops cleanly at a truncated tail."""
    with open(path, "rb") as f:
        head = f.read(9)
        if head[:5] != MAGIC:
            raise ValueError(f"{path} is not a tick log")
        (hlen,) = struct.unpack("<I", head[5:9])
        topics = json.loads(f.read(hlen))["topics"]
        dctx = zstd.ZstdDecompressor()
        while True:
            hdr = f.read(_BLOCK_HDR.size)
            if len(hdr) < _BLOCK_HDR.size:
                return
            clen, rlen = _BLOCK_HDR.unpack(hdr)
            comp = f.read(clen)
            if len(comp) < clen:  # torn block from a crash: ignore it
                return
            raw = memoryview(dctx.decompress(comp, max_output_size=rlen))
            off = 0
            while off < rlen:
                tid, n = _FRAME_HDR.unpack_from(raw, off)
                off += _FRAME_HDR.size
                yield topics[tid], bytes(raw[off:off + n])
                off += n


def read_dir(directory: Path) -> Iterator[tuple[str, bytes]]:
    for p in sorted(Path(directory).glob("*.qtlg")):
        yield from read_ticklog(p)


def stats(directory: Path) -> dict:
    from collections import Counter

    files = sorted(Path(directory).glob("*.qtlg"))
    counts: Counter = Counter()
    raw = 0
    for topic, payload in read_dir(directory):
        counts[topic] += 1
        raw += len(payload) + _FRAME_HDR.size
    on_disk = sum(os.path.getsize(p) for p in files)
    return {"files": len(files), "messages": dict(counts), "payload_bytes": raw,
            "disk_bytes": on_disk, "ratio": raw / on_disk if on_disk else None,
            "bytes_per_message_on_disk": on_disk / max(sum(counts.values()), 1)}


if __name__ == "__main__":
    import sys

    print(json.dumps(stats(Path(sys.argv[1] if len(sys.argv) > 1 else "data/ticks/ethusdt")), indent=2))
