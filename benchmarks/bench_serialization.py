"""Wire format: encode + decode round trip of one market-data message."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.common import table, timeit  # noqa: E402
from qforecast.schemas import BookTicker, Kline, decode_md, encode  # noqa: E402


def run() -> str:
    out = []
    msgs = {
        "BookTicker": BookTicker("ETHUSDT", 81859439942, 2433.88, 7.8788, 2433.89, 12.5, 123456789),
        "Kline": Kline("ETHUSDT", 1791475440000, 1791475499999, 2429.36, 2432.28, 2426.0, 2429.24,
                       663.8521, 11068, 415.6769, 1791475500012, 123456789),
    }
    for name, m in msgs.items():
        d = {f: getattr(m, f) for f in m.__struct_fields__}
        sizes = {"json": len(json.dumps(d)), "orjson": len(orjson.dumps(d)), "msgpack": len(encode(m))}
        rows = [
            ("json (stdlib) dict", timeit(lambda d=d: json.loads(json.dumps(d)), n=50_000)),
            ("orjson dict", timeit(lambda d=d: orjson.loads(orjson.dumps(d)), n=50_000)),
            ("msgspec msgpack typed struct", timeit(lambda m=m: decode_md(encode(m)), n=50_000)),
        ]
        out.append(f"**{name}** — payload bytes: json {sizes['json']}, orjson {sizes['orjson']}, "
                   f"msgpack (array-like struct) {sizes['msgpack']}\n\n"
                   + table(rows, baseline=rows[0][0]))
    return "\n\n".join(out)


if __name__ == "__main__":
    print(run())
