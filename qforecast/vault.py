# ruff: noqa: E501
"""Clean-test vault: data sets (coin sets or time windows) that no model may look at until a
pre-registered test opens them, once.

    python -m qforecast.vault status
    python -m qforecast.vault seal coins30 data/um/cube_coins30.npz --note "30 coins never used in design"
    python -m qforecast.vault seal fwd_q1 --start 2027-01-01 --end 2027-03-31 --note "forward quarter"

In code:  path = Vault().open("coins30", purpose="final test", prereg="research/prereg/final.json")

Rules enforced:
  * a sealed cube file is hashed; opening it checks the hash (the data did not change after sealing);
  * opening requires an existing pre-registration file; its sha256 is written next to the opening;
  * a set can be opened once; a second opening needs ``reopen=True`` and the set is then marked "used"
    (it stops being a clean test, which the status table shows);
  * a forward (time) vault cannot open before ``min_days`` of data exist after its start; a vault with an ``end``
    opens only after that day and covers [start, end].
The registry is a small JSON file: keep it under version control, it is the audit trail.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import time
from pathlib import Path

REGISTRY = Path("research/vault/registry.json")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())


class VaultError(RuntimeError):
    pass


class Vault:
    def __init__(self, registry: Path = REGISTRY, clock=time.time):
        self.path = Path(registry)
        self.clock = clock
        self.reg = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.reg, indent=1))

    def seal(self, name: str, cube: str | None = None, start: str | None = None, min_days: int = 90, note: str = "",
             end: str | None = None) -> dict:
        if name in self.reg:
            raise VaultError(f"{name} already sealed")
        e = {"sealed": _now(), "note": note, "openings": [], "status": "clean"}
        if cube:
            p = Path(cube)
            e.update({"kind": "coins", "cube": str(p).replace("\\", "/"), "sha256": _sha(p)})
        elif start:
            e.update({"kind": "forward", "start": start, "min_days": int(min_days)})
            if end:
                e["end"] = end
        else:
            raise VaultError("give a cube file or a forward start date")
        self.reg[name] = e
        self._save()
        return e

    def open(self, name: str, purpose: str, prereg: str, reopen: bool = False):
        """Coin vault: the checked cube path. Forward vault: (start_ms, end_ms or None). The opening is logged."""
        if name not in self.reg:
            raise VaultError(f"{name} is not in the vault")
        e = self.reg[name]
        pp = Path(prereg)
        if not pp.exists():
            raise VaultError(f"pre-registration {prereg} not found: write it before opening a clean test")
        if e["openings"] and not reopen:
            raise VaultError(f"{name} was already opened ({e['openings'][0]['purpose']}); pass reopen=True to use it again (it becomes 'used')")
        if e["kind"] == "coins":
            if _sha(Path(e["cube"])) != e["sha256"]:
                raise VaultError(f"{name}: cube file changed after sealing")
            out = Path(e["cube"])
        else:
            start = calendar.timegm(time.strptime(e["start"], "%Y-%m-%d"))
            now = self.clock()
            if now < start + e["min_days"] * 86400:
                raise VaultError(f"{name}: forward window has fewer than {e['min_days']} days of data yet")
            end = calendar.timegm(time.strptime(e["end"], "%Y-%m-%d")) if e.get("end") else None
            if end is not None and now < end + 86400:
                raise VaultError(f"{name}: forward window ends {e['end']}: open after that day is complete")
            out = (start * 1000, None if end is None else (end + 86400) * 1000)
        e["openings"].append({"at": _now(), "purpose": purpose, "prereg": str(pp).replace("\\", "/"), "prereg_sha": _sha(pp)[:12]})
        e["status"] = "used" if len(e["openings"]) > 1 else "opened once"
        self._save()
        return out

    def status(self) -> list[dict]:
        return [{"name": k, "kind": v["kind"], "status": v["status"], "sealed": v["sealed"], "openings": len(v["openings"]),
                 "note": v.get("note", "")} for k, v in self.reg.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    s = sub.add_parser("seal")
    s.add_argument("name")
    s.add_argument("cube", nargs="?")
    s.add_argument("--start")
    s.add_argument("--min-days", type=int, default=90)
    s.add_argument("--note", default="")
    s.add_argument("--end")
    a = ap.parse_args()
    v = Vault()
    if a.cmd == "seal":
        print(json.dumps(v.seal(a.name, a.cube, a.start, a.min_days, a.note, a.end), indent=1))
    else:
        for r in v.status():
            print(f"{r['name']:<16}{r['kind']:<9}{r['status']:<13}{r['openings']:>3} opened  sealed {r['sealed']}  {r['note']}")


if __name__ == "__main__":
    main()
