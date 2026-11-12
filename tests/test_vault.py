import json
import time

import numpy as np
import pytest

from qforecast.vault import Vault, VaultError


def test_vault_coin_set_opens_once_and_checks_hash(tmp_path):
    cube = tmp_path / "cube.npz"
    np.savez(cube, a=np.arange(3))
    pre = tmp_path / "prereg.json"
    v = Vault(tmp_path / "reg.json")
    v.seal("coins", str(cube))
    with pytest.raises(VaultError):
        v.open("coins", "test", str(tmp_path / "missing.json"))
    pre.write_text(json.dumps({"hypothesis": "x"}))
    assert v.open("coins", "test", str(pre)) == cube
    with pytest.raises(VaultError):
        v.open("coins", "again", str(pre))
    v.open("coins", "again", str(pre), reopen=True)
    assert v.status()[0]["status"] == "used"


def test_vault_rejects_changed_cube(tmp_path):
    cube = tmp_path / "cube.npz"
    np.savez(cube, a=np.arange(3))
    pre = tmp_path / "prereg.json"
    pre.write_text("{}")
    v = Vault(tmp_path / "reg.json")
    v.seal("coins", str(cube))
    np.savez(cube, a=np.arange(4))
    with pytest.raises(VaultError):
        v.open("coins", "test", str(pre))


def test_forward_vault_waits_for_its_window(tmp_path):
    pre = tmp_path / "prereg.json"
    pre.write_text("{}")
    start = time.time()
    v = Vault(tmp_path / "reg.json", clock=lambda: start)
    v.seal("fwd", start="2030-01-01", end="2030-03-31", min_days=60)
    with pytest.raises(VaultError):
        v.open("fwd", "test", str(pre))
    late = Vault(tmp_path / "reg.json", clock=lambda: 1_900_000_000.0)  # 2030-03-17: end not complete yet
    with pytest.raises(VaultError):
        late.open("fwd", "test", str(pre))
    done = Vault(tmp_path / "reg.json", clock=lambda: 1_910_000_000.0)  # 2030-07
    a, b = done.open("fwd", "test", str(pre))
    assert a < b
