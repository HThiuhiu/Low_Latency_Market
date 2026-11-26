"""Optional compiled hot paths (crates/qf_core, Rust + PyO3, module ``qforecast_core``).

Every accelerated function keeps its pure-Python reference next to it (``_..._py``), and
tests/test_rust_parity.py checks that both give the same numbers. Build and install:

    cd crates/qf_core && maturin build --release && pip install target/wheels/qforecast_core-*.whl

Without the wheel everything runs on the Python reference. ``QF_NO_RUST=1`` forces the Python
reference even when the wheel is installed (debugging, parity checks).
"""

from __future__ import annotations

import os

try:  # pragma: no cover - depends on the local build
    import qforecast_core as _rs
except ImportError:  # pragma: no cover
    _rs = None


def rust():
    """The compiled module, or None when it is not installed or disabled with QF_NO_RUST=1."""
    if _rs is None or os.environ.get("QF_NO_RUST", "") == "1":
        return None
    return _rs


def rust_fn(name: str):
    """One compiled function, or None (not installed, disabled, or an older wheel without it)."""
    rs = rust()
    return getattr(rs, name, None) if rs is not None else None
