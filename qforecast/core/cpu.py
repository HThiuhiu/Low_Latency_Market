"""CPU placement helpers: affinity pinning and Windows EcoQoS opt-out."""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("qforecast")


def pin_cpus(cpus: tuple | list) -> None:
    """Restrict the process to ``cpus`` (no-op when empty)."""
    if not cpus:
        return
    if hasattr(os, "sched_setaffinity"):  # Linux
        os.sched_setaffinity(0, set(cpus))
    else:  # Windows / macOS
        import psutil

        psutil.Process().cpu_affinity(list(cpus))
    log.info("pinned to CPUs %s", list(cpus))


def disable_power_throttling() -> bool:
    """Opt the current process out of Windows 11 EcoQoS ("power throttling").

    Windows tags background processes (no foreground window) as efficiency-class, and
    the hybrid-aware scheduler then keeps them on E-cores even when P-cores sit idle.
    On the dev machine this left P-cores 10-27% busy and made a 20-process batch job
    2-3x slower (benchmarks/bench_hetero.py). Setting ControlMask=EXECUTION_SPEED with
    StateMask=0 requests high-QoS scheduling. Returns True on success; no-op elsewhere.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                    ("StateMask", wintypes.ULONG)]

    process_power_throttling, execution_speed = 4, 0x1
    state = _State(1, execution_speed, 0)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Without explicit types ctypes truncates the 64-bit pseudo-handle -> ERROR_INVALID_HANDLE.
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    ok = bool(k32.SetProcessInformation(k32.GetCurrentProcess(), process_power_throttling,
                                        ctypes.byref(state), ctypes.sizeof(state)))
    if not ok:
        log.warning("could not disable power throttling (error %d)", ctypes.get_last_error())
    return ok
