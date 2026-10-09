import sys

from qforecast.config import Settings
from qforecast.core.cpu import disable_power_throttling, pin_cpus


def test_power_throttling_opt_out_is_safe_everywhere():
    ok = disable_power_throttling()
    if sys.platform == "win32":
        assert ok is True  # the 64-bit handle must be typed correctly or the call fails
    else:
        assert ok is False


def test_pin_cpus_noop_when_empty():
    pin_cpus(())  # must not raise or change affinity


def test_high_qos_is_on_by_default():
    assert Settings().high_qos is True
