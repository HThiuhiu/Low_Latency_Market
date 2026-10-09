import numpy as np
import pytest

from qforecast.core.ringbuffer import RingBuffer


def test_last_is_contiguous_and_ordered():
    rb = RingBuffer(capacity=5, width=2)
    for i in range(13):
        rb.push([i, -i])
        n = min(i + 1, 5)
        view = rb.last(n)
        assert view.flags.c_contiguous
        np.testing.assert_array_equal(view[:, 0], np.arange(i - n + 1, i + 1))
    assert len(rb) == 5 and rb.count == 13


def test_view_is_zero_copy_and_read_only():
    rb = RingBuffer(capacity=4, width=1)
    for i in range(6):
        rb.push([i])
    v = rb.last(3)
    assert np.shares_memory(v, rb._buf)
    with pytest.raises(ValueError):
        v[0, 0] = 99


def test_overwrite_last_and_bounds():
    rb = RingBuffer(capacity=3, width=1)
    rb.push([1])
    rb.push([2])
    rb.overwrite_last([7])
    np.testing.assert_array_equal(rb.last(2)[:, 0], [1, 7])
    with pytest.raises(ValueError):
        rb.last(3)
