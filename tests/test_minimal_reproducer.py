import pytest
from concurrent.futures import ProcessPoolExecutor

import numpy as np  # noqa: F401


def foo(arg):
    return True


@pytest.mark.parametrize("n_proc", [20] * 100)
def test_crash_races(n_proc):
    """Test the race conditions in reusable_executor crash handling"""
    # Test for external crash signal comming from neighbor
    # with various race setup
    executor = ProcessPoolExecutor(max_workers=n_proc)
    executor.map(id, range(n_proc))  # trigger the creation of the workers
    pids = list(executor._processes.keys())
    assert len(pids) == n_proc
    assert None not in pids

    res = executor.map(foo, [j for j in range(2 * n_proc)])
    assert all(list(res))
