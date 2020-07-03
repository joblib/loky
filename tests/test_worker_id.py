import os
import time
import pytest
import numpy as np
from collections import defaultdict
from loky import get_reusable_executor, get_worker_id


def random_sleep(k):
    rng = np.random.RandomState(seed=k)
    duration = rng.uniform(0, 0.05)
    t0 = time.time()
    time.sleep(duration)
    t1 = time.time()
    wid = get_worker_id()
    return (wid, t0, t1)


def test_worker_ids():
    """Test that worker IDs are always unique, with re-use over time"""
    num_workers = 4
    executor = get_reusable_executor(max_workers=num_workers, timeout=2)
    results = executor.map(random_sleep, range(100))

    all_intervals = defaultdict(list)
    for wid, t0, t1 in results:
        assert wid in set(range(num_workers))
        all_intervals[wid].append((t0, t1))

    for intervals in all_intervals.values():
        intervals = sorted(intervals)
        for i in range(len(intervals) - 1):
            assert intervals[i + 1][0] >= intervals[i][1]
