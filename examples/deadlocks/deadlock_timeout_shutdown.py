"""Deadlock with pickling errors
================================

This example highlights the fact that the ProcessPoolExecutor implementation
from concurrent.futures is not robust to pickling error (at least in versions
3.6 and lower).
"""


import time
from loky import ProcessPoolExecutor
from loky.backend import get_context


class SlowPickle:
    def __init__(self, delay=.1):
        self.delay = delay

    def __reduce__(self):
        time.sleep(self.delay)
        return SlowPickle, (self.delay,)


if __name__ == "__main__":
    ctx = get_context("spawn")
    o = SlowPickle()
    with ProcessPoolExecutor(max_workers=2, timeout=.01, context=ctx) as e:
        f = e.submit(id, SlowPickle())
    f.result()
