"""
Advanced Executor Setup
=======================

It is possible to reuse an executor even if it has a complicated setup. When
using the parameter `reuse=True`, the executor is resized if needed but the
arguments stay the same.
"""
from time import sleep
from loky import get_reusable_executor

INITIALIZER_STATUS = "uninitialized"


def initializer(self, x):
    global INITIALIZER_STATUS
    INITIALIZER_STATUS = x


def test_initializer(self, delay=0):
    print(delay)
    sleep(delay)

    global INITIALIZER_STATUS
    return INITIALIZER_STATUS


executor = get_reusable_executor(
    max_workers=2, initializer=initializer, initargs=('initialized',))

assert executor.submit(test_initializer).result() == 'initialized'

# With reuse=True, the executor use the same initializer
executor = get_reusable_executor(max_workers=4, reuse=True)
for x in executor.map(test_initializer, delay=.5):
    assert x == 'initialized'

# With reuse='auto', the initializer is not used anymore as a new executor is
# created.
executor = get_reusable_executor(max_workers=4)
for x in executor.map(test_initializer, delay=.1):
    assert x == 'uninitialized'
