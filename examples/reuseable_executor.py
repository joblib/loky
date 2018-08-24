"""
Advanced Executor Setup
=======================

It is possible to reuse an executor even if it has a complicated setup. When
using the parameter `reuse=True`, the executor is resized if needed but the
arguments stay the same.
"""
from time import sleep
import multiprocessing as mp
from loky import get_reusable_executor

INITIALIZER_STATUS = "uninitialized"


def initializer(x):
    global INITIALIZER_STATUS
    print("[{}] init".format(mp.current_process().name))
    INITIALIZER_STATUS = x


def return_initializer_status(delay=0):
    sleep(delay)

    global INITIALIZER_STATUS
    return INITIALIZER_STATUS


executor = get_reusable_executor(
    max_workers=2, initializer=initializer, initargs=('initialized',),
    context="loky", timeout=1000)

assert INITIALIZER_STATUS == "uninitialized"
executor.submit(return_initializer_status).result()
assert executor.submit(return_initializer_status).result() == 'initialized'

# With reuse=True, the executor use the same initializer
executor = get_reusable_executor(max_workers=4, reuse=True)
for x in executor.map(return_initializer_status, [.5] * 4):
    assert x == 'initialized'

# With reuse='auto', the initializer is not used anymore as a new executor
# is created.
executor = get_reusable_executor(max_workers=4)
for x in executor.map(return_initializer_status, [.1] * 4):
    assert x == 'uninitialized'
