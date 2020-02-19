"""
Reuse Executor
==============

This example highlights the ``loky`` API to reuse a ``ReusableProcessPool``.

The factory ``get_reusable_executor`` provides an executor. As long as this
executor is not in a broken state, it is reused for all the computation.
"""
import os
from loky import get_reusable_executor


def func_async(i):
    import os
    pid = os.getpid()
    return (2 * i, pid)


def test_1():
    executor = get_reusable_executor(max_workers=1)
    return executor.submit(func_async, 1)


def test_2():
    executor = get_reusable_executor(max_workers=1)
    return executor.submit(func_async, 2)


def test_3():
    executor = get_reusable_executor(max_workers=1)
    return executor.submit(func_async, 3)


f1 = test_1()
f2 = test_2()
f3 = test_3()

main_pid = os.getpid()
results = [f1.result(), f2.result(), f3.result()]

pids = [pid for _, pid in results]

for i, (val, pid) in enumerate(results):
    assert val == 2 * (i + 1)
    assert pid != main_pid
print("All the jobs were run in a process different from main process")

assert len(set(pids)) == 1
print("All the computation where run in a single `ProcessPoolExecutor` with "
      "worker pid={}".format(pids[0]))
