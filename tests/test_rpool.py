from __future__ import print_function
import os
import sys
import psutil
from time import sleep
from nose.tools import assert_raises
from backend.reusable_pool import get_reusable_pool, AbortedWorkerError


def wait_dead(pid, n_tries=1000, delay=0.001):
    """Wait for process to die"""
    for i in range(n_tries):
        if not psutil.pid_exists(pid):
            return
        sleep(delay)
    raise RuntimeError("Process %d failed to die for at least %0.3fs" %
                       (pid, delay * n_tries))


def crash():
    '''Induce a segfault in process
    '''
    import ctypes
    i = ctypes.c_char(b'a')
    j = ctypes.pointer(i)
    c = 0
    while True:
        j[c] = i
        c += 1
    j


def exit():
    '''Induce a sys exit in process
    '''
    sys.exit(1)


def work_sleep(arg):
    """Sleep for some time before returning
    and check if all the passed pid exist"""
    time, pids = arg
    sleep(time)
    res = True
    for p in pids:
        res &= psutil.pid_exists(p)
    return res


def kill_friend(pid):
    '''Try to kill the process associated to pid
    '''
    os.kill(pid, 9)


def raise_Error():
    '''Raise an Exception in process
    '''
    raise RuntimeError('bad except')


class CrashAtUnpickle(object):
    """Bad object that triggers a segfault at unpickling time."""
    def __reduce__(self):
        return crash, (), ()


def test_Rpool_crash():
    '''Test the crash handling in pool
    '''

    for func, err in [(crash, AbortedWorkerError),
                      (exit, AbortedWorkerError),
                      (raise_Error, RuntimeError)]:
        pool = get_reusable_pool(processes=2)
        res = pool.apply_async(func, tuple())
        assert_raises(err, res.get)

    pool = get_reusable_pool(processes=2)
    res = pool.apply_async(id, CrashAtUnpickle())
    assert_raises(AbortedWorkerError, res.get)

    # Test for external signal comming from neighbor
    for i in [1, 2, 5, 17]:
        pool = get_reusable_pool(processes=i)
        pids = [p.pid for p in pool._pool]
        assert len(pids) == i
        assert None not in pids
        res = pool.map(work_sleep, [(.001 * j, pids) for j in range(2 * i)])
        assert all(res)
        res = pool.map_async(kill_friend, pids[::-1])
        assert_raises(AbortedWorkerError, res.get)

        pool = get_reusable_pool(processes=i)
        pids = [p.pid for p in pool._pool]
        res = pool.imap(work_sleep, [(.001 * j, pids) for j in range(2 * i)])
        assert all(list(res))
        res = pool.imap(kill_friend, pids[::-1])
        assert_raises(AbortedWorkerError, list, res)

    pool = get_reusable_pool(processes=1)
    pool.terminate()


def test_Rpool_resize():
    '''Test the resize function in reusable_pool
    '''

    pool = get_reusable_pool(processes=2)

    # Decreasing the pool should drop a single process and keep one of the
    # old one as it is still in a good shape. The resize should not occur
    # while there are on going works.
    pids = [p.pid for p in pool._pool]
    res = pool.apply_async(work_sleep, ((.5, pids),))
    pool = get_reusable_pool(processes=1)
    assert res.get(), "Resize should wait for current processes to finish"
    assert len(pool._pool) == 1
    assert pool._pool[0].pid in pids

    # Requesting the same number of process should not impact the pool nor
    # kill the processed
    old_pid = pool._pool[0].pid
    unchanged_pool = get_reusable_pool(processes=1)
    assert len(unchanged_pool._pool) == 1
    assert unchanged_pool is pool
    assert unchanged_pool._pool[0].pid == old_pid

    # Growing the pool again should add a single process and keep the old
    # one as it is still in a good shape
    pool = get_reusable_pool(processes=2)
    assert len(pool._pool) == 2
    assert old_pid in [p.pid for p in pool._pool]

    pool.terminate()


def test_invalid_process_number():
    assert_raises(ValueError, get_reusable_pool, processes=0)
    assert_raises(ValueError, get_reusable_pool, processes=-1)


def test_deadlock_kill():
    """Create a deadlock in pool by killing the lock owners."""
    pool = get_reusable_pool(processes=1)
    pid = pool._pool[0].pid
    pool = get_reusable_pool(processes=2)
    os.kill(pid, 9)
    wait_dead(pid)

    pool = get_reusable_pool(processes=2)
    pool.apply(print, ('Pool recovered from the worker crash', ))
    pool.terminate()


def test_freeze():
    """Test no freeze on OSX with Accelerate"""
    import numpy as np
    a = np.random.randn(1000, 1000)
    np.dot(a, a)
    pool = get_reusable_pool(2)
    pool.apply(np.dot, (a, a))
