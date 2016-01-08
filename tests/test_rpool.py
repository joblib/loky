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
        try:
            os.kill(pid, 0)  # check that pid exists
        except OSError:
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


def work_sleep(time, pids):
    '''Sleep for some time before returning
    and check if all the passed pid exist
    '''
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
    pool = get_reusable_pool(processes=2)
    pids = [p.pid for p in pool._pool]
    assert None not in pids
    res = pool.map_async(kill_friend, pids[::-1])
    assert_raises(AbortedWorkerError, res.get)

    pool.terminate()


def test_Rpool_resize():
    '''Test the resize function in reusable_pool
    '''

    pool = get_reusable_pool(processes=2)
    pids = [p.pid for p in pool._pool]
    res = pool.apply_async(work_sleep, (.5, pids))
    pool = get_reusable_pool(processes=1)
    assert res.get(), "Resize does not wait for current processes to finish"
    pool = get_reusable_pool(processes=1)
    pool.terminate()
    assert_raises(ValueError, pool.resize, 0)
    pool = get_reusable_pool()


def test_deadlock_kill():
    '''Create a deadlock in pool by killing
    the lock owner.
    '''
    pool = get_reusable_pool(processes=1)
    pid = pool._pool[0].pid
    pool = get_reusable_pool(processes=2)
    os.kill(pid, 9)
    wait_dead(pid)

    pool = get_reusable_pool(processes=2)
    pool.apply(print, ('Pool recovered from the worker crash', ))
    pool.terminate()
