import os
import sys
import psutil
from time import sleep
from nose.tools import assert_raises
from backend.reusable_pool import get_reusable_pool, AbortedWorkerError


def crash():
    '''Induce a segfault in process
    '''
    import ctypes
    i = ctypes.c_char(b'a')
    j = ctypes.pointer(i)
    c = 0
    while True:
        j[c] = 1
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


def test_Rpool_crash():
    '''Test the crash handling in pool
    '''
    pool = get_reusable_pool(2)

    for func, err in [(crash, AbortedWorkerError),
                      (exit, AbortedWorkerError),
                      (raise_Error, RuntimeError)]:
        res = pool.apply_async(func, tuple())
        assert_raises(err, res.get)
        sleep(.1)

    # Test for external signal comming from neighbor
    pids = pool.starmap(os.getpid, [tuple()]*2)
    res = pool.map_async(kill_friend, pids[::-1])
    assert_raises(AbortedWorkerError, res.get)

    pool.terminate()


def test_Rpool_resize():
    '''Test the resize function in reusable_pool
    '''

    pool = get_reusable_pool(2)
    pids = pool.starmap(os.getpid, [tuple()]*2)
    res = pool.apply_async(work_sleep, (.5, pids))
    pool = get_reusable_pool(1)
    assert res.get(), "Resize does not wait for current processes to finish"
    pool = get_reusable_pool(1)
    pool.terminate()
    assert_raises(ValueError, pool.resize, 0)
    pool = get_reusable_pool()


def test_deadlock_kill():
    '''Create a deadlock in pool by killing
    the lock owner.
    '''
    pool = get_reusable_pool(1)
    pid = pool.apply(os.getpid, tuple())
    pool = get_reusable_pool(2)
    os.kill(pid, 9)
    sleep(.2)

    pool = get_reusable_pool(2)
    pool.apply(print, ('Pool recovered from the worker crash', ))
    pool.terminate()
