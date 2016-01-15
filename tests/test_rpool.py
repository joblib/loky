from __future__ import print_function
import os
import sys
import psutil
from time import sleep
from faulthandler import dump_traceback_later
from faulthandler import cancel_dump_traceback_later
from nose.tools import assert_raises
from nose import SkipTest
import pytest
from backend.reusable_pool import get_reusable_pool
from backend.reusable_pool import AbortedWorkerError, TerminatedPoolError
from multiprocessing import util
from multiprocessing.pool import MaybeEncodingError
from pickle import PicklingError, UnpicklingError

# Backward compat for python2 cPickle module
PICKLING_ERRORS = (PicklingError,)
try:
    import cPickle
    PICKLING_ERRORS += (cPickle.PicklingError,)
except ImportError:
    pass

# Activate multiprocessing logging
util.log_to_stderr()
util._logger.setLevel(10)


@pytest.yield_fixture
def exit_on_deadlock():
    dump_traceback_later(timeout=5, exit=True)
    yield
    cancel_dump_traceback_later()


def wait_dead(pid, n_tries=1000, delay=0.001):
    """Wait for process pid to die"""
    for i in range(n_tries):
        if not psutil.pid_exists(pid):
            return
        sleep(delay)
    raise RuntimeError("Process %d failed to die for at least %0.3fs" %
                       (pid, delay * n_tries))


def crash():
    """Induces a segfault"""
    import ctypes
    i = ctypes.c_char(b'a')
    j = ctypes.pointer(i)
    c = 0
    while True:
        j[c] = i
        c += 1


def exit():
    """Induces a sys exit with exitcode 1"""
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
    """Function that send SIGKILL at process pid"""
    os.kill(pid, 9)


def raise_error():
    """Function that raises an Exception in process"""
    raise RuntimeError('bad except')


def return_instance(cls):
    """Function that returns a instance of cls"""
    return cls()

def do_nothing(arg):
    """Function that return True, test passing argument"""
    return True


class CrashAtPickle(object):
    """Bad object that triggers a segfault at pickling time."""
    def __reduce__(self):
        crash()


class CrashAtUnpickle(object):
    """Bad object that triggers a segfault at unpickling time."""
    def __reduce__(self):
        return crash, (), ()


class ExitAtPickle(object):
    """Bad object that triggers a segfault at pickling time."""
    def __reduce__(self):
        exit()


class ExitAtUnpickle(object):
    """Bad object that triggers a process exit at unpickling time."""
    def __reduce__(self):
        return exit, (), ()


class ErrorAtPickle(object):
    """Bad object that triggers a segfault at pickling time."""
    def __reduce__(self):
        raise PicklingError("Error in pickle")


class ErrorAtUnpickle(object):
    """Bad object that triggers a process exit at unpickling time."""
    def __reduce__(self):
        return UnpicklingError, ("Error in unpickle"), ()


def sleep_identity(args):
    x, delay = args
    sleep(delay)
    return x

crash_cases = [
            # Check problem occuring while pickling a task in
            # the task_handler thread
            (id, (ExitAtPickle(),), AbortedWorkerError),
            (id, (ErrorAtPickle(),), PICKLING_ERRORS),
            # Check problem occuring while unpickling a task on workers
            (id, (ExitAtUnpickle(),), AbortedWorkerError),
            (id, (ErrorAtUnpickle(),), PICKLING_ERRORS),
            (id, (CrashAtUnpickle(),), AbortedWorkerError),
            # Check problem occuring during function execution on workers
            (crash, (), AbortedWorkerError),
            (exit, (), AbortedWorkerError),
            (raise_error, (), RuntimeError),
            # Check problem occuring while pickling a task result on workers
            (return_instance, (CrashAtPickle,), AbortedWorkerError),
            (return_instance, (ExitAtPickle,), AbortedWorkerError),
            (return_instance, (ErrorAtPickle,), MaybeEncodingError),
            # Check problem occuring while unpickling a task in
            # the result_handler thread
            (return_instance, (ExitAtUnpickle,), AbortedWorkerError),
            (return_instance, (ErrorAtUnpickle,), MaybeEncodingError),
]


@pytest.mark.parametrize("func, args, expected_err", crash_cases)
def test_rpool_crash(exit_on_deadlock, func, args, expected_err):
    pool = get_reusable_pool(processes=2)
    res = pool.apply_async(func, args)
    assert_raises(expected_err, res.get)

    # Check that the pool can still be recovered
    pool = get_reusable_pool(processes=2)
    assert pool.apply(sleep_identity, ((1, 0.),)) == 1


def test_terminate_kill(exit_on_deadlock):
    pool = get_reusable_pool(processes=5)
    res1 = pool.map_async(sleep_identity, [(i, 0.001) for i in range(50)])
    res2 = pool.map_async(sleep_identity, [(i, 0.001) for i in range(50)])
    assert res1.get() == list(range(50))
    # Send sentinels and join worker process.
    # We should get all the results
    pool.terminate()
    assert_raises(TerminatedPoolError, res2.get)


def test_crash(exit_on_deadlock):
    """Test the crash handling in pool"""
    # Test for external crash signal comming from neighbor
    # with various race setup
    for i in [1, 2, 5, 17]:
        print("# Processes ", i)
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

    # Clean terminate
    pool.terminate()


def test_rpool_resize(exit_on_deadlock):
    """Test the resize function in reusable_pool"""

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
    """Raise error on invalid process number"""
    assert_raises(ValueError, get_reusable_pool, processes=0)
    assert_raises(ValueError, get_reusable_pool, processes=-1)


def test_deadlock_kill(exit_on_deadlock):
    """Create a deadlock in pool by killing the lock owner."""
    pool = get_reusable_pool(processes=1)
    pid = pool._pool[0].pid
    pool = get_reusable_pool(processes=2)
    os.kill(pid, 9)
    wait_dead(pid)

    pool = get_reusable_pool(processes=2)
    pool.apply(print, ('Pool recovered from the worker crash', ))
    pool.terminate()


def test_freeze(exit_on_deadlock):
    """Test no freeze on OSX with Accelerate"""
    raise SkipTest('Known failure')
    import numpy as np
    a = np.random.randn(1000, 1000)
    np.dot(a, a)
    pool = get_reusable_pool(2)
    pool.apply(np.dot, (a, a))
