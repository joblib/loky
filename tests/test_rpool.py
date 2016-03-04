import os
import sys
import psutil
import warnings
from time import sleep, time
import pytest
from backend.reusable_pool import get_reusable_pool, CallbackError
from backend.reusable_pool import AbortedWorkerError, TerminatedPoolError
import multiprocessing as mp
from multiprocessing.pool import MaybeEncodingError
from pickle import PicklingError, UnpicklingError

# Backward compat for python2 cPickle module
PICKLING_ERRORS = (PicklingError,)
try:
    import cPickle
    PICKLING_ERRORS += (cPickle.PicklingError,)
    from multiprocessing import TimeoutError
except ImportError:
    pass

# Compat windows
try:
    from signal import SIGKILL
    # Increase time if the test is perform on a slow machine
    TIMEOUT = max(20 / mp.cpu_count(), 5)
except ImportError:
    from signal import SIGTERM as SIGKILL
    TIMEOUT = 20

# Compat windows and python2.7
try:
    from faulthandler import dump_traceback_later
    from faulthandler import cancel_dump_traceback_later
except ImportError:
    def dump_traceback_later(timeout=None, exit=None, file=None):
        pass

    def cancel_dump_traceback_later():
        pass

# Activate multiprocessing logging
mp.util.log_to_stderr()
mp.util._logger.setLevel(5)


@pytest.yield_fixture
def exit_on_deadlock():
    with open(".exit_on_lock", "w") as f:
        dump_traceback_later(timeout=TIMEOUT, exit=True, file=f)
        yield
        cancel_dump_traceback_later()


def wait_dead(worker, n_tries=1000, delay=0.001):
    """Wait for process pid to die"""
    for i in range(n_tries):
        if worker.exitcode is not None:
            return
        sleep(delay)
    raise RuntimeError("Process %d failed to die for at least %0.3fs" %
                       (worker.pid, delay * n_tries))


def crash():
    """Induces a segfault"""
    import faulthandler
    faulthandler._sigsegv()


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


def kill_friend(pid, delay=0):
    """Function that send SIGKILL at process pid"""
    sleep(delay)
    try:
        os.kill(pid, SIGKILL)
    except (PermissionError, ProcessLookupError) as e:
        if psutil.pid_exists(pid):
            mp.util.debug("Fail to kill an alive process?!?")
            raise e
        mp.util.debug("process {} was already dead".format(pid))


def raise_error(Err):
    """Function that raises an Exception in process"""
    raise Err()


def return_instance(cls):
    """Function that returns a instance of cls"""
    return cls()


def start_job(func, args):
    pool = get_reusable_pool(processes=2)
    pool.apply(func, args)


class SayWhenError(ValueError):
    pass


def exception_throwing_generator(total, when):
    for i in range(total):
        if i == when:
            raise SayWhenError("Somebody said when")
        yield i


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
        return crash, ()


class ExitAtPickle(object):
    """Bad object that triggers a segfault at pickling time."""
    def __reduce__(self):
        exit()


class ExitAtUnpickle(object):
    """Bad object that triggers a process exit at unpickling time."""
    def __reduce__(self):
        return exit, ()


class ErrorAtPickle(object):
    """Bad object that triggers a segfault at pickling time."""
    def __reduce__(self):
        raise PicklingError("Error in pickle")


class ErrorAtUnpickle(object):
    """Bad object that triggers a process exit at unpickling time."""
    def __reduce__(self):
        return raise_error, (UnpicklingError, )


def id_sleep(args):
    try:
        x, delay = args
    except TypeError:
        x, delay = args, 0
    sleep(delay)
    return x


class TestPoolDeadLock:

    crash_cases = [
                # Check problem occuring while pickling a task in
                (id, (ExitAtPickle(),), AbortedWorkerError),
                (id, (ErrorAtPickle(),), PICKLING_ERRORS),
                # Check problem occuring while unpickling a task on workers
                (id, (ExitAtUnpickle(),), AbortedWorkerError),
                (id, (ErrorAtUnpickle(),), AbortedWorkerError),
                (id, (CrashAtUnpickle(),), AbortedWorkerError),
                # Check problem occuring during function execution on workers
                (crash, (), AbortedWorkerError),
                (exit, (), AbortedWorkerError),
                (raise_error, (RuntimeError, ), RuntimeError),
                # Check problem occuring while pickling a task result
                # on workers
                (return_instance, (CrashAtPickle,), AbortedWorkerError),
                (return_instance, (ExitAtPickle,), AbortedWorkerError),
                (return_instance, (ErrorAtPickle,), MaybeEncodingError),
                # Check problem occuring while unpickling a task in
                # the result_handler thread
                (return_instance, (ExitAtUnpickle,), AbortedWorkerError),
                (return_instance, (ErrorAtUnpickle,), AbortedWorkerError),
    ]

    callback_crash_cases = [
                # Check problem occuring during function execution on workers
                # (crash, AbortedWorkerError),
                (exit, (), AbortedWorkerError),
                (raise_error, (RuntimeError, ), RuntimeError),
                (start_job, (id, (ExitAtPickle(),)), AbortedWorkerError),
                (start_job, (id, (ErrorAtPickle(),)), PICKLING_ERRORS),
                # Check problem occuring while unpickling a task on workers
                (start_job, (id, (ExitAtUnpickle(),)), AbortedWorkerError),
                (start_job, (id, (ErrorAtUnpickle(),)), AbortedWorkerError),
                (start_job, (id, (CrashAtUnpickle(),)), AbortedWorkerError),
                # Check problem occuring during function execution on workers
                (start_job, (crash, ()), AbortedWorkerError),
                (start_job, (exit, ()), AbortedWorkerError),
                (start_job, (raise_error, (RuntimeError, )), RuntimeError),
                # Check problem occuring while pickling a task
                # result on workers
                (start_job, (return_instance, (CrashAtPickle,)),
                 AbortedWorkerError),
                (start_job, (return_instance, (ExitAtPickle,)),
                 AbortedWorkerError),
                (start_job, (return_instance, (ErrorAtPickle,)),
                 MaybeEncodingError),
                # Check problem occuring while unpickling a task in
                # the result_handler thread
                (start_job, (return_instance, (ExitAtUnpickle,)),
                 AbortedWorkerError),
                (start_job, (return_instance, (ErrorAtUnpickle,)),
                 AbortedWorkerError),
    ]

    @pytest.mark.parametrize("func, args, expected_err", crash_cases)
    def test_crashes(self, exit_on_deadlock, func, args, expected_err):
        """Test various reusable_pool crash handling"""
        pool = get_reusable_pool(processes=2)
        res = pool.apply_async(func, args)
        with pytest.raises(expected_err):
            res.get()

        # Check that the pool can still be recovered
        pool = get_reusable_pool(processes=2)
        assert pool.apply(id_sleep, ((1, 0.),)) == 1
        pool.terminate()

    @pytest.mark.parametrize("func, args, expected_err", callback_crash_cases)
    def test_callback(self, exit_on_deadlock, func, args, expected_err):
        """Test the recovery from callback crash"""
        pool = get_reusable_pool(processes=2)
        res = pool.apply_async(id_sleep,
                               ((func, 0),),
                               callback=lambda f: f(*args))
        with pytest.raises(expected_err):
            try:
                res.get()
            except CallbackError as e:
                assert func == e.value
                raise e.err

        # Check that the pool can still be recovered
        pool = get_reusable_pool(processes=2)
        assert pool.apply(id_sleep, ((1, 0.),)) == 1
        pool.terminate()

    def test_deadlock_kill(self, exit_on_deadlock):
        """Test deadlock recovery for reusable_pool"""
        pool = get_reusable_pool(processes=1)
        worker = pool._pool[0]
        pool = get_reusable_pool(processes=2)
        os.kill(worker.pid, SIGKILL)
        wait_dead(worker)

        pool = get_reusable_pool(processes=2)
        assert pool.apply(id_sleep, ((1, 0.),)) == 1
        pool.terminate()

    @pytest.mark.parametrize("n_proc", [1, 2, 5, 13])
    def test_crash_races(self, exit_on_deadlock, n_proc):
        """Test the race conditions in reusable_pool crash handling"""
        # Test for external crash signal comming from neighbor
        # with various race setup
        mp.util.debug("Test race - # Processes = {}".format(n_proc))
        pool = get_reusable_pool(processes=n_proc)
        pids = [p.pid for p in pool._pool]
        assert len(pids) == n_proc
        assert None not in pids
        res = pool.map(work_sleep, [(.0001 * (j//2), pids)
                                    for j in range(2 * n_proc)])
        assert all(res)
        res = pool.map_async(kill_friend, pids[::-1])
        with pytest.raises(AbortedWorkerError):
            res.get()

        pool = get_reusable_pool(processes=n_proc)
        pids = [p.pid for p in pool._pool]
        res = pool.imap(work_sleep, [(.0001 * j, pids)
                                     for j in range(2 * n_proc)])
        assert all(list(res))
        res = pool.imap(kill_friend, pids[::-1])
        with pytest.raises(AbortedWorkerError):
            list(res)

        # Clean terminate
        pool.terminate()

    def test_imap_handle_iterable_exception(self, exit_on_deadlock):
        # The catch of the errors in imap generation depend on the
        # builded version of python
        expected_err = (SayWhenError,)
        if sys.version_info[:2] < (3, 5):
            expected_err += (AbortedWorkerError,)
        pool = get_reusable_pool(processes=2)
        it = pool.imap(id_sleep, exception_throwing_generator(10, 3), 1)
        with pytest.raises(expected_err) as exc_info:
            for i in range(3):
                assert next(it) == i
            next(it)
        if isinstance(exc_info, SayWhenError):
            assert i == 3

        # SayWhenError seen at start of problematic chunk's results
        pool = get_reusable_pool(processes=2)
        it = pool.imap(id_sleep, exception_throwing_generator(20, 7), 2)

        with pytest.raises(expected_err) as exc_info:
            for i in range(6):
                assert next(it) == i
            next(it)
        if isinstance(exc_info, SayWhenError):
            assert i == 6

        pool = get_reusable_pool(processes=2)
        it = pool.imap(id_sleep, exception_throwing_generator(20, 7), 4)

        with pytest.raises(expected_err) as exc_info:
            for i in range(4):
                assert next(it) == i
            next(it)
        if isinstance(exc_info, SayWhenError):
            assert i == 4
        pool.terminate()

    def test_imap_unordered_handle_iterable_exception(self, exit_on_deadlock):
        """Test correct hadeling of generator failure in crash"""
        # The catch of the errors in imap generation depend on the
        # builded version of python
        expected_err = (SayWhenError,)
        if sys.version_info[:2] < (3, 5):
            expected_err += (AbortedWorkerError,)
        pool = get_reusable_pool(processes=2)
        it = pool.imap_unordered(id_sleep, exception_throwing_generator(10, 3),
                                 1)
        expected_values = list(range(10))
        with pytest.raises(expected_err):
            # imap_unordered makes it difficult to anticipate the SayWhenError
            for i in range(10):
                value = next(it)
                assert value in expected_values
                expected_values.remove(value)

        pool = get_reusable_pool(processes=2)
        it = pool.imap_unordered(id_sleep, exception_throwing_generator(20, 7),
                                 2)
        expected_values = list(map(id_sleep, list(range(20))))
        with pytest.raises(expected_err):
            for i in range(20):
                value = next(it)
                assert value in expected_values
                expected_values.remove(value)

        # Clean terminate
        pool.terminate()


class TestTerminatePool:
    def test_terminate_kill(self, exit_on_deadlock):
        """Test reusable_pool termination handling"""
        pool = get_reusable_pool(processes=5)
        res1 = pool.map_async(id_sleep, [(i, 0.001) for i in range(50)])
        res2 = pool.map_async(id_sleep, [(i, 0.1) for i in range(50)])
        assert res1.get() == list(range(50))
        # We should get an error as the pool terminated before we fetched
        # the results from the operation.
        terminate = TimingWrapper(pool.terminate)
        terminate()
        assert terminate.elapsed < 0.5
        with pytest.raises(TerminatedPoolError):
            res2.get()

    def test_terminate_deadlock(self, exit_on_deadlock):
        """Test recovery if killed after resize call"""
        # Test the pool terminate call do not cause deadlock
        pool = get_reusable_pool(processes=2)
        pool.apply_async(kill_friend, (pool._pool[1].pid, .0))
        sleep(.01)
        pool.terminate()

        pool = get_reusable_pool(processes=2)
        pool.terminate()

    def test_terminate(self, exit_on_deadlock):

        pool = get_reusable_pool(processes=4)
        res = pool.map_async(
            sleep, [0.1 for i in range(10000)], chunksize=1
            )
        pool.terminate()
        join = TimingWrapper(pool.join)
        join()
        assert join.elapsed < 0.5
        with pytest.raises(TerminatedPoolError):
            res.get()


class TestResizeRpool:
    def test_rpool_resize(self, exit_on_deadlock):
        """Test reusable_pool resizing"""

        pool = get_reusable_pool(processes=2)

        # Decreasing the pool should drop a single process and keep one of the
        # old one as it is still in a good shape. The resize should not occur
        # while there are on going works.
        pids = [p.pid for p in pool._pool]
        res = pool.apply_async(work_sleep, ((.1, pids),))
        warnings.filterwarnings("always", category=UserWarning)
        with warnings.catch_warnings(record=True) as w:
            pool = get_reusable_pool(processes=1)
            assert res.get(), ("Resize should wait for current processes "
                               " to finish")
            assert len(pool._pool) == 1
            assert pool._pool[0].pid in pids
            assert len(w) == 1

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

    def test_kill_after_resize_call(self, exit_on_deadlock):
        """Test recovery if killed after resize call"""
        # Test the pool resizing called before a kill arrive
        pool = get_reusable_pool(processes=2)
        pool.apply_async(kill_friend, (pool._pool[1].pid, .1))
        pool = get_reusable_pool(processes=1)
        assert pool.apply(id_sleep, ((1, 0.),)) == 1
        pool.terminate()


def test_invalid_process_number():
    """Raise error on invalid process number"""

    with pytest.raises(ValueError):
        get_reusable_pool(processes=0)

    with pytest.raises(ValueError):
        get_reusable_pool(processes=-1)


@pytest.mark.skipif(True, reason="Known failure")
def test_freeze(exit_on_deadlock):
    """Test no freeze on OSX with Accelerate"""
    import numpy as np
    a = np.random.randn(1000, 1000)
    np.dot(a, a)
    pool = get_reusable_pool(2)
    pool.apply(np.dot, (a, a))
    pool.terminate()


class TimingWrapper(object):

    def __init__(self, func):
        self.func = func
        self.elapsed = None

    def __call__(self, *args, **kwds):
        t = time()
        try:
            return self.func(*args, **kwds)
        finally:
            self.elapsed = time() - t
