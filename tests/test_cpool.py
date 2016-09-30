import os
import sys
import psutil
import warnings
from time import sleep, time
import pytest
from loky.concurrent_pool import get_reusable_executor
import multiprocessing as mp
from loky.process_executor import BrokenProcessPool, ShutdownProcessPool
from pickle import PicklingError, UnpicklingError

# Backward compat for python2 cPickle module
PICKLING_ERRORS = (PicklingError,)
try:
    import cPickle
    PICKLING_ERRORS += (cPickle.PicklingError,)
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
if not mp.util._log_to_stderr:
    import logging
    log = mp.util.log_to_stderr(10)
    log.handlers[0].setFormatter(logging.Formatter(
        '[%(levelname)s||%(processName)s:%(threadName)s] %(message)s'))


@pytest.yield_fixture
def exit_on_deadlock():
    from sys import stderr
    dump_traceback_later(timeout=TIMEOUT, exit=True, file=stderr)
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
    pool = get_reusable_executor(max_workers=2)
    try:
        pool.submit(func, args)
    except Exception as e:
        # One should never call join before terminate: if the pool is broken,
        # (AbortedWorkerError was raised) the cleanup mechanism should be
        # triggered by the _worker_handler thread. Else we should call
        # terminate explicitly
        if not isinstance(e, BrokenProcessPool):
            pool.shutdown(wait=True)
        raise e
    finally:
        # _worker_handler thread is triggered every .1s
        sleep(.2)


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


def is_terminated_properly():
    pass


# class ReusableExecutorMixin:
#     worker_count = 5

#     def setup_method(self, method):
#         self.t1 = time.time()
#         try:
#             self.executor = get_reusable_executor(max_workers=2)
#         except NotImplementedError as e:
#             self.skipTest(str(e))
#         self._prime_executor()

#     def teardown_method(self, method):
#         self.executor.shutdown(wait=True)
#         dt = time.time() - self.t1
#         if test.support.verbose:
#             print("%.2fs" % dt, end=' ')
#         assert dt < 60, "synchronization issue: test lasted too long"

#     def _prime_executor(self):
#         # Make sure that the executor is ready to do work before running the
#         # tests. This should reduce the probability of timeouts in the tests.
#         futures = [self.executor.submit(time.sleep, 0.1)
#                    for _ in range(self.worker_count)]
#         for f in futures:
#             f.result()


class TestPoolDeadLock:

    crash_cases = [
                # Check problem occuring while pickling a task in
                (id, (ExitAtPickle(),), BrokenProcessPool),
                (id, (ErrorAtPickle(),), BrokenProcessPool),
                # Check problem occuring while unpickling a task on workers
                (id, (ExitAtUnpickle(),), BrokenProcessPool),
                (id, (ErrorAtUnpickle(),), BrokenProcessPool),
                (id, (CrashAtUnpickle(),), BrokenProcessPool),
                # Check problem occuring during function execution on workers
                (crash, (), BrokenProcessPool),
                (exit, (), SystemExit),
                (raise_error, (RuntimeError,), RuntimeError),
                # Check problem occuring while pickling a task result
                # on workers
                (return_instance, (CrashAtPickle,), BrokenProcessPool),
                (return_instance, (ExitAtPickle,), BrokenProcessPool),
                (return_instance, (ErrorAtPickle,), PicklingError),
                # Check problem occuring while unpickling a task in
                # the result_handler thread
                (return_instance, (ExitAtUnpickle,), BrokenProcessPool),
                (return_instance, (ErrorAtUnpickle,), UnpicklingError),
    ]

    # callback_crash_cases = [
    #             # Check problem occuring during function execution on workers
    #             # (crash, AbortedWorkerError),
    #             (exit, (), AbortedWorkerError),
    #             (raise_error, (RuntimeError, ), RuntimeError),
    #             (start_job, (id, (ExitAtPickle(),)), AbortedWorkerError),
    #             (start_job, (id, (ErrorAtPickle(),)), PICKLING_ERRORS),
    #             # Check problem occuring while unpickling a task on workers
    #             (start_job, (id, (ExitAtUnpickle(),)), AbortedWorkerError),
    #             (start_job, (id, (ErrorAtUnpickle(),)), AbortedWorkerError),
    #             (start_job, (id, (CrashAtUnpickle(),)), AbortedWorkerError),
    #             # Check problem occuring during function execution on workers
    #             (start_job, (crash, ()), AbortedWorkerError),
    #             (start_job, (exit, ()), AbortedWorkerError),
    #             (start_job, (raise_error, (RuntimeError, )), RuntimeError),
    #             # Check problem occuring while pickling a task
    #             # result on workers
    #             (start_job, (return_instance, (CrashAtPickle,)),
    #              AbortedWorkerError),
    #             (start_job, (return_instance, (ExitAtPickle,)),
    #              AbortedWorkerError),
    #             (start_job, (return_instance, (ErrorAtPickle,)),
    #              MaybeEncodingError),
    #             # Check problem occuring while unpickling a task in
    #             # the result_handler thread
    #             (start_job, (return_instance, (ExitAtUnpickle,)),
    #              AbortedWorkerError),
    #             (start_job, (return_instance, (ErrorAtUnpickle,)),
    #              AbortedWorkerError),
    # ]

    @pytest.mark.parametrize("func, args, expected_err", crash_cases)
    def test_crashes(self, exit_on_deadlock, func, args, expected_err):
        """Test various reusable_pool crash handling"""
        pool = get_reusable_executor(max_workers=2)
        res = pool.submit(func, *args)
        with pytest.raises(expected_err):
            res.result()

        # Check that the pool can still be recovered
        pool = get_reusable_executor(max_workers=2)
        assert pool.submit(id_sleep, (1, 0.)).result() == 1
        pool.shutdown(wait=True)

    # # @pytest.mark.skipif(True, reason="Known failure")
    # @pytest.mark.parametrize("func, args, expected_err", callback_crash_cases)
    # def test_callback(self, exit_on_deadlock, func, args, expected_err):
    #     """Test the recovery from callback crash"""
    #     pool = get_reusable_executor(max_workers=2)
    #     res = pool.apply_async(id_sleep,
    #                            ((func, 0),),
    #                            callback=lambda f: f(*args))
    #     with pytest.raises(expected_err):
    #         try:
    #             res.get()
    #         except CallbackError as e:
    #             assert func == e.value
    #             raise e.err

    #     # Check that the pool can still be recovered
    #     pool = get_reusable_executor(max_workers=2)
    #     assert pool.apply(id_sleep, ((1, 0.),)) == 1
    #     pool.shutdown(wait=True)
    #

    def test_deadlock_kill(self, exit_on_deadlock):
        """Test deadlock recovery for reusable_pool"""
        pool = get_reusable_executor(max_workers=1)
        worker = next(iter(pool._processes.values()))
        pool = get_reusable_executor(max_workers=2)
        os.kill(worker.pid, SIGKILL)
        wait_dead(worker)
        sleep(.1)

        pool = get_reusable_executor(max_workers=2)
        assert pool.submit(id_sleep, (1, 0.)).result() == 1
        pool.shutdown(wait=True)

    @pytest.mark.parametrize("n_proc", [1, 2, 5, 13])
    def test_crash_races(self, exit_on_deadlock, n_proc):
        """Test the race conditions in reusable_pool crash handling"""
        # Test for external crash signal comming from neighbor
        # with various race setup
        mp.util.debug("Test race - # Processes = {}".format(n_proc))
        pool = get_reusable_executor(max_workers=n_proc)
        pids = list(pool._processes.keys())
        assert len(pids) == n_proc
        assert None not in pids
        res = pool.map(work_sleep, [(.0001 * (j//2), pids)
                                    for j in range(2 * n_proc)])
        assert all(list(res))
        with pytest.raises(BrokenProcessPool):
            res = pool.map(kill_friend, pids[::-1])
            list(res)

        # Clean terminate
        pool.shutdown(wait=True)

    def test_imap_handle_iterable_exception(self, exit_on_deadlock):
        # The catch of the errors in imap generation depend on the
        # builded version of python
        pool = get_reusable_executor(max_workers=2)
        with pytest.raises(SayWhenError):
            pool.map(id_sleep, exception_throwing_generator(10, 3),
                     chunksize=1)

        # SayWhenError seen at start of problematic chunk's results
        pool = get_reusable_executor(max_workers=2)
        with pytest.raises(SayWhenError):
            pool.map(id_sleep, exception_throwing_generator(20, 7),
                     chunksize=2)

        pool = get_reusable_executor(max_workers=2)
        with pytest.raises(SayWhenError):
            pool.map(id_sleep, exception_throwing_generator(20, 7),
                     chunksize=4)

        pool.shutdown(wait=True)


class TestTerminatePool:
    def test_terminate_kill(self, exit_on_deadlock):
        """Test reusable_pool termination handling"""
        pool = get_reusable_executor(max_workers=5)
        res1 = pool.map(id_sleep, [(i, 0.001) for i in range(50)])
        res2 = pool.map(id_sleep, [(i, 0.1) for i in range(50)])
        assert list(res1) == list(range(50))
        # We should get an error as the pool.shutdownd before we fetched
        # the results from the operation.
        terminate = TimingWrapper(pool.shutdown)
        terminate(wait=True)
        assert terminate.elapsed < .5
        with pytest.raises(ShutdownProcessPool):
            list(res2)

    def test_terminate_deadlock(self, exit_on_deadlock):
        """Test recovery if killed after resize call"""
        # Test the pool.shutdown call do not cause deadlock
        pool = get_reusable_executor(max_workers=2)
        pool.submit(kill_friend, (next(iter(pool._processes.keys())), .0))
        sleep(.01)
        pool.shutdown(wait=True)

        pool = get_reusable_executor(max_workers=2)
        pool.shutdown(wait=True)

    def test_terminate(self, exit_on_deadlock):

        pool = get_reusable_executor(max_workers=4)
        res = pool.map(
            sleep, [0.1 for i in range(10000)], chunksize=1
            )
        shutdown = TimingWrapper(pool.shutdown)
        shutdown(wait=True)
        assert shutdown.elapsed < 0.5

        with pytest.raises(ShutdownProcessPool):
            list(res)


class TestResizeRpool:
    def test_rpool_resize(self, exit_on_deadlock):
        """Test reusable_pool resizing"""

        pool = get_reusable_executor(max_workers=2)

        # Decreasing the pool should drop a single process and keep one of the
        # old one as it is still in a good shape. The resize should not occur
        # while there are on going works.
        pids = list(pool._processes.keys())
        res = pool.submit(work_sleep, (.3, pids))
        warnings.filterwarnings("always", category=UserWarning)
        with warnings.catch_warnings(record=True) as w:
            pool = get_reusable_executor(max_workers=1)
            assert len(w) == 1
            assert res.result(), ("Resize should wait for current processes "
                                  " to finish")
            assert len(pool._processes) == 1
            assert next(iter(pool._processes.keys())) in pids

        # Requesting the same number of process should not impact the pool nor
        # kill the processed
        old_pid = next(iter((pool._processes.keys())))
        unchanged_pool = get_reusable_executor(max_workers=1)
        assert len(unchanged_pool._processes) == 1
        assert unchanged_pool is pool
        assert next(iter(unchanged_pool._processes.keys())) == old_pid

        # Growing the pool again should add a single process and keep the old
        # one as it is still in a good shape
        pool = get_reusable_executor(max_workers=2)
        assert len(pool._processes) == 2
        assert old_pid in list(pool._processes.keys())
        pool.shutdown(wait=True)

    def test_kill_after_resize_call(self, exit_on_deadlock):
        """Test recovery if killed after resize call"""
        # Test the pool resizing called before a kill arrive
        pool = get_reusable_executor(max_workers=2)
        pool.submit(kill_friend, (next(iter(pool._processes.keys())), .1))
        pool = get_reusable_executor(max_workers=1)
        assert pool.submit(id_sleep, (1, 0.)).result() == 1
        pool.shutdown(wait=True)


def test_invalid_process_number():
    """Raise error on invalid process number"""

    with pytest.raises(ValueError):
        get_reusable_executor(max_workers=0)

    with pytest.raises(ValueError):
        get_reusable_executor(max_workers=-1)


@pytest.mark.skipif(True, reason="Known failure")
def test_freeze(exit_on_deadlock):
    """Test no freeze on OSX with Accelerate"""
    import numpy as np
    a = np.random.randn(1000, 1000)
    np.dot(a, a)
    pool = get_reusable_executor(max_workers=2)
    pool.submit(np.dot, (a, a))
    pool.shutdown(wait=True)


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
