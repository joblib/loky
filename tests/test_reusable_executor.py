import os
import sys
import gc
import ctypes
import psutil
import pytest
import warnings
import threading
from time import sleep
from tempfile import mkstemp
from multiprocessing import util, current_process
from pickle import PicklingError, UnpicklingError

from loky.backend import get_context
from loky import get_reusable_executor
from loky.process_executor import BrokenExecutor, ShutdownExecutorError

from ._executor_mixin import ReusableExecutorMixin
from .utils import TimingWrapper, id_sleep, check_subprocess_call

# Compat windows
if sys.platform == "win32":
    from signal import SIGTERM as SIGKILL
    libc = ctypes.cdll.msvcrt
else:
    from signal import SIGKILL
    from ctypes.util import find_library
    libc = ctypes.CDLL(find_library("libc"))


try:
    import numpy as np
except ImportError:
    np = None

# Backward compat for python2 cPickle module
PICKLING_ERRORS = (PicklingError,)
try:
    import cPickle
    PICKLING_ERRORS += (cPickle.PicklingError,)
except ImportError:
    pass


def clean_warning_registry():
    """Safe way to reset warnings."""
    warnings.resetwarnings()
    reg = "__warningregistry__"
    for mod_name, mod in list(sys.modules.items()):
        if hasattr(mod, reg):
            getattr(mod, reg).clear()


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
    """Induces a sys exit with exitcode 0"""
    sys.exit(0)


def c_exit():
    """Induces a libc exit with exitcode 0"""
    libc.exit(0)


def check_pids_exist_then_sleep(arg):
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
            util.debug("Fail to kill an alive process?!?")
            raise e
        util.debug("process {} was already dead".format(pid))


def raise_error(Err):
    """Function that raises an Exception in process"""
    raise Err()


def return_instance(cls):
    """Function that returns a instance of cls"""
    return cls()


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


class CExitAtPickle(object):
    """Bad object that triggers a segfault at pickling time."""
    def __reduce__(self):
        c_exit()


class CExitAtUnpickle(object):
    """Bad object that triggers a process exit at unpickling time."""
    def __reduce__(self):
        return c_exit, ()


class ErrorAtPickle(object):
    """Bad object that raises an error at pickling time."""
    def __reduce__(self):
        raise PicklingError("Error in pickle")


class ErrorAtUnpickle(object):
    """Bad object that triggers a process exit at unpickling time."""
    def __reduce__(self):
        return raise_error, (UnpicklingError, )


class CrashAtGCInWorker(object):
    """Bad object that triggers a segfault at call item GC time"""
    def __del__(self):
        if current_process().name != "MainProcess":
            crash()


class CExitAtGCInWorker(object):
    """Exit worker at call item GC time"""
    def __del__(self):
        if current_process().name != "MainProcess":
            c_exit()


class SlowlyPickling(object):
    """Simulate slowly pickling object, e.g. large numpy array or dict"""
    def __init__(self, delay=1):
        self.delay = delay

    def __reduce__(self):
        sleep(self.delay)
        return SlowlyPickling, (self.delay,)


class TestExecutorDeadLock(ReusableExecutorMixin):

    crash_cases = [
        # Check problem occuring while pickling a task in
        (id, (ExitAtPickle(),), BrokenExecutor),
        (id, (ErrorAtPickle(),), PicklingError),
        # Check problem occuring while unpickling a task on workers
        (id, (ExitAtUnpickle(),), BrokenExecutor),
        (id, (CExitAtUnpickle(),), BrokenExecutor),
        (id, (ErrorAtUnpickle(),), BrokenExecutor),
        (id, (CrashAtUnpickle(),), BrokenExecutor),
        # Check problem occuring during function execution on workers
        (crash, (), BrokenExecutor),
        (exit, (), SystemExit),
        (c_exit, (), BrokenExecutor),
        (raise_error, (RuntimeError,), RuntimeError),
        # Check problem occuring while pickling a task result
        # on workers
        (return_instance, (CrashAtPickle,), BrokenExecutor),
        (return_instance, (ExitAtPickle,), BrokenExecutor),
        (return_instance, (CExitAtPickle,), BrokenExecutor),
        (return_instance, (ErrorAtPickle,), PicklingError),
        # Check problem occuring while unpickling a task in
        # the result_handler thread
        (return_instance, (ExitAtUnpickle,), BrokenExecutor),
        (return_instance, (ErrorAtUnpickle,), UnpicklingError),
    ]

    @pytest.mark.parametrize("func, args, expected_err", crash_cases)
    def test_crashes(self, func, args, expected_err):
        """Test various reusable_executor crash handling"""
        executor = get_reusable_executor(max_workers=2)
        res = executor.submit(func, *args)
        with pytest.raises(expected_err):
            res.result()

    @pytest.mark.parametrize("func, args, expected_exc", crash_cases)
    def test_in_callback_submit_with_crash(self, func, args, expected_exc):
        """Test the recovery from callback crash"""
        executor = get_reusable_executor(max_workers=2, timeout=12)

        def in_callback_submit(future):
            future2 = get_reusable_executor(
                max_workers=2, timeout=12).submit(func, *args)
            # Store the future of the job submitted in the callback to make it
            # easy to introspect.
            future.callback_future = future2
            future.callback_done.set()

        # Make sure the first submitted job last a bit to make sure that
        # the callback will be called in the queue manager thread and not
        # immediately in the main thread.
        delay = 0.1
        f = executor.submit(id_sleep, 42, delay)
        f.callback_done = threading.Event()
        f.add_done_callback(in_callback_submit)
        assert f.result() == 42
        if not f.callback_done.wait(timeout=3):
            raise AssertionError('callback not done before timeout')
        with pytest.raises(expected_exc):
            f.callback_future.result()

    def test_callback_crash_on_submit(self):
        """Errors in the callback execution directly in queue manager thread.

        This case can break the process executor and we want to make sure
        that we can detect the issue and recover by calling
        get_reusable_executor.
        """
        executor = get_reusable_executor(max_workers=2)

        # Make sure the first submitted job last a bit to make sure that
        # the callback will be called in the queue manager thread and not
        # immediately in the main thread.
        delay = 0.1
        f = executor.submit(id_sleep, 42, delay)
        f.add_done_callback(lambda _: exit())
        assert f.result() == 42
        with pytest.raises(BrokenExecutor):
            executor.submit(id_sleep, 42, 0.1).result()

        executor = get_reusable_executor(max_workers=2)
        f = executor.submit(id_sleep, 42, delay)
        f.add_done_callback(lambda _: raise_error())
        assert f.result() == 42
        assert executor.submit(id_sleep, 42, 0.).result() == 42

    def test_deadlock_kill(self):
        """Test deadlock recovery for reusable_executor"""
        executor = get_reusable_executor(max_workers=1, timeout=None)
        # trigger the spawning of the worker process
        executor.submit(sleep, 0.1)
        worker = next(iter(executor._processes.values()))
        with pytest.warns(UserWarning) as recorded_warnings:
            executor = get_reusable_executor(max_workers=2, timeout=None)
        assert len(recorded_warnings) == 1
        expected_msg = ("Trying to resize an executor with running jobs:"
                        " waiting for jobs completion before resizing.")
        assert recorded_warnings[0].message.args[0] == expected_msg
        os.kill(worker.pid, SIGKILL)
        wait_dead(worker)

        # wait for the executor to be able to detect the issue and set itself
        # in broken state:
        sleep(.5)
        with pytest.raises(BrokenExecutor):
            executor.submit(id_sleep, 42, 0.1).result()

        # the get_reusable_executor factory should be able to create a new
        # working instance
        executor = get_reusable_executor(max_workers=2, timeout=None)
        assert executor.submit(id_sleep, 42, 0.).result() == 42

    @pytest.mark.parametrize("n_proc", [1, 2, 5, 13])
    def test_crash_races(self, n_proc):
        """Test the race conditions in reusable_executor crash handling"""
        # Test for external crash signal comming from neighbor
        # with various race setup
        executor = get_reusable_executor(max_workers=n_proc, timeout=None)
        executor.map(id, range(n_proc))  # trigger the creation of the workers
        pids = list(executor._processes.keys())
        assert len(pids) == n_proc
        assert None not in pids
        res = executor.map(check_pids_exist_then_sleep,
                           [(.0001 * (j // 2), pids)
                            for j in range(2 * n_proc)])
        assert all(list(res))
        with pytest.raises(BrokenExecutor):
            res = executor.map(kill_friend, pids[::-1])
            list(res)

    def test_imap_handle_iterable_exception(self):
        # The catch of the errors in imap generation depend on the
        # builded version of python
        executor = get_reusable_executor(max_workers=2)
        with pytest.raises(SayWhenError):
            executor.map(id_sleep, exception_throwing_generator(10, 3),
                         chunksize=1)

        # SayWhenError seen at start of problematic chunk's results
        executor = get_reusable_executor(max_workers=2)
        with pytest.raises(SayWhenError):
            executor.map(id_sleep, exception_throwing_generator(20, 7),
                         chunksize=2)

        executor = get_reusable_executor(max_workers=2)
        with pytest.raises(SayWhenError):
            executor.map(id_sleep, exception_throwing_generator(20, 7),
                         chunksize=4)


class TestTerminateExecutor(ReusableExecutorMixin):

    def test_shutdown_kill(self):
        """Test reusable_executor termination handling"""
        from itertools import repeat
        executor = get_reusable_executor(max_workers=5)
        res1 = executor.map(id_sleep, range(100), repeat(.001))
        res2 = executor.map(id_sleep, range(100), repeat(1))
        assert list(res1) == list(range(100))

        shutdown = TimingWrapper(executor.shutdown)
        shutdown(wait=True, kill_workers=True)
        assert shutdown.elapsed < 5

        # We should get an error as the executor shutdowned before we fetched
        # all the results from the long running operation.
        with pytest.raises(ShutdownExecutorError):
            list(res2)

    def test_shutdown_deadlock(self):
        """Test recovery if killed after resize call"""
        # Test the executor.shutdown call do not cause deadlock
        executor = get_reusable_executor(max_workers=2, timeout=None)
        executor.map(id, range(2))  # start the worker processes
        executor.submit(kill_friend, (next(iter(executor._processes.keys())),
                                      .0))
        sleep(.01)
        executor.shutdown(wait=True)

    def test_kill_workers_on_new_options(self):
        # submit a long running job with no timeout
        executor = get_reusable_executor(max_workers=2, timeout=None)
        f = executor.submit(sleep, 10000)

        # change the constructor parameter while requesting not to wait
        # for the long running task to complete (the workers will get
        # shutdown forcibly)
        executor = get_reusable_executor(max_workers=2, timeout=5,
                                         kill_workers=True)
        with pytest.raises(ShutdownExecutorError):
            f.result()
        f2 = executor.submit(id_sleep, 42, 0)
        assert f2.result() == 42


class TestResizeExecutor(ReusableExecutorMixin):
    def test_reusable_executor_resize(self):
        """Test reusable_executor resizing"""

        executor = get_reusable_executor(max_workers=2, timeout=None)
        executor.map(id, range(2))

        # Decreasing the executor should drop a single process and keep one of
        # the old one as it is still in a good shape. The resize should not
        # occur while there are on going works.
        pids = list(executor._processes.keys())
        res1 = executor.submit(check_pids_exist_then_sleep, (.3, pids))
        clean_warning_registry()
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            executor = get_reusable_executor(max_workers=1, timeout=None)
            assert len(w) == 1
            expected_msg = "Trying to resize an executor with running jobs"
            assert expected_msg in str(w[0].message)
            assert res1.result(), ("Resize should wait for current processes "
                                   " to finish")
            assert len(executor._processes) == 1
            assert next(iter(executor._processes.keys())) in pids

        # Requesting the same number of process should not impact the executor
        # nor kill the processed
        old_pid = next(iter((executor._processes.keys())))
        unchanged_executor = get_reusable_executor(max_workers=1, timeout=None)
        assert len(unchanged_executor._processes) == 1
        assert unchanged_executor is executor
        assert next(iter(unchanged_executor._processes.keys())) == old_pid

        # Growing the executor again should add a single process and keep the
        # old one as it is still in a good shape
        executor = get_reusable_executor(max_workers=2, timeout=None)
        assert len(executor._processes) == 2
        assert old_pid in list(executor._processes.keys())

    def test_kill_after_resize_call(self):
        """Test recovery if killed after resize call"""
        # Test the executor resizing called before a kill arrive
        executor = get_reusable_executor(max_workers=2, timeout=None)
        executor.map(id, range(2))  # trigger the creation of worker processes
        pid = next(iter(executor._processes.keys()))
        executor.submit(kill_friend, (pid, .1))

        with pytest.warns(UserWarning) as recorded_warnings:
            executor = get_reusable_executor(max_workers=1, timeout=None)
        assert len(recorded_warnings) == 1
        expected_msg = ("Trying to resize an executor with running jobs:"
                        " waiting for jobs completion before resizing.")
        assert recorded_warnings[0].message.args[0] == expected_msg
        assert executor.submit(id_sleep, 42, 0.).result() == 42

    @pytest.mark.timeout(30 if sys.platform == "win32" else 15)
    def test_resize_after_timeout(self):
        executor = get_reusable_executor(max_workers=2, timeout=.001)
        assert executor.submit(id_sleep, 42, 0.).result() == 42
        sleep(.1)
        executor = get_reusable_executor(max_workers=8, timeout=.001)
        assert executor.submit(id_sleep, 42, 0.).result() == 42
        sleep(.1)
        executor = get_reusable_executor(max_workers=2, timeout=.001)
        assert executor.submit(id_sleep, 42, 0.).result() == 42


def test_invalid_process_number():
    """Raise error on invalid process number"""

    with pytest.raises(ValueError):
        get_reusable_executor(max_workers=0)

    with pytest.raises(ValueError):
        get_reusable_executor(max_workers=-1)


@pytest.mark.skipif(sys.platform == "win32", reason="No fork on windows")
@pytest.mark.skipif(sys.version_info <= (3, 4),
                    reason="No context before 3.4")
def test_invalid_context():
    """Raise error on invalid context"""

    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            get_reusable_executor(max_workers=2, context=get_context("fork"))


@pytest.mark.skipif(np is None, reason="requires numpy")
def test_osx_accelerate_freeze():
    """Test no freeze on OSX with Accelerate"""
    a = np.random.randn(1000, 1000)
    np.dot(a, a)
    executor = get_reusable_executor(max_workers=2)
    executor.submit(np.dot, (a, a))
    executor.shutdown(wait=True)


def test_call_item_gc_crash_or_exit():
    for bad_object in [CrashAtGCInWorker(), CExitAtGCInWorker()]:
        executor = get_reusable_executor(max_workers=1)
        f = executor.submit(id, bad_object)

        # The worker will sucessfully send back its result to the master
        # process before crashing so this future can always be collected:
        assert f.result() is not None

        # The executor should automatically detect that the worker has crashed
        # when processing subsequently dispatched tasks:
        with pytest.raises(BrokenExecutor):
            executor.submit(gc.collect).result()
            for r in executor.map(sleep, [.1] * 100):
                pass


def test_worker_timeout_with_slowly_pickling_objects(n_tasks=5):
    """Check that the worker timeout can be low without deadlocking

    In particular if dispatching call items to the queue is slow because of
    pickling large arguments, the executor should ensure that there is an
    appropriate amount of workers to move one and not get stalled.
    """
    for timeout, delay in [(0.01, 0.02), (0.01, 0.1), (0.1, 0.1)]:
        executor = get_reusable_executor(max_workers=2, timeout=timeout)
        results = list(executor.map(id, [SlowlyPickling(delay)] * n_tasks))
        assert len(results) == n_tasks


def test_pass_start_method_name_as_context():
    executor = get_reusable_executor(max_workers=2, context='loky')
    assert executor.submit(id, 42).result() >= 0

    with pytest.raises(ValueError):
        get_reusable_executor(max_workers=2, context='invalid_start_method')


def test_interactively_define_executor_no_main():
    # check that the init_main_module parameter works properly
    # when using -c option, we don't need the safeguard if __name__ ..
    # and thus test LokyProcess without the extra argument. For running
    # a script, it is necessary to use init_main_module=False.
    code = """if True:
        from loky import get_reusable_executor
        e = get_reusable_executor()
        e.submit(id, 42).result()
        print("ok")
    """
    cmd = [sys.executable]
    try:
        fid, filename = mkstemp(suffix="_joblib.py")
        os.close(fid)
        with open(filename, mode='wb') as f:
            f.write(code.encode('ascii'))
        cmd += [filename]
        check_subprocess_call(cmd, stdout_regex=r'ok', timeout=10)
    finally:
        os.unlink(filename)
