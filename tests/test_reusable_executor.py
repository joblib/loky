import os
import sys
import gc
import ctypes
import psutil
import pytest
import warnings
import threading
from time import sleep
from multiprocessing import util, current_process
from pickle import PicklingError, UnpicklingError
from distutils.version import LooseVersion

import loky
from loky import cpu_count
from loky import get_reusable_executor
from loky.process_executor import _RemoteTraceback, TerminatedWorkerError
from loky.process_executor import BrokenProcessPool, ShutdownExecutorError
from loky.reusable_executor import _ReusablePoolExecutor
import cloudpickle

from ._executor_mixin import ReusableExecutorMixin
from .utils import TimingWrapper, id_sleep, check_python_subprocess_call
from .utils import filter_match

cloudpickle_version = LooseVersion(cloudpickle.__version__)

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


def c_exit(exitcode=0):
    """Induces a libc exit with exitcode 0"""
    libc.exit(exitcode)


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


def raise_error(etype=UnpicklingError, message=None):
    """Function that raises an Exception in process"""
    raise etype(message)


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
    def __init__(self, fail=True):
        self.fail = fail

    def __reduce__(self):
        if self.fail:
            raise PicklingError("Error in pickle")
        else:
            return id, (42, )


class ErrorAtUnpickle(object):
    """Bad object that triggers a process exit at unpickling time."""
    def __init__(self, etype=UnpicklingError, message='the error message'):
        self.etype = etype
        self.message = message

    def __reduce__(self):
        return raise_error, (self.etype, self.message)


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


class TestExecutorDeadLock(ReusableExecutorMixin):

    crash_cases = [
        # Check problem occuring while pickling a task in
        (id, (ExitAtPickle(),), PicklingError, None),
        (id, (ErrorAtPickle(),), PicklingError, None),
        # Check problem occuring while unpickling a task on workers
        (id, (ExitAtUnpickle(),), BrokenProcessPool, r"SystemExit"),
        (id, (CExitAtUnpickle(),), TerminatedWorkerError, r"EXIT\(0\)"),
        (id, (ErrorAtUnpickle(),), BrokenProcessPool, r"UnpicklingError"),
        (id, (CrashAtUnpickle(),), TerminatedWorkerError, r"SIGSEGV"),
        # Check problem occuring during function execution on workers
        (crash, (), TerminatedWorkerError, r"SIGSEGV"),
        (exit, (), SystemExit, None),
        (c_exit, (), TerminatedWorkerError, r"EXIT\(0\)"),
        (raise_error, (RuntimeError,), RuntimeError, None),
        # Check problem occuring while pickling a task result
        # on workers
        (return_instance, (CrashAtPickle,), TerminatedWorkerError, r"SIGSEGV"),
        (return_instance, (ExitAtPickle,), SystemExit, None),
        (return_instance, (CExitAtPickle,), TerminatedWorkerError, r"EXIT\(0\)"),
        (return_instance, (ErrorAtPickle,), PicklingError, None),
        # Check problem occuring while unpickling a task in
        # the result_handler thread
        (return_instance, (ExitAtUnpickle,), BrokenProcessPool, r"SystemExit"),
        (return_instance, (ErrorAtUnpickle,), BrokenProcessPool,
         r"UnpicklingError"),
    ]

    @pytest.mark.parametrize("func, args, expected_err, match", crash_cases)
    def test_crashes(self, func, args, expected_err, match):
        """Test various reusable_executor crash handling"""
        executor = get_reusable_executor(max_workers=2)
        res = executor.submit(func, *args)

        match_err = None
        if expected_err is TerminatedWorkerError:
            match_err = filter_match(match)
            match = None
        with pytest.raises(expected_err, match=match_err) as exc_info:
            res.result()

        # For remote traceback, ensure that the cause contains the original
        # error
        if match is not None:
            with pytest.raises(_RemoteTraceback, match=match):
                raise exc_info.value.__cause__

    @pytest.mark.parametrize("func, args, expected_err, match", crash_cases)
    def test_in_callback_submit_with_crash(self, func, args, expected_err,
                                           match):
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

        match_err = None
        if expected_err is TerminatedWorkerError:
            match_err = filter_match(match)
            match = None
        with pytest.raises(expected_err, match=match_err) as exc_info:
            f.callback_future.result()

        # For remote traceback, ensure that the cause contains the original
        # error
        if match is not None:
            with pytest.raises(_RemoteTraceback, match=match):
                raise exc_info.value.__cause__

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
        assert executor.submit(id_sleep, 42, 0.1).result() == 42

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
        with pytest.raises(TerminatedWorkerError,
                           match=filter_match(r"SIGKILL")):
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
        with pytest.raises(TerminatedWorkerError,
                           match=filter_match(r"SIGKILL")):
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

    def test_queue_full_deadlock(self):
        executor = get_reusable_executor(max_workers=1)
        fs_fail = [executor.submit(do_nothing, ErrorAtPickle(True))
                   for i in range(100)]
        fs = [executor.submit(do_nothing, ErrorAtPickle(False))
              for i in range(100)]
        with pytest.raises(PicklingError):
            fs_fail[99].result()
        assert fs[99].result()

    def test_informative_error_when_fail_at_unpickle(self):
        executor = get_reusable_executor(max_workers=2)
        obj = ErrorAtUnpickle(RuntimeError, 'message raised in child')
        f = executor.submit(id, obj)

        with pytest.raises(BrokenProcessPool) as exc_info:
            f.result()
        assert 'RuntimeError' in str(exc_info.value.__cause__)
        assert 'message raised in child' in str(exc_info.value.__cause__)

    @pytest.mark.skipif(np is None, reason="requires numpy")
    def test_numpy_dot_parent_and_child_no_freeze(self):
        """Test that no freeze happens in child process when numpy's thread
        pool is started in the parent.
        """
        a = np.random.randn(1000, 1000)
        np.dot(a, a)  # trigger the thread pool init in the parent process
        executor = get_reusable_executor(max_workers=2)
        executor.submit(np.dot, a, a).result()
        executor.shutdown(wait=True)


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

    @pytest.mark.parametrize("bad_object, match", [
        (CrashAtGCInWorker, r"SIGSEGV"), (CExitAtGCInWorker, r"EXIT\(0\)")])
    def test_call_item_gc_crash_or_exit(self, bad_object, match):
        executor = get_reusable_executor(max_workers=1)
        bad_object = bad_object()
        f = executor.submit(id, bad_object)

        # The worker will successfully send back its result to the master
        # process before crashing so this future can always be collected:
        assert f.result() is not None

        # The executor should automatically detect that the worker has crashed
        # when processing subsequently dispatched tasks:
        with pytest.raises(TerminatedWorkerError, match=filter_match(match)):
            executor.submit(gc.collect).result()
            for r in executor.map(sleep, [.1] * 100):
                pass


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

    @pytest.mark.parametrize("reuse", [True, False])
    @pytest.mark.parametrize("kill_workers", [True, False])
    def test_reusable_executor_resize_many_times(self, kill_workers, reuse):
        # Tentative non-regression test for a deadlock when shutting down
        # the workers of an executor prior to resizing it.
        kwargs = {
            'timeout': None,
            'kill_workers': kill_workers,
            'reuse': reuse,
        }
        with warnings.catch_warnings(record=True):
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            for size in [12, 2, 1, 12, 6, 1, 8, 5]:
                executor = get_reusable_executor(max_workers=size, **kwargs)
                executor.map(sleep, [0.01] * 6)
                # Do not wait for the tasks to complete.
            executor.shutdown()

    def test_kill_after_resize_call(self):
        """Test recovery if killed after resize call"""
        # Test the executor resizing called before a kill arrive
        executor = get_reusable_executor(max_workers=2, timeout=None)
        executor.map(id, range(2))  # trigger the creation of worker processes
        pid = next(iter(executor._processes.keys()))
        executor.submit(kill_friend, (pid, .1))

        with pytest.warns(UserWarning) as recorded_warnings:
            warnings.simplefilter("always")
            executor = get_reusable_executor(max_workers=1, timeout=None)
        assert len(recorded_warnings) == 1
        expected_msg = ("Trying to resize an executor with running jobs:"
                        " waiting for jobs completion before resizing.")
        assert recorded_warnings[0].message.args[0] == expected_msg
        assert executor.submit(id_sleep, 42, 0.).result() == 42
        executor.shutdown()

    def test_resize_after_timeout(self):
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            executor = get_reusable_executor(max_workers=2, timeout=.001)
            assert executor.submit(id_sleep, 42, 0.).result() == 42
            sleep(.1)
            executor = get_reusable_executor(max_workers=8, timeout=.001)
            assert executor.submit(id_sleep, 42, 0.).result() == 42
            sleep(.1)
            executor = get_reusable_executor(max_workers=2, timeout=.001)
            assert executor.submit(id_sleep, 42, 0.).result() == 42

        if len(recorded_warnings) > 1:
            expected_msg = 'A worker stopped'
            assert expected_msg in recorded_warnings[0].message.args[0]


class TestGetReusableExecutor(ReusableExecutorMixin):

    def test_invalid_process_number(self):
        """Raise error on invalid process number"""

        with pytest.raises(ValueError):
            get_reusable_executor(max_workers=0)

        with pytest.raises(ValueError):
            get_reusable_executor(max_workers=-1)

        executor = get_reusable_executor()
        with pytest.raises(ValueError):
            executor._resize(max_workers=None)

    @pytest.mark.skipif(sys.platform == "win32", reason="No fork on windows")
    @pytest.mark.skipif(sys.version_info <= (3, 4),
                        reason="No context before 3.4")
    def test_invalid_context(self):
        """Raise error on invalid context"""

        with pytest.warns(UserWarning):
            with pytest.raises(ValueError):
                get_reusable_executor(max_workers=2, context="fork")

    def test_pass_start_method_name_as_context(self):
        executor = get_reusable_executor(max_workers=2, context='loky')
        assert executor.submit(id, 42).result() >= 0

        with pytest.raises(ValueError):
            get_reusable_executor(max_workers=2, context='bad_start_method')

    def test_interactively_defined_executor_no_main(self):
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
        check_python_subprocess_call(code, stdout_regex=r"ok")

    def test_reused_flag(self):
        executor, _ = _ReusablePoolExecutor.get_reusable_executor(
            max_workers=2
        )
        executor, reused = _ReusablePoolExecutor.get_reusable_executor(
            max_workers=2
        )
        assert reused
        executor.shutdown(kill_workers=True)
        executor, reused = _ReusablePoolExecutor.get_reusable_executor(
            max_workers=2
        )
        assert not reused

    @pytest.mark.xfail(cloudpickle_version >= LooseVersion("0.5.4") and
                       cloudpickle_version <= LooseVersion("0.7.0"),
                       reason="Known issue in cloudpickle")
    # https://github.com/cloudpipe/cloudpickle/pull/240
    def test_interactively_defined_nested_functions(self):
        # Check that it's possible to call nested interactively defined
        # functions and furthermore that changing the code interactively
        # is taken into account by the single worker process.
        code = """if True:
            from loky import get_reusable_executor
            e = get_reusable_executor(max_workers=1)

            # Force a start of the children process:
            e.submit(id, 42).result()

            # Test that it's possible to call interactively defined, nested
            # functions:

            def inner_func(x):
                return -x

            def outer_func(x):
                return inner_func(x)

            assert e.submit(outer_func, 1).result() == outer_func(1) == -1

            # Test that changes to the definition of the inner function are
            # taken into account in subsequent calls to the outer function.

            def inner_func(x):
                return x

            assert e.submit(outer_func, 1).result() == outer_func(1) == 1

            print("ok")
        """
        check_python_subprocess_call(code, stdout_regex=r"ok")

    def test_interactively_defined_recursive_functions(self):
        # Check that it's possible to call a recursive function defined
        # in a closure.
        # Also check that calling several function that stems from the same
        # factory with different closure states results in the expected result:
        # the function definitions should not collapse in the single worker
        # process.
        code = """if True:
            from loky import get_reusable_executor
            e = get_reusable_executor(max_workers=1)

            # Force a start of the children process:
            e.submit(id, 42).result()

            def make_func(seed):
                def func(x):
                    if x <= 0:
                        return seed
                    return func(x - 1) + 1
                return func

            func = make_func(0)
            assert e.submit(func, 5).result() == func(5) == 5

            func = make_func(1)
            assert e.submit(func, 5).result() == func(5) == 6

            print("ok")
        """
        check_python_subprocess_call(code, stdout_regex=r"ok")

    def test_compat_with_concurrent_futures_exception(self):
        # It should be possible to use a loky process pool executor as a dropin
        # replacement for a ProcessPoolExecutor, including when catching
        # exceptions:
        concurrent = pytest.importorskip('concurrent')
        from concurrent.futures.process import BrokenProcessPool as BPPExc

        with pytest.raises(BPPExc):
            get_reusable_executor(max_workers=2).submit(crash).result()
        e = get_reusable_executor(max_workers=2)
        f = e.submit(id, 42)

        # Ensure that loky.Future are compatible with concurrent.futures
        # (see #155)
        assert isinstance(f, concurrent.futures.Future)
        (done, running) = concurrent.futures.wait([f], timeout=15)
        assert len(running) == 0

    thread_configurations = [
        ('constant', 'clean_start'),
        ('constant', 'broken_start'),
        ('varying', 'clean_start'),
        ('varying', 'broken_start'),
    ]

    @pytest.mark.parametrize("workers, executor_state", thread_configurations)
    def test_reusable_executor_thread_safety(self, workers, executor_state):
        if executor_state == 'clean_start':
            # Create a new shared executor and ensures that it's workers are
            # ready:
            get_reusable_executor(reuse=False).submit(id, 42).result()
        else:
            # Break the shared executor before launching the threads:
            with pytest.raises(TerminatedWorkerError,
                               match=filter_match(r"SIGSEGV")):
                executor = get_reusable_executor(reuse=False)
                executor.submit(return_instance, CrashAtPickle).result()

        def helper_func(output_collector, max_workers=2, n_outer_steps=5,
                        n_inner_steps=10):
            with warnings.catch_warnings():  # ignore resize warnings
                warnings.simplefilter("always")
                executor = get_reusable_executor(max_workers=max_workers)
                for i in range(n_outer_steps):
                    results = executor.map(
                        lambda x: x ** 2, range(n_inner_steps))
                    expected_result = [x ** 2 for x in range(n_inner_steps)]
                    assert list(results) == expected_result
                output_collector.append('ok')

        if workers == 'constant':
            max_workers = [2] * 10
        else:
            max_workers = [(i % 4) + 1 for i in range(10)]
        # Use the same executor with the same number of workers concurrently
        # in different threads:
        output_collector = []
        threads = [threading.Thread(
            target=helper_func, args=(output_collector, w),
            name='test_thread_%02d_max_workers_%d' % (i, w))
            for i, w in enumerate(max_workers)]

        with warnings.catch_warnings(record=True):
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert output_collector == ['ok'] * len(threads)

    def test_reusable_executor_reuse_true(self):
        executor = get_reusable_executor(max_workers=3, timeout=42)
        executor.submit(id, 42).result()
        assert len(executor._processes) == 3
        assert executor._timeout == 42

        executor2 = get_reusable_executor(reuse=True)
        executor2.submit(id, 42).result()
        assert len(executor2._processes) == 3
        assert executor2._timeout == 42
        assert executor2 is executor

        executor3 = get_reusable_executor()
        executor3.submit(id, 42).result()
        assert len(executor3._processes) == cpu_count()
        assert executor3._timeout == 10
        assert executor3 is not executor

        executor4 = get_reusable_executor()
        assert executor4 is executor3


class TestExecutorInitializer(ReusableExecutorMixin):
    def _initializer(self, x):
        loky._initialized_state = x

    def _test_initializer(self, delay=0):
        sleep(delay)
        return getattr(loky, "_initialized_state", "uninitialized")

    def test_reusable_initializer(self):
        executor = get_reusable_executor(
            max_workers=2, initializer=self._initializer, initargs=('done',))

        assert executor.submit(self._test_initializer).result() == 'done'

        # when the initializer change, the executor is re-spawned
        executor = get_reusable_executor(
            max_workers=2, initializer=self._initializer, initargs=(42,))

        assert executor.submit(self._test_initializer).result() == 42

        # With reuse=True, the executor use the same initializer
        executor = get_reusable_executor(max_workers=4, reuse=True)
        for x in executor.map(self._test_initializer, delay=.1):
            assert x == 42

        # With reuse='auto', the initializer is not used anymore
        executor = get_reusable_executor(max_workers=4)
        for x in executor.map(self._test_initializer, delay=.1):
            assert x == 'uninitialized'
