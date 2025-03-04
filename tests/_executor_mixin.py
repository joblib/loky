import sys
import time
import math
import pytest
import threading
from time import sleep
import random

from loky import TimeoutError, get_reusable_executor
from loky.backend import get_context


try:
    import psutil

    psutil_exceptions = (psutil.NoSuchProcess, psutil.AccessDenied)
except ImportError:
    psutil = None
    psutil_exceptions = ()


# Set a large timeout as it should only be reached in case of deadlocks
TIMEOUT = 40

_test_event = None


def initializer_event(event):
    """Initializer that set a global test event for test synchronization"""
    global _test_event
    _test_event = event

    # Inject some randomness in the initialization to reveal race conditions.
    if random.random() < 0.2:
        sleep(random.random() * 0.1)  # 0-100ms


def _direct_children_with_cmdline(p):
    """Helper to fetch cmdline from children process list"""
    children_with_cmdline = []
    for c in p.children():
        try:
            cmdline = " ".join(c.cmdline())
            if not c.is_running() or not cmdline:
                # Under linux is_running() can return True even though
                # the command line data can no longer be read from
                # /proc/<pid>/cmdline. This looks like a race condition
                # between /proc/<pid>/stat and /proc/<pid>/cmdline
                # when the process is being terminated by the OS.
                continue
            children_with_cmdline.append((c, cmdline))
        except (OSError,) + psutil_exceptions:
            # These errors indicate that the process has terminated while
            # we were processing the info. Just discard it.
            pass
    return children_with_cmdline


def _running_children_pids_with_cmdline(p):
    all_children = _direct_children_with_cmdline(p)
    workers = [
        (c, cmdline)
        for c, cmdline in all_children
        if (
            "semaphore_tracker" not in cmdline
            and "resource_tracker" not in cmdline
            and "multiprocessing.forkserver" not in cmdline
        )
    ]

    forkservers = [
        c
        for c, cmdline in all_children
        if "multiprocessing.forkserver" in cmdline
    ]
    for fs in forkservers:
        workers += _direct_children_with_cmdline(fs)
    return [(w.pid, cmdline) for w, cmdline in workers]


def _check_subprocesses_number(
    executor=None,
    expected_process_number=None,
    expected_max_process_number=None,
    patience=10,
):
    if not psutil:
        # psutil is not installed, we cannot check the number of subprocesses
        return

    for trial_idx in range(patience):
        try:
            # Wait for terminating processes to disappear
            pids_cmdlines = _running_children_pids_with_cmdline(
                psutil.Process()
            )
            children_pids = {pid for pid, _ in pids_cmdlines}
            if executor is not None:
                worker_pids = set(executor._processes.keys())
                # Consistency check: all workers should be in the children list
                assert worker_pids.issubset(children_pids)
            else:
                # Bypass pids checks when executor has been garbage collected
                worker_pids = children_pids
            if expected_process_number is not None:
                assert (
                    len(children_pids) == expected_process_number
                ), pids_cmdlines
                assert (
                    len(worker_pids) == expected_process_number
                ), pids_cmdlines

            if expected_max_process_number is not None:
                assert (
                    len(children_pids) <= expected_max_process_number
                ), pids_cmdlines
                assert (
                    len(worker_pids) <= expected_max_process_number
                ), pids_cmdlines

            return
        except AssertionError as e:
            # Sometimes executor._processes or psutil seems to report
            # out-of-sync information. Let's wait a bit an try again later to see if:
            # - worker_pids is consistent with the psutil children list
            # - the number of children is consistent with the expected (max) number
            if trial_idx == patience - 1:
                raise e
            sleep(0.1)


def _check_executor_started(executor):
    # Submit a small job to make sure that the pool is an working state
    res = executor.submit(id, None)
    try:
        res.result(timeout=TIMEOUT)
    except TimeoutError:
        print(
            "\n" * 3,
            res.done(),
            executor._call_queue.empty(),
            executor._result_queue.empty(),
        )
        print(executor._processes)
        print(threading.enumerate())
        from faulthandler import dump_traceback

        dump_traceback()
        executor.submit(dump_traceback).result(TIMEOUT)
        raise RuntimeError("Executor took too long to run basic task.")


class ExecutorMixin:
    worker_count = 5

    @classmethod
    def setup_class(cls):
        print(f"setup class with {cls.context}")
        global _test_event
        if _test_event is None:
            _test_event = cls.context.Event()

    @classmethod
    def teardown_class(cls):
        print(f"teardown class with {cls.context}")
        global _test_event
        if _test_event is not None:
            _test_event = None

    @pytest.fixture(autouse=True)
    def setup_method(self):
        global _test_event
        assert _test_event is not None
        try:
            self.executor = self.executor_type(
                max_workers=self.worker_count,
                context=self.context,
                initializer=initializer_event,
                initargs=(_test_event,),
            )
        except NotImplementedError as e:
            self.skipTest(str(e))
        _check_executor_started(self.executor)
        _check_subprocesses_number(self.executor, self.worker_count)

    def teardown_method(self, method):
        # Make sure executor is not broken if it should not be
        executor = getattr(self, "executor", None)
        if executor is not None:
            try:
                # old pytest markers:
                expect_broken_pool = hasattr(method, "broken_pool")
                # new pytest markers:
                for mark in getattr(method, "pytestmark", []):
                    if mark.name == "broken_pool":
                        expect_broken_pool = True

                if expect_broken_pool:
                    for _ in range(10):
                        # The executor manager thread can take some time to
                        # mark the executor broken.
                        is_actually_broken = executor._flags.broken is not None
                        if is_actually_broken:
                            break
                        sleep(0.1)
                    else:
                        raise AssertionError(
                            "The executor was not flagged broken at the end of "
                            f" {method.__qualname__} as expected."
                        )
                else:
                    # Check that the executor is not broken right away to avoid
                    # wasting CI time. False negative should be very rare.
                    is_actually_broken = executor._flags.broken is not None
                    assert not is_actually_broken
            finally:
                # Always shutdown the executor, with SIGKILL if the executor
                # is actually broken.
                kill_workers = executor._flags.broken is not None
                t_start = time.time()
                executor.shutdown(wait=True, kill_workers=kill_workers)
                dt = time.time() - t_start
                assert dt < 10, "Executor took too long to shutdown"
                _check_subprocesses_number(executor, 0)

    def _prime_executor(self):
        # Make sure that the executor is ready to do work before running the
        # tests. This should reduce the probability of timeouts in the tests.
        futures = [
            self.executor.submit(time.sleep, 0.1)
            for _ in range(self.worker_count)
        ]
        for f in futures:
            f.result()

    @classmethod
    def check_no_running_workers(cls, patience=5, sleep_duration=0.01):
        if psutil is None:
            return

        deadline = time.time() + patience

        while time.time() <= deadline:
            time.sleep(sleep_duration)
            p = psutil.Process()
            workers = _running_children_pids_with_cmdline(p)
            if not workers:
                return

        # Patience exhausted: log the remaining workers command line and
        # raise error.
        print("Remaining worker processes command lines:", file=sys.stderr)
        for w, cmdline in workers:
            print(w.pid, w.status(), end="\n", file=sys.stderr)
            print(cmdline, end="\n\n", file=sys.stderr)
        raise AssertionError(
            f"Expected no more running worker processes but got {len(workers)}"
            f" after waiting {patience:0.3f}s."
        )


class ReusableExecutorMixin:
    def setup_method(self, method):
        default_start_method = get_context().get_start_method()
        assert default_start_method == "loky", default_start_method
        executor = get_reusable_executor(max_workers=2)
        _check_executor_started(executor)
        # There can be less than 2 workers because of the worker timeout
        _check_subprocesses_number(executor, expected_max_process_number=2)

        # Check that there no other running subprocesses beyond the workers
        # of the reusable executor.
        _check_subprocesses_number(
            executor=None, expected_max_process_number=2
        )

    def teardown_method(self, method):
        """Make sure the executor can be recovered after the tests"""
        executor = get_reusable_executor(max_workers=2)
        assert executor.submit(math.sqrt, 1).result() == 1
        # There can be less than 2 workers because of the worker timeout
        _check_subprocesses_number(executor, expected_max_process_number=2)

        # Check that there no other running subprocesses beyond the workers
        # of the reusable executor.
        _check_subprocesses_number(
            executor=None, expected_max_process_number=2
        )

    @classmethod
    def teardown_class(cls):
        executor = get_reusable_executor(max_workers=2)
        executor.shutdown(wait=True)
