from __future__ import print_function
import os
import sys
import time
import math
import psutil
import pytest
import threading

from loky._base import TimeoutError
from loky.backend import get_context
from loky import get_reusable_executor, cpu_count


# Compat Travis
CPU_COUNT = cpu_count()
if os.environ.get("TRAVIS_OS_NAME") is not None and sys.version_info < (3, 4):
    # Hard code number of cpu in travis as cpu_count return 32 whereas we
    # only access 2 cores.
    # This is done automatically by cpu_count for Python >= 3.4
    CPU_COUNT = 2

# Set a large timeout as it should only be reached in case of deadlocks
TIMEOUT = 40

_test_event = None


def initializer_event(event):
    """Initializer that set a global test event for test synchronization"""
    global _test_event
    _test_event = event


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
        except (OSError, psutil.NoSuchProcess, psutil.AccessDenied):
            # These errors indicate that the process has terminated while
            # we were processing the info. Just discard it.
            pass
    return children_with_cmdline


def _running_children_with_cmdline(p):
    all_children = _direct_children_with_cmdline(p)
    workers = [(c, cmdline) for c, cmdline in all_children
               if (u'semaphore_tracker' not in cmdline and
                   u'resource_tracker' not in cmdline and
                   u'multiprocessing.forkserver' not in cmdline)]

    forkservers = [c for c, cmdline in all_children
                   if u'multiprocessing.forkserver' in cmdline]
    for fs in forkservers:
        workers.extend(_direct_children_with_cmdline(fs))
    return workers


def _check_subprocesses_number(executor, expected_process_number=None,
                               expected_max_process_number=None, patience=100):
    # Wait for terminating processes to disappear
    children_cmdlines = _running_children_with_cmdline(psutil.Process())
    pids_cmdlines = [(c.pid, cmdline) for c, cmdline in children_cmdlines]
    children_pids = set(pid for pid, _ in pids_cmdlines)
    if executor is not None:
        worker_pids = set(executor._processes.keys())
    else:
        # Bypass pids checks when executor has been garbage
        # collected
        worker_pids = children_pids
    if expected_process_number is not None:
        try:
            assert len(children_pids) == expected_process_number, pids_cmdlines
            assert len(worker_pids) == expected_process_number, pids_cmdlines
            assert worker_pids == children_pids, pids_cmdlines
        except AssertionError:
            if expected_process_number != 0:
                raise
            # there is a race condition with the /proc/<pid>/ system clean up
            # and our utilization of psutil. The Process is considered alive by
            # psutil even though it have been terminated. Wait for the system
            # clean up in this case.
            for _ in range(patience):
                if len(_running_children_with_cmdline(psutil.Process())) == 0:
                    break
                time.sleep(.1)
            else:
                raise

    if expected_max_process_number is not None:
        assert len(children_pids) <= expected_max_process_number, pids_cmdlines
        assert len(worker_pids) <= expected_max_process_number, pids_cmdlines


def _check_executor_started(executor):
    # Submit a small job to make sure that the pool is an working state
    res = executor.submit(id, None)
    try:
        res.result(timeout=TIMEOUT)
    except TimeoutError:
        print('\n' * 3, res.done(), executor._call_queue.empty(),
              executor._result_queue.empty())
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
        print("setup class with {}".format(cls.context))
        global _test_event
        if _test_event is None:
            _test_event = cls.context.Event()

    @classmethod
    def teardown_class(cls):
        print("teardown class with {}".format(cls.context))
        global _test_event
        if _test_event is not None:
            _test_event = None

    @pytest.fixture(autouse=True)
    def setup_method(self):
        global _test_event
        assert _test_event is not None
        try:
            self.executor = self.executor_type(
                max_workers=self.worker_count, context=self.context,
                initializer=initializer_event, initargs=(_test_event,))
        except NotImplementedError as e:
            self.skipTest(str(e))
        _check_executor_started(self.executor)
        _check_subprocesses_number(self.executor, self.worker_count)

    def teardown_method(self, method):
        # Make sure executor is not broken if it should not be
        executor = getattr(self, 'executor', None)
        if executor is not None:
            expect_broken_pool = hasattr(method, "broken_pool")  # old pytest
            for mark in getattr(method, "pytestmark", []):
                if mark.name == "broken_pool":
                    expect_broken_pool = True
            is_actually_broken = executor._flags.broken is not None
            assert is_actually_broken == expect_broken_pool

            t_start = time.time()
            executor.shutdown(wait=True, kill_workers=True)
            dt = time.time() - t_start
            assert dt < 10, "Executor took too long to shutdown"
        _check_subprocesses_number(executor, 0)

    def _prime_executor(self):
        # Make sure that the executor is ready to do work before running the
        # tests. This should reduce the probability of timeouts in the tests.
        futures = [self.executor.submit(time.sleep, 0.1)
                   for _ in range(self.worker_count)]
        for f in futures:
            f.result()

    @classmethod
    def check_no_running_workers(cls, patience=5, sleep_duration=0.01):
        deadline = time.time() + patience

        while time.time() <= deadline:
            time.sleep(sleep_duration)
            p = psutil.Process()
            workers = _running_children_with_cmdline(p)
            if len(workers) == 0:
                return

        # Patience exhausted: log the remaining workers command line and
        # raise error.
        print("Remaining worker processes command lines:", file=sys.stderr)
        for w, cmdline in workers:
            print(w.pid, w.status(), end='\n', file=sys.stderr)
            print(cmdline, end='\n\n', file=sys.stderr)
        raise AssertionError(
            'Expected no more running worker processes but got %d after'
            ' waiting %0.3fs.'
            % (len(workers), patience))


class ReusableExecutorMixin:

    def setup_method(self, method):
        default_start_method = get_context().get_start_method()
        assert default_start_method == "loky", default_start_method
        executor = get_reusable_executor(max_workers=2)
        _check_executor_started(executor)
        # There can be less than 2 workers because of the worker timeout
        _check_subprocesses_number(executor, expected_max_process_number=2)

    def teardown_method(self, method):
        """Make sure the executor can be recovered after the tests"""
        executor = get_reusable_executor(max_workers=2)
        assert executor.submit(math.sqrt, 1).result() == 1
        # There can be less than 2 workers because of the worker timeout
        _check_subprocesses_number(executor, expected_max_process_number=2)

    @classmethod
    def teardown_class(cls):
        executor = get_reusable_executor(max_workers=2)
        executor.shutdown(wait=True)
