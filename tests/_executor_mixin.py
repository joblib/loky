import os
import sys
import time
import math
import psutil
import threading

from loky.reusable_executor import get_reusable_executor
from multiprocessing import cpu_count


# Compat Travis
CPU_COUNT = cpu_count()
if os.environ.get("TRAVIS_OS_NAME") is not None:
    # Hard code number of cpu in travis as cpu_count return 32 whereas we
    # only access 2 cores.
    CPU_COUNT = 2

# Compat windows
try:
    # Increase time if the test is perform on a slow machine
    TIMEOUT = max(20 / CPU_COUNT, 5)
except ImportError:
    TIMEOUT = 20


def _running_children_with_cmdline(p):
    """Helper to fetch cmdline from children process list"""
    children_with_cmdline = []
    for c in p.children():
        try:
            if not c.is_running():
                continue
            cmdline = " ".join(c.cmdline())
            children_with_cmdline.append((c, cmdline))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return children_with_cmdline


def _check_subprocesses_number(executor, expected_process_number=None,
                               expected_max_process_number=None):
    children_cmdlines = _running_children_with_cmdline(psutil.Process())
    children_pids = set(c.pid for c, cmdline in children_cmdlines
                        if ("semaphore_tracker" not in cmdline
                            and "forkserver" not in cmdline))
    worker_pids = set(executor._processes.keys())
    if expected_process_number is not None:
        assert len(children_pids) == expected_process_number
        assert len(worker_pids) == expected_process_number
        assert worker_pids == children_pids
    if expected_max_process_number is not None:
        assert len(children_pids) <= expected_max_process_number
        assert len(worker_pids) <= expected_max_process_number


def _check_executor_started(executor):
    # Submit a small job to make sure that the pool is an working state
    res = executor.submit(id, None)
    try:
        res.result(timeout=TIMEOUT)
    except TimeoutError:
        print('\n'*3, res.done(), executor._call_queue.empty(),
              executor._result_queue.empty())
        print(executor._processes)
        print(threading.enumerate())
        from faulthandler import dump_traceback
        dump_traceback()
        executor.submit(dump_traceback).result(TIMEOUT)
        raise RuntimeError("Executor took too long to run basic task.")


class ExecutorMixin:
    worker_count = 5

    def setup_method(self, method):
        try:
            self.executor = self.executor_type(
                max_workers=self.worker_count, context=self.context,
                kill_on_shutdown=not hasattr(method, 'wait_on_shutdown'))
        except NotImplementedError as e:
            self.skipTest(str(e))
        _check_executor_started(self.executor)
        if (sys.version_info < (3, 4)
                or self.context.get_start_method() != "forkserver"):
            _check_subprocesses_number(self.executor, self.worker_count)

    def teardown_method(self, method):
        # Make sure is not broken if it should not be
        assert hasattr(method, 'broken_pool') != (not self.executor._broken)
        t_start = time.time()
        self.executor.shutdown(wait=True)
        dt = time.time() - t_start
        assert dt < 10, "Executor took too long to shutdown"
        _check_subprocesses_number(self.executor, 0)

    def _prime_executor(self):
        # Make sure that the executor is ready to do work before running the
        # tests. This should reduce the probability of timeouts in the tests.
        futures = [self.executor.submit(time.sleep, 0.1)
                   for _ in range(self.worker_count)]
        for f in futures:
            f.result()


class ReusableExecutorMixin:

    def setup_method(self, method):
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
