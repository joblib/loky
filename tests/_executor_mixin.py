import os
import time
import math
import psutil
import threading

from loky._base import TimeoutError
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


def _direct_children_with_cmdline(p):
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


def _running_children_with_cmdline(p):
    all_children = _direct_children_with_cmdline(p)
    workers = [(c, cmdline) for c, cmdline in all_children
               if (u'semaphore_tracker' not in cmdline and
                   u'multiprocessing.forkserver' not in cmdline)]

    forkservers = [c for c, cmdline in all_children
                   if u'multiprocessing.forkserver' in cmdline]
    for fs in forkservers:
        workers.extend(_direct_children_with_cmdline(fs))
    return workers


def _check_subprocesses_number(executor, expected_process_number=None,
                               expected_max_process_number=None):
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
        assert len(children_pids) == expected_process_number, pids_cmdlines
        assert len(worker_pids) == expected_process_number, pids_cmdlines
        assert worker_pids == children_pids, pids_cmdlines
    if expected_max_process_number is not None:
        assert len(children_pids) <= expected_max_process_number, pids_cmdlines
        assert len(worker_pids) <= expected_max_process_number, pids_cmdlines


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
                max_workers=self.worker_count, context=self.context)
        except NotImplementedError as e:
            self.skipTest(str(e))
        _check_executor_started(self.executor)
        _check_subprocesses_number(self.executor, self.worker_count)

    def teardown_method(self, method):
        # Make sure is not broken if it should not be
        executor = getattr(self, 'executor', None)
        if executor is not None:
            assert hasattr(method, 'broken_pool') != (
                not self.executor._flags.broken)
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
