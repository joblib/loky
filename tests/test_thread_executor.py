import os
import re
import sys
import threading
from collections import Counter

import pytest

from . import _executor_mixin
from ._test_process_executor import (
    ExecutorTest,
    mul,
    WaitTests,
    ExecutorShutdownTest,
    AsCompletedTests
)

if sys.version_info[:2] < (3, 3):
    import loky._base as futures
else:
    from concurrent import futures


class TestsThreadExecutor(ExecutorTest, _executor_mixin.ThreadExecutorMixin):
    def test_map_submits_without_iteration(self):
        """Tests verifying issue 11777."""
        finished = []

        def record_finished(n):
            finished.append(n)

        self.executor.map(record_finished, range(10))
        self.executor.shutdown(wait=True)
        assert Counter(list(finished)) == Counter(list(range(10)))

    def test_default_workers(self):
        executor = self.executor_type()
        if sys.version_info[:2] > (3, 8):
            expected = min(32, (os.cpu_count() or 1) + 4)
        else:
            expected = min(32, (os.cpu_count() or 1) * 5)
        assert executor._max_workers == expected

    def test_saturation(self):
        executor = self.executor_type(4)

        def acquire_lock(lock):
            lock.acquire()

        sem = threading.Semaphore(0)
        for i in range(15 * executor._max_workers):
            executor.submit(acquire_lock, sem)
        assert len(executor._threads) == executor._max_workers
        for i in range(15 * executor._max_workers):
            sem.release()
        executor.shutdown(wait=True)

    @pytest.mark.skipif(
        sys.version_info[:2] < (3, 8),
        reason="idle threads re-usage was introduced in Python3.8",
    )
    def test_idle_thread_reuse(self):
        executor = self.executor_type()
        executor.submit(mul, 21, 2).result()
        executor.submit(mul, 6, 7).result()
        executor.submit(mul, 3, 14).result()
        assert len(executor._threads) == 1
        executor.shutdown(wait=True)


class TestsThreadWait(WaitTests, _executor_mixin.ThreadExecutorMixin):
    def test_pending_calls_race(self):
        # Issue #14406: multi-threaded race condition when waiting on all
        # futures.
        event = threading.Event()

        def future_func():
            event.wait()

        oldswitchinterval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            fs = {self.executor.submit(future_func) for i in range(100)}
            event.set()
            futures.wait(fs, return_when=futures.ALL_COMPLETED)
        finally:
            sys.setswitchinterval(oldswitchinterval)


class TestsThreadExecutorShutdown(
    ExecutorShutdownTest, _executor_mixin.ThreadExecutorMixin
):
    def prime_executor(self):
        pass

    def test_context_manager_shutdown(self):
        with self.executor as e:
            executor = e
            assert list(e.map(abs, range(-5, 5))) == [
                5,
                4,
                3,
                2,
                1,
                0,
                1,
                2,
                3,
                4,
            ]

        for t in executor._threads:
            t.join()

    def test_del_shutdown(self):
        self.executor.map(abs, range(-5, 5))
        threads = self.executor._threads
        del self.executor

        for t in threads:
            t.join()

    def test_thread_names_assigned(self):
        executor = self.executor_type(
            max_workers=5, thread_name_prefix="SpecialPool"
        )
        executor.map(abs, range(-5, 5))
        threads = executor._threads
        del executor

        for t in threads:
            assert re.match(r"^SpecialPool_[0-4]$", t.name)
            t.join()

    def test_thread_names_default(self):
        executor = self.executor_type(max_workers=5)
        executor.map(abs, range(-5, 5))
        threads = executor._threads
        del executor

        for t in threads:
            # Ensure that our default name is reasonably sane and unique when
            # no thread_name_prefix was supplied.
            assert re.match(r"ThreadPoolExecutor-\d+_[0-4]$", t.name)
            t.join()

    def test_thread_terminate(self):
        executor = self.executor_type(max_workers=5)

        def acquire_lock(lock):
            lock.acquire()

        sem = threading.Semaphore(0)
        for i in range(3):
            executor.submit(acquire_lock, sem)
        assert len(executor._threads) == 3
        for i in range(3):
            sem.release()
        executor.shutdown()
        for t in executor._threads:
            t.join()


class TestsThreadAsCompleted(
    AsCompletedTests, _executor_mixin.ThreadExecutorMixin
):
    pass
