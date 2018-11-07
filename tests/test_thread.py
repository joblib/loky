import test.support

import contextlib
import itertools
import logging

# from logging.handlers import QueueHandler
# from multiprocessing import get_context
import os
import multiprocessing as mp
# import queue
import re
import sys
import threading
from tempfile import mkstemp
import time
import unittest
import weakref
from pickle import PicklingError

import pytest

from loky import _base
from loky.thread import ThreadPoolExecutor
from loky._base import (
    PENDING, RUNNING, CANCELLED, CANCELLED_AND_NOTIFIED, FINISHED, Future,
    BrokenExecutor)
from loky.backend.handlers import QueueHandler
from loky.backend.context import get_context
from loky.backend.compat import queue

from ._executor_mixin import ThreadExecutorMixin
from .utils import captured_stderr, check_subprocess_call


def create_future(state=PENDING, exception=None, result=None):
    f = Future()
    f._state = state
    f._exception = exception
    f._result = result
    return f


PENDING_FUTURE = create_future(state=PENDING)
RUNNING_FUTURE = create_future(state=RUNNING)
CANCELLED_FUTURE = create_future(state=CANCELLED)
CANCELLED_AND_NOTIFIED_FUTURE = create_future(state=CANCELLED_AND_NOTIFIED)
EXCEPTION_FUTURE = create_future(state=FINISHED, exception=OSError())
SUCCESSFUL_FUTURE = create_future(state=FINISHED, result=42)

INITIALIZER_STATUS = 'uninitialized'


def mul(x, y):
    return x * y


def sleep_and_raise(t):
    time.sleep(t)
    raise Exception('this is an exception')


def sleep_and_print(t, msg):
    time.sleep(t)
    print(msg)
    sys.stdout.flush()


def init(x):
    global INITIALIZER_STATUS
    INITIALIZER_STATUS = x


def get_init_status():
    return INITIALIZER_STATUS


def init_fail(log_queue=None):
    if log_queue is not None:
        logger = logging.getLogger('concurrent.futures')
        logger.addHandler(QueueHandler(log_queue))
        logger.setLevel('CRITICAL')
        logger.propagate = False
    time.sleep(0.1)  # let some futures be scheduled
    raise ValueError('error in initializer')


class MyObject(object):
    def my_method(self):
        pass


class EventfulGCObj():
    def __init__(self, ctx):
        mgr = get_context(ctx).Manager()
        self.event = mgr.Event()

    def __del__(self):
        self.event.set()


def make_dummy_object(_):
    return MyObject()


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self._thread_key = test.support.threading_setup()

    def tearDown(self):
        test.support.reap_children()
        test.support.threading_cleanup(*self._thread_key)


class ThreadPoolMixin(ThreadExecutorMixin):
    executor_type = ThreadPoolExecutor


def create_executor_tests(mixin, bases=(BaseTestCase,),
                          executor_mixins=(ThreadPoolMixin,)):
    def strip_mixin(name):
        if name.endswith(('Mixin', 'Tests')):
            return name[:-5]
        elif name.endswith('Test'):
            return name[:-4]
        else:
            return name

    for exe in executor_mixins:
        name = ("%s%sTest"
                % (strip_mixin(exe.__name__), strip_mixin(mixin.__name__)))
        cls = type(name, (mixin,) + (exe,) + bases, {})
        globals()[name] = cls


class InitializerMixin(ThreadExecutorMixin, object):
    worker_count = 2

    def setUp(self):
        global INITIALIZER_STATUS
        INITIALIZER_STATUS = 'uninitialized'
        self.executor_kwargs = dict(initializer=init,
                                    initargs=('initialized',))
        super(InitializerMixin, self).setUp()

    def test_initializer(self):
        futures = [self.executor.submit(get_init_status)
                   for _ in range(self.worker_count)]

        for f in futures:
            self.assertEqual(f.result(), 'initialized')


class FailingInitializerMixin(ThreadExecutorMixin, object):
    worker_count = 2

    def setUp(self):
        # In a thread pool, the child shares our logging setup
        # (see _assert_logged())
        self.mp_context = None
        self.log_queue = None
        self.executor_kwargs = dict(initializer=init_fail)
        super(FailingInitializerMixin, self).setUp()

    def test_initializer(self):
        with self._assert_logged('ValueError: error in initializer'):
            try:
                future = self.executor.submit(get_init_status)
            except BrokenExecutor:
                # Perhaps the executor is already broken
                pass
            else:
                with self.assertRaises(BrokenExecutor):
                    future.result()
            # At some point, the executor should break
            t1 = time.time()
            while not self.executor._broken:
                if time.time() - t1 > 5:
                    self.fail("executor not broken after 5 s.")
                time.sleep(0.01)
            # ... and from this point submit() is guaranteed to fail
            with self.assertRaises(BrokenExecutor):
                self.executor.submit(get_init_status)

    def _prime_executor(self):
        pass

    @contextlib.contextmanager
    def _assert_logged(self, msg):
        if self.log_queue is not None:
            yield
            output = []
            try:
                while True:
                    output.append(self.log_queue.get_nowait().getMessage())
            except queue.Empty:
                pass
        else:
            with captured_stderr() as stderr:
                import logging
                log = logging.getLogger("concurrent.futures")
                log.addHandler(logging.StreamHandler())
                # used to be called within a self.assertLogs context manager
                yield
                output = stderr.getvalue().split('\n')
                del log.handlers[:]

        self.assertTrue(any(msg in line for line in output),
                        output)


create_executor_tests(InitializerMixin)
create_executor_tests(FailingInitializerMixin)


class ExecutorShutdownTest:
    def test_run_after_shutdown(self):
        self.executor.shutdown()
        self.assertRaises(RuntimeError,
                          self.executor.submit,
                          pow, 2, 5)

    def test_interpreter_shutdown(self):
        # Test the atexit hook for shutdown of worker threads and processes
        code = """if 1:
            from loky.thread import {executor_type}
            from time import sleep
            from tests.test_thread import sleep_and_print
            if __name__ == "__main__":
                t = {executor_type}(5)
                t.submit(sleep_and_print, 1.0, "apple")
        """.format(executor_type=self.executor_type.__name__)
        cmd = [sys.executable]
        try:
            fid, filename = mkstemp(suffix="_joblib.py")
            os.close(fid)
            with open(filename, mode='wb') as f:
                f.write(code.encode('ascii'))
            cmd += [filename]
            out, err = check_subprocess_call(
                    cmd, stdout_regex=r'apple', timeout=10)
        finally:
            os.unlink(filename)

    @pytest.mark.skipif(
            sys.version_info[:4] == (3, 7, 0, 'alpha'),
            reason="The current py37-dev version of travis of python is python"
            " 3.7.0a4. This test fails for this specific version of python,"
            " however this may be a python bug because it does not fail for"
            " python3.7.0b1.")
    def test_submit_after_interpreter_shutdown(self):
        # Test the atexit hook for shutdown of worker threads and processes
        code = """if 1:
            import atexit
            @atexit.register
            def run_last():
                try:
                    t.submit(id, None)
                except RuntimeError:
                    print("runtime-error")
                    raise
            from loky.thread import {executor_type}
            if __name__ == "__main__":
                t = {executor_type}(5)
            """.format(executor_type=self.executor_type.__name__)
        cmd = [sys.executable]
        try:
            fid, filename = mkstemp(suffix="_joblib.py")
            os.close(fid)
            with open(filename, mode='wb') as f:
                f.write(code.encode('ascii'))
            cmd += [filename]
            out, err = check_subprocess_call(cmd, timeout=10)
            # Errors in atexit hooks don't change the process exit code, check
            # stderr manually.
            assert out.strip() == "runtime-error"
            assert "RuntimeError: cannot schedule new futures" in err
        finally:
            os.unlink(filename)

    def test_hang_issue12364(self):
        fs = [self.executor.submit(time.sleep, 0.1) for _ in range(50)]
        self.executor.shutdown()
        for f in fs:
            f.result()


class ThreadPoolShutdownTest(ThreadPoolMixin, ExecutorShutdownTest,
                             BaseTestCase):
    def _prime_executor(self):
        pass

    def test_threads_terminate(self):
        self.executor.submit(mul, 21, 2)
        self.executor.submit(mul, 6, 7)
        self.executor.submit(mul, 3, 14)
        self.assertEqual(len(self.executor._threads), 3)
        self.executor.shutdown()
        for t in self.executor._threads:
            t.join()

    def test_context_manager_shutdown(self):
        with ThreadPoolExecutor(max_workers=5) as e:
            executor = e
            self.assertEqual(list(e.map(abs, range(-5, 5))),
                             [5, 4, 3, 2, 1, 0, 1, 2, 3, 4])

        for t in executor._threads:
            t.join()

    def test_del_shutdown(self):
        executor = ThreadPoolExecutor(max_workers=5)
        executor.map(abs, range(-5, 5))
        threads = executor._threads
        del executor

        for t in threads:
            t.join()

    def test_thread_names_assigned(self):
        executor = ThreadPoolExecutor(
            max_workers=5, thread_name_prefix='SpecialPool')
        executor.map(abs, range(-5, 5))
        threads = executor._threads
        del executor

        for t in threads:
            r = re.compile('^SpecialPool_[0-4]$')
            assert r.search(t.name)  # TODO be more verbose
            t.join()

    def test_thread_names_default(self):
        executor = ThreadPoolExecutor(max_workers=5)
        executor.map(abs, range(-5, 5))
        threads = executor._threads
        del executor

        for t in threads:
            # Ensure that our default name is reasonably sane and unique when
            # no thread_name_prefix was supplied.
            # invalid escape char?
            r = re.compile('ThreadPoolExecutor-\d+_[0-4]')
            assert r.search(t.name)
            t.join()


class WaitTests:

    def test_first_completed(self):
        future1 = self.executor.submit(mul, 21, 2)
        future2 = self.executor.submit(time.sleep, 1.5)

        done, not_done = _base.wait(
                [CANCELLED_FUTURE, future1, future2],
                return_when=_base.FIRST_COMPLETED)

        self.assertEqual(set([future1]), done)
        self.assertEqual(set([CANCELLED_FUTURE, future2]), not_done)

    def test_first_completed_some_already_completed(self):
        future1 = self.executor.submit(time.sleep, 1.5)

        finished, pending = _base.wait(
                 [CANCELLED_AND_NOTIFIED_FUTURE, SUCCESSFUL_FUTURE, future1],
                 return_when=_base.FIRST_COMPLETED)

        self.assertEqual(
                set([CANCELLED_AND_NOTIFIED_FUTURE, SUCCESSFUL_FUTURE]),
                finished)
        self.assertEqual(set([future1]), pending)

    def test_first_exception(self):
        future1 = self.executor.submit(mul, 2, 21)
        future2 = self.executor.submit(sleep_and_raise, 1.5)
        future3 = self.executor.submit(time.sleep, 3)

        finished, pending = _base.wait(
                [future1, future2, future3],
                return_when=_base.FIRST_EXCEPTION)

        self.assertEqual(set([future1, future2]), finished)
        self.assertEqual(set([future3]), pending)

    def test_first_exception_some_already_complete(self):
        future1 = self.executor.submit(divmod, 21, 0)
        future2 = self.executor.submit(time.sleep, 1.5)

        finished, pending = _base.wait(
                [SUCCESSFUL_FUTURE,
                 CANCELLED_FUTURE,
                 CANCELLED_AND_NOTIFIED_FUTURE,
                 future1, future2],
                return_when=_base.FIRST_EXCEPTION)

        self.assertEqual(set([SUCCESSFUL_FUTURE,
                              CANCELLED_AND_NOTIFIED_FUTURE,
                              future1]), finished)
        self.assertEqual(set([CANCELLED_FUTURE, future2]), pending)

    def test_first_exception_one_already_failed(self):
        future1 = self.executor.submit(time.sleep, 2)

        finished, pending = _base.wait(
                 [EXCEPTION_FUTURE, future1],
                 return_when=_base.FIRST_EXCEPTION)

        self.assertEqual(set([EXCEPTION_FUTURE]), finished)
        self.assertEqual(set([future1]), pending)

    def test_all_completed(self):
        future1 = self.executor.submit(divmod, 2, 0)
        future2 = self.executor.submit(mul, 2, 21)

        finished, pending = _base.wait(
                [SUCCESSFUL_FUTURE,
                 CANCELLED_AND_NOTIFIED_FUTURE,
                 EXCEPTION_FUTURE,
                 future1,
                 future2],
                return_when=_base.ALL_COMPLETED)

        self.assertEqual(set([SUCCESSFUL_FUTURE,
                              CANCELLED_AND_NOTIFIED_FUTURE,
                              EXCEPTION_FUTURE,
                              future1,
                              future2]), finished)
        self.assertEqual(set(), pending)

    def test_timeout(self):
        future1 = self.executor.submit(mul, 6, 7)
        future2 = self.executor.submit(time.sleep, 6)

        finished, pending = _base.wait(
                [CANCELLED_AND_NOTIFIED_FUTURE,
                 EXCEPTION_FUTURE,
                 SUCCESSFUL_FUTURE,
                 future1, future2],
                timeout=5,
                return_when=_base.ALL_COMPLETED)

        self.assertEqual(set([CANCELLED_AND_NOTIFIED_FUTURE,
                              EXCEPTION_FUTURE,
                              SUCCESSFUL_FUTURE,
                              future1]), finished)
        self.assertEqual(set([future2]), pending)


class ThreadPoolWaitTests(ThreadPoolMixin, WaitTests, BaseTestCase):

    def test_pending_calls_race(self):
        # Issue #14406: multi-threaded race condition when waiting on all
        # futures.
        event = threading.Event()

        def future_func():
            event.wait()
        oldcheckinterval = sys.getcheckinterval()
        sys.setcheckinterval(1)
        try:
            fs = {self.executor.submit(future_func) for i in range(100)}
            event.set()
            _base.wait(fs, return_when=_base.ALL_COMPLETED)
        finally:
            sys.setcheckinterval(oldcheckinterval)


class ExecutorTest:
    # Executor.shutdown() and context manager usage is tested by
    # ExecutorShutdownTest.
    def test_submit(self):
        future = self.executor.submit(pow, 2, 8)
        self.assertEqual(256, future.result())

    def test_submit_keyword(self):
        future = self.executor.submit(mul, 2, y=8)
        self.assertEqual(16, future.result())

    def test_map(self):
        self.assertEqual(
                list(self.executor.map(pow, range(10), range(10))),
                list(map(pow, range(10), range(10))))

        self.assertEqual(
                list(self.executor.map(pow, range(10), range(10),
                                       chunksize=3)),
                list(map(pow, range(10), range(10))))

    def test_map_exception(self):
        i = self.executor.map(divmod, [1, 1, 1, 1], [2, 3, 0, 5])
        self.assertEqual(next(i), (0, 1))
        self.assertEqual(next(i), (0, 1))
        with self.assertRaises(ZeroDivisionError):
            next(i)

    def test_map_timeout(self):
        results = []
        try:
            for i in self.executor.map(time.sleep,
                                       [0, 0, 6],
                                       timeout=5):
                results.append(i)
        except _base.TimeoutError:
            pass
        else:
            self.fail('expected TimeoutError')

        self.assertEqual([None, None], results)

    def test_shutdown_race_issue12456(self):
        # Issue #12456: race condition at shutdown where trying to post a
        # sentinel in the call queue blocks (the queue is full while processes
        # have exited).
        self.executor.map(str, [2] * (self.worker_count + 1))
        self.executor.shutdown()

    @test.support.cpython_only
    def test_no_stale_references(self):
        # Issue #16284: check that the executors don't unnecessarily hang onto
        # references.
        my_object = MyObject()
        my_object_collected = threading.Event()
        my_object_callback = weakref.ref(
            my_object, lambda obj: my_object_collected.set())
        # Deliberately discarding the future.
        self.executor.submit(my_object.my_method)
        del my_object

        collected = my_object_collected.wait(timeout=5.0)
        assert collected, "Stale reference not collected within timeout."

    def test_max_workers_negative(self):
        for number in (0, -1):
            with pytest.raises(ValueError, match=r"max_workers must be "
                                                 "greater than 0"):
                self.executor_type(max_workers=number)

    # def test_free_reference(self):
    #     # Issue #14406: Result iterator should not keep an internal
    #     # reference to result objects.
    #     for obj in self.executor.map(make_dummy_object, range(10)):
    #         wr = weakref.ref(obj)
    #         del obj
    #         self.assertIsNone(wr())


class ThreadPoolExecutorTest(ThreadPoolMixin, ExecutorTest, BaseTestCase):
    def test_map_submits_without_iteration(self):
        """Tests verifying issue 11777."""
        finished = []

        def record_finished(n):
            finished.append(n)

        self.executor.map(record_finished, range(10))
        self.executor.shutdown(wait=True)

        assert len(finished) == 10 and sorted(finished) == list(range(10))

    def test_default_workers(self):
        executor = self.executor_type()
        self.assertEqual(executor._max_workers,
                         (mp.cpu_count() or 1) * 5)
