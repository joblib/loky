try:
    import test.support

    # Skip tests if _multiprocessing wasn't built.
    test.support.import_module('_multiprocessing')
    # Skip tests if sem_open implementation is broken.
    test.support.import_module('multiprocessing.synchronize')
    # import threading after _multiprocessing to raise a more revelant error
    # message: "No module named _multiprocessing" if multiprocessing is not
    # compiled without thread support.
    test.support.import_module('threading')
except ImportError:
    pass


# from test.support.script_helper import assert_python_ok
from loky import process_executor

import sys
import threading
import time
import pytest
import weakref
import loky._base as futures
from loky._base import (PENDING, RUNNING, CANCELLED, CANCELLED_AND_NOTIFIED,
                        FINISHED, Future)
from loky.process_executor import BrokenExecutor
import multiprocessing as mp


def create_future(state=PENDING, exception=None, result=None):
    f = Future()
    f._state = state
    f._exception = exception
    f._result = result
    return f


@pytest.yield_fixture
def exit_on_deadlock():
    try:
        TIMEOUT = 5
        from faulthandler import dump_traceback_later
        from faulthandler import cancel_dump_traceback_later
        from sys import stderr
        dump_traceback_later(timeout=TIMEOUT, exit=True, file=stderr)
        yield
        cancel_dump_traceback_later()
    except ImportError:
        yield


PENDING_FUTURE = create_future(state=PENDING)
RUNNING_FUTURE = create_future(state=RUNNING)
CANCELLED_FUTURE = create_future(state=CANCELLED)
CANCELLED_AND_NOTIFIED_FUTURE = create_future(state=CANCELLED_AND_NOTIFIED)
EXCEPTION_FUTURE = create_future(state=FINISHED, exception=OSError())
SUCCESSFUL_FUTURE = create_future(state=FINISHED, result=42)


def mul(x, y):
    return x * y


def sleep_and_raise(t):
    time.sleep(t)
    raise Exception('this is an exception')


def sleep_and_print(t, msg):
    time.sleep(t)
    print(msg)
    sys.stdout.flush()


class MyObject(object):
    def my_method(self):
        pass


class ExecutorMixin:
    worker_count = 5

    def setup_method(self, method):
        self.t1 = time.time()
        try:
            self.executor = self.executor_type(
                max_workers=self.worker_count, context=self.context,
                kill_on_shutdown=not hasattr(method, 'wait_on_shutdown'))
        except NotImplementedError as e:
            self.skipTest(str(e))
        self._prime_executor()

    def teardown_method(self, method):
        self.executor.shutdown(wait=True)
        dt = time.time() - self.t1
        print("%.2fs" % dt)
        assert dt < 60, "synchronization issue: test lasted too long"

    def _prime_executor(self):
        # Make sure that the executor is ready to do work before running the
        # tests. This should reduce the probability of timeouts in the tests.
        futures = [self.executor.submit(time.sleep, 0.1)
                   for _ in range(self.worker_count)]
        for f in futures:
            f.result()


class ExecutorShutdownTest:
    def test_run_after_shutdown(self, exit_on_deadlock):
        self.executor.shutdown()
        with pytest.raises(RuntimeError):
            self.executor.submit(pow, 2, 5)

    @pytest.mark.skipif(sys.version_info < (3, 4),
                        reason="requires python>3.4")
    def test_interpreter_shutdown(self, exit_on_deadlock):
        from .script_helper import assert_python_ok
        # Test the atexit hook for shutdown of worker threads and processes
        rc, out, err = assert_python_ok('-c', """if 1:
            from loky.process_executor import {executor_type}
            from time import sleep
            from tests._test_process_executor import sleep_and_print
            t = {executor_type}(5)
            t.submit(sleep_and_print, 1.0, "apple")
            """.format(executor_type=self.executor_type.__name__))
        # Errors in atexit hooks don't change the process exit code, check
        # stderr manually.
        assert not err
        assert out.strip() == b"apple"

    @pytest.mark.wait_on_shutdown
    def test_hang_issue12364(self, exit_on_deadlock):
        fs = [self.executor.submit(time.sleep, 0.1) for _ in range(50)]
        self.executor.shutdown()
        for f in fs:
            f.result()

    def test_processes_terminate(self, exit_on_deadlock):
        self.executor.submit(mul, 21, 2)
        self.executor.submit(mul, 6, 7)
        self.executor.submit(mul, 3, 14)
        assert len(self.executor._processes) == 5
        processes = self.executor._processes
        self.executor.shutdown()

        for p in processes.values():
            p.join()

    def test_context_manager_shutdown(self, exit_on_deadlock):
        with process_executor.ProcessPoolExecutor(max_workers=5) as e:
            processes = e._processes
            assert list(e.map(abs, range(-5, 5))) == \
                [5, 4, 3, 2, 1, 0, 1, 2, 3, 4]

        for p in processes.values():
            p.join()

    def test_del_shutdown(self, exit_on_deadlock):
        executor = process_executor.ProcessPoolExecutor(max_workers=5)
        list(executor.map(abs, range(-5, 5)))
        queue_management_thread = executor._queue_management_thread
        processes = executor._processes
        del executor

        queue_management_thread.join()
        for p in processes.values():
            p.join()


class WaitTests:

    def test_first_completed(self, exit_on_deadlock):
        future1 = self.executor.submit(mul, 21, 2)
        future2 = self.executor.submit(time.sleep, 1.5)

        done, not_done = futures.wait(
                [CANCELLED_FUTURE, future1, future2],
                return_when=futures.FIRST_COMPLETED)

        assert set([future1]) == done
        assert set([CANCELLED_FUTURE, future2]) == not_done

    def test_first_completed_some_already_completed(self, exit_on_deadlock):
        future1 = self.executor.submit(time.sleep, 1.5)

        finished, pending = futures.wait(
                 [CANCELLED_AND_NOTIFIED_FUTURE, SUCCESSFUL_FUTURE, future1],
                 return_when=futures.FIRST_COMPLETED)

        assert (set([CANCELLED_AND_NOTIFIED_FUTURE, SUCCESSFUL_FUTURE]) ==
                finished)
        assert set([future1]) == pending

    def test_first_exception(self, exit_on_deadlock):
        future1 = self.executor.submit(mul, 2, 21)
        future2 = self.executor.submit(sleep_and_raise, 1.5)
        future3 = self.executor.submit(time.sleep, 3)

        finished, pending = futures.wait(
                [future1, future2, future3],
                return_when=futures.FIRST_EXCEPTION)

        assert set([future1, future2]) == finished
        assert set([future3]) == pending

    def test_first_exception_some_already_complete(self, exit_on_deadlock):
        future1 = self.executor.submit(divmod, 21, 0)
        future2 = self.executor.submit(time.sleep, 1.5)

        finished, pending = futures.wait(
                [SUCCESSFUL_FUTURE,
                 CANCELLED_FUTURE,
                 CANCELLED_AND_NOTIFIED_FUTURE,
                 future1, future2],
                return_when=futures.FIRST_EXCEPTION)

        assert set([SUCCESSFUL_FUTURE, CANCELLED_AND_NOTIFIED_FUTURE,
                    future1]) == finished
        assert set([CANCELLED_FUTURE, future2]) == pending

    def test_first_exception_one_already_failed(self, exit_on_deadlock):
        future1 = self.executor.submit(time.sleep, 2)

        finished, pending = futures.wait(
                 [EXCEPTION_FUTURE, future1],
                 return_when=futures.FIRST_EXCEPTION)

        assert set([EXCEPTION_FUTURE]) == finished
        assert set([future1]) == pending

    def test_all_completed(self, exit_on_deadlock):
        future1 = self.executor.submit(divmod, 2, 0)
        future2 = self.executor.submit(mul, 2, 21)

        finished, pending = futures.wait(
                [SUCCESSFUL_FUTURE,
                 CANCELLED_AND_NOTIFIED_FUTURE,
                 EXCEPTION_FUTURE,
                 future1,
                 future2],
                return_when=futures.ALL_COMPLETED)

        assert set([SUCCESSFUL_FUTURE, CANCELLED_AND_NOTIFIED_FUTURE,
                    EXCEPTION_FUTURE, future1, future2]) == finished
        assert set() == pending

    def test_timeout(self, exit_on_deadlock):
        future1 = self.executor.submit(mul, 6, 7)
        future2 = self.executor.submit(time.sleep, 2)

        finished, pending = futures.wait(
                [CANCELLED_AND_NOTIFIED_FUTURE,
                 EXCEPTION_FUTURE,
                 SUCCESSFUL_FUTURE,
                 future1, future2],
                timeout=1,
                return_when=futures.ALL_COMPLETED)

        assert set([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE,
                    SUCCESSFUL_FUTURE, future1]) == finished
        assert set([future2]) == pending


class AsCompletedTests:
    # TODO(brian@sweetapp.com): Should have a test with a non-zero timeout.
    def test_no_timeout(self, exit_on_deadlock):
        future1 = self.executor.submit(mul, 2, 21)
        future2 = self.executor.submit(mul, 7, 6)

        completed = set(futures.as_completed(
                [CANCELLED_AND_NOTIFIED_FUTURE,
                 EXCEPTION_FUTURE,
                 SUCCESSFUL_FUTURE,
                 future1, future2]))
        assert set([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE,
                    SUCCESSFUL_FUTURE, future1, future2]) == completed

    def test_zero_timeout(self, exit_on_deadlock):
        future1 = self.executor.submit(time.sleep, 2)
        completed_futures = set()
        try:
            for future in futures.as_completed(
                    [CANCELLED_AND_NOTIFIED_FUTURE,
                     EXCEPTION_FUTURE,
                     SUCCESSFUL_FUTURE,
                     future1],
                    timeout=0):
                completed_futures.add(future)
        except futures.TimeoutError:
            pass

        assert set([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE,
                    SUCCESSFUL_FUTURE]) == completed_futures

    def test_duplicate_futures(self, exit_on_deadlock):
        # Issue 20367. Duplicate futures should not raise exceptions or give
        # duplicate responses.
        future1 = self.executor.submit(time.sleep, .2)
        completed = [f for f in futures.as_completed([future1, future1])]
        assert len(completed) == 1


class ExecutorTest:
    # Executor.shutdown() and context manager usage is tested by
    # ExecutorShutdownTest.
    def test_submit(self, exit_on_deadlock):
        future = self.executor.submit(pow, 2, 8)
        assert 256 == future.result()

    def test_submit_keyword(self, exit_on_deadlock):
        future = self.executor.submit(mul, 2, y=8)
        assert 16 == future.result()

    def test_map(self, exit_on_deadlock):
        assert list(self.executor.map(pow, range(10), range(10))) ==\
            list(map(pow, range(10), range(10)))

    def test_map_exception(self, exit_on_deadlock):
        i = self.executor.map(divmod, [1, 1, 1, 1], [2, 3, 0, 5])
        assert next(i), (0 == 1)
        assert next(i), (0 == 1)
        with pytest.raises(ZeroDivisionError):
            next(i)

    def test_map_timeout(self, exit_on_deadlock):
        results = []
        try:
            for i in self.executor.map(time.sleep,
                                       [0, 0, 3],
                                       timeout=1):
                results.append(i)
        except futures.TimeoutError:
            pass
        else:
            self.fail('expected TimeoutError')

        assert [None, None] == results

    def test_shutdown_race_issue12456(self, exit_on_deadlock):
        # Issue #12456: race condition at shutdown where trying to post a
        # sentinel in the call queue blocks (the queue is full while processes
        # have exited).
        self.executor.map(str, [2] * (self.worker_count + 1))
        self.executor.shutdown()

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

    def test_max_workers_negative(self, exit_on_deadlock):
        for number in (0, -1):
            with pytest.raises(ValueError) as infos:
                self.executor_type(max_workers=number)
            assert infos.value.args[0] == "max_workers must be greater than 0"

    def test_killed_child(self, exit_on_deadlock):
        # When a child process is abruptly terminated, the whole pool gets
        # "broken".
        futures = [self.executor.submit(time.sleep, 3)]
        # Get one of the processes, and terminate (kill) it
        p = next(iter(self.executor._processes.values()))
        p.terminate()
        for fut in futures:
            with pytest.raises(BrokenExecutor):
                fut.result()
        # Submitting other jobs fails as well.
        with pytest.raises(BrokenExecutor):
            self.executor.submit(pow, 2, 8)

    def test_map_chunksize(self, exit_on_deadlock):
        def bad_map():
            list(self.executor.map(pow, range(40), range(40), chunksize=-1))

        ref = list(map(pow, range(40), range(40)))
        assert list(self.executor.map(pow, range(40), range(40), chunksize=6)
                    ) == ref
        assert list(self.executor.map(pow, range(40), range(40), chunksize=50)
                    ) == ref
        assert list(self.executor.map(pow, range(40), range(40), chunksize=40)
                    ) == ref
        with pytest.raises(ValueError):
            bad_map()

    @classmethod
    def _test_traceback(cls):
        raise RuntimeError(123)  # some comment

    def test_traceback(self, exit_on_deadlock):
        # We want ensure that the traceback from the child process is
        # contained in the traceback raised in the main process.
        future = self.executor.submit(self._test_traceback)
        with pytest.raises(Exception) as cm:
            future.result()

        exc = cm.value
        assert type(exc) is RuntimeError
        assert exc.args == (123,)
        cause = exc.__cause__
        assert type(cause) is process_executor._RemoteTraceback
        assert 'raise RuntimeError(123)  # some comment' in cause.tb


class TestsFuture:
    def test_done_callback_with_result(self):
        callback_result = [None]

        def fn(callback_future):
            callback_result[0] = callback_future.result()

        f = Future()
        f.add_done_callback(fn)
        f.set_result(5)
        assert 5 == callback_result[0]

    def test_done_callback_with_exception(self):
        callback_exception = [None]

        def fn(callback_future):
            callback_exception[0] = callback_future.exception()

        f = Future()
        f.add_done_callback(fn)
        f.set_exception(Exception('test'))
        assert ('test',) == callback_exception[0].args

    def test_done_callback_with_cancel(self):
        was_cancelled = [None]

        def fn(callback_future):
            was_cancelled[0] = callback_future.cancelled()

        f = Future()
        f.add_done_callback(fn)
        assert f.cancel()
        assert was_cancelled[0]

    # @pytest.mark.skip(reason="Known failure")
    def test_done_callback_raises(self):
        # with captured_stderr() as stderr:
        raising_was_called = [False]
        fn_was_called = [False]

        def raising_fn(callback_future):
            raising_was_called[0] = True
            raise Exception('doh!')

        def fn(callback_future):
            fn_was_called[0] = True

        f = Future()
        f.add_done_callback(raising_fn)
        f.add_done_callback(fn)
        f.set_result(5)
        assert raising_was_called
        assert fn_was_called
        # assert 'Exception: doh!' in stderr.getvalue()

    def test_done_callback_already_successful(self):
        callback_result = [None]

        def fn(callback_future):
            callback_result[0] = callback_future.result()

        f = Future()
        f.set_result(5)
        f.add_done_callback(fn)
        assert 5 == callback_result[0]

    def test_done_callback_already_failed(self):
        callback_exception = [None]

        def fn(callback_future):
            callback_exception[0] = callback_future.exception()

        f = Future()
        f.set_exception(Exception('test'))
        f.add_done_callback(fn)
        assert ('test',) == callback_exception[0].args

    def test_done_callback_already_cancelled(self):
        was_cancelled = [None]

        def fn(callback_future):
            was_cancelled[0] = callback_future.cancelled()

        f = Future()
        assert f.cancel()
        f.add_done_callback(fn)
        assert was_cancelled[0]

    def test_repr(self, exit_on_deadlock):
        import re
        assert re.match('<Future at 0x[0-9a-f]+ state=pending>',
                        repr(PENDING_FUTURE)).pos > -1
        assert re.match('<Future at 0x[0-9a-f]+ state=running>',
                        repr(RUNNING_FUTURE)).pos > -1
        assert re.match('<Future at 0x[0-9a-f]+ state=cancelled>',
                        repr(CANCELLED_FUTURE)).pos > -1
        assert re.match('<Future at 0x[0-9a-f]+ state=cancelled>',
                        repr(CANCELLED_AND_NOTIFIED_FUTURE)).pos > -1
        assert re.match('<Future at 0x[0-9a-f]+ state=finished raised '
                        'OSError>', repr(EXCEPTION_FUTURE)).pos > -1
        assert re.match('<Future at 0x[0-9a-f]+ state=finished returned int>',
                        repr(SUCCESSFUL_FUTURE)).pos > -1

    def test_cancel(self, exit_on_deadlock):
        f1 = create_future(state=PENDING)
        f2 = create_future(state=RUNNING)
        f3 = create_future(state=CANCELLED)
        f4 = create_future(state=CANCELLED_AND_NOTIFIED)
        f5 = create_future(state=FINISHED, exception=OSError())
        f6 = create_future(state=FINISHED, result=5)

        assert f1.cancel()
        assert f1._state == CANCELLED

        assert not f2.cancel()
        assert f2._state == RUNNING

        assert f3.cancel()
        assert f3._state == CANCELLED

        assert f4.cancel()
        assert f4._state == CANCELLED_AND_NOTIFIED

        assert not f5.cancel()
        assert f5._state == FINISHED

        assert not f6.cancel()
        assert f6._state == FINISHED

    def test_cancelled(self, exit_on_deadlock):
        assert not PENDING_FUTURE.cancelled()
        assert not RUNNING_FUTURE.cancelled()
        assert CANCELLED_FUTURE.cancelled()
        assert CANCELLED_AND_NOTIFIED_FUTURE.cancelled()
        assert not EXCEPTION_FUTURE.cancelled()
        assert not SUCCESSFUL_FUTURE.cancelled()

    def test_done(self, exit_on_deadlock):
        assert not PENDING_FUTURE.done()
        assert not RUNNING_FUTURE.done()
        assert CANCELLED_FUTURE.done()
        assert CANCELLED_AND_NOTIFIED_FUTURE.done()
        assert EXCEPTION_FUTURE.done()
        assert SUCCESSFUL_FUTURE.done()

    def test_running(self, exit_on_deadlock):
        assert not PENDING_FUTURE.running()
        assert RUNNING_FUTURE.running()
        assert not CANCELLED_FUTURE.running()
        assert not CANCELLED_AND_NOTIFIED_FUTURE.running()
        assert not EXCEPTION_FUTURE.running()
        assert not SUCCESSFUL_FUTURE.running()

    def test_result_with_timeout(self, exit_on_deadlock):
        with pytest.raises(futures.TimeoutError):
            PENDING_FUTURE.result(timeout=0)
        with pytest.raises(futures.TimeoutError):
            RUNNING_FUTURE.result(timeout=0)
        with pytest.raises(futures.CancelledError):
            CANCELLED_FUTURE.result(timeout=0)
        with pytest.raises(futures.CancelledError):
            CANCELLED_AND_NOTIFIED_FUTURE.result(timeout=0)
        with pytest.raises(OSError):
            EXCEPTION_FUTURE.result(timeout=0)
        assert SUCCESSFUL_FUTURE.result(timeout=0) == 42

    def test_result_with_success(self, exit_on_deadlock):
        # TODO(brian@sweetapp.com): This test is timing dependent.
        def notification():
            # Wait until the main thread is waiting for the result.
            time.sleep(1)
            f1.set_result(42)

        f1 = create_future(state=PENDING)
        t = threading.Thread(target=notification)
        t.start()

        assert f1.result(timeout=5) == 42

    def test_result_with_cancel(self, exit_on_deadlock):
        # TODO(brian@sweetapp.com): This test is timing dependent.
        def notification():
            # Wait until the main thread is waiting for the result.
            time.sleep(1)
            f1.cancel()

        f1 = create_future(state=PENDING)
        t = threading.Thread(target=notification)
        t.start()

        with pytest.raises(futures.CancelledError):
            f1.result(timeout=5)

    def test_exception_with_timeout(self, exit_on_deadlock):
        with pytest.raises(futures.TimeoutError):
            PENDING_FUTURE.exception(timeout=0)
        with pytest.raises(futures.TimeoutError):
            RUNNING_FUTURE.exception(timeout=0)
        with pytest.raises(futures.CancelledError):
            CANCELLED_FUTURE.exception(timeout=0)
        with pytest.raises(futures.CancelledError):
            CANCELLED_AND_NOTIFIED_FUTURE.exception(timeout=0)
        assert isinstance(EXCEPTION_FUTURE.exception(timeout=0), OSError)
        assert SUCCESSFUL_FUTURE.exception(timeout=0) == None

    def test_exception_with_success(self, exit_on_deadlock):
        def notification():
            # Wait until the main thread is waiting for the exception.
            time.sleep(1)
            with f1._condition:
                f1._state = FINISHED
                f1._exception = OSError()
                f1._condition.notify_all()

        f1 = create_future(state=PENDING)
        t = threading.Thread(target=notification)
        t.start()

        assert isinstance(f1.exception(timeout=5), OSError)
