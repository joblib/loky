from __future__ import print_function
import psutil
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
from math import sqrt
from collections import defaultdict
from threading import Thread
import traceback
import loky._base as futures
from loky._base import (PENDING, RUNNING, CANCELLED, CANCELLED_AND_NOTIFIED,
                        FINISHED, Future)
from loky.process_executor import BrokenExecutor
from ._executor_mixin import _running_children_with_cmdline


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


class ExecutorShutdownTest:
    def test_run_after_shutdown(self):
        self.executor.shutdown()
        with pytest.raises(RuntimeError):
            self.executor.submit(pow, 2, 5)

    @pytest.mark.skipif(sys.version_info < (3, 4),
                        reason="requires python>3.4")
    def test_interpreter_shutdown(self):
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
    def test_hang_issue12364(self):
        fs = [self.executor.submit(time.sleep, 0.1) for _ in range(50)]
        self.executor.shutdown()
        for f in fs:
            f.result()

    def test_processes_terminate(self):
        self.executor.submit(mul, 21, 2)
        self.executor.submit(mul, 6, 7)
        self.executor.submit(mul, 3, 14)
        assert len(self.executor._processes) == 5
        processes = self.executor._processes
        self.executor.shutdown()

        for p in processes.values():
            p.join()

    def test_context_manager_shutdown(self):
        with process_executor.ProcessPoolExecutor(max_workers=5) as e:
            processes = e._processes
            assert list(e.map(abs, range(-5, 5))) == \
                [5, 4, 3, 2, 1, 0, 1, 2, 3, 4]

        for p in processes.values():
            p.join()

    def test_del_shutdown(self):
        executor = process_executor.ProcessPoolExecutor(max_workers=5)
        list(executor.map(abs, range(-5, 5)))
        queue_management_thread = executor._queue_management_thread
        processes = executor._processes
        del executor

        queue_management_thread.join()
        for p in processes.values():
            p.join()


class WaitTests:

    def test_first_completed(self):
        future1 = self.executor.submit(mul, 21, 2)
        future2 = self.executor.submit(time.sleep, 1.5)

        done, not_done = futures.wait(
                [CANCELLED_FUTURE, future1, future2],
                return_when=futures.FIRST_COMPLETED)

        assert set([future1]) == done
        assert set([CANCELLED_FUTURE, future2]) == not_done

    def test_first_completed_some_already_completed(self):
        future1 = self.executor.submit(time.sleep, 1.5)

        finished, pending = futures.wait(
                 [CANCELLED_AND_NOTIFIED_FUTURE, SUCCESSFUL_FUTURE, future1],
                 return_when=futures.FIRST_COMPLETED)

        assert (set([CANCELLED_AND_NOTIFIED_FUTURE, SUCCESSFUL_FUTURE]) ==
                finished)
        assert set([future1]) == pending

    def test_first_exception(self):
        future1 = self.executor.submit(mul, 2, 21)
        future2 = self.executor.submit(sleep_and_raise, 1.5)
        future3 = self.executor.submit(time.sleep, 3)

        finished, pending = futures.wait(
                [future1, future2, future3],
                return_when=futures.FIRST_EXCEPTION)

        assert set([future1, future2]) == finished
        assert set([future3]) == pending

    def test_first_exception_some_already_complete(self):
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

    def test_first_exception_one_already_failed(self):
        future1 = self.executor.submit(time.sleep, 2)

        finished, pending = futures.wait(
                 [EXCEPTION_FUTURE, future1],
                 return_when=futures.FIRST_EXCEPTION)

        assert set([EXCEPTION_FUTURE]) == finished
        assert set([future1]) == pending

    def test_all_completed(self):
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

    def test_timeout(self):
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
    def test_no_timeout(self):
        future1 = self.executor.submit(mul, 2, 21)
        future2 = self.executor.submit(mul, 7, 6)

        completed = set(futures.as_completed(
                [CANCELLED_AND_NOTIFIED_FUTURE,
                 EXCEPTION_FUTURE,
                 SUCCESSFUL_FUTURE,
                 future1, future2]))
        assert set([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE,
                    SUCCESSFUL_FUTURE, future1, future2]) == completed

    def test_zero_timeout(self):
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

    def test_duplicate_futures(self):
        # Issue 20367. Duplicate futures should not raise exceptions or give
        # duplicate responses.
        future1 = self.executor.submit(time.sleep, .2)
        completed = [f for f in futures.as_completed([future1, future1])]
        assert len(completed) == 1


class ExecutorTest:
    # Executor.shutdown() and context manager usage is tested by
    # ExecutorShutdownTest.
    def test_submit(self):
        future = self.executor.submit(pow, 2, 8)
        assert 256 == future.result()

    def test_submit_keyword(self):
        future = self.executor.submit(mul, 2, y=8)
        assert 16 == future.result()

    def test_map(self):
        assert list(self.executor.map(pow, range(10), range(10))) ==\
            list(map(pow, range(10), range(10)))

    def test_map_exception(self):
        i = self.executor.map(divmod, [1, 1, 1, 1], [2, 3, 0, 5])
        assert next(i), (0 == 1)
        assert next(i), (0 == 1)
        with pytest.raises(ZeroDivisionError):
            next(i)

    def test_map_timeout(self):
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

    def test_shutdown_race_issue12456(self):
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

    def test_max_workers_negative(self):
        for number in (0, -1):
            with pytest.raises(ValueError) as infos:
                self.executor_type(max_workers=number)
            assert infos.value.args[0] == "max_workers must be greater than 0"

    def test_killed_child(self):
        # When a child process is abruptly terminated, the whole pool gets
        # "broken".
        future = self.executor.submit(time.sleep, 30)
        # Get one of the processes, and terminate (kill) it
        p = next(iter(self.executor._processes.values()))
        p.terminate()
        with pytest.raises(BrokenExecutor):
            future.result()
        # Submitting other jobs fails as well.
        with pytest.raises(BrokenExecutor):
            self.executor.submit(pow, 2, 8)

    def test_map_chunksize(self):
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

    def test_traceback(self):
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

    def _test_thread_safety(self, thread_idx, results):
        try:
            # submit a mix of very simple tasks with map and submit,
            # cancel some of them and check the results
            map_future_1 = self.executor.map(sqrt, range(40), timeout=10)
            if thread_idx % 2 == 0:
                # Make it more likely for scheduling threads to overtake one
                # another
                time.sleep(0.001)
            submit_futures = [self.executor.submit(time.sleep, 0.0001)
                              for i in range(20)]
            for i, f in enumerate(submit_futures):
                if i % 2 == 0:
                    f.cancel()
            map_future_2 = self.executor.map(sqrt, range(40), timeout=10)

            assert list(map_future_1) == [sqrt(x) for x in range(40)]
            assert list(map_future_2) == [sqrt(i) for i in range(40)]
            for i, f in enumerate(submit_futures):
                if i % 2 == 1 or not f.cancelled():
                    assert f.result(timeout=10) is None
            results[thread_idx] = 'ok'
        except Exception:
            # Ensure that py.test can report the content of the exception
            results[thread_idx] = traceback.format_exc()

#
# The following tests are new additions to the test suite originally backported
# from the Python 3 concurrent.futures package.
#

    def test_thread_safety(self):
        # Check that our process-pool executor can be shared to schedule work
        # by concurrent threads
        threads = []
        results = [None] * 10
        for i in range(len(results)):
            threads.append(Thread(target=self._test_thread_safety,
                                  args=(i, results)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for result in results:
            if result != "ok":
                raise AssertionError(result)

    @classmethod
    def return_inputs(cls, *args):
        return args

    def test_submit_from_callback(self):
        collected = defaultdict(list)
        executor = self.executor

        def _collect_and_submit_next(future):
            name, count = future.result()
            collected[name].append(count)
            if count > 0:
                future = executor.submit(self.return_inputs, name, count - 1)
                future.add_done_callback(_collect_and_submit_next)

        # Start 3 concurrent callbacks chains
        fa = executor.submit(self.return_inputs, 'chain a', 100)
        fa.add_done_callback(_collect_and_submit_next)
        fb = executor.submit(self.return_inputs, 'chain b', 50)
        fb.add_done_callback(_collect_and_submit_next)
        fc = executor.submit(self.return_inputs, 'chain c', 60)
        fc.add_done_callback(_collect_and_submit_next)
        assert fa.result() == ('chain a', 100)
        assert fb.result() == ('chain b', 50)
        assert fc.result() == ('chain c', 60)

        # Wait a maximum of 5s for the asynchronous callback chains to complete
        patience = 500
        while True:
            if (collected['chain a'] == list(range(100, -1, -1)) and
                    collected['chain b'] == list(range(50, -1, -1)) and
                    collected['chain c'] == list(range(60, -1, -1))):
                # the recursive callback chains have completed successfully
                break
            elif patience < 0:
                raise AssertionError("callback submit chains stalled at: %r"
                                     % collected)
            else:
                patience -= 1
                time.sleep(0.01)

    @classmethod
    def check_no_running_workers(cls, patience=5, sleep_duration=0.01):
        deadline = time.time() + patience

        while time.time() <= deadline:
            time.sleep(sleep_duration)
            p = psutil.Process()
            all_children = _running_children_with_cmdline(p)
            workers = [(c, cmdline) for c, cmdline in all_children
                       if (u'semaphore_tracker' not in cmdline and
                           u'multiprocessing.forkserver' not in cmdline)]

            forkservers = [c for c, cmdline in all_children
                           if u'multiprocessing.forkserver' in cmdline]
            for fs in forkservers:
                workers.extend(_running_children_with_cmdline(fs))
            if len(workers) == 0:
                return

        # Patience exhausted: log the remaining workers command line and
        # raise error.
        print("Remaining worker processes command lines:", file=sys.stderr)
        for w, cmdline in workers:
            print(cmdline, end='\n\n', file=sys.stderr)
        raise AssertionError(
            'Expected no more running worker processes but got %d after'
            ' waiting %0.3fs.'
            % (len(workers), patience))

    def test_worker_timeout(self):
        self.executor.shutdown(wait=True)
        self.check_no_running_workers(patience=5)
        try:
            self.executor = self.executor_type(
                max_workers=4, context=self.context, timeout=0.01)
        except NotImplementedError as e:
            self.skipTest(str(e))

        for i in range(5):
            # Trigger worker spawn for lazy executor implementations
            for result in self.executor.map(id, range(8)):
                pass

            # Check that all workers shutdown (via timeout) when waiting a bit:
            # note that the effictive time for Python process to completely
            # shutdown can vary a lot especially on loaded CI machines with and
            # the atexit callbacks that writes test coverage data to disk.
            # Let's be patient.
            self.check_no_running_workers(patience=5)
