from __future__ import print_function
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

import os
import sys
import threading
import time
import pytest
import weakref
import faulthandler
import shutil
import tempfile
from math import sqrt
from collections import defaultdict
from threading import Thread
import traceback
from loky._base import (PENDING, RUNNING, CANCELLED, CANCELLED_AND_NOTIFIED,
                        FINISHED, Future)
from loky.process_executor import BrokenExecutor, LokyRecursionError
from .test_reusable_executor import ErrorAtPickle
from .test_reusable_executor import ExitAtPickle
from .utils import id_sleep, check_subprocess_call

if sys.version_info[:2] < (3, 3):
    import loky._base as futures
else:
    from concurrent import futures


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


def sleep_and_print(t, msg):
    time.sleep(t)
    print(msg)
    sys.stdout.flush()


def sleep_and_return(delay, x):
    time.sleep(delay)
    return x


def sleep_and_write(t, filename, msg):
    time.sleep(t)
    with open(filename, 'wb') as f:
        f.write(str(msg).encode('utf-8'))


class MyObject(object):
    def __init__(self, value=0):
        self.value = value

    def my_method(self):
        pass


class ExecutorShutdownTest:

    def test_run_after_shutdown(self):
        self.executor.shutdown()
        with pytest.raises(RuntimeError):
            self.executor.submit(pow, 2, 5)

    def test_shutdown_with_pickle_error(self):
        self.executor.shutdown()
        with self.executor_type(max_workers=4) as e:
            e.submit(id, ErrorAtPickle())

    def test_shutdown_with_sys_exit_at_pickle(self):
        self.executor.shutdown()
        with self.executor_type(max_workers=4) as e:
            e.submit(id, ExitAtPickle())

    def test_interpreter_shutdown(self):
        # Free ressources to avoid random timeout in CI
        self.executor.shutdown(wait=True, kill_workers=True)

        tempdir = tempfile.mkdtemp(prefix='loky_')
        try:
            n_jobs = 4
            code = """if True:
                from loky.process_executor import {executor_type}
                from loky.backend import get_context
                from tests._test_process_executor import sleep_and_write

                context = get_context("{start_method}")
                e = {executor_type}({n_jobs}, context=context)
                e.submit(id, 42).result()

                task_ids = list(range(2 * {n_jobs}))
                filenames = ['{tempdir}/task_{{:02}}.log'.format(i)
                             for i in task_ids]
                e.map(sleep_and_write, [0.1] * 2 * {n_jobs},
                      filenames, task_ids)

                # Do not wait for the results: garbage collect executor and
                # shutdown main Python interpreter while letting the worker
                # processes finish in the background.
            """
            code = code.format(executor_type=self.executor_type.__name__,
                               start_method=self.context.get_start_method(),
                               n_jobs=n_jobs,
                               tempdir=tempdir.replace("\\", "/"))
            stdout, stderr = check_subprocess_call(
                [sys.executable, "-c", code], timeout=10)

            # On OSX, remove UserWarning for broken semaphores
            if sys.platform == "darwin":
                stderr = [e for e in stderr.strip().split("\n")
                          if "increase its maximal value" not in e]
            assert len(stderr) == 0 or stderr[0] == ''

            # The workers should have completed their work before the main
            # process exits:
            expected_filenames = ['task_%02d.log' % i
                                  for i in range(2 * n_jobs)]

            # Apparently files can take some time to appear under windows
            # on AppVeyor
            for retry_idx in range(20):
                filenames = sorted(os.listdir(tempdir))
                if len(filenames) != len(expected_filenames):
                    time.sleep(1)
                else:
                    break

            assert filenames == expected_filenames
            for i, filename in enumerate(filenames):
                with open(os.path.join(tempdir, filename), 'rb') as f:
                    assert int(f.read().strip()) == i
        finally:
            shutil.rmtree(tempdir)

    def test_hang_issue12364(self):
        fs = [self.executor.submit(time.sleep, 0.01) for _ in range(50)]
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

    def test_processes_terminate_on_executor_gc(self):
        results = self.executor.map(sleep_and_return,
                                    [0.1] * 10, range(10))
        assert len(self.executor._processes) == 5
        processes = self.executor._processes
        executor_flags = self.executor._flags

        # The following should trigger GC and therefore shutdown of workers.
        # However the shutdown wait for all the pending jobs to complete
        # first.
        executor_reference = weakref.ref(self.executor)
        self.executor = None

        # Make sure that there is not other reference to the executor object.
        # We have to be patient as _thread_management_worker might have a
        # reference when we deleted self.executor.
        t_deadline = time.time() + 1
        while executor_reference() is not None and time.time() < t_deadline:
            time.sleep(0.001)
        assert executor_reference() is None

        # The remaining jobs should still be processed in the background
        for result, expected in zip(results, range(10)):
            assert result == expected

        # Once all pending jobs have completed the executor and threads should
        # terminate automatically.
        self.check_no_running_workers(patience=2)
        assert executor_flags.shutdown, processes
        assert not executor_flags.broken, processes

    @classmethod
    def _wait_and_crash(cls, event):
        event.wait()
        faulthandler._sigsegv()

    def test_processes_crash_handling_after_executor_gc(self):
        # Start 5 easy jobs on 5 workers
        results = self.executor.map(sleep_and_return,
                                    [0.01] * 5, range(5))

        # Enqueue a job that will trigger a crash of one of the workers.
        # Make sure this crash does not happen before the non-failing jobs
        # have returned their results by using and multiprocessing Event
        # instance
        if self.context.get_start_method() != "fork":
            manager = self.context.Manager()
            event = manager.Event()
        else:
            manager = None
            event = self.context.Event()

        crash_result = self.executor.submit(self._wait_and_crash, event)
        assert len(self.executor._processes) == 5
        processes = self.executor._processes
        executor_flags = self.executor._flags

        # The following should trigger the GC and therefore shutdown of
        # workers. However the shutdown wait for all the pending jobs to
        # complete first.
        executor_reference = weakref.ref(self.executor)
        self.executor = None

        # Make sure that there is not other reference to the executor object.
        assert executor_reference() is None

        # The remaining jobs should still be processed in the background
        for result, expected in zip(results, range(5)):
            assert result == expected

        # Let the crash job know that it can crash now
        event.set()

        # The crashing job should be executed after the non-failing jobs
        # have completed. The crash should be detected.
        with pytest.raises(BrokenExecutor):
            crash_result.result()

        if manager is not None:
            manager.shutdown()
            manager.join()

        # The executor flag should have been set at this point.
        assert executor_flags.broken, processes

        # Once all pending jobs have completed the executor and threads should
        # terminate automatically.
        self.check_no_running_workers(patience=2)

    def test_context_manager_shutdown(self):
        with self.executor_type(max_workers=5, context=self.context) as e:
            processes = e._processes
            assert list(e.map(abs, range(-5, 5))) == \
                [5, 4, 3, 2, 1, 0, 1, 2, 3, 4]

        for p in processes.values():
            p.join()

    def test_del_shutdown(self):
        executor = self.executor_type(max_workers=5, context=self.context)
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

        done, not_done = futures.wait([CANCELLED_FUTURE, future1, future2],
                                      return_when=futures.FIRST_COMPLETED)

        assert set([future1]) == done
        assert set([CANCELLED_FUTURE, future2]) == not_done

    def test_first_completed_some_already_completed(self):
        future1 = self.executor.submit(time.sleep, 1.5)

        finished, pending = futures.wait([CANCELLED_AND_NOTIFIED_FUTURE,
                                          SUCCESSFUL_FUTURE, future1],
                                         return_when=futures.FIRST_COMPLETED)

        assert (set([CANCELLED_AND_NOTIFIED_FUTURE, SUCCESSFUL_FUTURE]) ==
                finished)
        assert set([future1]) == pending

    @classmethod
    def wait_and_raise(cls, ev, t):
        ev.wait(t)
        raise Exception('this is an exception')

    def test_first_exception(self):
        manager = self.context.Manager()
        event = manager.Event()
        future1 = self.executor.submit(mul, 2, 21)
        future2 = self.executor.submit(self.wait_and_raise, event, 1.5)
        future3 = self.executor.submit(time.sleep, 3)

        def cb_done(f):
            event.set()
        future1.add_done_callback(cb_done)

        finished, pending = futures.wait([future1, future2, future3],
                                         return_when=futures.FIRST_EXCEPTION)

        assert event.is_set()

        assert set([future1, future2]) == finished
        assert set([future3]) == pending

        manager.shutdown()

    def test_first_exception_some_already_complete(self):
        future1 = self.executor.submit(divmod, 21, 0)
        future2 = self.executor.submit(time.sleep, 1.5)

        finished, pending = futures.wait([SUCCESSFUL_FUTURE, CANCELLED_FUTURE,
                                          CANCELLED_AND_NOTIFIED_FUTURE,
                                          future1, future2],
                                         return_when=futures.FIRST_EXCEPTION)

        assert set([SUCCESSFUL_FUTURE, CANCELLED_AND_NOTIFIED_FUTURE,
                    future1]) == finished
        assert set([CANCELLED_FUTURE, future2]) == pending

    def test_first_exception_one_already_failed(self):
        future1 = self.executor.submit(time.sleep, 2)

        finished, pending = futures.wait([EXCEPTION_FUTURE, future1],
                                         return_when=futures.FIRST_EXCEPTION)

        assert set([EXCEPTION_FUTURE]) == finished
        assert set([future1]) == pending

    def test_all_completed(self):
        future1 = self.executor.submit(divmod, 2, 0)
        future2 = self.executor.submit(mul, 2, 21)

        finished, pending = futures.wait([SUCCESSFUL_FUTURE, EXCEPTION_FUTURE,
                                          CANCELLED_AND_NOTIFIED_FUTURE,
                                          future1, future2],
                                         return_when=futures.ALL_COMPLETED)

        assert set([SUCCESSFUL_FUTURE, CANCELLED_AND_NOTIFIED_FUTURE,
                    EXCEPTION_FUTURE, future1, future2]) == finished
        assert set() == pending

    def test_timeout(self):
        # Make sure the executor has already started to avoid timeout happening
        # before future1 returns
        assert self.executor.submit(id_sleep, 42).result() == 42

        future1 = self.executor.submit(mul, 6, 7)
        future2 = self.executor.submit(time.sleep, 5)

        finished, pending = futures.wait([CANCELLED_AND_NOTIFIED_FUTURE,
                                          EXCEPTION_FUTURE, SUCCESSFUL_FUTURE,
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

        completed = set(futures.as_completed([CANCELLED_AND_NOTIFIED_FUTURE,
                                              EXCEPTION_FUTURE,
                                              SUCCESSFUL_FUTURE,
                                              future1, future2]))
        assert set([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE,
                    SUCCESSFUL_FUTURE, future1, future2]) == completed

    def test_zero_timeout(self):
        future1 = self.executor.submit(time.sleep, 2)
        completed_futures = set()
        with pytest.raises(futures.TimeoutError):
            for future in futures.as_completed(
                    [CANCELLED_AND_NOTIFIED_FUTURE,
                     EXCEPTION_FUTURE,
                     SUCCESSFUL_FUTURE,
                     future1],
                    timeout=0):
                completed_futures.add(future)

        assert set([CANCELLED_AND_NOTIFIED_FUTURE, EXCEPTION_FUTURE,
                    SUCCESSFUL_FUTURE]) == completed_futures

    def test_duplicate_futures(self):
        # Issue 20367. Duplicate futures should not raise exceptions or give
        # duplicate responses.
        future1 = self.executor.submit(time.sleep, .1)
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
        with pytest.raises(futures.TimeoutError):
            for i in self.executor.map(time.sleep,
                                       [0, 0, 5],
                                       timeout=1):
                results.append(i)

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
        collect = threading.Event()
        _ = weakref.ref(my_object, lambda obj: collect.set())  # noqa
        # Deliberately discarding the future.
        self.executor.submit(my_object.my_method)
        del my_object

        collected = collect.wait(timeout=5.0)
        assert collected, "Stale reference not collected within timeout."

    def test_max_workers_negative(self):
        for number in (0, -1):
            with pytest.raises(ValueError) as infos:
                self.executor_type(max_workers=number)
            assert infos.value.args[0] == "max_workers must be greater than 0"

    @pytest.mark.broken_pool
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

    #
    # The following tests are new additions to the test suite originally
    # backported from the Python 3 concurrent.futures package.
    #

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

    @pytest.mark.timeout(50 if sys.platform == "win32" else 25)
    def test_worker_timeout(self):
        self.executor.shutdown(wait=True)
        self.check_no_running_workers(patience=5)
        timeout = getattr(self, 'min_worker_timeout', .01)
        try:
            self.executor = self.executor_type(
                max_workers=4, context=self.context, timeout=timeout)
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

    @classmethod
    def reducer_in(cls, obj):
        return MyObject, (obj.value + 5, )

    @classmethod
    def reducer_out(cls, obj):
        return MyObject, (7 * obj.value, )

    def test_serialization(self):
        """Test custom serialization for process_executor"""
        self.executor.shutdown(wait=True)

        # Use non commutative operation to check correct order
        job_reducers = {}
        job_reducers[MyObject] = self.reducer_in
        result_reducers = {}
        result_reducers[MyObject] = self.reducer_out

        # Create a new executor to ensure that we did not mess with the
        # existing module level serialization
        executor = self.executor_type(
            max_workers=2, context=self.context, job_reducers=job_reducers,
            result_reducers=result_reducers
        )
        self.executor = self.executor_type(max_workers=2, context=self.context)

        obj = MyObject(1)
        ret_obj = self.executor.submit(self.return_inputs, obj).result()[0]
        ret_obj_custom = executor.submit(self.return_inputs, obj).result()[0]

        assert ret_obj.value == 1
        assert ret_obj_custom.value == 42

        executor.shutdown(wait=True)

    @classmethod
    def _test_max_depth(cls, max_depth=10, ctx=None):
        if max_depth == 0:
            return 42
        executor = cls.executor_type(1, context=ctx)
        f = executor.submit(cls._test_max_depth, max_depth - 1, ctx)
        try:
            return f.result()
        finally:
            executor.shutdown(wait=True, kill_workers=True)

    def test_max_depth(self):
        from loky.process_executor import MAX_DEPTH
        if self.context.get_start_method() == 'fork':
            # For 'fork', we do not allow nested process as the threads ends
            # up in messy states
            with pytest.raises(LokyRecursionError):
                self._test_max_depth(max_depth=2, ctx=self.context)
            return

        assert self._test_max_depth(max_depth=MAX_DEPTH,
                                    ctx=self.context) == 42

        with pytest.raises(LokyRecursionError):
            self._test_max_depth(max_depth=MAX_DEPTH + 1, ctx=self.context)
