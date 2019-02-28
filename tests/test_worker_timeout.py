import sys
import time
import pytest
import threading
import warnings
import multiprocessing as mp
from loky.backend import get_context
from loky import ProcessPoolExecutor
from loky import get_reusable_executor
from loky.backend.queues import SimpleQueue
from loky.backend.reduction import dumps


class SlowlyPickling(object):
    """Simulate slowly pickling object, e.g. large numpy array or dict"""
    def __init__(self, delay=1):
        self.delay = delay

    def __reduce__(self):
        time.sleep(self.delay)
        return SlowlyPickling, (self.delay,)


class DelayedSimpleQueue(SimpleQueue):
    def __init__(self, reducers=None, ctx=None, delay=.1):
        super(DelayedSimpleQueue, self).__init__(reducers=reducers, ctx=ctx)
        self.out_queue = SimpleQueue(ctx=ctx)
        self._inner_reader = self._reader
        self._reader = self.out_queue._reader

        self._start_thread(delay)

    # Overload _start_thread to correctly call our custom _feed
    def _start_thread(self, delay):
        self._thread = threading.Thread(
            target=DelayedSimpleQueue._feed,
            args=(self._rlock, self._inner_reader, self.out_queue._writer,
                  delay),
            name='QueueDeFeederThread'
        )
        self._thread.daemon = True

        self._thread.start()

    def get(self):
        return self.out_queue.get()

    def close(self):
        self.put(None)
        self._thread.join()

    @staticmethod
    def _feed(readlock, reader, writer, delay):

        PICKLE_NONE = dumps(None)

        while True:
            with readlock:
                res = reader.recv_bytes()
            if res == PICKLE_NONE:
                break
            time.sleep(delay)
            writer.send_bytes(res)
        mp.util.debug("Defeeder clean exit")


class TestTimeoutExecutor():
    def test_worker_timeout_mock(self):
        timeout = .001
        context = get_context()
        executor = ProcessPoolExecutor(
            max_workers=4, context=context, timeout=timeout)
        result_queue = DelayedSimpleQueue(ctx=context, delay=.001)
        executor._result_queue = result_queue

        with pytest.warns(UserWarning,
                          match=r'^A worker stopped while some jobs'):
            for i in range(5):
                # Trigger worker spawn for lazy executor implementations
                for result in executor.map(id, range(8)):
                    pass

        executor.shutdown()
        result_queue.close()

    def test_worker_timeout_with_slowly_pickling_objects(self, n_tasks=5):
        """Check that the worker timeout can be low without deadlocking

        In particular if dispatching call items to the queue is slow because of
        pickling large arguments, the executor should ensure that there is an
        appropriate amount of workers to move one and not get stalled.
        """
        with pytest.warns(UserWarning,
                          match=r'^A worker stopped while some jobs'):
            for timeout, delay in [(0.01, 0.02), (0.01, 0.1), (0.1, 0.1),
                                   (0.001, .1)]:
                executor = get_reusable_executor(max_workers=2,
                                                 timeout=timeout)
                # First make sure the executor is started
                executor.submit(id, 42).result()
                results = list(executor.map(
                    id, [SlowlyPickling(delay)] * n_tasks))
                assert len(results) == n_tasks

    def test_worker_timeout_shutdown_deadlock(self):
        """Check that worker timeout don't cause deadlock when shutting down.
        """
        with warnings.catch_warnings(record=True) as record:
            with get_reusable_executor(max_workers=2, timeout=.001) as e:
                # First make sure the executor is started
                e.submit(id, 42).result()

                # Then give a task that will take more time to be pickled so
                # we are sure that the worker timeout.
                f = e.submit(id, SlowlyPickling(1))
        f.result()

        # The warning detection is unreliable on pypy
        if not hasattr(sys, "pypy_version_info"):
            assert len(record) > 0, "No warnings was emitted."
            msg = record[0].message.args[0]
            assert 'A worker stopped while some jobs' in msg
