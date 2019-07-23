import multiprocessing as mp
import threading
import weakref
from concurrent.futures import _base, thread
from concurrent.futures.thread import ThreadPoolExecutor

_next_thread_executor_id = 0
_thread_executor_kwargs = None
_thread_executor = None


def _get_next_thread_executor_id():
    """Ensure that each successive executor instance has a unique monotonic id.

    The purpose of this monotonic id is to help debug and test automated
    instance creation.
    """
    global _next_thread_executor_id
    thread_executor_id = _next_thread_executor_id
    _next_thread_executor_id += 1
    return thread_executor_id


def _worker(executor_reference, work_queue, initializer, initargs):
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical("Exception in initializer:", exc_info=True)
            executor = executor_reference()
            if executor is not None:
                executor._initializer_failed()
            return
    try:
        while True:
            work_item = work_queue.get(block=True)
            if work_item is not None:
                work_item.run()
                # Delete references to object. See issue16284
                del work_item
                continue
            executor = executor_reference()
            # Exit if:
            #   - The interpreter is shutting down OR
            #   - The executor that owns the worker has been collected OR
            #   - The executor that owns the worker has been shutdown.
            if thread._shutdown or executor is None or executor._shutdown:
                # Flag the executor as shutting down as early as possible if it
                # is not gc-ed yet.
                if executor is not None:
                    executor._shutdown = True
                # Notice other workers
                work_queue.put(None)
                del executor
                return
            else:
                # Leave the thread pool. This comes from a resize event.
                return
    except BaseException:
        _base.LOGGER.critical("Exception in worker", exc_info=True)


class _ReusableThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(
        self,
        max_workers=None,
        executor_id=0,
        thread_name_prefix="",
        initializer=None,
        initargs=(),
    ):
        super(_ReusableThreadPoolExecutor, self).__init__(
            max_workers=max_workers, initializer=initializer,
            initargs=initargs, thread_name_prefix=thread_name_prefix
        )
        self.executor_id = executor_id

    def _resize(self, max_workers):
        if max_workers is None:
            raise ValueError("Trying to resize with max_workers=None")
        elif max_workers == self._max_workers:
            return

        nb_children_alive = sum(t.is_alive() for t in self._threads)

        for _ in range(max_workers, nb_children_alive):
            self._work_queue.put(None)

        # The original ThreadPoolExecutor of concurrent.futures adds threads
        # lazily during `executor.submit` calls. We stick to this behavior in
        # the case where threads should be added to the pool (max_workers >
        # nb_children_alive)
        self._max_workers = max_workers

    def _adjust_thread_count(self):
        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        # TODO(bquinlan): Should avoid creating new threads if there are more
        # idle threads than items in the work queue.
        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (
                self._thread_name_prefix or self,
                num_threads,
            )
            # use our custom _worker function as a target, that can kill
            # where workers can die without shutting down the executor
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
            )
            t.daemon = True
            t.start()
            self._threads.add(t)
            thread._threads_queues[t] = self._work_queue


def get_reusable_thread_executor(
    max_workers=None, reuse="auto", initializer=None, initargs=()
):
    """Return the current _ReusableThreadPoolExectutor instance.

    Start a new instance if it has not been started already or if the previous
    instance was left in a broken state.

    If the previous instance does not have the requested number of workers, the
    executor is dynamically resized to adjust the number of workers prior to
    returning.

    Reusing a singleton instance spares the overhead of starting new worker
    threads and re-executing initializer functions each time.

    ``max_workers`` controls the maximum number of tasks that can be running in
    parallel in worker threads. By default this is set to the number of
    5 times the number of CPUs on the host.

    When provided, the ``initializer`` is run first in newly created
    threads with argument ``initargs``.
    """
    global _thread_executor, _thread_executor_kwargs
    thread_executor = _thread_executor

    if max_workers is None:
        if reuse is True and thread_executor is not None:
            max_workers = thread_executor._max_workers
        else:
            max_workers = (mp.cpu_count() or 1) * 5
    elif max_workers <= 0:
        raise ValueError(
            "max_workers must be greater than 0, got {}.".format(max_workers)
        )

    kwargs = dict(initializer=initializer, initargs=initargs)
    if thread_executor is None:
        mp.util.debug(
            "Create a thread_executor with max_workers={}.".format(max_workers)
        )
        executor_id = _get_next_thread_executor_id()
        _thread_executor_kwargs = kwargs
        _thread_executor = thread_executor = _ReusableThreadPoolExecutor(
            max_workers=max_workers, executor_id=executor_id, **kwargs
        )
    else:
        if reuse == "auto":
            reuse = kwargs == _thread_executor_kwargs
        if thread_executor._broken or thread_executor._shutdown or not reuse:
            if thread_executor._broken:
                reason = "broken"
            elif thread_executor._shutdown:
                reason = "shutdown"
            else:
                reason = "arguments have changed"
            mp.util.debug(
                "Creating a new thread_executor with max_workers={} as the "
                "previous instance cannot be reused ({}).".format(
                    max_workers, reason
                )
            )
            thread_executor.shutdown(wait=True)
            _thread_executor = thread_executor = _thread_executor_kwargs = None
            # Recursive call to build a new instance
            return get_reusable_thread_executor(
                max_workers=max_workers, **kwargs
            )
        else:
            mp.util.debug(
                "Reusing existing thread_executor with "
                "max_workers={}.".format(thread_executor._max_workers)
            )
            thread_executor._resize(max_workers)

    return thread_executor
