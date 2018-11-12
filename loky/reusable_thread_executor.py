# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ThreadPoolExecutor."""

__author__ = 'Brian Quinlan (brian@sweetapp.com)'

import threading
import weakref
import sys


if sys.version_info[:2] >= (3, 7):
    from concurrent.futures import _base, thread
    from concurrent.futures import ThreadPoolExecutor
else:
    from loky import _base, thread
    from loky.thread import ThreadPoolExecutor


def _worker(executor_reference, work_queue, initializer, initargs):
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical('Exception in initializer:', exc_info=True)
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
                # leave the thread pool. This comes from a resize event
                return
    except BaseException:
        _base.LOGGER.critical('Exception in worker', exc_info=True)


class _ReusableThreadPoolExecutor(ThreadPoolExecutor):

    def __init__(self, max_workers=None, executor_id=0, thread_name_prefix='',
                 initializer=None, initargs=()):
        super(_ReusableThreadPoolExecutor, self).__init__(
                max_workers=max_workers,
                initializer=initializer, initargs=initargs)
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
            thread_name = '%s_%d' % (self._thread_name_prefix or self,
                                     num_threads)
            # use our custom _worker function as a target, that can kill
            # where workers can die without shutting down the executor
            t = threading.Thread(name=thread_name, target=_worker,
                                 args=(weakref.ref(self, weakref_cb),
                                       self._work_queue,
                                       self._initializer,
                                       self._initargs))
            t.daemon = True
            t.start()
            self._threads.add(t)
            thread._threads_queues[t] = self._work_queue
