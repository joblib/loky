# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ProcessPoolExecutor.

The follow diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |    => |        |  => | Call Q    | => |         |
|          |     +----------+       |        |     +-----------+    |         |
|          |     | ...      |       |        |     | ...       |    |         |
|          |     | 6        |       |        |     | 5, call() |    |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     |        |     | 4, result |    |         |
|          |     | ...        |     |        |     | 3, except |    |         |
+----------+     +------------+     +--------+     +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""


import atexit
import time
import os
from concurrent.futures import _base
import queue
from queue import Full
import multiprocessing as mp
try:
    from multiprocessing import get_context
except ImportError:
    def get_context():
        return mp
from multiprocessing.connection import wait
import threading
import weakref
from functools import partial
import itertools
import traceback


__author__ = 'Brian Quinlan (brian@sweetapp.com)'

# Workers are created as daemon threads and processes. This is done to allow
# the interpreter to exit when there are still idle processes in a
# ProcessPoolExecutor's process pool (i.e. shutdown() was not called). However,
# allowing workers to die with the interpreter has two undesirable properties:
#   - The workers would still be running during interpretor shutdown,
#     meaning that they would fail in unpredictable ways.
#   - The workers could be killed while evaluating a work item, which could
#     be bad if the callable being evaluated has external side-effects e.g.
#     writing to a file.
#
# To work around this problem, an exit handler is installed which tells the
# workers to exit when their work queues are empty and then waits until the
# threads/processes finish.

_threads_queues = weakref.WeakKeyDictionary()
_shutdown = False


def _is_crashed(thread):
    """helper to check if a thread is started for any version of python"""
    if thread is None:
        return False
    if hasattr(thread, "_started"):
        return thread._started.is_set() and not thread.is_alive()
    # Backward compat for python 2.7
    return thread._Thread__started.is_set() and not thread.is_alive()


def _python_exit():
    global _shutdown
    _shutdown = True
    items = list(_threads_queues.items())
    for t, q in items:
        q.put(None)
    for t, q in items:
        t.join()

# Controls how many more calls than processes will be queued in the call queue.
# A smaller number will mean that processes spend more time idle waiting for
# work while a larger number will make Future.cancel() succeed less frequently
# (Futures in the call queue cannot be cancelled).
EXTRA_QUEUED_CALLS = 1

# Hack to embed stringification of remote traceback in local traceback


class _RemoteTraceback(Exception):

    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        return self.tb


class _ExceptionWithTraceback:

    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        self.tb = '\n"""\n%s"""' % tb

    def __reduce__(self):
        return _rebuild_exc, (self.exc, self.tb)


def _rebuild_exc(exc, tb):
    exc.__cause__ = _RemoteTraceback(tb)
    return exc


class _WorkItem(object):

    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class _ResultItem(object):

    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result


class _CallItem(object):

    def __init__(self, work_id, fn, args, kwargs):
        self.work_id = work_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


def _get_chunks(chunksize, *iterables):
    """ Iterates over zip()ed iterables in chunks. """
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def _process_chunk(fn, chunk):
    """ Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _process_worker(call_queue, result_queue):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A multiprocessing.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A multiprocessing.Queue of _ResultItems that will written
            to by the worker.
        shutdown: A multiprocessing.Event that will be set as a signal to the
            worker that it should exit when call_queue is empty.
    """
    while True:
        try:
            call_item = call_queue.get(block=True)
        except BaseException as e:
            traceback.print_exc()
            return
        if call_item is None:
            # Wake up queue management thread
            result_queue.put(os.getpid())
            return
        try:
            r = call_item.fn(*call_item.args, **call_item.kwargs)
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, e.__traceback__)
            result_queue.put(_ResultItem(call_item.work_id, exception=exc))
        else:
            try:
                result_queue.put(_ResultItem(call_item.work_id, result=r))
            except BaseException as e:
                traceback.print_exc()
                return


def _add_call_item_to_queue(pending_work_items,
                            running_work_item,
                            work_ids,
                            call_queue):
    """Fills call_queue with _WorkItems from pending_work_items.

    This function never blocks.

    Args:
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids: A queue.Queue of work ids e.g. Queue([5, 6, ...]). Work ids
            are consumed and the corresponding _WorkItems from
            pending_work_items are transformed into _CallItems and put in
            call_queue.
        call_queue: A multiprocessing.Queue that will be filled with _CallItems
            derived from _WorkItems.
    """
    while True:
        if call_queue.full():
            return
        try:
            work_id = work_ids.get(block=False)
        except queue.Empty:
            return
        else:
            work_item = pending_work_items[work_id]

            if work_item.future.set_running_or_notify_cancel():
                call_queue.put(_CallItem(work_id,
                                         work_item.fn,
                                         work_item.args,
                                         work_item.kwargs),
                               block=True)
                running_work_item += [work_id]
            else:
                del pending_work_items[work_id]
                continue


def _queue_management_worker(executor_reference,
                             processes,
                             pending_work_items,
                             work_ids_queue,
                             call_queue,
                             result_queue,
                             com_queue,
                             kill_on_shutdown=False):
    """Manages the communication between this process and the worker processes.

    This function is run in a local thread.

    Args:
        executor_reference: A weakref.ref to the ProcessPoolExecutor that owns
            this thread. Used to determine if the ProcessPoolExecutor has been
            garbage collected and that this function can exit.
        process: A list of the multiprocessing.Process instances used as
            workers.
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids_queue: A queue.Queue of work ids e.g. Queue([5, 6, ...]).
        call_queue: A multiprocessing.Queue that will be filled with _CallItems
            derived from _WorkItems for processing by the process workers.
        result_queue: A multiprocessing.Queue of _ResultItems generated by the
            process workers.
        com_queue: A multiprocessing.SimpleQueue for communication purposes
            between the main Thread and this one to avoid deadlock.
        kill_on_shutdown: A bool modifying the behavior of the executor. If
            set to True, all pending jobs will be canceled on shutdown for an
            immediate completion.
    """
    executor = None
    running_work_item = []

    def shutting_down():
        return _shutdown or executor is None or executor._shutdown_thread

    def shutdown_worker():
        # This is an upper bound
        nb_children_alive = sum(p.is_alive() for p in processes.values())
        try:
            for i in range(0, nb_children_alive):
                call_queue.put_nowait(None)
        except queue.Full:
            pass
        # Release the queue's resources as soon as possible.
        call_queue.close()
        # If .join() is not called on the created processes then
        # some multiprocessing.Queue methods may deadlock on Mac OS X.

        for p in processes.values():
            p.join()

    reader = result_queue._reader
    com = com_queue._reader

    while True:
        _add_call_item_to_queue(pending_work_items,
                                running_work_item,
                                work_ids_queue,
                                call_queue)

        sentinels = [p.sentinel for p in processes.values()]
        assert sentinels
        ready = wait([reader, com] + sentinels)
        # broken = not call_queue._thread.is_alive()
        # broken |= any([p.exitcode for p in processes.values()])
        if reader in ready:
            try:
                result_item = reader.recv()
            except Exception as exc:
                result_item = None
                for work_id in running_work_item:
                    work_item = pending_work_items.pop(work_id, None)
                    if work_item is not None:
                        work_item.future.set_exception(exc)
                        del work_item
                running_work_item.clear()

        elif com in ready:
            result_item = com.recv()
        else:
            # Mark the process pool broken so that submits fail right now.
            executor = executor_reference()
            if executor is not None:
                executor._broken = True
                executor._shutdown_thread = True
                executor = None
            # All futures in flight must be marked failed
            for work_id, work_item in pending_work_items.items():
                work_item.future.set_exception(
                    BrokenProcessPool(
                        "A process in the process pool was "
                        "terminated abruptly while the future was "
                        "running or pending."
                    ))
                # Delete references to object. See issue16284
                del work_item
            pending_work_items.clear()
            # Terminate remaining workers forcibly: the queues or their
            # locks may be in a dirty state and block forever.
            for p in processes.values():
                mp.util.debug('terminate process {}'.format(p.name))
                try:
                    p.terminate()
                except ProcessLookupError:
                    pass
            shutdown_worker()
            return
        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID
            # (avoids marking the executor broken)
            assert shutting_down()
            p = processes.pop(result_item)
            p.join()
            if not processes:
                shutdown_worker()
                return
        elif result_item is not None:
            work_item = pending_work_items.pop(result_item.work_id, None)
            # work_item can be None if another process terminated (see above)
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                # Delete references to object. See issue16284
                del work_item
            running_work_item.remove(result_item.work_id)
        # Check whether we should start shutting down.
        executor = executor_reference()
        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that owns this worker has been collected OR
        #   - The executor that owns this worker has been shutdown.
        if shutting_down():
            if kill_on_shutdown:
                while pending_work_items:
                    _, work_item = pending_work_items.popitem()
                    work_item.future.set_exception(ShutdownProcessPool(
                        "The Executor was shutdown before this job could "
                        "complete."))
                    del work_item
                # Terminate remaining workers forcibly: the queues or their
                # locks may be in a dirty state and block forever.
                for p in processes.values():
                    p.terminate()
                shutdown_worker()
            try:
                # Since no new work items can be added, it is safe to shutdown
                # this thread if there are no pending work items.
                if not pending_work_items:
                    shutdown_worker()
                    return
            except Full:
                # This is not a problem: we will eventually be woken up (in
                # result_queue.get()) and be able to send a sentinel again.
                pass
        executor = None


def _management_worker(executor_reference, queue_management_thread, processes,
                       pending_work_items, work_ids, call_queue, result_queue):
    while True:
        broken_qm = not queue_management_thread.is_alive()

        if broken_qm:
            broken = (call_queue._thread is not None and
                      not call_queue._thread.is_alive())
            broken |= any([p.exitcode for p in processes.values()])
            if not broken:
                _shutdown_crash(executor_reference, processes,
                                pending_work_items,
                                BrokenProcessPool(
                                    "The QueueManagerThread was terminated "
                                    "abruptly while the future was running or "
                                    "pending. This is due to a result "
                                    "unpickling error."
                                ))
            return
        elif _is_crashed(call_queue._thread):
            _shutdown_crash(executor_reference, processes, pending_work_items,
                            BrokenProcessPool(
                                "The QueueFeederThread was terminated abruptly"
                                " while feeding a new job. This is due to a "
                                "job pickling error."
                            ))
            return

        time.sleep(.1)


def _shutdown_crash(executor_reference, processes, pending_work_items, exc):
    executor = executor_reference()
    if executor is not None:
        executor._broken = True
        executor._shutdown_thread = True
        executor = None
    # Terminate remaining workers forcibly: the queues or their
    # locks may be in a dirty state and block forever.
    for p in processes.values():
        p.terminate()
    # All futures in flight must be marked failed
    for work_id, work_item in pending_work_items.items():
        work_item.future.set_exception(exc)
        # Delete references to object. See issue16284
        del work_item
    pending_work_items.clear()


_system_limits_checked = False
_system_limited = None


def _check_system_limits():
    global _system_limits_checked, _system_limited
    if _system_limits_checked:
        if _system_limited:
            raise NotImplementedError(_system_limited)
    _system_limits_checked = True
    try:
        nsems_max = os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, ValueError):
        # sysconf not available or setting not available
        return
    if nsems_max == -1:
        # indetermined limit, assume that limit is determined
        # by available memory only
        return
    if nsems_max >= 256:
        # minimum number of semaphores available
        # according to POSIX
        return
    _system_limited = ("system provides too few semaphores (%d available, "
                       "256 necessary)" % nsems_max)
    raise NotImplementedError(_system_limited)


class BrokenProcessPool(RuntimeError):

    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """


class ShutdownProcessPool(RuntimeError):

    """
    Raised when a ProcessPoolExecutor is shutdown while a future was in the
    running or pending state.
    """


class ProcessPoolExecutor(_base.Executor):

    def __init__(self, max_workers=None, context=None,
                 kill_on_shutdown=False):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: The maximum number of processes that can be used to
                execute the given calls. If None or not given then as many
                worker processes will be created as the machine has processors.
        """
        _check_system_limits()

        if max_workers is None:
            self._max_workers = os.cpu_count() or 1
        else:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")

            self._max_workers = max_workers

        # Parameters of this executor
        self._ctx = context or get_context()
        self._kill_on_shutdown = kill_on_shutdown

        # Make the call queue slightly larger than the number of processes to
        # prevent the worker processes from idling. But don't make it too big
        # because futures in the call queue cannot be cancelled.
        self._call_queue = self._ctx.Queue(self._max_workers +
                                           EXTRA_QUEUED_CALLS)
        # Killed worker processes can produce spurious "broken pipe"
        # tracebacks in the queue's own worker thread. But we detect killed
        # processes anyway, so silence the tracebacks.
        self._call_queue._ignore_epipe = True
        self._result_queue = self._ctx.SimpleQueue()
        self._com_queue = self._ctx.SimpleQueue()
        self._work_ids = queue.Queue()
        self._management_thread = None
        self._queue_management_thread = None
        # Map of pids to processes
        self._processes = {}

        # Shutdown is a two-step process.
        self._shutdown_thread = False
        self._shutdown_lock = threading.Lock()
        self._broken = False
        self._queue_count = 0
        self._pending_work_items = {}

    def _start_queue_management_thread(self):
        # When the executor gets lost, the weakref callback will wake up
        # the queue management thread.
        def weakref_cb(_, q=self._com_queue):
            q.put(None)
        if self._queue_management_thread is None:
            # Start the processes so that their sentinels are known.
            self._adjust_process_count()
            self._queue_management_thread = threading.Thread(
                target=_queue_management_worker,
                args=(weakref.ref(self, weakref_cb),
                      self._processes,
                      self._pending_work_items,
                      self._work_ids,
                      self._call_queue,
                      self._result_queue,
                      self._com_queue,
                      self._kill_on_shutdown),
                name="QueueManager")
            self._queue_management_thread.daemon = True
            self._queue_management_thread.start()
            _threads_queues[self._queue_management_thread] = self._com_queue

    def _start_thread_management_thread(self):
        # When the executor gets lost, the weakref callback will wake up
        # the queue management thread.
        if self._management_thread is None:
            # Start the processes so that their sentinels are known.
            self._management_thread = threading.Thread(
                target=_management_worker,
                args=(weakref.ref(self),
                      self._queue_management_thread,
                      self._processes,
                      self._pending_work_items,
                      self._work_ids,
                      self._call_queue,
                      self._result_queue),
                name="ThreadManager")
            self._management_thread.daemon = True
            self._management_thread.start()

    def _adjust_process_count(self):
        for _ in range(len(self._processes), self._max_workers):
            p = self._ctx.Process(
                target=_process_worker,
                args=(self._call_queue,
                      self._result_queue))
            p.start()
            self._processes[p.pid] = p

    def submit(self, fn, *args, **kwargs):
        with self._shutdown_lock:
            if self._broken:
                raise BrokenProcessPool('A child process terminated abruptly, '
                                        'the process pool is not usable anymore')
            if self._shutdown_thread:
                raise RuntimeError(
                    'cannot schedule new futures after shutdown')

            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._com_queue.put(None)

            self._start_queue_management_thread()
            self._start_thread_management_thread()
            return f
    submit.__doc__ = _base.Executor.submit.__doc__

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = super().map(partial(_process_chunk, fn),
                              _get_chunks(chunksize, *iterables),
                              timeout=timeout)
        return itertools.chain.from_iterable(results)

    def shutdown(self, wait=True):
        mp.util.debug('shuting down the executor')
        if self._kill_on_shutdown:
            pass
        with self._shutdown_lock:
            self._shutdown_thread = True
        if self._queue_management_thread:
            # Wake up queue management thread
            self._com_queue.put(None)
            if wait and self._queue_management_thread.is_alive():
                self._queue_management_thread.join()
        # To reduce the risk of opening too many files, remove references to
        # objects that use file descriptors.
        self._queue_management_thread = None
        self._call_queue = None
        self._result_queue = None
        self._com_queue = None
        self._processes = None
    shutdown.__doc__ = _base.Executor.shutdown.__doc__

atexit.register(_python_exit)
