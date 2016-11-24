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
import sys
import multiprocessing as mp
import threading
import weakref
from functools import partial
import itertools
import traceback
from . import _base
from loky.backend.connection import wait

# Compatibility for python2.7
if sys.version_info[:2] > (2, 7):
    import queue
    from queue import Full, Empty
    from _pickle import PicklingError
else:
    import Queue as queue
    from Queue import Full, Empty
    from pickle import PicklingError
    ProcessLookupError = OSError

if sys.version_info < (3, 4):
    from loky import backend

    def get_context():
        return backend
else:
    from multiprocessing import get_context

__author__ = 'Thomas Moreau (thomas.moreau.2010@gmail.com)'


# Specific exit code to be used by worker processes that stop because of a
# timeout. This value was chose to be:
# - outside or reserved exit code:
#   http://www.tldp.org/LDP/abs/html/exitcodes.html#FTN.AEN23629
# - outside of codes typically used by C programs /usr/include/sysexits.h
# - because it's 42 * 2
WORKER_TIMEOUT_EXIT_CODE = 84

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

_threads_wakeup = weakref.WeakKeyDictionary()
_global_shutdown = False


class _Sentinel:
    __slot__ = ["_state"]

    def __init__(self):
        self._state = False

    def set(self):
        self._state = True

    def get_and_unset(self):
        s = self._state
        if s:
            self._state = False
        return s


def _thread_has_stopped(thread):
    """Helper to check if a previously started thread has stopped

     This helper should work for any version of python.
     """
    if thread is None:
        return False
    if hasattr(thread, "_started"):
        return thread._started.is_set() and not thread.is_alive()
    # Backward compat for python 2.7
    return thread._Thread__started.is_set() and not thread.is_alive()


def _clear_list(ll):
    if sys.version_info < (3, 3):
        del ll[:]
    else:
        ll.clear()


def _python_exit():
    global _global_shutdown
    _global_shutdown = True
    items = list(_threads_wakeup.items())
    for t, wakeup in items:
        if t.is_alive():
            wakeup.set()
    for t, _ in items:
        t.join()

# Controls how many more calls than processes will be queued in the call queue.
# A smaller number will mean that processes spend more time idle waiting for
# work while a larger number will make Future.cancel() succeed less frequently
# (Futures in the call queue cannot be cancelled).
EXTRA_QUEUED_CALLS = 1


class _RemoteTraceback(Exception):
    """Embed stringification of remote traceback in local traceback
    """
    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        return self.tb


class _ExceptionWithTraceback(BaseException):

    def __init__(self, exc, tb=None):
        if tb is None:
            _, _, tb = sys.exc_info()
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

    __slots__ = ["future", "fn", "args", "kwargs"]

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

    def __repr__(self):
        return "CallItem({}, {}, {}, {})".format(
            self.work_id, self.fn, self.args, self.kwargs)


def _get_chunks(chunksize, *iterables):
    """ Iterates over zip()ed iterables in chunks. """
    if sys.version_info < (3, 3):
        it = itertools.izip(*iterables)
    else:
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


def _process_worker(call_queue, result_queue, timeout=None):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A multiprocessing.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A multiprocessing.Queue of _ResultItems that will written
            to by the worker.
        timeout: maximum time to wait for a new item in the call_queue. If that
            time is expired, the worker will shutdown.
    """
    mp.util.debug('worker started with timeout=%s' % timeout)
    while True:
        try:
            call_item = call_queue.get(block=True, timeout=timeout)
        except Empty:
            mp.util.info("shutting down worker after timeout %0.3fs"
                         % timeout)
            sys.exit(WORKER_TIMEOUT_EXIT_CODE)
        except BaseException as e:
            traceback.print_exc()
            sys.exit(1)
        if call_item is None:
            # Notifiy queue management thread about clean worker shutdown
            mp.util.info("shutting down worker on sentinel")
            result_queue.put(os.getpid())
            return
        try:
            r = call_item.fn(*call_item.args, **call_item.kwargs)
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, getattr(e, "__traceback__", None))
            result_queue.put(_ResultItem(call_item.work_id, exception=exc))
        else:
            try:
                result_queue.put(_ResultItem(call_item.work_id, result=r))
            except PicklingError as e:
                tb = getattr(e, "__traceback__", None)
                exc = _ExceptionWithTraceback(e, tb)
                result_queue.put(_ResultItem(call_item.work_id, exception=exc))
            except BaseException as e:
                traceback.print_exc()
                sys.exit(1)


def _add_call_item_to_queue(pending_work_items,
                            running_work_items,
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
        except Empty:
            return
        else:
            work_item = pending_work_items[work_id]

            if work_item.future.set_running_or_notify_cancel():
                call_queue.put(_CallItem(work_id,
                                         work_item.fn,
                                         work_item.args,
                                         work_item.kwargs),
                               block=True)
                running_work_items += [work_id]
            else:
                del pending_work_items[work_id]
                continue


def _queue_management_worker(executor_reference,
                             processes,
                             pending_work_items,
                             work_ids_queue,
                             call_queue,
                             result_queue,
                             wakeup,
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
        wakeup: A multiprocessing.Connection for communication purposes
            between the main Thread and this one, to avoid deadlock.
        kill_on_shutdown: A bool modifying the behavior of the executor. If
            set to True, all pending jobs will be canceled on shutdown for an
            immediate completion.
    """
    executor = None
    running_work_items = []

    def shutting_down():
        return _global_shutdown or executor is None or executor._shutting_down

    def shutdown_all_workers():
        # This is an upper bound
        nb_children_alive = sum(p.is_alive() for p in processes.values())
        try:
            for i in range(0, nb_children_alive):
                call_queue.put_nowait(None)
        except (Full, AssertionError):
            pass
        # Release the queue's resources as soon as possible.
        call_queue.close()

        # If .join() is not called on the created processes then
        # some multiprocessing.Queue methods may deadlock on Mac OS X.
        for p in processes.values():
            p.join()
        mp.util.debug("queue management thread clean shutdown of worker "
                      "processes: {}".format(processes))

    result_reader = result_queue._reader
    _poll_timeout = .001

    while True:
        _add_call_item_to_queue(pending_work_items,
                                running_work_items,
                                work_ids_queue,
                                call_queue)
        # Wait for a result to be ready in the result_queue while checking
        # that worker process are still running.
        result_item = None
        count = 0
        while not wakeup.get_and_unset():
            if sys.platform == "win32" and sys.version_info < (3, 3):
                # Process objects do not have a builtin sentinel attribute that
                # can be passed directly to the 'wait' function (which does a
                # 'select' under the hood). Instead we check for dead processes
                # manually from time to time.
                count += 1
                ready = wait([result_reader], timeout=_poll_timeout)
                if count == 10:
                    count = 0
                    ready += [p for p in processes.values()
                              if not p.is_alive()]
            else:
                worker_sentinels = [p.sentinel for p in processes.values()]
                ready = wait([result_reader] + worker_sentinels,
                             timeout=_poll_timeout)
            if len(ready) > 0:
                break
        else:
            ready = []
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
            except Exception as exc:
                result_item = None
                for work_id in running_work_items:
                    work_item = pending_work_items.pop(work_id, None)
                    if work_item is not None:
                        work_item.future.set_exception(exc)
                        del work_item
                del running_work_items[:]
            ready.pop(ready.index(result_reader))

        executor = executor_reference()

        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID on request by the
            # executor.shutdown method: we should not mark the executor
            # as broken.
            p = processes.pop(result_item, None)
            if p is not None:
                p.join()
        elif result_item is not None:
            work_item = pending_work_items.pop(result_item.work_id, None)
            # work_item can be None if another process terminated
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                # Delete references to object. See issue16284
                del work_item
            running_work_items.remove(result_item.work_id)

        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that owns this worker has been collected OR
        #   - The executor that owns this worker has been shutdown.
        if shutting_down():
            if kill_on_shutdown and (executor is None or not executor._broken):

                while pending_work_items:
                    _, work_item = pending_work_items.popitem()
                    work_item.future.set_exception(ShutdownExecutor(
                        "The executor was shutdown before this job could "
                        "complete."))
                    del work_item
                # Terminate remaining workers forcibly
                mp.util.debug('terminating all remaining worker process on'
                              ' executor shutdown')
                while processes:
                    _, p = processes.popitem()
                    p.terminate()
                    p.join()
                shutdown_all_workers()
                return

            # Since no new work items can be added, it is safe to shutdown
            # this thread if there are no pending work items.
            if not pending_work_items:
                shutdown_all_workers()
                return
        executor = None


def _management_worker(executor_reference, queue_management_thread, processes,
                       pending_work_items, work_ids, call_queue, result_queue):

    executor = None

    def shutting_down():
        return _global_shutdown or executor is None or executor._shutting_down

    while True:
        if sys.platform == "win32" and sys.version_info < (3, 3):
            # Process objects do not have a builtin sentinel attribute that
            # can be passed directly to the 'wait' function (which does a
            # 'select' under the hood). Instead we just sleep and check
            # the process state later.
            sleep(0.1)
        else:
            # Let's watch the worker sentinels to handle terminate process
            # without having to wait till the timeout.
            worker_sentinels = [p.sentinel for p in processes.values()]
            wait(worker_sentinels, timeout=0.1)

        executor = executor_reference()
        if shutting_down():
            # clean shutdown handled by the queue_management_thread
            return

        # Fetch the process objects for the recently stopped workers.
        stopped_workers = [p for p in processes.values() if not p.is_alive()]
        for stopped_worker in stopped_workers:
            if stopped_worker.exitcode == WORKER_TIMEOUT_EXIT_CODE:
                # Workers stopped by timeout have a specific return code: they
                # should be collected but the executor should not be marked as
                # broken
                processes.pop(stopped_worker.pid).join()
            elif stopped_worker.exitcode == 0 and shutting_down():
                # This is a regular clean shutdown typically handled by the
                # queue management thread.
                continue
            else:
                cause_msg = ("A worker process managed by the executor was"
                             " terminated abruptly while the future was"
                             " running or pending.")
                _handle_executor_crash(executor_reference, processes,
                                       pending_work_items, call_queue,
                                       cause_msg)
                return

        if _thread_has_stopped(call_queue._thread) and not shutting_down():
            # The call queue feature thread was stopped unexpectedly. This is
            # typically caused by unpicklable jobs.
            cause_msg = ("The executor was terminated abruptly. This can"
                         " be caused by an pickling issue when dispatching"
                         " a job to a worker process.")
            _handle_executor_crash(executor_reference, processes,
                                   pending_work_items, call_queue,
                                   cause_msg)
            return

        if (_thread_has_stopped(queue_management_thread)
                and not shutting_down()):
            # the queue manager thread has stopped while the feed thread
            # and all worker process are still running: this is a crash
            # of the queue manager thread it-self.
            cause_msg = ("The executor was terminated abruptly. This can"
                         " be caused by an unpickling error of a job"
                         " result received from a worker process.")
            _handle_executor_crash(executor_reference, processes,
                                   pending_work_items, call_queue,
                                   cause_msg)
            return
        executor = None


def _handle_executor_crash(executor_reference, processes, pending_work_items,
                           call_queue, cause_msg):
    mp.util.info(cause_msg)
    executor = executor_reference()
    if executor is not None:
        executor._broken = True
        executor._shutting_down = True
        executor = None
    call_queue.close()
    # All futures in flight must be marked failed
    while pending_work_items:
        _, work_item = pending_work_items.popitem()
        work_item.future.set_exception(BrokenExecutor(cause_msg))
        # Delete references to object. See issue16284
        del work_item
    pending_work_items.clear()

    # Terminate remaining workers forcibly: the queues or their
    # locks may be in a dirty state and block forever.
    while processes:
        _, p = processes.popitem()
        p.terminate()
        p.join()

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


class BrokenExecutor(RuntimeError):
    """Raised when a process in a ProcessPoolExecutor terminated abruptly

    This exception is raised when fetching the result from a running future
    issued by an executor in a broken state.
    """


class ShutdownExecutor(RuntimeError):
    """Raised when a ProcessPoolExecutor was shutdown prior to completion

    This exception is raised when fetching the result from a running future
    issued by an executor that was concurrently shutdown.
    """


class ProcessPoolExecutor(_base.Executor):

    def __init__(self, max_workers=None, context=None,
                 timeout=None, kill_on_shutdown=False):
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
                raise ValueError("max_workers must be greater than 0, got %s."
                                 % max_workers)

            self._max_workers = max_workers

        # Parameters of this executor
        self._ctx = context or get_context()
        mp.util.debug("using context {}".format(self._ctx))
        self._kill_on_shutdown = kill_on_shutdown
        self._timeout = timeout

        self._setup_queue()
        # Connection to wakeup QueueManager thread
        self._queue_management_wakeup = _Sentinel()
        self._work_ids = queue.Queue()
        self._management_thread = None
        self._queue_management_thread = None

        # Map of pids to processes
        self._processes = {}

        # Shutdown is a two-step process.
        self._shutting_down = False
        self._shutdown_lock = threading.Lock()
        self._broken = False
        self._queue_count = 0
        self._pending_work_items = {}
        mp.util.debug('PoolProcessExecutor is setup')

    def _setup_queue(self):
        # Make the call queue slightly larger than the number of processes to
        # prevent the worker processes from idling. But don't make it too big
        # because futures in the call queue cannot be cancelled.
        self._call_queue = self._ctx.Queue(2 * self._max_workers +
                                           EXTRA_QUEUED_CALLS)
        # Killed worker processes can produce spurious "broken pipe"
        # tracebacks in the queue's own worker thread. But we detect killed
        # processes anyway, so silence the tracebacks.
        self._call_queue._ignore_epipe = True
        try:
            self._result_queue = self._ctx.SimpleQueue()
        except AttributeError:
            self._result_queue = mp.queues.SimpleQueue()

    def _start_queue_management_thread(self):
        if len(self._processes) != self._max_workers:
            self._start_missing_workers()
        if self._queue_management_thread is None:
            mp.util.debug('starting queue management thread')
            # Start the processes so that their sentinels are known.
            self._queue_management_thread = threading.Thread(
                target=_queue_management_worker,
                args=(weakref.ref(self),
                      self._processes,
                      self._pending_work_items,
                      self._work_ids,
                      self._call_queue,
                      self._result_queue,
                      self._queue_management_wakeup,
                      self._kill_on_shutdown),
                name="QueueManager")
            self._queue_management_thread.daemon = True
            self._queue_management_thread.start()

    def _start_thread_management_thread(self):
        # When the executor gets lost, the weakref callback will wake up
        # the queue management thread.
        def weakref_cb(_, wakeup=self._queue_management_wakeup):
            wakeup.set()
        if self._management_thread is None:
            mp.util.debug('starting thread management thread')
            # Start the processes so that their sentinels are known.
            self._management_thread = threading.Thread(
                target=_management_worker,
                args=(weakref.ref(self, weakref_cb),
                      self._queue_management_thread,
                      self._processes,
                      self._pending_work_items,
                      self._work_ids,
                      self._call_queue,
                      self._result_queue),
                name="ThreadManager")
            self._management_thread.daemon = True
            self._management_thread.start()
            _threads_wakeup[
                self._queue_management_thread] = self._queue_management_wakeup

    def _start_missing_workers(self):
        previous_count = len(self._processes)
        for _ in range(previous_count, self._max_workers):
            p = self._ctx.Process(
                target=_process_worker,
                args=(self._call_queue,
                      self._result_queue,
                      self._timeout))
            p.start()
            self._processes[p.pid] = p
        new_count = len(self._processes)
        mp.util.debug('starting new worker processes from {} to {}: {}'
                      .format(previous_count, new_count, self._processes))

    def submit(self, fn, *args, **kwargs):
        with self._shutdown_lock:
            if self._broken:
                raise BrokenExecutor('A child process terminated abruptly, '
                                     'the process pool is not usable anymore')
            if self._shutting_down:
                raise RuntimeError(
                    'cannot schedule new futures after shutdown')

            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._queue_management_wakeup.set()

            self._start_queue_management_thread()
            self._start_thread_management_thread()
            return f
    submit.__doc__ = _base.Executor.submit.__doc__

    def map(self, fn, *iterables, **kwargs):
        # timeout=None, chunksize=1):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a
                time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        timeout = kwargs.get('timeout', None)
        chunksize = kwargs.get('chunksize', 1)
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = super(ProcessPoolExecutor, self).map(
            partial(_process_chunk, fn), _get_chunks(chunksize, *iterables),
            timeout=timeout)
        return itertools.chain.from_iterable(results)

    def shutdown(self, wait=True):
        mp.util.debug('shutting down executor %s' % self)
        with self._shutdown_lock:
            self._shutting_down = True
        if self._queue_management_thread is not None:
            # Wake up queue management thread
            self._queue_management_wakeup.set()
            if wait and self._queue_management_thread.is_alive():
                self._queue_management_thread.join()
        if self._management_thread is not None:
            if wait and self._management_thread.is_alive():
                self._management_thread.join()
        if self._call_queue:
            self._call_queue.close()
            self._call_queue.join_thread()
        if wait:
            for p in self._processes.values():
                p.join()
        # To reduce the risk of opening too many files, remove references to
        # objects that use file descriptors.
        self._queue_management_thread = None
        self._management_thread = None
        self._call_queue = None
        self._result_queue = None
        self._processes.clear()
    shutdown.__doc__ = _base.Executor.shutdown.__doc__

atexit.register(_python_exit)
