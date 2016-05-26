from multiprocessing.pool import Pool, RUN, TERMINATE
from multiprocessing.pool import ApplyResult, MapResult, IMapIterator
import multiprocessing as mp
import threading
import warnings
import os
import sys
from time import sleep

__all__ = ['get_reusable_pool', 'TerminatedPoolError', 'AbortedWorkerError']

# Additional state to flag the pool as unusable
# with unsafe terminate function
BROKEN = 3

# Exit messages for Broken pools
CRASH_WORKER = ("A process was killed during the execution "
                "of multiprocessing job.")
CRASH_RESULT_HANDLER = ("The result handler crashed. This can be caused by "
                        "an unpickling error.")
CRASH_TASK_HANDLER = ("The task handler crashed. This is probably "
                      "due to a result pickling error.")

# Protect the queue from being reused in different threads
_local = threading.local()


def mapstar(args):
    return list(map(*args))


def _is_started(thread):
    """helper to check if a thread is started for any version of python"""
    if hasattr(thread, "_started"):
        return thread._started.is_set()
    # Backward compat for python 2.7
    return thread._Thread__started.is_set()


def get_reusable_pool(*args, **kwargs):
    """Return a the current ReusablePool. Start a new one if needed"""
    _pool = getattr(_local, '_pool', None)
    _id_pool = getattr(_local, '_id_pool', 0)
    processes = kwargs.get('processes')
    if _pool is None:
        mp.util.debug("Create a pool with size {}.".format(processes))
        _local._pool = _pool = _ReusablePool(*args, id_pool=_id_pool,
                                             **kwargs)
        _local._id_pool = _id_pool + 1
    else:
        _pool._maintain_pool(timeout=None)
        if _pool._state != RUN:
            mp.util.debug("Create a new pool with {} processes as the "
                          "previous one was in state {}"
                          "".format(processes, _pool._state))
            _pool.terminate()
            _pool.join()
            _local._pool = _pool = None
            return get_reusable_pool(*args, **kwargs)
        else:
            if _pool._resize(processes):
                mp.util.debug("Resized existing pool to target size.")
                return _pool
            mp.util.debug("Failed to resize existing pool to target size {}."
                          "".format(processes))
            _pool.terminate()
            _pool.join()
            _local._pool = _pool = None
            return get_reusable_pool(*args, **kwargs)

    return _pool


class _ReusablePool(Pool):
    """A Pool, not tolerant to fault and reusable"""
    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None, id_pool=0):
        self.maintain_lock = mp.Lock()
        self.id_pool = id_pool
        if sys.version_info[:2] >= (3, 4):
            super(_ReusablePool, self).__init__(
                processes, initializer, initargs, maxtasksperchild, context)
        else:
            super(_ReusablePool, self).__init__(
                processes, initializer, initargs, maxtasksperchild)
        self._result_handler.name = 'ResultHandler-{}'.format(id_pool)
        self._task_handler.name = 'TaskHandler-{}'.format(id_pool)
        self._worker_handler.name = 'WorkerHandler-{}'.format(id_pool)
        mp.util.debug("Pool{} started.".format(self.id_pool))
        mp.util.debug("Pool{} ident: WH: {:x}  - TH: {:x}  - RH: {:x}".format(
            id_pool, self._worker_handler.ident,  self._task_handler.ident,
            self._result_handler.ident))

    def _has_started_thread(self, thread_name):
        """
        helper function asserting if a pool thread has been created
        and started
        """
        thread = getattr(self, thread_name, None)
        return thread is not None and _is_started(thread)

    def _join_exited_workers(self):
        """
        Cleanup after any worker processes which have exited due to reaching
        their specified lifetime.  Returns True if any workers were cleaned up.
        Also detect if a worker stopped unexpectedly and clean up the pool in
        this case
        """
        cleaned = False
        for worker in self._pool[:]:
            if worker.exitcode is not None:
                # worker exited
                mp.util.debug('Cleaning up worker %r' % worker)
                worker.join()
                cleaned = True
                # If a worker has stopped unexpectedely, clean the pool
                # to avoid deadlocks
                if worker.exitcode != 0:
                    mp.util.sub_warning(
                        "Pool{} might be corrupted. Worker exited with "
                        "error code {}".format(self.id_pool, worker.exitcode))
                    self._clean_up_crash(cause_msg=CRASH_WORKER,
                                         exitcode=worker.exitcode)
                    raise _BrokenPoolError(worker.exitcode)

                # If the worker exited cleanly, juste remove it
                self._pool.remove(worker)

        # Make sure the handler threads did not crashed
        if (self._has_started_thread("_result_handler") and
                not self._result_handler.is_alive()):
            self._clean_up_crash(cause_msg=CRASH_RESULT_HANDLER)
            raise _BrokenPoolError(worker.exitcode)
        if (self._has_started_thread("_task_handler") and
                not self._task_handler.is_alive()):
            self._clean_up_crash(cause_msg=CRASH_TASK_HANDLER)
            raise _BrokenPoolError(worker.exitcode)

        return cleaned

    def terminate(self):
        """Terminate the pool. This does not wait for job completion"""
        mp.util.debug('terminating pool')

        # make sure the state of the pool is up to date
        self._maintain_pool(timeout=None)

        self._state = TERMINATE

        # call _terminate with a lock to avoid concurrent call with
        # clean_up_crash
        with self.maintain_lock:
            self._terminate()

    def _maintain_pool(self, timeout=.01):
        """Clean up any exited workers and start replacements for them"""
        if self.maintain_lock.acquire(timeout=timeout):
            try:
                    if (self._state != BROKEN and (
                            self._join_exited_workers() or
                            self._processes >= len(self._pool))):
                        self._repopulate_pool()
            except _BrokenPoolError:
                pass
            self.maintain_lock.release()
        else:
            mp.util.debug("Could not maintain pool..")

    def _clean_up_workers(self, cause_msg, exitcode=None):
        """Clean up deadlocks due to a worker crashing"""

        # If the calling thread is not worker_handler, make sure it will
        # not restart the workers
        if (self._has_started_thread("_worker_handler") and
                threading.current_thread() is not self._worker_handler and
                self._worker_handler.is_alive()):
            self._worker_handler._state = TERMINATE
            self._worker_handler.join()

        # Terminate and join the workers
        if self._pool and hasattr(self._pool[0], 'terminate'):
            mp.util.debug('terminating workers')
            for p in self._pool:
                if p.exitcode is None:
                    p.terminate()
                mp.util.debug('cleaning up worker %d' % p.pid)
                p.join()

        # Kill the wlock of the outqueue to avoid deadlock with _task_handler
        # sentinelling the _result handler
        self._outqueue._wlock = None
        if sys.version_info[:2] < (3, 4):
            self._outqueue._make_methods()

    def _clean_up_crash(self, cause_msg, exitcode=None):
        """
        Clean up the state of the pool in case of crash from a worker or one of
        the handlers
        """
        if self._state == BROKEN:
            return

        # Flag the pool as broken
        mp.util.debug('clean up broken pool{}'.format(self.id_pool))
        self._state = BROKEN

        if exitcode is not None:
            self._clean_up_workers(cause_msg, exitcode)
        self._task_handler._state = TERMINATE

        mp.util.debug("send sentinel for task_handler")
        self._taskqueue.put(None)

        # Flag all the _cached job as failed due to aborted worker
        mp.util.debug("flag the cache as broken")
        _ReusablePool._flag_cache_broken(
            self._cache, AbortedWorkerError(cause_msg, exitcode))

        self._terminate()

    def _resize(self, processes=None):
        """Resize the pool to the desired number of processes"""
        if processes is None:
            processes = os.cpu_count() or 1
        # Make sure we require a valid number of processes.
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")

        # Return if the current size is already good
        if self._processes == processes:
            return True

        # Wait for the current job to finish and ensure that they did not
        # break the pool in the process
        self._wait_job_complete()
        if self._state != RUN:
            return False

        self._processes = processes
        if len(self._pool) > processes:
            # Sentinel excess workers, they will terminate and
            # be collected asynchronously
            for worker in self._pool[processes:]:
                self._inqueue.put(None)

        # Make sure that the pool as the expected number of workers
        # up and ready
        while processes != len(self._pool) and self._state == RUN:
            self._maintain_pool(timeout=None)

        # Broken signals could have been missed on the wait_job_complete
        # due to delay in OS signal propagation, assert that the pool
        # is still running
        if self._state != RUN:
            return False

        assert processes == len(self._pool), (
            "Resize pool failed. Got {} extra  processes"
            "".format(processes - len(self._pool)))
        return True

    def _wait_job_complete(self):
        """Wait for the cache to be empty before resizing the pool."""
        # Issue a warning to the user about the bad effect of this usage.
        if len(self._cache) > 0:
            warnings.warn("You are trying to resize a working pool. "
                          "The pool will wait until the jobs are "
                          "finished and then resize it. This can "
                          "slow down your code.", UserWarning)
            mp.util.debug("Pool{} waiting for job completion before resize"
                          "".format(self.id_pool))
        # Wait for the completion of the jobs
        while len(self._cache) > 0:
            sleep(.1)

        # Ensure finishing jobs haven't broken the pool
        self._maintain_pool(timeout=None)

    def join(self):
        mp.util.debug('joining Pool{}'.format(self.id_pool))
        assert self._state in (BROKEN, TERMINATE)
        mp.util.debug('Pool state:\n{}, _state:{}\n{}, _state:{}\n'
                      '{}, _state:{}\n{}'.format(
                        self._worker_handler, self._worker_handler._state,
                        self._task_handler, self._task_handler._state,
                        self._result_handler, self._result_handler._state,
                        self._pool))
        self._worker_handler.join()
        self._task_handler.join()
        self._result_handler.join()
        for p in self._pool:
            p.join()
        mp.util.debug('Pool{} terminated cleanly'.format(self.id_pool))

    @classmethod
    def _terminate_pool(cls, taskqueue, inqueue, outqueue, pool,
                        worker_handler, task_handler, result_handler, cache):
        """
        Overload the _terminate_pool method to handle outqueue and cache
        cleaning and avoid deadlocks. This method is guaranteed to only
        be called once.
        """

        mp.util.debug('finalizing pool')
        # Flush inqueue to ensure that the task_handler can put the sentinel
        # in it without hanging forever. This also avoids waiting for jobs
        # completion when calling terminate
        mp.util.debug('helping task handler/workers to finish')
        cls._empty_queue(inqueue, task_handler, pool, taskqueue)

        # If the result handler crashed, the sentinel might deadlock
        # due to the outqueue being full
        if not result_handler.is_alive():
            cls._empty_queue(outqueue, task_handler, [result_handler])

        # Terminate the managing threads. We must wait for the worker_handler
        # to exit before terminating workers because we don't want workers to
        # be restarted behind our back
        worker_handler._state = TERMINATE
        task_handler._state = TERMINATE
        mp.util.debug('joining worker handler')
        if threading.current_thread() is not worker_handler:
            worker_handler.join()
        mp.util.debug('joining task handler')
        if threading.current_thread() is not task_handler:
            task_handler.join()

        # Terminate the workers
        if pool and hasattr(pool[0], 'terminate'):
            mp.util.debug('terminating workers')
            for p in pool:
                if p.exitcode is None:
                    p.terminate()
                mp.util.debug('cleaning up worker %d' % p.pid)
                p.join()

        # At this point, there is no work done anymore so we can flag all the
        # remaining jobs in cache as broken by the terminate call
        cls._flag_cache_broken(cache, TerminatedPoolError())
        result_handler._state = TERMINATE

        # send a sentinel to make sure the result_handler do wait forever
        # on the outqueue
        outqueue._wlock = None
        if sys.version_info[:2] < (3, 4):
            outqueue._make_methods()
        outqueue.put(None)
        # outqueue.put(None)

        mp.util.debug('joining result handler')
        if threading.current_thread() is not result_handler:
            result_handler.join()

    @staticmethod
    def _empty_queue(queue, writer, readers, taskqueue=None):
        """
        Empty a queue to ensure that writer threads will not hang forever in a
        put call. This kills the reader threads if the _rlock cannot be
        acquired.
        """
        acquire = True
        # We use a timeout to detect queue that was locked by a dead
        # process and therefor will never be unlocked.
        if not queue._rlock.acquire(timeout=.05):
            mp.util.debug("queue is locked when terminating. "
                          "The pool is probably broken.")
            acquire = False
            # Terminate the readers of the queue as the read will be unsafe
            if readers and hasattr(readers[0], 'terminate'):
                mp.util.debug('terminating readers of the queue')
                for p in readers:
                    if p.exitcode is None:
                        p.terminate()
                        p.join()
        while (writer.is_alive() and (queue._reader.poll() or
                                      (taskqueue and not taskqueue.empty()))):
            queue._reader.recv_bytes()
            sleep(0)
        if acquire:
            queue._rlock.release()

    @staticmethod
    def _flag_cache_broken(cache, err):
        """Flag all the cached job with the given error"""
        mp.util.debug("flag cache as broken with err: {}".format(err))
        for k in list(cache.keys()):
            cache[k]._flag(err)

    def map_async(self, func, iterable, chunksize=None, callback=None,
                  error_callback=None):
        """
        Asynchronous version of `map()` method.
        Overloaded to avoid compatibility issues with name change from 2.7
        """
        return self._map_async(func, iterable, mapstar, chunksize, callback,
                               error_callback)

    def _map_async(self, func, iterable, mapper, chunksize=None, callback=None,
                   error_callback=None):
        """
        Helper function to implement map, starmap and their async counterparts.
        Overloaded to use RobustMapResult instead of MapResult to permit better
        handling of callback raising Exceptions
        """
        if self._state != RUN:
            raise ValueError("Pool not running")
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)

        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) * 4)
            if extra:
                chunksize += 1
        if len(iterable) == 0:
            chunksize = 0

        task_batches = Pool._get_tasks(func, iterable, chunksize)
        result = RobustMapResult(self._cache, chunksize, len(iterable),
                                 callback, error_callback=error_callback)
        self._taskqueue.put((((result._job, i, mapper, (x,), {})
                              for i, x in enumerate(task_batches)), None))
        return result

    def apply_async(self, func, args=(), kwds={}, callback=None,
                    error_callback=None):
        """
        Asynchronous version of `apply()` method.
        Overloaded to use RobustApplyResult instead of ApplyResult to permit
        better handling of callback raising Exceptions
        """
        if self._state != RUN:
            raise ValueError("Pool not running")
        result = RobustApplyResult(self._cache, callback, error_callback)
        self._taskqueue.put(([(result._job, None, func, args, kwds)], None))
        return result

    def imap(self, func, iterable, chunksize=1):
        """
        Equivalent of `map()` -- can be MUCH slower than `Pool.map()`.
        Overloaded to use RobustIMapIterator instead of IMapIterator to permit
        better handling of callback raising Exceptions
        """
        if self._state != RUN:
            raise ValueError("Pool not running")
        if chunksize == 1:
            result = RobustIMapIterator(self._cache)
            self._taskqueue.put((((result._job, i, func, (x,), {})
                                  for i, x in enumerate(iterable)),
                                 result._set_length))
            return result
        else:
            assert chunksize > 1
            task_batches = Pool._get_tasks(func, iterable, chunksize)
            result = RobustIMapIterator(self._cache)
            self._taskqueue.put((((result._job, i, mapstar, (x,), {})
                                  for i, x in enumerate(task_batches)),
                                result._set_length))
            return (item for chunk in result for item in chunk)

    def imap_unordered(self, func, iterable, chunksize=1):
        """
        Like `imap()` method but ordering of results is arbitrary.
        Overloaded to use RobustIMapUnorderedIterator instead of
        IMapUnorderedIterator to permit better handling of callback raising
        Exceptions
        """
        if self._state != RUN:
            raise ValueError("Pool not running")
        if chunksize == 1:
            result = RobustIMapUnorderedIterator(self._cache)
            self._taskqueue.put((((result._job, i, func, (x,), {})
                                 for i, x in enumerate(iterable)),
                                result._set_length))
            return result
        else:
            assert chunksize > 1
            task_batches = Pool._get_tasks(func, iterable, chunksize)
            result = RobustIMapUnorderedIterator(self._cache)
            self._taskqueue.put((((result._job, i, mapstar, (x,), {})
                                 for i, x in enumerate(task_batches)),
                                result._set_length))
            return (item for chunk in result for item in chunk)


def callback_call(job, callback):
    try:
        callback(job._value)
    except Exception as e:
        err = CallbackError(e, job._value)
        job._value = err
        job._success = False


class RobustApplyResult(ApplyResult):

    def __init__(self, cache, callback, error_callback):
        if sys.version_info[:2] >= (3, 3):
            ApplyResult.__init__(self, cache, callback, error_callback)
        else:
            ApplyResult.__init__(self, cache, callback)

    def _set(self, i, obj):
        self._success, self._value = obj
        if self._callback and self._success:
            callback_call(self, self._callback)

        if (hasattr(self, '_error_callback') and self._error_callback and
                not self._success):
            callback_call(self, self._error_callback)
        self._notify()
        del self._cache[self._job]

    def _flag(self, err):
        self._set(0, (False, err))

    def _notify(self):
        if sys.version_info[:2] >= (3, 3):
            self._event.set()
        else:
            self._cond.acquire()
            try:
                self._ready = True
                self._cond.notify()
            finally:
                self._cond.release()

AsyncResult = RobustApplyResult       # create alias -- see #17805


class RobustMapResult(MapResult):

    def __init__(self, cache, chunksize, length, callback, error_callback):
        if sys.version_info[:2] >= (3, 3):
            MapResult.__init__(self, cache, chunksize, length, callback,
                               error_callback)
        else:
            MapResult.__init__(self, cache, chunksize, length, callback)

    def _set(self, i, success_result):
        self._number_left -= 1
        success, result = success_result
        if success and self._success:
            self._value[i*self._chunksize:(i+1)*self._chunksize] = result
            if self._number_left == 0:
                if self._callback:
                    callback_call(self, self._callback)
                del self._cache[self._job]
                self._notify()
        else:
            self._success = False
            self._value = result
            if hasattr(self, '_error_callback') and self._error_callback:
                callback_call(self, self._error_callback)
            del self._cache[self._job]
            self._notify()

    def _notify(self):
        if sys.version_info[:2] >= (3, 3):
            self._event.set()
        else:
            self._cond.acquire()
            try:
                self._ready = True
                self._cond.notify()
            finally:
                self._cond.release()

    def _flag(self, err):
        self._success = False
        self._value = err
        if (hasattr(self, '_error_callback') and self._error_callback and
                not self._success):
            callback_call(self, self._error_callback)
        del self._cache[self._job]
        self._notify()


class RobustIMapIterator(IMapIterator):
    def _flag(self, err):
        # We can set the length to whatever as this operation should be called
        # after the task handeler finished
        if self._length is None:
            self._set_length(self._index+1)
        with self._cond:
            while self._index < self._length:
                if self._index in self._unsorted:
                    obj = self._unsorted.pop(self._index)
                    self._items.append(obj)
                else:
                    self._items.append((False, err))
                self._index += 1
                self._cond.notify()
            del self._cache[self._job]


class RobustIMapUnorderedIterator(RobustIMapIterator):

    def _set(self, i, obj):
        with self._cond:
            self._items.append(obj)
            self._index += 1
            self._cond.notify()
            if self._index == self._length:
                del self._cache[self._job]


class AbortedWorkerError(Exception):
    """A worker was aborted"""
    def __init__(self, msg, exitcode):
        super(AbortedWorkerError, self).__init__()
        self.msg = msg
        self.exitcode = exitcode
        self.args = [repr(self)]

    def __repr__(self):
        format_exitcode = ''
        if self.exitcode is not None and self.exitcode < 0:
            format_exitcode = ' with signal {}'.format(-self.exitcode)
        if self.exitcode is not None and self.exitcode > 0:
            format_exitcode = ' with exitcode {}'.format(self.exitcode)
        return self.msg + format_exitcode


class _BrokenPoolError(Exception):
    """A worker was aborted"""
    def __init__(self, exitcode):
        super(_BrokenPoolError, self).__init__()
        self.exitcode = exitcode
        self.args = [repr(self)]

    def __repr__(self):
        return ('The pool is rendered unusable by a procecss that exited'
                ' with exitcode {}'.format(self.exitcode))


class TerminatedPoolError(Exception):
    """A worker was aborted"""
    def __init__(self):
        super(TerminatedPoolError, self).__init__()
        self.args = [repr(self)]

    def __repr__(self):
        return ('The pool was terminated while the task could complete')


class CallbackError(Exception):
    """The callback have failed in some ways and wrap value and
    error together in results
    """
    def __init__(self, e, value):
        super(CallbackError, self).__init__()
        self.args = [repr(e)]
        self.err = e
        self.value = value

    def __repr__(self):
        return ('The callback for the jobs raised an error {}\n'
                'The result is still accessible in this Error as'
                'in the filed result')


if __name__ == '__main__':
    # This will cause a deadlock
    pool = get_reusable_pool(1)
    pid = pool.apply(os.getpid, tuple())
    pool = get_reusable_pool(2)
    os.kill(pid, 15)

    pool.terminate()
    pool.join()
