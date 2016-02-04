from multiprocessing.pool import Pool, RUN, TERMINATE
import multiprocessing as mp
import threading
import warnings
import os
import sys
from time import sleep

import threading
# Additional state to flag the pool as unusable
# with unsafe terminate function
BROKEN = 3

#
CRASH_WORKER = ("A process was killed during the execution "
                "a multiprocessing job.")
CRASH_RESULT_HANDLER = ("The result handler crashed. This is probably"
                        "due to a result unpickling error.")
CRASH_TASK_HANDLER = ("The task handler crashed. This is probably"
                      "due to a result pickling error.")

# Protect the queue fro being reused
_local = threading.local()
_local._id_pool = 0


# Check if a thread has been started
def _is_started(thread):
    if hasattr(thread, "_started"):
        return thread._started.is_set()
    # Backward compat for python 2.7
    return thread._Thread__started.is_set()


def get_reusable_pool(*args, **kwargs):
    _pool = getattr(_local, '_pool', None)
    processes = kwargs.get('processes')
    if _pool is None:
        _local._pool = _pool = _ReusablePool(*args, id_pool=_local._id_pool,
                                             **kwargs)
        _local._id_pool += 1
    else:
        _pool._maintain_pool()
        if _pool._state != RUN:
            mp.util.debug("Create a new pool with {} processes as the "
                          "previous one was in state {}"
                          "".format(processes, _pool._state))
            with _pool.maintain_lock:
                _pool.terminate()
            _local._pool = None
            return get_reusable_pool(*args, **kwargs)
        else:
            if _pool._resize(processes):
                mp.util.debug("Resized existing pool to target size.")
                return _pool
            mp.util.debug("Failed to resize existing pool to target size.")
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
        self._worker_handler.name = 'WorkerrHandler-{}'.format(id_pool)
        mp.util.debug("Just created pool #{}".format(self.id_pool))

    def _has_started_thread(self, thread_name):
        thread = getattr(self, thread_name, None)
        if thread is not None:
            return _is_started(thread)
        return False

    def _join_exited_workers(self):
        """Cleanup after any worker processes which have exited due to reaching
        their specified lifetime.  Returns True if any workers were cleaned up.
        """
        cleaned = False
        for worker in self._pool[:]:
            if worker.exitcode is not None:
                # worker exited
                mp.util.debug('Cleaning up worker %r' % worker)
                worker.join()
                cleaned = True
                if worker.exitcode != 0:
                    mp.util.debug('A worker have failed in some ways, we will '
                                  'flag all current jobs as failed')
                    self._clean_up_crash(cause_msg=CRASH_WORKER,
                                         exitcode=worker.exitcode)
                    mp.util.sub_warning(
                        "Pool might be corrupted. Worker exited with "
                        "error code {}".format(worker.exitcode))
                    raise _BrokenPoolError(worker.exitcode)
                self._pool.remove(worker)
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
        mp.util.debug("Called terminate on pool #{}".format(self.id_pool))
        if self._state != BROKEN:
            self._maintain_pool()
        if self._state != BROKEN:
            super(_ReusablePool, self).terminate()
        else:
            n_tries = 1000
            delay = .001
            for i in range(n_tries):
                alive = False
                for p in self._pool:
                    alive |= p.exitcode is None
                alive |= self._result_handler.is_alive()
                alive |= self._task_handler.is_alive()
                if not alive:
                    return
                sleep(delay)
            # TODO - kill -9 ?
            mp.util.sub_warning("Terminate was called on a BROKEN pool but "
                                "some processes were still alive.")

    def _maintain_pool(self):
        """Clean up any exited workers and start replacements for them.
        """
        try:
            with self.maintain_lock:
                if (self._state != BROKEN and (
                        self._join_exited_workers() or
                        self._processes >= len(self._pool))):
                    self._repopulate_pool()
        except _BrokenPoolError:
            pass

    def _clean_up_crash(self, cause_msg, exitcode=None):
        if self._state == BROKEN:
            return

        # Flag the pool as broken
        self._state = BROKEN

        # Terminate the worker handler thread
        mp.util.debug("set terminate state for worker_handler and workers")
        self._worker_handler._state = TERMINATE
        for p in self._pool:
            p.terminate()

        # Flag all the _cached job as failed due to aborted worker
        mp.util.debug("flag the cache as broken")
        _ReusablePool._flag_cache_broken(
            self._cache, AbortedWorkerError(cause_msg, exitcode))

        # Terminate result handler by sentinel
        mp.util.debug("set terminate state for result_handler")
        self._result_handler._state = TERMINATE

        # This avoids deadlock caused by putting a sentinel in the outqueue
        # as it might be locked by a dead worker
        mp.util.debug("send sentinel for result_handler")
        self._outqueue._wlock = None
        if sys.version_info[:2] < (3, 4):
            self._outqueue._make_methods()
        self._outqueue.put(None)

        # Terminate tasks handler by sentinel
        mp.util.debug("send sentinel for task_handler")
        self._taskqueue.put(None)

        mp.util.debug("end _clean_up_crash")

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
            self._maintain_pool()

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
        # Wait for the completion of the jobs
        while len(self._cache) > 0:
            sleep(.1)

        # Ensure finishing jobs haven't broken the pool
        self._maintain_pool()

    @classmethod
    def _terminate_pool(cls, taskqueue, inqueue, outqueue, pool,
                        worker_handler, task_handler, result_handler, cache):
        """Overload the _terminate_pool method to handle outqueue and cache
        cleaning and avoid deadlocks.
        """
        # this is guaranteed to only be called once
        mp.util.debug('finalizing pool {}'.format(
            result_handler.name.split('-')[-1]))

        worker_handler._state = TERMINATE
        task_handler._state = TERMINATE

        mp.util.debug('helping task handler/workers to finish')
        # Flush both inqueue and outqueue to ensure that the task_handler
        # does not get blocked.
        cls._help_stuff_finish(inqueue, outqueue, task_handler, result_handler,
                               pool, cache)

        assert result_handler.is_alive() or len(cache) == 0

        result_handler._state = TERMINATE
        outqueue.put(None)                  # sentinel

        # We must wait for the worker handler to exit before terminating
        # workers because we don't want workers to be restarted behind our back
        mp.util.debug('joining worker handler')
        if threading.current_thread() is not worker_handler:
            worker_handler.join()

        # Terminate workers which haven't already finished.
        if pool and hasattr(pool[0], 'terminate'):
            mp.util.debug('terminating workers')
            for p in pool:
                if p.exitcode is None:
                    p.terminate()

        mp.util.debug('joining task handler')
        if threading.current_thread() is not task_handler:
            task_handler.join()

        mp.util.debug('joining result handler')
        if threading.current_thread() is not result_handler:
            result_handler.join()
        # Flag all the _cached job as terminated has the result handler
        # already exited. This avoid waiting for result forever.
        cls._flag_cache_broken(cache, TerminatedPoolError())

        if pool and hasattr(pool[0], 'terminate'):
            mp.util.debug('joining pool workers')
            for p in pool:
                if p.is_alive():
                    # worker has not yet exited
                    mp.util.debug('cleaning up worker %d' % p.pid)
                    p.join()

    @staticmethod
    def _help_stuff_finish(inqueue, outqueue, task_handler, result_handler,
                           pool, cache):
        """Ensure the sentinel can be sent by emptying the communication queues.
        """
        # task_handler may be blocked trying to put items on inqueue
        # or sentinel in outqueue.
        mp.util.debug("removing tasks from inqueue until task "
                      "handler finished")
        _ReusablePool._empty_queue(inqueue, task_handler)

        # at this point, no worker should be running and thus we flag the
        # remaining cache as with the TerminatedPoolError and kill remaining
        # workers
        _ReusablePool._flag_cache_broken(cache, TerminatedPoolError)
        if pool and hasattr(pool[0], 'terminate'):
            mp.util.debug('terminating workers')
            for p in pool:
                if p.exitcode is None:
                    p.terminate()

        mp.util.debug("removing tasks from outqueue until task "
                      "handler finished")
        # Ensure that the results handler quit before emptying the outqueue
        # to avoid simultaneaous call to read on outqueue
        outqueue._wlock = None
        if sys.version_info[:2] < (3, 4):
            outqueue._make_methods()
        while result_handler.is_alive():
            outqueue.put(None)
            mp.util.debug("Sent sentinel to result_handler")
            sleep(.001)
        _ReusablePool._empty_queue(outqueue, task_handler)

    @staticmethod
    def _empty_queue(queue, task_handler):
        """Empty a communication queue to ensure that maintainer threads will
        not hang forever.
        """
        # We use a timeout to detect queue that was locked by a dead
        # process and therefor will never be unlocked.
        if not queue._rlock.acquire(timeout=.1):
            mp.util.debug("queue is locked when terminating. "
                          "The pool is probably broken.")
        while task_handler.is_alive() and queue._reader.poll():
            queue._reader.recv_bytes()
            sleep(0)

    @staticmethod
    def _flag_cache_broken(cache, err):
        """Flag all the cached job with the given error"""
        success, value = (False, err)
        for k in list(cache.keys()):
            result = cache[k]
            # Handle the iterator map case to completly clean the cache
            if hasattr(result, '_index'):
                while result._index != result._length:
                    result._set(result._index, (success, value))
            else:
                result._set(0, (success, value))


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

if __name__ == '__main__':
    # This will cause a deadlock
    pool = get_reusable_pool(1)
    pid = pool.apply(os.getpid, tuple())
    pool = get_reusable_pool(2)
    os.kill(pid, 15)

    pool.terminate()
