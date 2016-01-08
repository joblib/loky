from __future__ import print_function
from multiprocessing.pool import Pool, RUN, TERMINATE
import multiprocessing as mp
import threading
import os
import sys
from time import sleep

BROKEN = 3
DEBUG = True

_local = threading.local()


def get_reusable_pool(*args, **kwargs):
    _pool = getattr(_local, '_pool', None)
    if _pool is None:
        _local._pool = _pool = _ReusablePool(*args, **kwargs)
    else:
        _pool._maintain_pool()
        if _pool._state != RUN:
            if DEBUG:
                print("DEBUG   - Create a new pool as the previous one"
                      " was in state {}".format(_pool._state))
            with _pool.maintain_lock:
                _pool.terminate()
            _local._pool = None
            return get_reusable_pool(*args, **kwargs)
        else:
            _pool._resize(kwargs.get('processes'))
    return _pool


class _ReusablePool(Pool):
    """A Pool, not tolerant to fault and reusable"""
    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None):
        self.maintain_lock = mp.Lock()
        if sys.version_info[:2] >= (3, 4):
            super(_ReusablePool, self).__init__(
                processes, initializer, initargs, maxtasksperchild, context)
        else:
            super(_ReusablePool, self).__init__(
                processes, initializer, initargs, maxtasksperchild)

    def _join_exited_workers(self):
        """Cleanup after any worker processes which have exited due to reaching
        their specified lifetime.  Returns True if any workers were cleaned up.
        """
        cleaned = False
        for i in reversed(range(len(self._pool))):
            worker = self._pool[i]
            if worker.exitcode is not None:
                # worker exited
                mp.util.debug('cleaning up worker %d' % i)
                worker.join()
                cleaned = True
                if worker.exitcode != 0:
                    mp.util.debug('A worker have failed in some ways, we will '
                                  'flag all current jobs as failed')
                    for k in list(self._cache.keys()):
                        self._cache[k]._set(i, (False, AbortedWorkerError(
                            'A process was killed during the execution '
                            'a multiprocessing job.', worker.exitcode)))
                    self._clean_up_crash()
                    print(
                        "WARNING - Pool might be corrupted, restart it if you "
                        "need a new queue \n" + " " * 10 +
                        "Worker exited with error "
                        "code {}".format(worker.exitcode))
                    raise BrokenPoolError(worker.exitcode)
                del self._pool[i]
        return cleaned

    def terminate(self):
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
            print('DEBUG   - Terminate was called on a BROKEN pool but some'
                  ' processes was still alive.')

    def _maintain_pool(self):
        """Clean up any exited workers and start replacements for them.
        """
        try:
            with self.maintain_lock:
                if ((self._join_exited_workers() or
                        (self._processes >= len(self._pool))) and
                        self._state != BROKEN):
                    self._repopulate_pool()
        except BrokenPoolError:
            pass

    def _clean_up_crash(self):
        if self._state == BROKEN:
            return
        # Terminate tasks handler by sentinel
        self._taskqueue.put(None)

        # Terminate result handler by sentinel
        #self._outqueue.put(None)
        self._result_handler._state = TERMINATE

        # Terminate the worker handler thread
        threading.current_thread()._state = TERMINATE
        for p in self._pool:
            p.terminate()

        # Flag the pool as broken
        self._state = BROKEN

    def _wait_complete(self):
        if len(self._cache) > 0:
            print("WARNING - You are trying to resize a working pool. "
                  "The pool will wait until\nthe jobs are finished and "
                  "then resize it. This can slow down your code and "
                  "you\nshould not do it!")
        while len(self._cache) > 0:
            sleep(.1)

    def _resize(self, processes=None):
        """Resize the pool to the desired number of processes
        """
        if processes is None:
            processes = os.cpu_count() or 1
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")
        if self._processes == processes:
            return
        self._wait_complete()
        self._processes = processes
        if len(self._pool) > processes:
            # Sentinel excess workers, they will terminate and
            # be collected asynchronously
            for worker in self._pool[processes:]:
                self._inqueue.put(None)

        # Make sure that the pool as the expected number of workers
        # up and ready
        while processes != len(self._pool):
            self._maintain_pool()

        assert processes == len(self._pool), (
            "Resize pool failed. Got {} extra  processes"
            "".format(processes - len(self._pool)))

    @staticmethod
    def _help_stuff_finish(inqueue, task_handler, size):
        # task_handler may be blocked trying to put items on inqueue
        mp.util.debug("removing tasks from inqueue until task "
                      "handler finished")
        # We use a timeout to detect inqueues that was locked by a dead
        # process and therefor will never be unlocked
        if inqueue._rlock.acquire(timeout=.1):
            while task_handler.is_alive() and inqueue._reader.poll():
                inqueue._reader.recv()
                sleep(0)
        else:
            print("WARNING - Unusual finish, the pool might have crashed")
            while task_handler.is_alive() and inqueue._reader.poll():
                inqueue._reader.recv()
                sleep(0)

    @staticmethod
    def _handle_tasks(taskqueue, put, outqueue, pool, cache):
        thread = threading.current_thread()

        for taskseq, set_length in iter(taskqueue.get, None):
            task = None
            i = -1
            try:
                for i, task in enumerate(taskseq):
                    if thread._state == RUN:
                        mp.util.debug(
                            'task handler found thread._state != RUN')
                        break
                    try:
                        put(task)
                    except Exception as e:
                        job, ind = task[:2]
                        try:
                            cache[job]._set(ind, (False, e))
                        except KeyError:
                            pass
                else:
                    if set_length:
                        mp.util.debug('doing set_length()')
                        set_length(i+1)
                    continue
                break
            except Exception as ex:
                job, ind = task[:2] if task else (0, 0)
                if job in cache:
                    cache[job]._set(ind + 1, (False, ex))
                if set_length:
                    mp.util.debug('doing set_length()')
                    set_length(i+1)
        else:
            mp.util.debug('task handler got sentinel')

        try:
            if pool._state != BROKEN:
                # tell result handler to finish when cache is empty
                mp.util.debug('task handler sending sentinel '
                              'to result handler')
                outqueue.put(None)
                # tell workers there is no more work
                mp.util.debug('task handler sending sentinel to workers')
                for p in pool:
                    put(None)
        except OSError:
            mp.util.debug('task handler got OSError when sending sentinels')

        mp.util.debug('task handler exiting')


class AbortedWorkerError(Exception):
    """A worker was aborted"""
    def __init__(self, msg, exitcode):
        super(AbortedWorkerError, self).__init__()
        self.msg = msg
        self.exitcode = exitcode
        self.args = [repr(self)]

    def __repr__(self):
        return '{} with exitcode {}'.format(self.msg, self.exitcode)


class BrokenPoolError(Exception):
    """A worker was aborted"""
    def __init__(self, exitcode):
        super(BrokenPoolError, self).__init__()
        self.exitcode = exitcode
        self.args = [repr(self)]

    def __repr__(self):
        return ('The pool is rendered unusable by a procecss that exited'
                ' with exitcode {}'.format(self.exitcode))

if __name__ == '__main__':
    # This will cause a deadlock
    pool = get_reusable_pool(1)
    pid = pool.apply(os.getpid, tuple())
    pool = get_reusable_pool(2)
    os.kill(pid, 15)

    pool.terminate()
