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
    elif _pool._state != RUN:
        if DEBUG:
            print("DEBUG   - Create a new pool as the previous one"
                  " was in state {}".format(_pool._state))
        _pool.terminate()
        _local._pool = None
        return get_reusable_pool(*args, **kwargs)
    else:
        _pool.resize(kwargs.get('processes'))
        return _pool
    return _pool


class _ReusablePool(Pool):
    """A Pool, not tolerant to fault and reusable"""
    def __init__(self, timeout=10, processes=None, initializer=None,
                 initargs=(), maxtasksperchild=None, context=None):
        if sys.version_info[:2] >= (3, 4):
            super(_ReusablePool, self).__init__(
                processes, initializer, initargs, maxtasksperchild, context)
        else:
            super(_ReusablePool, self).__init__(
                processes, initializer, initargs, maxtasksperchild)
        self.timeout = timeout

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
                mp.util.debug('A worker have failed in some ways, we will '
                              'flag all current jobs as failed')
                for k in list(self._cache.keys()):
                    self._cache[k]._set(i, (False, AbortedWorkerError(
                        'A process was killed during the execution '
                        'a multiprocessing job.', worker.exitcode)))
                if worker.exitcode in [-9, -15]:
                    self._clean_up_crash()
                    print(
                        "WARNING - Pool might be corrupted, restart it if you "
                        "need a new queue \n" + " "*10 +
                        "Worker exited with error "
                        "code {}".format(worker.exitcode))
                cleaned = True
                del self._pool[i]
        return cleaned

    def _clean_up_crash(self):
        # Terminate tasks handler by sentinel
        self._taskqueue.put(None)

        # Terminate result handler by sentinel
        self._outqueue.put(None)

        # Terminate the worker handler thread
        threading.current_thread()._state = TERMINATE
        for p in self._pool:
            p.terminate()

        # Flag the pool as broken
        self._state = BROKEN

    def _wait_complete(self):
        if len(self._cache) > 0:
            print("WARNING!!! : You are trying to resize a working pool. "
                  "The pool will wait until\nthe jobs are finished and "
                  "then resize it. This can slow down your code and "
                  "you\nshould not do it!")
        while len(self._cache) > 0:
            sleep(.1)

    def resize(self, processes=None):
        if processes is None:
            processes = os.cpu_count() or 1
        if processes < 1:
            raise ValueError("Number of processes must be at least 1")
        if self._processes == processes:
            return
        self._wait_complete()
        self._processes = processes
        self._join_exited_workers()
        self._repopulate_pool()
        if len(self._pool) > processes:
            for worker in self._pool[processes:]:
                self._inqueue.put(None)
            while processes != len(self._pool):
                self._join_exited_workers()

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


class AbortedWorkerError(Exception):
    """A worker was aborted"""
    def __init__(self, msg, exitcode):
        super(AbortedWorkerError, self).__init__()
        self.msg = msg
        self.exitcode = exitcode
        self.args = [repr(self)]

    def __repr__(self):
        return '{} with exitcode {}'.format(self.msg, self.exitcode)

if __name__ == '__main__':
    # This will cause a deadlock
    pool = get_reusable_pool(1)
    pid = pool.apply(os.getpid, tuple())
    pool = get_reusable_pool(2)
    os.kill(pid, 15)

    pool.terminate()

