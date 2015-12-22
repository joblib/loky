from multiprocessing.pool import Pool
import multiprocessing as mp
from multiprocessing.pool import ExceptionWithTraceback, MaybeEncodingError
import threading
import os
from time import sleep

_local = threading.local()


def get_reusable_pool(n_jobs=None):
    _pool = getattr(_local, '_pool', None)
    if _pool is None:
        _local._pool = _pool = _ReusablePool(processes=n_jobs)
    elif _pool._state != 0:
        _pool.terminate()
        _local._pool = None
        return get_reusable_pool(n_jobs)
    else:
        _pool.resize(n_jobs)
        return _pool
    return _pool


class _ReusablePool(Pool):
    """A Pool, not tolerant to fault and reusable"""
    def __init__(self, timeout=10, processes=None, initializer=None,
                 initargs=(), maxtasksperchild=None, context=None):
        self.pids = []
        super(_ReusablePool, self).__init__(
            processes, initializer, initargs,
            maxtasksperchild, context)
        self.timeout = timeout
        self.pids = self.starmap(os.getpid, [tuple()]*len(self._pool))

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
                cleaned = True
                del self._pool[i]
        if not self._is_conscistent():
            self._worker_handler._state = 2
            self._task_handler._state = 2
            self._result_handler._state = 2
        return cleaned

    def _is_conscistent(self):
        import psutil
        for pid in self.pids:
            if not psutil.pid_exists(pid):
                for pid in self.pids:
                    os.kill(pid, 15)
                return False

    def _maintain_pool(self):
        """Clean up any exited workers and start replacements for them.
        """
        if self._join_exited_workers():
            self._repopulate_pool()

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
        self.pids = self.starmap(os.getpid, [tuple()]*len(self._pool))

    # @staticmethod
    # def _help_stuff_finish(inqueue, task_handler, size):
    #     # task_handler may be blocked trying to put items on inqueue
    #     print("No problem!!!!!!!\n\n\n")
    #     util.debug('removing tasks from inqueue until task handler finished')
    #     for i in range(10):
    #         sleep(.01)
    #         print(inqueue._rlock)
    #     inqueue._rlock.acquire()
    #     while task_handler.is_alive() and inqueue._reader.poll():
    #         inqueue._reader.recv()
    #         time.sleep(0)


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

