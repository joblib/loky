import os
import sys
import time
import warnings
import threading
import multiprocessing as mp

from ._base import TimeoutError
from .process_executor import ProcessPoolExecutor

__all__ = ['get_reusable_executor']


# Singleton executor and id management
_executor_id_lock = threading.Lock()
_next_executor_id = 0
_executor = None
_executor_args = None


def _get_next_executor_id():
    """Ensure that each successive executor instance has a unique, monotonic id.

    The purpose of this monotonic id is to help debug and test automated
    instance creation.
    """
    global _next_executor_id
    with _executor_id_lock:
        executor_id = _next_executor_id
        _next_executor_id += 1
        return executor_id


def get_reusable_executor(max_workers=None, context=None,
                          timeout=None, kill_on_shutdown=True):
    """Return a the current ReusablePool. Start a new one if needed"""
    global _executor, _executor_args
    executor = _executor
    if executor is None:
        mp.util.debug("Create a executor with size {}.".format(max_workers))
        # assert len(mp.active_children()) == 0
        pool_id = _get_next_executor_id()
        _executor_args = dict(context=context, timeout=timeout,
                              kill_on_shutdown=kill_on_shutdown)
        _executor = executor = ReusablePoolExecutor(
            max_workers=max_workers, context=context, timeout=timeout,
            kill_on_shutdown=kill_on_shutdown, pool_id=pool_id)
    else:
        args = dict(context=context, timeout=timeout,
                    kill_on_shutdown=kill_on_shutdown)
        if (executor._broken or executor._shutdown_thread
                or args != _executor_args):
            if executor._broken:
                reason = "broken"
            elif executor._shutdown_thread:
                reason = "shutdown"
            else:
                reason = "arguments have changed"
            mp.util.debug("Create a new executor with {} processes as the "
                          "previous one was unusable ({})"
                          .format(max_workers, reason))
            executor.shutdown(wait=True)
            _executor = executor = _executor_args = None
            # Recursive call to build a new instance
            return get_reusable_executor(max_workers=max_workers, **args)
        else:
            if max_workers is not None and max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            if max_workers is None or executor._resize(max_workers):
                mp.util.debug("Reused existing executor with max_worker={}."
                              .format(executor._max_workers))
                return executor
            mp.util.debug("Failed to resize existing executor to max_worker={}"
                          ".".format(max_workers))
            # Resizing failed: shutdown the current instance and create a new
            # instance from scratch.
            executor.shutdown(wait=True)
            _executor = executor = _executor_args = None
            return get_reusable_executor(max_workers=max_workers, **args)

    return executor


class ReusablePoolExecutor(ProcessPoolExecutor):
    def __init__(self, max_workers=None, context=None, timeout=None,
                 kill_on_shutdown=True, pool_id=0):
        if context is None and sys.version_info[:2] > (3, 3):
            context = mp.get_context('spawn')
        super(ReusablePoolExecutor, self).__init__(
            max_workers=max_workers, context=context, timeout=timeout,
            kill_on_shutdown=kill_on_shutdown)
        self.pool_id = pool_id

    def _resize(self, max_workers):
        if max_workers is None or max_workers == self._max_workers:
            return True
        self._wait_job_complete()
        mw = self._max_workers
        self._max_workers = max_workers
        for _ in range(max_workers, mw):
            self._call_queue.put(None)
        while len(self._processes) > max_workers and not self._broken:
            time.sleep(.001)

        self._adjust_process_count()
        while not all([p.is_alive() for p in self._processes.values()]):
            time.sleep(.001)

        return True

    def _wait_job_complete(self):
        """Wait for the cache to be empty before resizing the pool."""
        # Issue a warning to the user about the bad effect of this usage.
        if len(self._pending_work_items) > 0:
            warnings.warn("You are trying to resize a working pool. "
                          "The pool will wait until the jobs are "
                          "finished and then resize it. This can "
                          "slow down your code.", UserWarning)
            mp.util.debug("Pool{} waiting for job completion before resize"
                          "".format(self.pool_id))
        # Wait for the completion of the jobs
        while len(self._pending_work_items) > 0:
            time.sleep(.1)


if __name__ == '__main__':
    # This will cause a deadlock
    pool = get_reusable_executor(1)
    pid = pool.apply(os.getpid, tuple())
    pool = get_reusable_executor(2)
    os.kill(pid, 15)

    pool.terminate()
    pool.join()
