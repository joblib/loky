from .process_executor import ProcessPoolExecutor
import multiprocessing as mp
import threading
import warnings
import os
import time
from concurrent.futures import TimeoutError

__all__ = ['get_reusable_pool']


# Protect the queue from being reused in different threads
_local = threading.local()
_pool_id_lock = mp.Lock()
_next_pool_id = 0


def _get_next_pool_id():
    global _next_pool_id
    with _pool_id_lock:
        pool_id = _next_pool_id
        _next_pool_id += 1
        return pool_id


def get_reusable_pool(max_workers=None, context=None, kill_on_shutdown=True):
    """Return a the current ReusablePool. Start a new one if needed"""
    _pool = getattr(_local, '_pool', None)
    _args = getattr(_local, '_args', None)
    if _pool is None:
        mp.util.debug("Create a pool with size {}.".format(max_workers))
        pool_id = _get_next_pool_id()
        _local._pool = _pool = ReusablePoolExecutor(
            max_workers=max_workers, context=context,
            kill_on_shutdown=kill_on_shutdown, pool_id=pool_id)
        _local._args = dict(context=context, kill_on_shutdown=kill_on_shutdown)
        res = _pool.submit(time.sleep, 0.01)
        try:
            res.result(timeout=2)
        except TimeoutError:
            print(_pool._processes)
            print(_pool._call_queue.empty())
            print(_pool._call_queue.qsize())
            print(_pool._pending_work_items[0].future)
            raise RuntimeError
    else:
        args = dict(context=context, kill_on_shutdown=kill_on_shutdown)
        if _pool._broken or _pool._shutdown_thread or args != _args:
            mp.util.debug("Create a new pool with {} processes as the "
                          "previous one was unusable".format(max_workers))
            _pool.shutdown()
            _local._pool = _pool = None
            _local._args = _args = None
            return get_reusable_pool(max_workers=max_workers, context=context,
                                     kill_on_shutdown=kill_on_shutdown)
        else:
            if _pool._resize(max_workers):
                mp.util.debug("Resized existing pool to target size.")
                return _pool
            mp.util.debug("Failed to resize existing pool to target size {}."
                          "".format(max_workers))
            _pool.terminate()
            _pool.join()
            _local._pool = _pool = None
            _local._args = _args = None
            return get_reusable_pool(max_workers=max_workers, context=context,
                                     kill_on_shutdown=kill_on_shutdown, )

    return _pool


class ReusablePoolExecutor(ProcessPoolExecutor):
    def __init__(self, max_workers=None, context=None, kill_on_shutdown=True,
                 pool_id=0):
        super(ReusablePoolExecutor, self).__init__(
            max_workers=max_workers, context=context,
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

    def apply(self, fn, *args, **kwargs):
        r = self.submit(fn, *args, **kwargs)
        return r.result()

    def apply_async(self, fn, *args, **kwargs):
        r = self.submit(fn, *args, **kwargs)
        return BinderResult(r)

    def map_async(self, fn, *args, **kwargs):
        r = self.map(fn, *args, **kwargs)
        return r


class BinderResult():
    def __init__(self, res):
        self.res = res

    def get(self):
        return self.res.result()


if __name__ == '__main__':
    # This will cause a deadlock
    pool = get_reusable_pool(1)
    pid = pool.apply(os.getpid, tuple())
    pool = get_reusable_pool(2)
    os.kill(pid, 15)

    pool.terminate()
    pool.join()
