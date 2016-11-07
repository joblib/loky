import sys
import time
import warnings
import threading
import multiprocessing as mp

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
    """Return the current ReusableExectutor instance.

    Start a new one if it has not been started already or if the previous
    instance was left in a broken state.

    If the previous instance does not have the requested number of workers, the
    executor is dynamically resized to adjust the number of workers prior to
    returning.

    Reusing a singleton instance spares the overhead of starting new worker
    processes and importing common python packages each time.
    """
    global _executor, _executor_args
    executor = _executor
    if executor is None:
        mp.util.debug("Create a executor with max_workers={}."
                      .format(max_workers))
        executor_id = _get_next_executor_id()
        _executor_args = dict(context=context, timeout=timeout,
                              kill_on_shutdown=kill_on_shutdown)
        _executor = executor = ReusablePoolExecutor(
            max_workers=max_workers, context=context, timeout=timeout,
            kill_on_shutdown=kill_on_shutdown, executor_id=executor_id)
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
            mp.util.debug("Creating a new executor with max_workers={} as the "
                          "previous instance was unusable ({})."
                          .format(max_workers, reason))
            executor.shutdown(wait=True)
            _executor = executor = _executor_args = None
            # Recursive call to build a new instance
            return get_reusable_executor(max_workers=max_workers, **args)
        else:
            if max_workers is not None and max_workers <= 0:
                raise ValueError("max_workers must be greater than 0, got {}."
                                 .format(max_workers))

            mp.util.debug("Reusing existing executor with max_worker={}."
                          .format(executor._max_workers))
            if max_workers is not None:
                executor._resize(max_workers)

    return executor


class ReusablePoolExecutor(ProcessPoolExecutor):
    def __init__(self, max_workers=None, context=None, timeout=None,
                 kill_on_shutdown=True, executor_id=0):
        if context is None and sys.version_info[:2] > (3, 3):
            context = mp.get_context('spawn')
        super(ReusablePoolExecutor, self).__init__(
            max_workers=max_workers, context=context, timeout=timeout,
            kill_on_shutdown=kill_on_shutdown)
        self.executor_id = executor_id

    def _resize(self, max_workers):
        if max_workers is None or max_workers == self._max_workers:
            return True
        self._wait_job_completion()
        mw = self._max_workers
        self._max_workers = max_workers
        for _ in range(max_workers, mw):
            self._call_queue.put(None)
        while len(self._processes) > max_workers and not self._broken:
            time.sleep(1e-3)

        self._adjust_process_count()
        while not all([p.is_alive() for p in self._processes.values()]):
            time.sleep(1e-3)

        return True

    def _wait_job_completion(self):
        """Wait for the cache to be empty before resizing the pool."""
        # Issue a warning to the user about the bad effect of this usage.
        if len(self._pending_work_items) > 0:
            warnings.warn("Trying to resize an executor with running jobs: "
                          "waiting for jobs completion before resizing.",
                          UserWarning)
            mp.util.debug("Executor {} waiting for jobs completion before"
                          " resizing".format(self.executor_id))
        # Wait for the completion of the jobs
        while len(self._pending_work_items) > 0:
            time.sleep(1e-3)
