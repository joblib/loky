import os


def get_worker_id():
    """Get the worker ID of the current process.
    
    For a `ReusableExectutor` with `max_workers=n`, the worker ID is in the
    range [0..n). This is suited for reuse of persistent objects such as GPU
    IDs. This function only works at the first level of parallelization (i.e.
    not for nested parallelization). Resizing the `ReusableExectutor` will
    result in unpredictable return values.
    
    Returns -1 when the process is not a worker.
    """
    return int(os.environ.get('LOKY_WORKER_ID', -1))
