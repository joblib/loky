from time import sleep
try:
    from time import monotonic
except ImportError:
    from time import time as monotonic


def wait(connections, processes, timeout=None):
    """Backward compat for python2.7

    This function wait for either:
    * one connections in object_list is ready for read
    * one process i processes is not alive
    * timeout is reached. not that this function has a precision of 2msec
    """
    if timeout is not None:
        deadline = monotonic() + timeout

    count = 0
    ready = []
    while True:
        count += 1
        if count == 10:
            count = 0
            if not all([p.is_alive() for p in processes]):
                return []
        # We cannot use select as in windows it only support sockets
        ready = [c for c in connections if c.poll(0)]
        if len(ready) > 0:
            return ready
        sleep(.001)
        if deadline - monotonic() <= 0:
            return []
