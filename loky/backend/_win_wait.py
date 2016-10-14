""" Windows systems wait compat for python2.7
"""

from time import sleep
import ctypes
try:
    from time import monotonic
except ImportError:
    # Backward old for crappy old Python that did not have cross-platform
    # monotonic clock by default.

    # TODO: do we want to add support for cygwin at some point? See:
    # https://github.com/atdt/monotonic/blob/master/monotonic.py
    GetTickCount64 = ctypes.windll.kernel32.GetTickCount64
    GetTickCount64.restype = ctypes.c_ulonglong

    def monotonic():
        """Monotonic clock, cannot go backward."""
        return GetTickCount64() / 1000.0


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
            dead_processes = [p for p in processes if not p.is_alive()]
            if len(dead_processes) > 0:
                return dead_processes
        # We cannot use select as in windows it only support sockets
        ready = [c for c in connections if c.poll(0)]
        if len(ready) > 0:
            return ready
        sleep(.001)
        if timeout is not None and deadline - monotonic() <= 0:
            return []
