import sys
try:
    import _winapi
except ImportError:
    from _multiprocessing import win32 as _winapi
try:
    WAIT_OBJECT_0 = _winapi.WAIT_OBJECT_0
except AttributeError:
    import _subprocess
    WAIT_OBJECT_0 = _subprocess.WAIT_OBJECT_0
try:
    WAIT_ABANDONED_0 = _winapi.WAIT_ABANDONED_0
    WAIT_TIMEOUT = _winapi.WAIT_TIMEOUT
except AttributeError:
    # value found in https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032(v=vs.85).aspx
    WAIT_ABANDONED_0 = 0x00000080L
    WAIT_TIMEOUT = 0x00000102L
INFINITE = _winapi.INFINITE


def _exhaustive_wait(handles, timeout):
    # Return ALL handles which are currently signaled.  (Only
    # returning the first signaled might create starvation issues.)
    L = list(handles)
    ready = []
    while L:
        res = _winapi.WaitForMultipleObjects(L, False, timeout)
        if res == WAIT_TIMEOUT:
            break
        elif WAIT_OBJECT_0 <= res < WAIT_OBJECT_0 + len(L):
            res -= WAIT_OBJECT_0
        elif WAIT_ABANDONED_0 <= res < WAIT_ABANDONED_0 + len(L):
            res -= WAIT_ABANDONED_0
        else:
            raise RuntimeError('Should not get here')
        ready.append(L[res])
        L = L[res + 1:]
        timeout = 0
    return ready

try:
    _ready_errors = {_winapi.ERROR_BROKEN_PIPE,
                     _winapi.ERROR_NETNAME_DELETED}
except AttributeError:
    _ready_errors = {109, 64}

if sys.version_info[:2] > (2, 7):
    def wait(object_list, timeout=None):
        '''
        Wait till an object in object_list is ready/readable.

        Returns list of those objects which are ready/readable.
        '''
        if timeout is None:
            timeout = INFINITE
        elif timeout < 0:
            timeout = 0
        else:
            timeout = int(timeout * 1000 + 0.5)

        object_list = list(object_list)
        waithandle_to_obj = {}
        ov_list = []
        ready_objects = set()
        ready_handles = set()

        try:
            for o in object_list:
                try:
                    fileno = getattr(o, 'fileno')
                except AttributeError:
                    waithandle_to_obj[o.__index__()] = o
                else:
                    # start an overlapped read of length zero
                    try:
                        ov, err = _winapi.ReadFile(fileno(), 0, True)
                    except OSError as e:
                        err = e.winerror
                        if err not in _ready_errors:
                            raise
                    if err == _winapi.ERROR_IO_PENDING:
                        ov_list.append(ov)
                        waithandle_to_obj[ov.event] = o
                    else:
                        # If o.fileno() is an overlapped pipe handle and
                        # err == 0 then there is a zero length message
                        # in the pipe, but it HAS NOT been consumed.
                        ready_objects.add(o)
                        timeout = 0

            ready_handles = _exhaustive_wait(waithandle_to_obj.keys(),
                                             timeout)
        finally:
            # request that overlapped reads stop
            for ov in ov_list:
                ov.cancel()

            # wait for all overlapped reads to stop
            for ov in ov_list:
                try:
                    _, err = ov.GetOverlappedResult(True)
                except OSError as e:
                    err = e.winerror
                    if err not in _ready_errors:
                        raise
                if err != _winapi.ERROR_OPERATION_ABORTED:
                    o = waithandle_to_obj[ov.event]
                    ready_objects.add(o)
                    if err == 0:
                        # If o.fileno() is an overlapped pipe handle then
                        # a zero length message HAS been consumed.
                        if hasattr(o, '_got_empty_message'):
                            o._got_empty_message = True

        ready_objects.update(waithandle_to_obj[h] for h in ready_handles)
        return [p for p in object_list if p in ready_objects]
else:
    from time import sleep
    def wait(object_list, processes, timeout=None):
        count = 0
        ready = []
        while True:
            count += 1
            if count == 100:
                count = 0
                if not all([p.is_alive() for p in processes]):
                    return []
            ready = [o for o in object_list if o.poll(0)]
            if len(ready) > 0:
                return ready
            sleep(.001)

