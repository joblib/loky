###############################################################################
# Extra reducers for Windows system and connections objects
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/reduction.py (17/02/2017)
#  * Add adapted reduction for LokyProcesses and socket/PipeConnection
#
import os
import sys
import socket
from .reduction import dispatch_table


if sys.platform == 'win32':
    import _winapi
    from multiprocessing.connection import PipeConnection

    class DupHandle(object):
        def __init__(self, handle, access, pid=None):
            # duplicate handle for process with given pid
            if pid is None:
                pid = os.getpid()
            proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, pid)
            try:
                self._handle = _winapi.DuplicateHandle(
                    _winapi.GetCurrentProcess(),
                    handle, proc, access, False, 0)
            finally:
                _winapi.CloseHandle(proc)
            self._access = access
            self._pid = pid

        def detach(self):
            # retrieve handle from process which currently owns it
            if self._pid == os.getpid():
                return self._handle
            proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False,
                                       self._pid)
            try:
                return _winapi.DuplicateHandle(
                    proc, self._handle, _winapi.GetCurrentProcess(),
                    self._access, False, _winapi.DUPLICATE_CLOSE_SOURCE)
            finally:
                _winapi.CloseHandle(proc)

    def reduce_pipe_connection(conn):
        access = ((_winapi.FILE_GENERIC_READ if conn.readable else 0) |
                  (_winapi.FILE_GENERIC_WRITE if conn.writable else 0))
        dh = DupHandle(conn.fileno(), access)
        return rebuild_pipe_connection, (dh, conn.readable, conn.writable)

    def rebuild_pipe_connection(dh, readable, writable):
        from multiprocessing.connection import PipeConnection
        handle = dh.detach()
        return PipeConnection(handle, readable, writable)
    dispatch_table[PipeConnection] = reduce_pipe_connection

    from multiprocessing.reduction import _reduce_socket
    dispatch_table[socket.socket] = _reduce_socket
