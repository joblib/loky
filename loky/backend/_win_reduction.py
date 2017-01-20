#
# Module which deals with pickling of objects.
#
# multiprocessing/reduction.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#
import os
import sys
import socket
from .reduction import register

# Windows
if sys.version_info[:2] < (3, 3):
    import _subprocess as _winapi

    from _multiprocessing import win32
    _winapi.OpenProcess = win32.OpenProcess

    from _multiprocessing import Connection, PipeConnection
else:
    import _winapi

    from multiprocessing.connection import Connection, PipeConnection


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


# make Connection pickable
def reduce_connection(conn):
    rh = DupHandle(conn.fileno(), _winapi.DUPLICATE_SAME_ACCESS)
    return rebuild_connection, (rh, conn.readable, conn.writable)


def rebuild_connection(reduced_handle, readable, writable):
    handle = reduced_handle.detach()
    return Connection(handle, readable=readable, writable=writable)


register(Connection, reduce_connection)


def reduce_pipe_connection(conn):
    access = ((_winapi.FILE_GENERIC_READ if conn.readable else 0) |
              (_winapi.FILE_GENERIC_WRITE if conn.writable else 0))
    dh = DupHandle(conn.fileno(), access)
    return rebuild_pipe_connection, (dh, conn.readable, conn.writable)


def rebuild_pipe_connection(dh, readable, writable):
    from .connection import PipeConnection
    handle = dh.detach()
    return PipeConnection(handle, readable, writable)


register(PipeConnection, reduce_pipe_connection)


# make sockets pickable
def _reduce_socket(s):
    from multiprocessing.resource_sharer import DupSocket
    return _rebuild_socket, (DupSocket(s),)


def _rebuild_socket(ds):
    return ds.detach()


register(socket.socket, _reduce_socket)
