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
    _winapi.CloseHandle = win32.CloseHandle
    _winapi.FILE_GENERIC_READ = win32.GENERIC_READ
    _winapi.FILE_GENERIC_WRITE = win32.GENERIC_WRITE

    # Value found at
    # https://msdn.microsoft.com/en-us/library/windows/desktop/ms684880(v=vs.85).aspx
    _winapi.PROCESS_DUP_HANDLE = 0x0040

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


# # make Connection pickable
# def reduce_connection(conn):
#     rh = DupHandle(conn.fileno(), _winapi.DUPLICATE_SAME_ACCESS)
#     return rebuild_connection, (rh, conn.readable, conn.writable)


# def rebuild_connection(reduced_handle, readable, writable):
#     handle = reduced_handle.detach()
#     return Connection(handle, readable=readable, writable=writable)


# register(Connection, reduce_connection)


def reduce_pipe_connection(conn):
    access = ((_winapi.FILE_GENERIC_READ if conn.readable else 0) |
              (_winapi.FILE_GENERIC_WRITE if conn.writable else 0))
    print(access)
    dh = DupHandle(conn.fileno(), access)
    return rebuild_pipe_connection, (dh, conn.readable, conn.writable)


def rebuild_pipe_connection(dh, readable, writable):
    from multiprocessing.connection import PipeConnection
    handle = dh.detach()
    return PipeConnection(handle, readable, writable)


register(PipeConnection, reduce_pipe_connection)


# make sockets pickable
if sys.version_info[:2] > (2, 7):
    def _reduce_socket(s):
        from multiprocessing.resource_sharer import DupSocket
        return _rebuild_socket, (DupSocket(s),)

    def _rebuild_socket(ds):
        return ds.detach()
else:
    from multiprocessing.reduction import reduce_socket as _reduce_socket
    # def fromfd(handle, family, type_, proto=0):
    #     s = socket.fromfd(handle, family, type_, proto)
    #     if s.__class__ is not socket.socket:
    #         s = socket.socket(_sock=s)
    #     return s

    # def _reduce_socket(s):
    #     access = _winapi.FILE_GENERIC_READ | _winapi.FILE_GENERIC_WRITE
    #     reduced_handle = DupHandle(s.fileno(), 0)
    #     return _rebuild_socket, (reduced_handle, s.family, s.type, s.proto)

    # def _rebuild_socket(reduced_handle, family, type_, proto):
    #     handle = reduced_handle.detach()
    #     s = fromfd(handle, family, type_, proto)
    #     _winapi.CloseHandle(handle)
    #     return s


register(socket.socket, _reduce_socket)
