#
# Module which deals with pickling of objects.
#
# multiprocessing/reduction.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#
try:
    import copyreg
except ImportError:
    import copy_reg as copyreg
import functools
import io
import os
import pickle
import socket
import sys

__all__ = ['send_handle', 'recv_handle', 'ExecPickler', 'register', 'dump']


HAVE_SEND_HANDLE = (sys.platform == 'win32' or
                    (hasattr(socket, 'CMSG_LEN') and
                     hasattr(socket, 'SCM_RIGHTS') and
                     hasattr(socket.socket, 'sendmsg')))


def _mk_inheritable(fd):
    if sys.version_info[:2] > (3, 3):
        if sys.platform == 'win32':
            # Change to Windwos file handle
            import msvcrt
            fdh = msvcrt.get_osfhandle(fd)
            os.set_handle_inheritable(fdh, True)
            return fdh
        else:
            os.set_inheritable(fd, True)
            return fd
    elif sys.platform == 'win32':
        # TODO: find a hack??
        # Not yet working
        import msvcrt
        import _subprocess

        curproc = _subprocess.GetCurrentProcess()
        fdh = msvcrt.get_osfhandle(fd)
        fdh = _subprocess.DuplicateHandle(
            curproc, fdh, curproc, 0,
            True,  # set inheritable FLAG
            _subprocess.DUPLICATE_SAME_ACCESS)
        return fdh
    else:
        return fd


#
# Pickler subclass
#

class ExecPickler(pickle.Pickler):
    '''Pickler subclass used by multiprocessing.'''
    if sys.version_info < (3, 3):
        dispatch = pickle.Pickler.dispatch.copy()

        @classmethod
        def register(cls, type, reduce):
            '''Register a reduce function for a type.'''
            def dispatcher(self, obj):
                rv = reduce(obj)
                self.save_reduce(obj=obj, *rv)
            cls.dispatch[type] = dispatcher

        @classmethod
        def loads(self, buf, loads=pickle.loads):
            if isinstance(buf, io.BytesIO):
                buf = buf.getvalue()
            return loads(buf)

    else:
        _extra_reducers = {}
        _copyreg_dispatch_table = copyreg.dispatch_table

        def __init__(self, *args):
            pickle.Pickler.__init__(self,   *args)
            # super(ExecPickler, self).__init__(*args)
            self.dispatch_table = self._copyreg_dispatch_table.copy()
            self.dispatch_table.update(self._extra_reducers)

        @classmethod
        def register(cls, type, reduce):
            '''Register a reduce function for a type.'''
            cls._extra_reducers[type] = reduce

        loads = pickle.loads

    @classmethod
    def dumps(cls, obj, protocol=None):
        buf = io.BytesIO()
        cls(buf, protocol).dump(obj)
        if sys.version_info < (3, 3):
            return buf.getvalue()
        return buf.getbuffer()


register = ExecPickler.register


def dump(obj, file, protocol=None):
    '''Replacement for pickle.dump() using ExecPickler.'''
    ExecPickler(file, protocol).dump(obj)

#
# Platform specific definitions
#

if sys.platform == 'win32':
    # Windows
    __all__ += ['DupHandle', 'duplicate', 'steal_handle']
    import _winapi

    def duplicate(handle, target_process=None, inheritable=False):
        '''Duplicate a handle.  (target_process is a handle not a pid!)'''
        if target_process is None:
            target_process = _winapi.GetCurrentProcess()
        return _winapi.DuplicateHandle(
            _winapi.GetCurrentProcess(), handle, target_process,
            0, inheritable, _winapi.DUPLICATE_SAME_ACCESS)

    def steal_handle(source_pid, handle):
        '''Steal a handle from process identified by source_pid.'''
        source_process_handle = _winapi.OpenProcess(
            _winapi.PROCESS_DUP_HANDLE, False, source_pid)
        try:
            return _winapi.DuplicateHandle(
                source_process_handle, handle,
                _winapi.GetCurrentProcess(), 0, False,
                _winapi.DUPLICATE_SAME_ACCESS | _winapi.DUPLICATE_CLOSE_SOURCE)
        finally:
            _winapi.CloseHandle(source_process_handle)

    def send_handle(conn, handle, destination_pid):
        '''Send a handle over a local connection.'''
        dh = DupHandle(handle, _winapi.DUPLICATE_SAME_ACCESS, destination_pid)
        conn.send(dh)

    def recv_handle(conn):
        '''Receive a handle over a local connection.'''
        return conn.recv().detach()

    class DupHandle(object):
        '''Picklable wrapper for a handle.'''
        def __init__(self, handle, access, pid=None):
            if pid is None:
                # We just duplicate the handle in the current process and
                # let the receiving process steal the handle.
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
            '''Get the handle.  This should only be called once.'''
            # retrieve handle from process which currently owns it
            if self._pid == os.getpid():
                # The handle has already been duplicated for this process.
                return self._handle
            # We must steal the handle from the process whose pid is self._pid.
            proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False,
                                       self._pid)
            try:
                return _winapi.DuplicateHandle(
                    proc, self._handle, _winapi.GetCurrentProcess(),
                    self._access, False, _winapi.DUPLICATE_CLOSE_SOURCE)
            finally:
                _winapi.CloseHandle(proc)

else:
    # Unix
    __all__ += ['DupFd']

    def DupFd(fd):
        '''Return a wrapper for an fd.'''

        from .popen_exec import get_spawning_popen
        popen = get_spawning_popen()
        return popen.DupFd(popen.duplicate_for_child(fd))

#
# Try making some callable types picklable
#

def _reduce_method(m):
    if m.__self__ is None:
        return getattr, (m.__class__, m.__func__.__name__)
    else:
        return getattr, (m.__self__, m.__func__.__name__)
class _C:
    def f(self):
        pass
    @classmethod
    def h(cls):
        pass
register(type(_C().f), _reduce_method)
register(type(_C.h), _reduce_method)


def _reduce_method_descriptor(m):
    return getattr, (m.__objclass__, m.__name__)
register(type(list.append), _reduce_method_descriptor)
register(type(int.__add__), _reduce_method_descriptor)


def _reduce_partial(p):
    return _rebuild_partial, (p.func, p.args, p.keywords or {})
def _rebuild_partial(func, args, keywords):
    return functools.partial(func, *args, **keywords)
register(functools.partial, _reduce_partial)

#
# Make sockets picklable
#

if sys.platform == 'win32':
    def _reduce_socket(s):
        from multiprocessing.resource_sharer import DupSocket
        return _rebuild_socket, (DupSocket(s),)
    def _rebuild_socket(ds):
        return ds.detach()
    register(socket.socket, _reduce_socket)

else:
    def _reduce_socket(s):
        df = DupFd(s.fileno())
        return _rebuild_socket, (df, s.family, s.type, s.proto)
    def _rebuild_socket(df, family, type, proto):
        fd = df.detach()
        return socket.socket(family, type, proto, fileno=fd)
    register(socket.socket, _reduce_socket)

if sys.version_info >= (3, 3):
    from multiprocessing.connection import Connection
else:
    from _multiprocessing import Connection


def reduce_connection(conn):
    df = DupFd(conn.fileno())
    return rebuild_connection, (df, conn.readable, conn.writable)


def rebuild_connection(df, readable, writable):
    fd = df.detach()
    return Connection(fd, readable, writable)

register(Connection, reduce_connection)
