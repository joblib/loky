import sys
import socket
from .reduction import register


def DupFd(fd):
    '''Return a wrapper for an fd.'''

    from .popen_loky import get_spawning_popen
    popen = get_spawning_popen()
    return popen.DupFd(popen.duplicate_for_child(fd))


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
