"""Ctypes implementation for posix semaphore

only work for fork_exec
OSX -> no semaphore with value > 1

"""

import os
import sys
import errno
import ctypes
import tempfile
import threading
from multiprocessing import util
from ctypes.util import find_library

SEM_FAILURE = 0
if sys.platform == 'darwin':
    SEM_FAILURE = -1

RECURSIVE_MUTEX = 0
SEMAPHORE = 1


pthread = ctypes.CDLL(find_library('pthread'), use_errno=True)
# pthread.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int,
#                              ctypes.c_int, ctypes.c_int]
pthread.sem_open.restype = ctypes.c_void_p
pthread.sem_close.argtypes = [ctypes.c_void_p]
pthread.sem_wait.argtypes = [ctypes.c_void_p]
pthread.sem_trywait.argtypes = [ctypes.c_void_p]
pthread.sem_post.argtypes = [ctypes.c_void_p]
pthread.sem_getvalue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
pthread.sem_unlink.argtypes = [ctypes.c_char_p]

try:
    from threading import get_ident
except ImportError:
    def get_ident():
        return threading.current_thread().ident


def sem_unlink(name):
    if pthread.sem_unlink(name) < 0:
        raiseFromErrno()


def _sem_open(name, o_flag, perm=None, value=None):
    if perm is None:
        assert value is None, "Wrong number of argument (either 2 or 4)"
        return pthread.sem_open(ctypes.c_char_p(name), o_flag)
    else:
        assert value is not None, "Wrong number of argument (either 2 or 4)"
        return pthread.sem_open(ctypes.c_char_p(name), ctypes.c_int(o_flag),
                                ctypes.c_int(perm), ctypes.c_int(value))


class SemLock(object):
    """ctypes wrapper to the unix semaphore"""

    _rand = tempfile._RandomNameSequence()

    def __init__(self, kind, value, maxvalue, n=None, u=None, name=None):
        assert n is None
        self.count = 0
        self.ident = 0
        self.kind = kind
        self.maxvalue = maxvalue
        if name:
            self.name = name
            self.handle = _sem_open(name, 0)
            if self.handle == SEM_FAILURE:
                raise FileExistsError('cannot find name for semaphore')
        else:
            for i in range(100):
                self.name = self._make_name()
                self.handle = _sem_open(
                    self.name, os.O_CREAT | os.O_EXCL, 384, value)
                if self.handle != SEM_FAILURE:
                    break
            else:
                raise FileExistsError('cannot find name for semaphore')

            util.debug('created semlock with handle %s and name %s'
                       % (self.handle, self.name))

    @staticmethod
    def _make_name():
        return str.encode('/mp-%s' % next(SemLock._rand))

    def _is_mine(self):
        return self.count > 0 and get_ident() == self.ident

    def acquire(self, blocking=True, timeout=None):
        if blocking and timeout is None:
            res = pthread.sem_wait(self.handle)
        elif not blocking:
            res = pthread.sem_trywait(self.handle)
        else:
            res = pthread.sem_trywait(self.handle)
            if res < 0:
                e = ctypes.get_errno()
                if e == errno.EINTR:
                    return None
                elif e == errno.EAGAIN:
                    return False
                raiseFromErrno()
            return True
        if res < 0:
            e = ctypes.get_errno()
            if e == errno.EINTR:
                return None
            elif e == errno.EAGAIN:
                return False
            raiseFromErrno()
        self.count += 1
        self.ident = get_ident()
        return True

    def release(self, *args):
        if self.kind == RECURSIVE_MUTEX:
            assert self.is_mine(), (
                "attempt to release recursive lock not owned by thread")
            if self.count > 1:
                self.count -= 1
                return
        else:
            if sys.platform == 'darwin':
                # Handle broken get_value for mac ==> only Lock will work
                # as sem_get_value do not work properly
                if self.maxvalue == 1:
                    if pthread.sem_trywait(self.handle) < 0:
                        e = ctypes.get_errno()
                        if e != errno.EAGAIN:
                            raise OSError(e, errno.errorcode[e])
                    else:
                        if pthread.sem_post(self.handle) < 0:
                            raiseFromErrno()
                        else:
                            raise ValueError(
                                "semaphore or lock realeased too many times")
                else:
                    import warnings
                    warnings.warn("semaphore are broken on OSX, release might "
                                  "increase its maximal value", RuntimeWarning)
            else:
                value = ctypes.pointer(ctypes.c_int(-1))
                if pthread.sem_getvalue(self.handle, value) < 0:
                    raiseFromErrno()
                elif value.contents.value >= self.maxvalue:
                        raise ValueError(
                            "semaphore or lock realeased too many times")

        if pthread.sem_post(self.handle) < 0:
            raiseFromErrno()

        self.count -= 1

    def _is_zero(self):
        if sys.platform == 'darwin':
            # Handle broken get_value for mac ==> only Lock will work
            # as sem_get_value do not work properly
            if pthread.sem_trywait(self.handle) < 0:
                e = ctypes.get_errno()
                if e == errno.EAGAIN:
                    return True
                raise OSError(e, errno.errorcode[e])
            else:
                if pthread.sem_post(self.handle) < 0:
                    raiseFromErrno()
                return False
        else:
            value = ctypes.pointer(ctypes.c_int(-1))
            if pthread.sem_getvalue(self.handle, value) < 0:
                raiseFromErrno()
            return value.contents.value == 0


def raiseFromErrno():
    e = ctypes.get_errno()
    raise OSError(e, errno.errorcode[e])
