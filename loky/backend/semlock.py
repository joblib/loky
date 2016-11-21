"""Ctypes implementation for posix semaphore

only work for fork_exec
OSX -> no semaphore with value > 1

"""

import os
import sys
import time
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


class timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


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
if sys.platform != "darwin":
    pthread.sem_timedwait.argtypes = [ctypes.c_void_p,
                                      ctypes.POINTER(timespec)]

try:
    from threading import get_ident
except ImportError:
    def get_ident():
        return threading.current_thread().ident


if sys.version_info[:2] < (3, 3):
    FileExistsError = OSError


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


def _sem_timedwait(handle, timeout):
    t_start = time.time()
    if sys.platform != "darwin":
        sec = int(timeout)
        tv_sec = int(t_start)
        nsec = int(1e9 * (timeout - sec) + .5)
        tv_nsec = int(1e9 * (t_start - tv_sec) + .5)
        deadline = timespec(sec+tv_sec, nsec+tv_nsec)
        deadline.tv_sec += int(deadline.tv_nsec / 1000000000)
        deadline.tv_nsec %= 1000000000
        return pthread.sem_timedwait(handle, ctypes.pointer(deadline))

    # PERFORMANCE WARNING
    # No sem_timedwait on OSX so we implement our own method. This method can
    # dergade performances has the wait can have a latency up to 20 msecs
    deadline = t_start + timeout
    delay = 0
    now = time.time()
    while True:
        # Poll the sem file
        res = pthread.sem_trywait(handle)
        if res == 0:
            return 0
        else:
            e = ctypes.get_errno()
            if e != errno.EAGAIN:
                raiseFromErrno()

        # check for timeout
        now = time.time()
        if now > deadline:
            ctypes.set_errno(errno.ETIMEDOUT)
            return -1

        # calculate how much time left and check the delay is not too long
        # -- maximum is 20 msecs
        difference = (deadline - now)
        delay = min(delay, 20e-3, difference)

        # Sleep and increase delay
        time.sleep(delay)
        delay += 1e-3


class SemLock(object):
    """ctypes wrapper to the unix semaphore"""

    _rand = tempfile._RandomNameSequence()

    def __init__(self, kind, value, maxvalue, name=None, unlink_now=False):
        self.count = 0
        self.ident = 0
        self.kind = kind
        self.maxvalue = maxvalue
        self.name = name.encode('ascii')
        self.handle = _sem_open(
            self.name, os.O_CREAT | os.O_EXCL, 384, value)

        if self.handle == SEM_FAILURE:
            raise FileExistsError('cannot find name for semaphore')

    def __del__(self):
        try:
            res = pthread.sem_close(self.handle)
            assert res == 0, "Issue while closing semaphores"
        except AttributeError:
            pass

    def _is_mine(self):
        return self.count > 0 and get_ident() == self.ident

    def acquire(self, blocking=True, timeout=None):
        if self.kind == RECURSIVE_MUTEX and self._is_mine():
            self.count += 1
            return True

        if blocking and timeout is None:
            res = pthread.sem_wait(self.handle)
        elif not blocking or timeout <= 0:
            res = pthread.sem_trywait(self.handle)
        else:
            res = _sem_timedwait(self.handle, timeout)
        if res < 0:
            e = ctypes.get_errno()
            if e == errno.EINTR:
                return None
            elif e in [errno.EAGAIN, errno.ETIMEDOUT]:
                return False
            raiseFromErrno()
        self.count += 1
        self.ident = get_ident()
        return True

    def release(self, *args):
        if self.kind == RECURSIVE_MUTEX:
            assert self._is_mine(), (
                "attempt to release recursive lock not owned by thread")
            if self.count > 1:
                self.count -= 1
                return
            assert self.count == 1
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
                value = self._get_value()
                if value >= self.maxvalue:
                    raise ValueError(
                        "semaphore or lock realeased too many times")

        if pthread.sem_post(self.handle) < 0:
            raiseFromErrno()

        self.count -= 1

    def _get_value(self):
        value = ctypes.pointer(ctypes.c_int(-1))
        if pthread.sem_getvalue(self.handle, value) < 0:
            raiseFromErrno()
        return value.contents.value

    def _count(self):
        return self.count

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

    def _after_fork(self):
        self.count = 0

    @staticmethod
    def _rebuild(handle, kind, maxvalue, name):
        self = SemLock.__new__(SemLock)
        self.count = 0
        self.ident = 0
        self.kind = kind
        self.maxvalue = maxvalue
        self.name = name
        self.handle = _sem_open(name, 0)
        if self.handle == SEM_FAILURE:
            raise FileNotFoundError('cannot find semaphore named %s' % name)
        return self


def raiseFromErrno():
    e = ctypes.get_errno()
    raise OSError(e, errno.errorcode[e])
