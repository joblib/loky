import os
import sys
import multiprocessing as mp
try:
    from multiprocessing.process import BaseProcess
    from multiprocessing.context import BaseContext
except ImportError:
    from multiprocessing.process import Process as BaseProcess
    BaseContext = object


class PosixExecProcess(BaseProcess):
    _start_method = 'exec'

    @staticmethod
    def _Popen(process_obj):
        from .popen_exec import Popen
        return Popen(process_obj)

    if sys.version_info < (3, 3):
        def start(self):
            '''
            Start child process
            '''
            from multiprocessing.process import _current_process, _cleanup
            assert self._popen is None, 'cannot start a process twice'
            assert self._parent_pid == os.getpid(), \
                'can only start a process object created by current process'
            _cleanup()
            self._popen = self._Popen(self)
            self._sentinel = self._popen.sentinel
            _current_process._children.add(self)

        @property
        def sentinel(self):
            '''
            Return a file descriptor (Unix) or handle (Windows) suitable for
            waiting for process termination.
            '''
            try:
                return self._sentinel
            except AttributeError:
                raise ValueError("process not started")

    if sys.version_info < (3, 4):
        def __init__(self, group=None, target=None, name=None, args=(),
                     kwargs={}, daemon=None):
            if sys.version_info < (3, 3):
                super(PosixExecProcess, self).__init__(
                    group=group, target=target, name=name, args=args,
                    kwargs=kwargs)
                self.daemon = daemon
            else:
                super(PosixExecProcess, self).__init__(
                    group=group, target=target, name=name, args=args,
                    kwargs=kwargs, daemon=daemon)
            self.authkey = self.authkey

        @property
        def authkey(self):
            return self._authkey

        @authkey.setter
        def authkey(self, authkey):
            '''
            Set authorization key of process
            '''
            self._authkey = AuthenticationKey(authkey)


class ExecContext(BaseContext):
    _name = 'exec'
    Process = PosixExecProcess


#
# We subclass bytes to avoid accidental transmission of auth keys over network
#

class AuthenticationKey(bytes):
    def __reduce__(self):
        from .popen_exec import is_spawning
        if not is_spawning():
            raise TypeError(
                'Pickling an AuthenticationKey object is '
                'disallowed for security reasons'
                )
        return AuthenticationKey, (bytes(self),)


class WinExecProcess(BaseProcess):
    """subclass Process for windows access to sentinel"""
    def start(self):
        super(WinExecProcess, self).start()
        self.sentinel = int(self._popen._handle)

try:
    from multiprocessing import context
    context._concrete_contexts['loky'] = ExecContext()
except ImportError:
    pass
