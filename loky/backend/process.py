import os
import sys
if sys.version_info > (3, 4):
    from multiprocessing.process import BaseProcess
else:
    from multiprocessing.process import Process as BaseProcess


class PosixLokyProcess(BaseProcess):
    _start_method = 'loky'

    @staticmethod
    def _Popen(process_obj):
        from .popen_loky import Popen
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
                super(PosixLokyProcess, self).__init__(
                    group=group, target=target, name=name, args=args,
                    kwargs=kwargs)
                self.daemon = daemon
            else:
                super(PosixLokyProcess, self).__init__(
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


#
# We subclass bytes to avoid accidental transmission of auth keys over network
#

class AuthenticationKey(bytes):
    def __reduce__(self):
        from .popen_loky import is_spawning
        if not is_spawning():
            raise TypeError(
                'Pickling an AuthenticationKey object is '
                'disallowed for security reasons'
                )
        return AuthenticationKey, (bytes(self),)
