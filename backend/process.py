import os
import sys
import multiprocessing as mp
try:
    from multiprocessing.process import BaseProcess
    from multiprocessing.context import BaseContext
except ImportError:
    from multiprocessing.process import Process as BaseProcess
    BaseContext = object


class ExecProcess(BaseProcess):
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


class ExecContext(BaseContext):
    _name = 'exec'
    Process = ExecProcess
if sys.version_info < (3, 4):
    from .backend import synchronize
    mp.synchronize.SemLock = synchronize.SemLock
    mp.synchronize.sem_unlink = synchronize.sem_unlink
    mp.synchronize = synchronize
try:
    from multiprocessing import context
    context._concrete_contexts['exec'] = ExecContext()
    mp.set_start_method('spawn', force=True)
except ImportError:
    print("BIGpass\n\n\n\n\n\n\n\n")
    pass
