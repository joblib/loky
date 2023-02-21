###############################################################################
# Popen for LokyProcess.
#
# author: Thomas Moreau and Olivier Grisel
#
import os
import sys
import signal
from io import BytesIO
from multiprocessing import util
from multiprocessing.connection import wait
from multiprocessing.context import set_spawning_popen

from . import reduction, resource_tracker, spawn


__all__ = ["Popen"]


#
# Wrapper for an fd used while launching a process
#


class _DupFd:
    def __init__(self, fd):
        self.fd = reduction._mk_inheritable(fd)

    def detach(self):
        return self.fd


#
# Start child process using subprocess.Popen
#


class Popen:
    method = "loky"
    DupFd = _DupFd

    def __init__(self, process_obj):
        sys.stdout.flush()
        sys.stderr.flush()
        self.returncode = None
        self._fds = []
        self._launch(process_obj)

    def duplicate_for_child(self, fd):
        self._fds.append(fd)
        return reduction._mk_inheritable(fd)

    def poll(self, flag=os.WNOHANG):
        if self.returncode is None:
            while True:
                try:
                    pid, sts = os.waitpid(self.pid, flag)
                except OSError:
                    # Child process not yet created. See #1731717
                    # e.errno == errno.ECHILD == 10
                    return None
                else:
                    break
            if pid == self.pid:
                if os.WIFSIGNALED(sts):
                    self.returncode = -os.WTERMSIG(sts)
                else:
                    assert os.WIFEXITED(sts)
                    self.returncode = os.WEXITSTATUS(sts)
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if timeout is not None:
                if not wait([self.sentinel], timeout):
                    return None
            # This shouldn't block if wait() returned successfully.
            return self.poll(os.WNOHANG if timeout == 0.0 else 0)
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            try:
                os.kill(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                if self.wait(timeout=0.1) is None:
                    raise

    def _launch(self, process_obj):
        tracker_fd = resource_tracker._resource_tracker.getfd()

        fp = BytesIO()
        set_spawning_popen(self)
        try:
            prep_data = spawn.get_preparation_data(
                process_obj._name,
                getattr(process_obj, "init_main_module", True),
            )
            reduction.dump(prep_data, fp)
            reduction.dump(process_obj, fp)

        finally:
            set_spawning_popen(None)

        try:
            parent_r, child_w = os.pipe()
            child_r, parent_w = os.pipe()

            reduction._mk_inheritable(child_r)
            reduction._mk_inheritable(child_w)
            reduction._mk_inheritable(tracker_fd)
            cmd_python = spawn.get_command_line(
                fd=child_r, process_name=process_obj.name
            )
            self._fds += [child_r, child_w, tracker_fd]
            if sys.version_info >= (3, 8) and os.name == "posix":
                mp_tracker_fd = prep_data["mp_tracker_args"]["fd"]
                self.duplicate_for_child(mp_tracker_fd)

            from .fork_exec import fork_exec

            pid = fork_exec(cmd_python, self._fds, env=process_obj.env)
            util.debug(
                f"launched python with pid {pid} and cmd:\n{cmd_python}"
            )
            self.sentinel = parent_r

            method = "getbuffer"
            if not hasattr(fp, method):
                method = "getvalue"
            with os.fdopen(parent_w, "wb") as f:
                f.write(getattr(fp, method)())
            self.pid = pid
        finally:
            if parent_r is not None:
                util.Finalize(self, os.close, (parent_r,))
            for fd in (child_r, child_w):
                if fd is not None:
                    os.close(fd)

    @staticmethod
    def thread_is_spawning():
        return True
