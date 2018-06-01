import os
import sys
import errno
import signal
import threading
import subprocess


def _flag_current_thread_clean_exit():
    """Put a ``_clean_exit`` flag on the current thread"""
    thread = threading.current_thread()
    thread._clean_exit = True


def safe_terminate(process):
    """Terminate a process and its children.
    """

    _safe_terminate(process.pid)
    process.join()


def _safe_terminate(pid):
    """Terminate the children of a process before killing this process.
    """

    if sys.platform == "win32":
        # On windows, the taskkill function with option `/T` terminate a given
        # process pid and its children.
        try:
            subprocess.check_output(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stderr=None)
        except subprocess.CalledProcessError as e:
            # In windows, taskkill return 1 for permission denied and 128 for
            # no process found.
            if e.returncode not in [1, 128]:
                raise
            elif e.returncode == 1:
                # Try to kill the process with a signal if taskkill was denied
                # permission.
                os.kill(pid, signal.SIGTERM)

    else:
        try:
            children_pids = subprocess.check_output(
                ["ps", "-o", "pid=", "--ppid", str(pid)],
                stderr=None
            )
        except subprocess.CalledProcessError as e:
            # `ps` returns 1 when no child process has been found
            children_pids = b''

        # Decode the result, split the cpid and remove the trailing line
        children_pids = children_pids.decode().split('\n')[:-1]
        for cpid in children_pids:
            cpid = int(cpid)
            _safe_terminate(cpid)

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            # if OSError is raised with [Errno 3] no such process, the process
            # is already terminated, else, raise the error
            if e.errno != errno.ESRCH:
                raise
