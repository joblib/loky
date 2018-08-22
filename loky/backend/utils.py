import os
import sys
import errno
import signal
import warnings
import threading
import subprocess
import multiprocessing as mp
try:
    import psutil
except ImportError:
    psutil = None


def _flag_current_thread_clean_exit():
    """Put a ``_clean_exit`` flag on the current thread"""
    thread = threading.current_thread()
    thread._clean_exit = True


def recursive_terminate(process, use_psutil=True):
    mp.util.debug("recursive_kill for process {}".format(process))
    if use_psutil and psutil is not None:
        _recursive_terminate_with_psutil(process)
    else:
        _recursive_terminate_without_psutil(process)


def _recursive_terminate_with_psutil(process, retries=5):
    mp.util.debug("psutil.kill process {}".format(process.pid))
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return

    mp.util.debug("Found children = {}".format(children))
    # Kill the children in reverse order to avoid killing the parents before
    # the children in cases where the tree is nested.
    for child in children[::-1]:
        try:
            mp.util.debug("psutil.kill child process {}".format(child.pid))
            child.kill()
        except psutil.NoSuchProcess as e:
            import traceback
            traceback.print_exc()
            pass

    process.terminate()
    process.join()


def _recursive_terminate_without_psutil(process):
    """Terminate a process and its descendants.
    """
    try:
        _recursive_terminate(process.pid)
    except OSError as e:
        warnings.warn("Failed to kill subprocesses on this platform. Please"
                      "install psutil: https://github.com/giampaolo/psutil")
        # In case we cannot introspect the children, we fall back to the
        # classic Process.terminate.
        process.terminate()
    process.join()


def _recursive_terminate(pid):
    """Recursively kill the descendants of a process before killing it.
    """
    mp.util.debug("nopsutil.kill child process {}".format(pid))

    if sys.platform == "win32":
        # On windows, the taskkill function with option `/T` terminate a given
        # process pid and its children.
        try:
            subprocess.check_output(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stderr=None)
        except subprocess.CalledProcessError as e:
            # In windows, taskkill return 1 for permission denied and 128, 255
            # for no process found.
            if e.returncode not in [1, 128, 255]:
                raise
            elif e.returncode == 1:
                # Try to kill the process without its descendants if taskkill
                # was denied permission. If this fails too, with an error
                # different from process not found, let the top level function
                # raise a warning and retry to kill the process.
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError as e:
                    if e.errno != errno.ESRCH:
                        raise

    else:
        try:
            children_pids = subprocess.check_output(
                ["pgrep", "-P", str(pid)],
                stderr=None
            )
        except subprocess.CalledProcessError as e:
            # `ps` returns 1 when no child process has been found
            if e.returncode == 1:
                children_pids = b''
            else:
                raise

        # Decode the result, split the cpid and remove the trailing line
        children_pids = children_pids.decode().split('\n')[:-1]
        for cpid in children_pids:
            cpid = int(cpid)
            _recursive_terminate(cpid)

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            # if OSError is raised with [Errno 3] no such process, the process
            # is already terminated, else, raise the error and let the top
            # level function raise a warning and retry to kill the process.
            if e.errno != errno.ESRCH:
                raise
