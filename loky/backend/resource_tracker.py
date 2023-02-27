###############################################################################
# Server process to keep track of unlinked resources, like folders and
# semaphores and clean them.
#
# author: Thomas Moreau
#
# adapted from multiprocessing/semaphore_tracker.py  (17/02/2017)
#  * include custom spawnv_passfds to start the process
#  * add some VERBOSE logging
#
# TODO: multiprocessing.resource_tracker was contributed to Python 3.8 so
# once loky drops support for Python 3.7 it might be possible to stop
# maintaining this loky-specific fork. As a consequence, it might also be
# possible to stop maintaining the loky.backend.synchronize fork of
# multiprocessing.synchronize.

#
# On Unix we run a server process which keeps track of unlinked
# resources. The server ignores SIGINT and SIGTERM and reads from a
# pipe. The resource_tracker implements a reference counting scheme: each time
# a Python process anticipates the shared usage of a resource by another
# process, it signals the resource_tracker of this shared usage, and in return,
# the resource_tracker increments the resource's reference count by 1.
# Similarly, when access to a resource is closed by a Python process, the
# process notifies the resource_tracker by asking it to decrement the
# resource's reference count by 1.  When the reference count drops to 0, the
# resource_tracker attempts to clean up the underlying resource.

# Finally, every other process connected to the resource tracker has a copy of
# the writable end of the pipe used to communicate with it, so the resource
# tracker gets EOF when all other processes have exited. Then the
# resource_tracker process unlinks any remaining leaked resources (with
# reference count above 0)

# For semaphores, this is important because the system only supports a limited
# number of named semaphores, and they will not be automatically removed till
# the next reboot.  Without this resource tracker process, "killall python"
# would probably leave unlinked semaphores.

# Note that this behavior differs from CPython's resource_tracker, which only
# implements list of shared resources, and not a proper refcounting scheme.
# Also, CPython's resource tracker will only attempt to cleanup those shared
# resources once all procsses connected to the resouce tracker have exited.


import os
import shutil
import sys
import signal
import warnings
import threading
from multiprocessing import util
from _multiprocessing import sem_unlink

from . import spawn

if sys.platform == "win32":
    import _winapi
    import msvcrt
    from .spawn import duplicate_in_child_process


__all__ = ["ensure_running", "register", "unregister"]

_HAVE_SIGMASK = hasattr(signal, "pthread_sigmask")
_IGNORED_SIGNALS = (signal.SIGINT, signal.SIGTERM)

_CLEANUP_FUNCS = {"folder": shutil.rmtree, "file": os.unlink}

if os.name == "posix":
    _CLEANUP_FUNCS["semlock"] = sem_unlink


VERBOSE = False


class ResourceTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._fd = None
        self._pid = None

    def getfd(self):
        self.ensure_running()
        return self._fd

    def ensure_running(self):
        """Make sure that resource tracker process is running.

        This can be run from any process.  Usually a child process will use
        the resource created by its parent."""
        with self._lock:
            if self._fd is not None:
                # resource tracker was launched before, is it still running?
                if self._check_alive():
                    # => still alive
                    return
                # => dead, launch it again
                os.close(self._fd)
                if os.name == "posix":
                    try:
                        # At this point, the resource_tracker process has been
                        # killed or crashed. Let's remove the process entry
                        # from the process table to avoid zombie processes.
                        os.waitpid(self._pid, 0)
                    except OSError:
                        # The process was terminated or is a child from an
                        # ancestor of the current process.
                        pass
                self._fd = None
                self._pid = None

                warnings.warn(
                    "resource_tracker: process died unexpectedly, "
                    "relaunching.  Some folders/sempahores might "
                    "leak."
                )

            if sys.platform == "win32":
                r, whandle = _winapi.CreatePipe(None, 0)
                w = os.fdopen(msvcrt.open_osfhandle(whandle, 0), mode="wb")
                fds_to_pass = [r]
            else:
                r, w = os.pipe()
                fds_to_pass = [r]
                try:
                    fds_to_pass.append(sys.stderr.fileno())
                except Exception:
                    pass

            try:
                # process will out live us, so no need to wait on pid
                cmd = spawn.get_command_line(
                    main_prog=main,
                    pipe_handle=r,
                    parent_pid=os.getpid(),
                    verbose=int(VERBOSE),
                )
                util.debug(f"launching resource tracker: {cmd}")
                # bpo-33613: Register a signal mask that will block the
                # signals.  This signal mask will be inherited by the child
                # that is going to be spawned and will protect the child from a
                # race condition that can make the child die before it
                # registers signal handlers for SIGINT and SIGTERM. The mask is
                # unregistered after spawning the child.
                try:
                    if _HAVE_SIGMASK:
                        signal.pthread_sigmask(
                            signal.SIG_BLOCK, _IGNORED_SIGNALS
                        )
                    pid = spawnv_passfds(cmd, fds_to_pass)
                finally:
                    if _HAVE_SIGMASK:
                        signal.pthread_sigmask(
                            signal.SIG_UNBLOCK, _IGNORED_SIGNALS
                        )
            except BaseException:
                os.close(w)
                raise
            else:
                self._fd = w
                self._pid = pid
            finally:
                if sys.platform == "win32":
                    _winapi.CloseHandle(r)
                else:
                    os.close(r)

    def _check_alive(self):
        """Check for the existence of the resource tracker process."""
        try:
            self._send("PROBE", "", "")
        except BrokenPipeError:
            return False
        else:
            return True

    def register(self, name, rtype):
        """Register a named resource, and increment its refcount."""
        self.ensure_running()
        self._send("REGISTER", name, rtype)

    def unregister(self, name, rtype):
        """Unregister a named resource with resource tracker."""
        self.ensure_running()
        self._send("UNREGISTER", name, rtype)

    def maybe_unlink(self, name, rtype):
        """Decrement the refcount of a resource, and delete it if it hits 0"""
        self.ensure_running()
        self._send("MAYBE_UNLINK", name, rtype)

    def _send(self, cmd, name, rtype):
        if len(name) > 512:
            # posix guarantees that writes to a pipe of less than PIPE_BUF
            # bytes are atomic, and that PIPE_BUF >= 512
            raise ValueError("name too long")
        msg = f"{cmd}:{name}:{rtype}\n".encode("ascii")
        nbytes = os.write(self._fd, msg)
        assert nbytes == len(msg)


_resource_tracker = ResourceTracker()
ensure_running = _resource_tracker.ensure_running
register = _resource_tracker.register
maybe_unlink = _resource_tracker.maybe_unlink
unregister = _resource_tracker.unregister
getfd = _resource_tracker.getfd


def main(pipe_handle, parent_pid, verbose=0):
    """Run resource tracker."""
    # Make sure the arguments have the right type as they are
    # passed as strings through the command line.
    pipe_handle, parent_pid = int(pipe_handle), int(parent_pid)
    verbose = int(verbose)
    if sys.platform == "win32":
        handle, parent_sentinel = duplicate_in_child_process(
            pipe_handle, parent_pid
        )
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
    else:
        fd = pipe_handle

    # protect the process from ^C and "killall python" etc
    if verbose:
        util.log_to_stderr(level=util.DEBUG)

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if _HAVE_SIGMASK:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _IGNORED_SIGNALS)

    for f in (sys.stdin, sys.stdout):
        try:
            f.close()
        except Exception:
            pass

    if verbose:
        util.debug("Main resource tracker is running")

    registry = {rtype: {} for rtype in _CLEANUP_FUNCS.keys()}
    try:
        # keep track of registered/unregistered resources
        with open(fd, "rb", closefd=True) as f:
            while True:
                line = f.readline()
                if line == b"":  # EOF
                    break
                try:
                    splitted = line.strip().decode("ascii").split(":")
                    # name can potentially contain separator symbols (for
                    # instance folders on Windows)
                    cmd, name, rtype = (
                        splitted[0],
                        ":".join(splitted[1:-1]),
                        splitted[-1],
                    )

                    if cmd == "PROBE":
                        continue

                    if rtype not in _CLEANUP_FUNCS:
                        raise ValueError(
                            f"Cannot register {name} for automatic cleanup: "
                            f"unknown resource type ({rtype}). Resource type "
                            "should be one of the following: "
                            f"{list(_CLEANUP_FUNCS.keys())}"
                        )

                    if cmd == "REGISTER":
                        if name not in registry[rtype]:
                            registry[rtype][name] = 1
                        else:
                            registry[rtype][name] += 1

                        if verbose:
                            util.debug(
                                "[ResourceTracker] incremented refcount of "
                                f"{rtype} {name} "
                                f"(current {registry[rtype][name]})"
                            )
                    elif cmd == "UNREGISTER":
                        del registry[rtype][name]
                        if verbose:
                            util.debug(
                                f"[ResourceTracker] unregister {name} {rtype}:"
                                f" registry({len(registry)})"
                            )
                    elif cmd == "MAYBE_UNLINK":
                        registry[rtype][name] -= 1
                        if verbose:
                            util.debug(
                                "[ResourceTracker] decremented refcount of "
                                f"{rtype} {name} "
                                f"(current {registry[rtype][name]})"
                            )

                        if registry[rtype][name] == 0:
                            del registry[rtype][name]
                            try:
                                if verbose:
                                    util.debug(
                                        f"[ResourceTracker] unlink {name}"
                                    )
                                _CLEANUP_FUNCS[rtype](name)
                            except Exception as e:
                                warnings.warn(
                                    f"resource_tracker: {name}: {e!r}"
                                )

                    else:
                        raise RuntimeError(f"unrecognized command {cmd!r}")
                except BaseException:
                    try:
                        sys.excepthook(*sys.exc_info())
                    except BaseException:
                        pass
    finally:
        # all processes have terminated; cleanup any remaining resources
        def _unlink_resources(rtype_registry, rtype):
            if rtype_registry:
                try:
                    warnings.warn(
                        "resource_tracker: There appear to be "
                        f"{len(rtype_registry)} leaked {rtype} objects to "
                        "clean up at shutdown"
                    )
                except Exception:
                    pass
            for name in rtype_registry:
                # For some reason the process which created and registered this
                # resource has failed to unregister it. Presumably it has
                # died.  We therefore clean it up.
                try:
                    _CLEANUP_FUNCS[rtype](name)
                    if verbose:
                        util.debug(f"[ResourceTracker] unlink {name}")
                except Exception as e:
                    warnings.warn(f"resource_tracker: {name}: {e!r}")

        for rtype, rtype_registry in registry.items():
            if rtype == "folder":
                continue
            else:
                _unlink_resources(rtype_registry, rtype)

        # The default cleanup routine for folders deletes everything inside
        # those folders recursively, which can include other resources tracked
        # by the resource tracker). To limit the risk of the resource tracker
        # attempting to delete twice a resource (once as part of a tracked
        # folder, and once as a resource), we delete the folders after all
        # other resource types.
        if "folder" in registry:
            _unlink_resources(registry["folder"], "folder")

    if verbose:
        util.debug("resource tracker shut down")


#
# Start a program with only specified fds kept open
#


def spawnv_passfds(cmd, passfds):
    """Spawn the resource tracker in a platform specific way.

    For posix platforms, make the passfds inheritable and use fork_exec.

    For windows platforms, passfds is only used to clean up the handles in
    case of failure, the inheritance of the handles will be taken care in the
    child process through _winapi.OpenProcess.
    """
    passfds = sorted(passfds)
    if sys.platform != "win32":
        errpipe_read, errpipe_write = os.pipe()
        try:
            from .reduction import _mk_inheritable
            from .fork_exec import fork_exec

            _pass = [_mk_inheritable(fd) for fd in passfds]
            return fork_exec(cmd, _pass)
        finally:
            os.close(errpipe_read)
            os.close(errpipe_write)
    else:
        exe = cmd[0]
        cmd = " ".join(f'"{x}"' for x in cmd)
        try:
            _, ht, pid, _ = _winapi.CreateProcess(
                exe, cmd, None, None, False, 0, None, None, None
            )
            _winapi.CloseHandle(ht)
        except BaseException:
            _winapi.CloseHandle(passfds[0])
            pass
        return pid
