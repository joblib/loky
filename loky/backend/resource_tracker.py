###############################################################################
# Server process to keep track of unlinked resources, like folders and
# semaphores and clean them.
#
# author: Thomas Moreau
#
# Adapted from multiprocessing/resource_tracker.py
#  * add some VERBOSE logging,
#  * add support to track folders,
#  * add Windows support,
#  * refcounting scheme to avoid unlinking resources still in use.
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
# resources once all processes connected to the resource tracker have exited.


import os
import shutil
import sys
import signal
import warnings
from multiprocessing import util
import base64
import json

PY_GREATER_THAN_311 = sys.version_info[:2] >= (3, 11)

if PY_GREATER_THAN_311:
    from .stdlib_py314_resource_tracker import (
        ResourceTracker as _ResourceTracker,
    )
else:
    # CPython >= 3.11 re-entrancy support code relies on
    # threading.RLock._recursion_count which does not exist for Python <= 3.10
    from .stdlib_py310_resource_tracker import (
        ResourceTracker as _ResourceTracker,
    )

from . import spawn

if sys.platform == "win32":
    import _winapi
    import msvcrt
    from multiprocessing.reduction import duplicate


__all__ = ["ensure_running", "register", "unregister"]

_HAVE_SIGMASK = hasattr(signal, "pthread_sigmask")
_IGNORED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def cleanup_noop(name):
    raise RuntimeError("noop should never be registered or cleaned up")


_CLEANUP_FUNCS = {
    "noop": cleanup_noop,
    "folder": shutil.rmtree,
    "file": os.unlink,
}

if os.name == "posix":
    import _multiprocessing

    # Use sem_unlink() to clean up named semaphores.
    #
    # sem_unlink() may be missing if the Python build process detected the
    # absence of POSIX named semaphores. In that case, no named semaphores were
    # ever opened, so no cleanup would be necessary.
    if hasattr(_multiprocessing, "sem_unlink"):
        _CLEANUP_FUNCS.update(
            {
                "semlock": _multiprocessing.sem_unlink,
            }
        )


VERBOSE = False


class ResourceTracker(_ResourceTracker):
    """Resource tracker with refcounting scheme.

    This class is an extension of the multiprocessing ResourceTracker class
    which implements a reference counting scheme to avoid unlinking shared
    resources still in use in other processes.

    This feature is notably used by `joblib.Parallel` to share temporary
    folders and memory mapped files between the main process and the worker
    processes.

    The actual implementation of the refcounting scheme is in the main
    function, which is run in a dedicated process.
    """

    def maybe_unlink(self, name, rtype):
        """Decrement the refcount of a resource, and delete it if it hits 0"""
        self._send("MAYBE_UNLINK", name, rtype)

    def _teardown_dead_process(self):
        if os.name == "posix":
            if PY_GREATER_THAN_311:
                super()._teardown_dead_process()
            else:
                # Python 3.10 doesn't have _teardown_dead_process,
                # this is copied from Python 3.14.7
                os.close(self._fd)

                # Clean-up to avoid dangling processes.
                try:
                    # _pid can be None if this process is a child from another
                    # python process, which has started the resource_tracker.
                    if self._pid is not None:
                        os.waitpid(self._pid, 0)
                except ChildProcessError:
                    # The resource_tracker has already been terminated.
                    pass
                self._fd = None
                self._pid = None
                self._exitcode = None

                warnings.warn(
                    "resource_tracker: process died unexpectedly, "
                    "relaunching.  Some resources might leak."
                )
        else:
            # TODO what is the right thing to do Windows? Probably larsoner PR has some answers.
            os.close(self._fd)
            # All 3 lines copied from stdlib _teardown_dead_processes
            self._fd = None
            self._pid = None
            self._exitcode = None

            warnings.warn(
                "resource_tracker: process died unexpectedly, "
                "relaunching.  Some resources might leak."
            )

    # To minimize the diff with stdlib ResourceTracker._launch
    # fmt: off
    def _launch(self):
        # This is copied from Python 3.14.7 with loky additions/modifications
        # mostly for Windows support and logging.
        # Added or changed lines have a comment that starts with "# loky:"
        fds_to_pass = []
        try:
            fds_to_pass.append(sys.stderr.fileno())
        except Exception:
            pass
        r, w = os.pipe()
        # loky: Windows support
        if sys.platform == "win32":
            _r = duplicate(msvcrt.get_osfhandle(r), inheritable=True)
            os.close(r)
            r = _r

        try:
            fds_to_pass.append(r)
            # process will out live us, so no need to wait on pid
            exe = spawn.get_executable()
            args = [
                exe,
                *util._args_from_interpreter_flags(),
                '-c',
                # loky: use loky main function rather than stdlib one
                f'from {main.__module__} import main; main({r}, {VERBOSE})'
            ]
            # loky: logging
            util.debug(f"launching resource tracker: {args}")
            # bpo-33613: Register a signal mask that will block the signals.
            # This signal mask will be inherited by the child that is going
            # to be spawned and will protect the child from a race condition
            # that can make the child die before it registers signal handlers
            # for SIGINT and SIGTERM. The mask is unregistered after spawning
            # the child.
            prev_sigmask = None
            try:
                if _HAVE_SIGMASK:
                    prev_sigmask = signal.pthread_sigmask(signal.SIG_BLOCK, _IGNORED_SIGNALS)
                # loky: call loky spawnv_passfds which supports Windows
                pid = spawnv_passfds(exe, args, fds_to_pass)
            finally:
                if prev_sigmask is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, prev_sigmask)
        except:
            os.close(w)
            raise
        else:
            self._fd = w
            self._pid = pid
        finally:
            # loky: Windows support
            if sys.platform == "win32":
                _winapi.CloseHandle(r)
            else:
                os.close(r)

    if not PY_GREATER_THAN_311:
        # for Python 3.10 need to override ensure_running since _launch does
        # not exist and overriding _launch doesn't do anything
        def ensure_running(self):
            self._ensure_running_and_write()

        # Copied from Python 3.14.7 except that re-entrant code has been removed
        def _ensure_running_and_write(self, msg=None):
            with self._lock:
                # resource tracker was launched before, is it still running?
                if self._fd is not None:
                    if msg is None:
                        to_send = self._make_probe_message()
                    else:
                        to_send = msg
                    try:
                        self._write(to_send)
                    except OSError:
                        self._teardown_dead_process()
                        self._launch()

                    msg = None  # message was sent in probe
                else:
                    self._launch()

            if msg is not None:
                self._write(msg)

        # Helper function for _ensure_running_and_write copied from Python 3.14.7
        def _write(self, msg):
            nbytes = os.write(self._fd, msg)
            assert nbytes == len(msg), f"{nbytes=} != {len(msg)=}"

        # Helper function for _ensure_running_and_write copied from Python
        # 3.14.7 and simplified since Python 3.10 use the simplest message
        # format
        def _make_probe_message(self):
            return b'PROBE:0:noop\n'
    # fmt: on

    def __del__(self):
        # Python 3.10 ResourceTracker does not have a __del__
        if not PY_GREATER_THAN_311:
            return

        if os.name == "posix":
            super().__del__()
        else:
            # TODO What is the right thing to do on Windows? Probably larsoner PR has some answers
            try:
                # use timeout=None which avoids WNOHANG which doesn't exist on Windows
                self._stop(use_blocking_lock=False)
            except ChildProcessError:
                # ignore error due to trying to clean up child process which has already been
                # shutdown on windows. See https://github.com/joblib/loky/pull/450
                pass


_resource_tracker = ResourceTracker()
ensure_running = _resource_tracker.ensure_running
register = _resource_tracker.register
maybe_unlink = _resource_tracker.maybe_unlink
unregister = _resource_tracker.unregister
getfd = _resource_tracker.getfd

# Copied from Python 3.14.7
# fmt: off
def _decode_message(line):
    if line.startswith(b'{'):
        try:
            obj = json.loads(line.decode('ascii'))
        except Exception as e:
            raise ValueError("malformed resource_tracker message: %r" % (line,)) from e

        cmd = obj["cmd"]
        rtype = obj["rtype"]
        b64  = obj.get("base64_name", "")

        if not isinstance(cmd, str) or not isinstance(rtype, str) or not isinstance(b64, str):
            raise ValueError("malformed resource_tracker fields: %r" % (obj,))

        try:
            name = base64.urlsafe_b64decode(b64).decode('utf-8', 'surrogateescape')
        except ValueError as e:
            raise ValueError("malformed resource_tracker base64_name: %r" % (b64,)) from e
    else:
        cmd, rest = line.strip().decode('ascii').split(':', maxsplit=1)
        name, rtype = rest.rsplit(':', maxsplit=1)
    return cmd, rtype, name
# fmt: on

# fmt: off
# The main function has been copied from Python 3.14.7 and modified, mostly for
# Windows support, logging and refcount functionality.
# Added or changed lines have a comment that starts with "# loky:"
# loky: add verbose argument for logging
def main(fd, verbose=0):
    '''Run resource tracker.'''

    # loky: logging
    if verbose:
        util.log_to_stderr(level=util.DEBUG)

    # protect the process from ^C and "killall python" etc
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if _HAVE_SIGMASK:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _IGNORED_SIGNALS)

    for f in (sys.stdin, sys.stdout):
        try:
            f.close()
        except Exception:
            pass

    # loky: logging
    if verbose:
        util.debug("Main resource tracker is running")

    # loky: change for refcount functionality we want a dict[str, dict] rather
    # than a dict[str, set] so that cache[folder]['resource'] is the refcount
    # associated with it
    cache = {rtype: dict() for rtype in _CLEANUP_FUNCS.keys()}
    exit_code = 0

    try:
        # loky: Windows support
        if sys.platform == "win32":
            fd = msvcrt.open_osfhandle(fd, os.O_RDONLY)
        # keep track of registered/unregistered resources
        with open(fd, 'rb') as f:
            for line in f:
                try:
                    cmd, rtype, name = _decode_message(line)
                    cleanup_func = _CLEANUP_FUNCS.get(rtype, None)
                    if cleanup_func is None:
                        raise ValueError(
                            f'Cannot register {name} for automatic cleanup: '
                            f'unknown resource type ({rtype})'
                            # loky: additional info for possible keys
                            '. Resource type should be one of the following: '
                            f'{list(_CLEANUP_FUNCS.keys())}'
                        )

                    if cmd == 'REGISTER':
                        # loky: refcount functionality
                        if name not in cache[rtype]:
                            cache[rtype][name] = 1
                        else:
                            cache[rtype][name] += 1

                        # loky: logging
                        if verbose:
                            util.debug(
                                '[ResourceTracker] incremented refcount of '
                                f'{rtype} {name} '
                                f'(current {cache[rtype][name]})'
                            )
                    elif cmd == 'UNREGISTER':
                        # loky: refcount functionality
                        del cache[rtype][name]
                        # loky: logging
                        if verbose:
                            util.debug(
                                f'[ResourceTracker] unregister {name} {rtype}: '
                                f'cache({len(cache)})'
                            )
                    elif cmd == 'PROBE':
                        pass
                    # loky: refcount functionality with logging
                    elif cmd == 'MAYBE_UNLINK':
                        cache[rtype][name] -= 1
                        if verbose:
                            util.debug(
                                '[ResourceTracker] decremented refcount of '
                                f'{rtype} {name} '
                                f'(current {cache[rtype][name]})'
                            )

                        if cache[rtype][name] == 0:
                            del cache[rtype][name]
                            try:
                                if verbose:
                                    util.debug(
                                        f'[ResourceTracker] unlink {name}'
                                    )
                                _CLEANUP_FUNCS[rtype](name)
                            except Exception as e:
                                warnings.warn(
                                    f"resource_tracker: {name}: {e!r}"
                                )

                    else:
                        raise RuntimeError('unrecognized command %r' % cmd)
                except Exception:
                    # TODO I followed the stdlib here and changed the exception
                    # class to be Exception instead of BaseException. Maybe
                    # loky had a reason to always print the back-trace even in
                    # BaseException case???
                    exit_code = 3
                    try:
                        sys.excepthook(*sys.exc_info())
                    except:
                        pass
    finally:
        # all processes have terminated; cleanup any remaining resources

        # loky: loky wants to clean ressources first and folder last because
        # there can be tracked resources inside tracked folders.
        # _unlink_resources is the stdlib code with some additional logging, it
        # is called for all resources except folders and then at the end for
        # all folders
        def _unlink_resources(rtype_cache, rtype):
            if rtype_cache:
                try:
                    exit_code = 1
                    if rtype == 'dummy':
                        # The test 'dummy' resource is expected to leak.
                        # We skip the warning (and *only* the warning) for it.
                        pass
                    else:
                        warnings.warn(
                            f'resource_tracker: There appear to be '
                            f'{len(rtype_cache)} leaked {rtype} objects to '
                            f'clean up at shutdown: {rtype_cache}'
                        )
                except Exception:
                    pass
            for name in rtype_cache:
                # For some reason the process which created and registered this
                # resource has failed to unregister it. Presumably it has
                # died.  We therefore unlink it.
                try:
                    try:
                        _CLEANUP_FUNCS[rtype](name)
                        # loky: logging
                        if verbose:
                            util.debug(f'[ResourceTracker] unlink {name}')
                    except Exception as e:
                        exit_code = 2
                        # loky: %r instead of %s for exception logging and
                        # (TODO is this really necessary, I guess you get
                        # the exact exception type with %r)
                        # also %s instead of %r for name, one reason may be
                        # for Windows %r doubles the backslashes for path
                        # ressources
                        warnings.warn('resource_tracker: %s: %r' % (name, e))
                finally:
                    pass

        # The default cleanup routine for folders deletes everything inside
        # those folders recursively, which can include other resources tracked
        # by the resource tracker). To limit the risk of the resource tracker
        # attempting to delete twice a resource (once as part of a tracked
        # folder, and once as a resource), we delete the folders after all
        # other resource types.
        for rtype, rtype_cache in cache.items():
            if rtype == 'folder':
                continue
            _unlink_resources(rtype_cache, rtype)

        if 'folder' in cache:
            _unlink_resources(cache['folder'], 'folder')

    # loky: logging
    if verbose:
        util.debug("resource tracker shut down")

    # TODO add exit_code to _unlink_resource + exit_code management with 2-stage clean-up
    # This can be done in a further PR, since we didn't have any kind of exit_code before
# fmt: on


def spawnv_passfds(path, args, passfds):
    if sys.platform != "win32":
        # loky: TODO not sure why encoding is needed since stdlib does not do
        # it, maybe Windows ... git blame points at
        # https://github.com/joblib/loky/pull/429 but couldn't find any clear
        # reason
        args = [arg.encode("utf-8") for arg in args]
        path = path.encode("utf-8")
        return util.spawnv_passfds(path, args, passfds)
    else:
        # loky: Windows support
        passfds = sorted(passfds)
        cmd = " ".join(f'"{x}"' for x in args)
        try:
            _, ht, pid, _ = _winapi.CreateProcess(
                path, cmd, None, None, True, 0, None, None, None
            )
            _winapi.CloseHandle(ht)
        except BaseException:
            pass
        return pid
