###############################################################################
# Launch a subprocess using forkexec and make sure only the needed fd are
# shared in the two process.
#
# author: Thomas Moreau and Olivier Grisel
#
import sys
import os
import subprocess


def close_fds(keep_fds):  # pragma: no cover
    """Close all the file descriptors except those in keep_fds."""

    # Make sure to keep stdout and stderr open for logging purpose. Do not
    # close stdin either, otherwise it can break calls to subprocess.run in the
    # child process (at least on macOS).
    keep_fds = {*keep_fds, 0, 1, 2}

    # We try to retrieve all the open fds
    try:
        open_fds = {int(fd) for fd in os.listdir("/proc/self/fd")}
    except FileNotFoundError:
        import resource

        max_nfds = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        open_fds = {*range(max_nfds)}

    for i in open_fds - keep_fds:
        try:
            os.close(i)
        except OSError:
            pass


def fork_exec(cmd, keep_fds, env=None):
    import _posixsubprocess

    # Encoded command args as bytes:
    cmd = [arg.encode("utf-8") for arg in cmd]

    # Copy the environment variables to set in the child process (also encoded
    # as bytes).
    env = env or {}
    env = {**os.environ, **env}
    encoded_env = []
    for key, value in env.items():
        encoded_env.append(f"{key}={value}".encode("utf-8"))

    # Fds with fileno larger than 3 (stdin=0, stdout=1, stderr=2) are be closed
    # in the child process, except for those passed in keep_fds.
    keep_fds = tuple(sorted(map(int, keep_fds)))
    errpipe_read, errpipe_write = os.pipe()

    # The default way to close fds implemented in _posixsubprocess.fork_exec does
    # not seem to catch them all. We use preexec_fn to close all fds except the
    # ones we want to keep.
    def preexec_fn():
        close_fds(keep_fds)

    # VFORK is not supported on older Python versions.
    if hasattr(subprocess, "_USE_VFORK"):
        allow_vfork = [subprocess._USE_VFORK]
    else:
        allow_vfork = []

    try:
        return _posixsubprocess.fork_exec(
            cmd,
            [sys.executable.encode("utf-8")],
            True,
            keep_fds,
            None,
            encoded_env,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            errpipe_read,
            errpipe_write,
            False,
            False,
            -1,
            None,
            None,
            None,
            -1,
            preexec_fn,
            *allow_vfork,
        )
    finally:
        os.close(errpipe_read)
        os.close(errpipe_write)
