###############################################################################
# Launch a subprocess using forkexec and make sure only the needed fd are
# shared in the two process.
#
# author: Thomas Moreau and Olivier Grisel
#
import sys
import os
import subprocess


def fork_exec(cmd, keep_fds, env=None):
    import _posixsubprocess

    # Encoded command args as bytes:
    cmd = [os.fsencode(arg) for arg in cmd]

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

    # VFORK is not supported on older Python versions.
    if hasattr(subprocess, "_USE_VFORK"):
        # Python 3.11 and later
        pgid_to_set = [-1]
        allow_vfork = [subprocess._USE_VFORK]
    else:
        pgid_to_set = []
        allow_vfork = []

    try:
        return _posixsubprocess.fork_exec(
            cmd,
            [cmd[0]],
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
            *pgid_to_set,
            None,
            None,
            None,
            -1,
            None,
            *allow_vfork,
        )
    finally:
        os.close(errpipe_read)
        os.close(errpipe_write)
