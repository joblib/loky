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

    # make sure fds are inheritable
    [os.set_inheritable(fd, True) for fd in keep_fds]

    # Encoded command args as bytes:
    cmd = [os.fsencode(arg) for arg in cmd]

    # Copy the environment variables to set in the child process (also encoded
    # as bytes).
    env = env or {}
    env = {**os.environ, **env}
    encoded_env = []
    for key, value in env.items():
        encoded_env.append(os.fsencode(f"{key}={value}"))

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
            cmd,  # args
            cmd[0:1],  # executable_list
            True,  # close_fds
            keep_fds,  # pass_fds
            None,  # cwd
            encoded_env,  # env
            -1,  # p2cread
            -1,  # p2cwrite
            -1,  # c2pread
            -1,  # c2pwrite
            -1,  # errread
            -1,  # errwrite
            errpipe_read,  # errpipe_read
            errpipe_write,  # errpipe_write
            False,  # restore_signal
            False,  # call_setsid
            *pgid_to_set,  # pgid_to_set
            None,  # gid
            None,  # extra_groups
            None,  # uid
            -1,  # child_umask
            None,  # preexec_fn
            *allow_vfork,  # extra flag if vfork is available
        )
    finally:
        os.close(errpipe_read)
        os.close(errpipe_write)
