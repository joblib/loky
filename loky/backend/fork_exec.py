import os
import sys

if sys.platform == "darwin" and sys.version_info < (3, 3):
     FileNotFoundError = OSError


def close_fds(keep_fds):  # pragma: no cover
    keep_fds = set(keep_fds).union([0, 1, 2])
    try:
        open_fds = set([int(fd) for fd in os.listdir('/proc/self/fd')])
    except FileNotFoundError:
        import resource
        max_nfds = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        open_fds = set([fd for fd in range(3, max_nfds)])
    for i in open_fds-keep_fds:
        try:
            os.close(i)
        except OSError:
            pass


def fork_exec(cmd, keep_fds):

    pid = os.fork()
    if pid == 0:  # pragma: no cover
        close_fds(keep_fds)
        os.execv(sys.executable, cmd)
    else:
        return pid
