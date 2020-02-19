import re
import os
import sys
import time
import pytest
import warnings
import threading
import subprocess
import contextlib
from tempfile import mkstemp, mkdtemp, NamedTemporaryFile
from loky.backend import resource_tracker
from loky.backend.semlock import SemLock, _sem_open

try:
    FileNotFoundError = FileNotFoundError
except NameError:  # FileNotFoundError is Python 3-only
    from loky.backend.semlock import FileNotFoundError


if sys.version_info[0] == 2:
    class TimeoutError(OSError):
        pass


def resource_unlink(name, rtype):
    resource_tracker._CLEANUP_FUNCS[rtype](name)


def create_resource(rtype):
    if rtype == "folder":
        return mkdtemp(dir=os.getcwd())

    elif rtype == "semlock":
        name = "loky-%i-%s" % (os.getpid(), next(SemLock._rand))
        _ = SemLock(1, 1, 1, name)
        return name
    elif rtype == "file":
        tmpfile = NamedTemporaryFile(delete=False)
        tmpfile.close()
        return tmpfile.name
    else:
        raise ValueError("Resource type %s not understood" % rtype)


def resource_exists(name, rtype):
    if rtype in ["folder", "file"]:
        return os.path.exists(name)
    elif rtype == "semlock":
        # On OSX, semaphore are not visible in the file system, we must
        # try to open the semaphore to check if it exists.
        from loky.backend.semlock import pthread
        try:
            h = _sem_open(name.encode('ascii'))
            pthread.sem_close(h)
            return True
        except FileNotFoundError:
            return False
    else:
        raise ValueError("Resource type %s not understood" % rtype)


@contextlib.contextmanager
def captured_output(stream_name):
    """Return a context manager used by captured_stdout/stdin/stderr
    that temporarily replaces the sys stream *stream_name* with a StringIO."""
    import io
    orig_stdout = getattr(sys, stream_name)
    s = io.StringIO()
    if sys.version_info[:2] < (3, 3):
        import types
        s.wrt = s.write

        def write(self, msg):
            self.wrt(unicode(msg))
        s.write = types.MethodType(write, s)

    setattr(sys, stream_name, s)
    try:
        yield getattr(sys, stream_name)
    finally:
        setattr(sys, stream_name, orig_stdout)


def captured_stderr():
    """Capture the output of sys.stderr:

       with captured_stderr() as stderr:
           print("hello", file=sys.stderr)
       self.assertEqual(stderr.getvalue(), "hello\\n")
    """
    return captured_output("stderr")


#
# Wrapper
#

class TimingWrapper(object):

    def __init__(self, func):
        self.func = func
        self.elapsed = None

    def __call__(self, *args, **kwds):
        t = time.time()
        try:
            return self.func(*args, **kwds)
        finally:
            self.elapsed = time.time() - t

    def assert_timing_almost_equal(self, delay):
        assert round(self.elapsed - delay, 1) == 0

    def assert_timing_almost_zero(self):
        self.assert_timing_almost_equal(0.0)


#
# helper functions
#

def id_sleep(x, delay=0):
    """sleep for delay seconds and return its first argument"""
    time.sleep(delay)
    return x


def check_subprocess_call(cmd, timeout=1, stdout_regex=None,
                          stderr_regex=None, env=None):
    """Runs a command in a subprocess with timeout in seconds.

    Also checks returncode is zero, stdout if stdout_regex is set, and
    stderr if stderr_regex is set.
    """
    if env is not None:
        env_ = os.environ.copy()
        env_.update(env)
        env = env_
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)

    def kill_process():
        warnings.warn("Timeout running {}".format(cmd))
        proc.kill()

    timer = threading.Timer(timeout, kill_process)
    try:
        timer.start()
        stdout, stderr = proc.communicate()

        if sys.version_info[0] >= 3:
            stdout, stderr = stdout.decode(), stderr.decode()
        if proc.returncode == -9:
            message = (
                'Subprocess timeout after {}s.\nStdout:\n{}\n'
                'Stderr:\n{}').format(timeout, stdout, stderr)
            raise TimeoutError(message)
        elif proc.returncode != 0:
            message = (
                'Non-zero return code: {}.\nStdout:\n{}\n'
                'Stderr:\n{}').format(
                    proc.returncode, stdout, stderr)
            raise ValueError(message)

        if (stdout_regex is not None and
                not re.search(stdout_regex, stdout)):
            raise ValueError(
                "Unexpected stdout: {!r} does not match:\n{!r}".format(
                    stdout_regex, stdout))
        if (stderr_regex is not None and
                not re.search(stderr_regex, stderr)):
            raise ValueError(
                "Unexpected stderr: {!r} does not match:\n{!r}".format(
                    stderr_regex, stderr))

        return stdout, stderr

    finally:
        timer.cancel()


def skip_func(msg):
    def test_func(*args, **kwargs):
        pytest.skip(msg)
    return test_func


# A decorator to run tests only when numpy is available
try:
    import numpy  # noqa F401

    def with_numpy(func):
        """A decorator to skip tests requiring numpy."""
        return func

except ImportError:
    def with_numpy(func):
        """A decorator to skip tests requiring numpy."""
        return skip_func('Test require numpy')


# A decorator to run tests only when numpy is available
try:
    from _openmp_test_helper.parallel_sum import parallel_sum  # noqa F401

    def with_parallel_sum(func):
        """A decorator to skip tests if parallel_sum is not compiled."""
        return func

    def _run_openmp_parallel_sum(*args):
        return parallel_sum(*args)

except ImportError:
    def with_parallel_sum(func):
        """A decorator to skip tests if parallel_sum is not compiled."""
        return skip_func('Test requires parallel_sum to be compiled')

    _run_openmp_parallel_sum = None


def check_python_subprocess_call(code, stdout_regex=None):
    cmd = [sys.executable]
    try:
        fid, filename = mkstemp(suffix="_joblib.py")
        os.close(fid)
        with open(filename, mode='wb') as f:
            f.write(code.encode('ascii'))
        cmd += [filename]
        check_subprocess_call(cmd, stdout_regex=stdout_regex, timeout=10)
    finally:
        os.unlink(filename)


def filter_match(match, start_method=None):
    if sys.platform == "win32":
        return

    if start_method == "forkserver" and sys.version_info < (3, 7):
        return "UNKNOWN"

    return match
