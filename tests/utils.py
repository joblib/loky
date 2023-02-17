import re
import os
import sys
import time
import pytest
import warnings
import threading
import subprocess
import contextlib
from tempfile import mkstemp, mkdtemp, NamedTemporaryFile, _RandomNameSequence
from _multiprocessing import SemLock as _SemLock
from _multiprocessing import sem_unlink

from loky.backend import resource_tracker


_rand_name = _RandomNameSequence()


def resource_unlink(name, rtype):
    resource_tracker._CLEANUP_FUNCS[rtype](name)


def create_resource(rtype):
    if rtype == "folder":
        return mkdtemp(dir=os.getcwd())
    elif rtype == "semlock":
        name = f"test-loky-{os.getpid()}-{next(_rand_name)}"
        _SemLock(1, 1, 1, name, False)
        return name
    elif rtype == "file":
        tmpfile = NamedTemporaryFile(delete=False)
        tmpfile.close()
        return tmpfile.name
    else:
        raise ValueError(f"Resource type {rtype} not understood")


def resource_exists(name, rtype):
    if rtype in ["folder", "file"]:
        return os.path.exists(name)
    elif rtype == "semlock":
        try:
            _SemLock(1, 1, 1, name, False)
            sem_unlink(name)
            return False
        except OSError:
            return True
    else:
        raise ValueError(f"Resource type {rtype} not understood")


@contextlib.contextmanager
def captured_output(stream_name):
    """Return a context manager used by captured_stdout/stdin/stderr
    that temporarily replaces the sys stream *stream_name* with a StringIO."""
    import io

    orig_stdout = getattr(sys, stream_name)
    s = io.StringIO()

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


class TimingWrapper:
    def __init__(self, func):
        self.func = func
        self.elapsed = None

    def __call__(self, *args, **kwds):
        t = time.time()
        try:
            return self.func(*args, **kwds)
        finally:
            self.elapsed = time.time() - t

    def assert_timing_lower_than(self, delay):
        msg = (
            f"expected duration lower than {delay:.3f}s, "
            f"got {self.elapsed:.3f}s"
        )
        assert self.elapsed < delay, msg

    def assert_timing_almost_zero(self):
        self.assert_timing_lower_than(0.1)


#
# helper functions
#


def id_sleep(x, delay=0):
    """sleep for delay seconds and return its first argument"""
    time.sleep(delay)
    return x


def check_subprocess_call(
    cmd, timeout=1, stdout_regex=None, stderr_regex=None, env=None
):
    """Runs a command in a subprocess with timeout in seconds.

    Also checks returncode is zero, stdout if stdout_regex is set, and
    stderr if stderr_regex is set.
    """
    if env is not None:
        env = {**os.environ, **env}

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True
    )

    def kill_process():
        warnings.warn(f"Timeout running {cmd}")
        proc.kill()

    timer = threading.Timer(timeout, kill_process)
    try:
        timer.start()
        stdout, stderr = proc.communicate()

        if proc.returncode == -9:
            message = (
                f"Subprocess timeout after {timeout}s.\nStdout:\n{stdout}\n"
                f"Stderr:\n{stderr}"
            )
            raise TimeoutError(message)
        elif proc.returncode != 0:
            message = (
                f"Non-zero return code: {proc.returncode}.\nStdout:\n{stdout}"
                f"\nStderr:\n{stderr}"
            )
            raise ValueError(message)

        if stdout_regex is not None and not re.search(stdout_regex, stdout):
            raise ValueError(
                f"Unexpected stdout: {stdout_regex!r} does not match:"
                f"\n{stdout!r}"
            )
        if stderr_regex is not None and not re.search(stderr_regex, stderr):
            raise ValueError(
                f"Unexpected stderr: {stderr_regex!r} does not "
                f"match:\n{stderr!r}"
            )

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
        return skip_func("Test require numpy")


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
        return skip_func("Test requires parallel_sum to be compiled")

    _run_openmp_parallel_sum = None


def check_python_subprocess_call(code, stdout_regex=None, timeout=10):
    cmd = [sys.executable]
    try:
        fid, filename = mkstemp(suffix="_joblib.py")
        os.close(fid)
        with open(filename, mode="w") as f:
            f.write(code)
        cmd += [filename]
        check_subprocess_call(cmd, stdout_regex=stdout_regex, timeout=timeout)
    finally:
        os.unlink(filename)


def filter_match(match):
    if sys.platform == "win32":
        return None
    return match
