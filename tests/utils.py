import re
import sys
import time
import warnings
import threading
import subprocess
import contextlib

if sys.version_info[0] == 2:
    class TimeoutError(OSError):
        pass


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
                          stderr_regex=None):
    """Runs a command in a subprocess with timeout in seconds.

    Also checks returncode is zero, stdout if stdout_regex is set, and
    stderr if stderr_regex is set.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

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
