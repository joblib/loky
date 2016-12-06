import sys
import time
import contextlib


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
