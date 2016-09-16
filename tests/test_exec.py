import os
import sys
import time
from loky import backend
import multiprocessing
import pytest
import signal

DELTA = 0.1


class TestExec:
    Process = backend.Process
    current_process = staticmethod(multiprocessing.current_process)
    active_children = staticmethod(multiprocessing.active_children)
    # Pool = staticmethod(multiprocessing.Pool)
    Pipe = staticmethod(backend.Pipe)
    Queue = staticmethod(backend.Queue)
    # JoinableQueue = staticmethod(multiprocessing.JoinableQueue)
    # Lock = staticmethod(multiprocessing.Lock)
    # RLock = staticmethod(multiprocessing.RLock)
    # Semaphore = staticmethod(multiprocessing.Semaphore)
    # BoundedSemaphore = staticmethod(multiprocessing.BoundedSemaphore)
    # Condition = staticmethod(multiprocessing.Condition)
    Event = staticmethod(backend.Event)
    # Barrier = staticmethod(multiprocessing.Barrier)
    # Value = staticmethod(multiprocessing.Value)
    # Array = staticmethod(multiprocessing.Array)
    # RawValue = staticmethod(multiprocessing.RawValue)
    # RawArray = staticmethod(multiprocessing.RawArray)

    def test_current(self):

        current = self.current_process()
        authkey = current.authkey

        assert current.is_alive()
        assert not current.daemon
        assert isinstance(authkey, bytes)
        assert len(authkey) > 0
        assert current.ident == os.getpid()
        assert current.exitcode is None

    @pytest.mark.skipif(sys.version_info < (3, 3),
                        reason="requires python3.3")
    def test_daemon_argument(self):

        # By default uses the current process's daemon flag.
        proc0 = self.Process(target=self._test_process)
        assert proc0.daemon == self.current_process().daemon
        proc1 = self.Process(target=self._test_process, daemon=True)
        assert proc1.daemon
        proc2 = self.Process(target=self._test_process, daemon=False)
        assert not proc2.daemon

    @classmethod
    def _test_process(cls, q, *args, **kwds):
        multiprocessing.util.debug("Start test properly")
        current = cls.current_process()
        q.put(args)
        q.put(kwds, timeout=1)
        q.put(current.name, timeout=1)
        q.put(bytes(current.authkey))
        q.put(current.pid)
        multiprocessing.util.debug("Finishe test properly")

    # @pytest.mark.skip(reason="Known failure")
    def test_process(self):
        q = self.Queue()
        args = (q, 1, 2)
        kwargs = {'hello': 23, 'bye': 2.54}
        name = 'SomeProcess'
        p = self.Process(
            target=self._test_process, args=args, kwargs=kwargs, name=name
            )
        p.daemon = True
        current = self.current_process()

        assert p.authkey == current.authkey
        assert not p.is_alive()
        assert p.daemon
        assert p not in self.active_children()
        assert type(self.active_children()) is list
        assert p.exitcode is None

        p.start()

        assert p.exitcode is None
        assert p.is_alive()
        assert p in self.active_children()

        assert q.get() == args[1:]
        assert q.get() == kwargs
        assert q.get() == p.name
        assert q.get() == current.authkey
        assert q.get() == p.pid

        p.join()

        assert p.exitcode == 0
        assert not p.is_alive()
        assert p not in self.active_children()

    @classmethod
    def _test_terminate(cls):
        time.sleep(100)

    def test_terminate(self):

        p = self.Process(target=self._test_terminate)
        p.daemon = True
        p.start()

        assert p.is_alive()
        assert p in self.active_children()
        assert p.exitcode is None

        join = TimingWrapper(p.join)

        assert join(0) is None
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        assert p.is_alive()

        assert join(-1) is None
        self.assertTimingAlmostEqual(join.elapsed, 0.0)
        assert p.is_alive()

        # XXX maybe terminating too soon causes the problems on Gentoo...
        time.sleep(1)

        p.terminate()

        if hasattr(signal, 'alarm'):
            # On the Gentoo buildbot waitpid() often seems to block forever.
            # We use alarm() to interrupt it if it blocks for too long.
            def handler(*args):
                raise RuntimeError('join took too long: %s' % p)
            old_handler = signal.signal(signal.SIGALRM, handler)
            try:
                signal.alarm(10)
                assert join() is None
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            assert join() is None

        self.assertTimingAlmostEqual(join.elapsed, 0.0)

        assert not p.is_alive()
        assert p not in self.active_children()

        p.join()

        # XXX sometimes get p.exitcode == 0 on Windows ...
        #assert p.exitcode == -signal.SIGTERM

    def test_cpu_count(self):
        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 1
        assert type(cpus) is int
        assert cpus >= 1

    def test_active_children(self):
        assert type(self.active_children()) == list

        p = self.Process(target=time.sleep, args=(DELTA,))
        assert p not in self.active_children()

        p.daemon = True
        p.start()
        assert p in self.active_children()

        p.join()
        assert p not in self.active_children()

    @classmethod
    def _test_recursion(cls, wconn, l):
        wconn.send(l)
        if len(l) < 2:
            for i in range(2):
                p = cls.Process(
                    target=cls._test_recursion, args=(wconn, l+[i])
                    )
                p.start()
                p.join()

    def test_recursion(self):
        rconn, wconn = self.Pipe(duplex=False)
        self._test_recursion(wconn, [])

        time.sleep(DELTA)
        result = []
        while rconn.poll():
            a = rconn.recv()
            result.append(a)

        expected = [
            [],
            [0],
            [0, 0],
            [0, 1],
            [1],
            [1, 0],
            [1, 1]
            ]
        assert result == expected

    @classmethod
    def _test_sentinel(cls, event):
        event.wait(10.0)

    def test_sentinel(self):
        event = self.Event()
        p = self.Process(target=self._test_sentinel, args=(event,))
        with pytest.raises(ValueError):
            p.sentinel
        p.start()
        sentinel = p.sentinel
        assert isinstance(sentinel, int)
        assert not wait_for_handle(sentinel, timeout=0.0)
        event.set()
        p.join()
        assert wait_for_handle(sentinel, timeout=1)

    @staticmethod
    def assertTimingAlmostEqual(t, g):
        assert round(t-g, 1) == 0

#
#
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


def wait_for_handle(handle, timeout):
    from loky.backend.connection import wait
    if timeout is not None and timeout < 0.0:
        timeout = None
    return wait([handle], timeout)
