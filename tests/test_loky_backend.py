import os
import sys
import time
import pytest
import signal
import multiprocessing

from loky import backend
from .utils import TimingWrapper

try:
    from ._openmp.parallel_sum import parallel_sum
except ImportError:
    parallel_sum = None

DELTA = 0.1


class TestLokyBackend:
    # loky processes
    Process = backend.Process
    current_process = staticmethod(multiprocessing.current_process)
    active_children = staticmethod(multiprocessing.active_children)

    # interprocess communication objects
    Pipe = staticmethod(backend.Pipe)
    Queue = staticmethod(backend.Queue)
    Manager = staticmethod(backend.Manager)

    # synchronization primitives
    Lock = staticmethod(backend.Lock)
    RLock = staticmethod(backend.RLock)
    Semaphore = staticmethod(backend.Semaphore)
    BoundedSemaphore = staticmethod(backend.BoundedSemaphore)
    Condition = staticmethod(backend.Condition)
    Event = staticmethod(backend.Event)

    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to
        setup_class.
        """
        # TODO: remove all child processes
        for child_process in cls.active_children():
            child_process.terminate()
            child_process.join()

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
    def _test_terminate(cls, ev):
        # Notify the main process that child process started
        ev.set()
        time.sleep(100)

    def test_terminate(self):

        mgr = self.Manager()
        ev = mgr.Event()

        p = self.Process(target=self._test_terminate, args=(ev, ))
        p.daemon = True
        p.start()

        assert p.is_alive()
        assert p in self.active_children()
        assert p.exitcode is None

        join = TimingWrapper(p.join)

        assert join(0) is None
        join.assert_timing_almost_zero()
        assert p.is_alive()

        assert join(-1) is None
        join.assert_timing_almost_zero()
        assert p.is_alive()

        # wait for child process to be fully setup
        ev.wait(1)

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

        join.assert_timing_almost_zero()

        assert not p.is_alive()
        assert p not in self.active_children()

        p.join()

        # XXX sometimes get p.exitcode == 0 on Windows ...
        # assert p.exitcode == -signal.SIGTERM

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
        assert p.exitcode == 0

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
                assert p.exitcode == 0

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

    @pytest.mark.skipif(
        sys.platform == 'win32' and sys.version_info[:2] < (3, 4),
        reason="test require python 3.4+")
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
        assert p.exitcode == 0
        assert wait_for_handle(sentinel, timeout=1)

    @classmethod
    def _high_number_Pipe(cls):
        """Create a Pipe with 2 high numbered file descriptors"""
        fds = []
        for _ in range(50):
            r, w = os.pipe()
            fds += [r, w]
        r, w = cls.Pipe(duplex=False)
        for fd in fds:
            os.close(fd)
        return r, w

    @classmethod
    def _test_sync_object_handleling(cls, started, stop, conn, w):
        """Check validity of parents args and Create semaphores to clean up

        started, stop: Event
            make sure the main Process use lsof when this Process is setup
        conn: Connection
            an open pipe that should be closed at exit
        w: int
            fileno of the writable end of the Pipe, it should be closed
        """
        to_clean_up = [cls.Semaphore(0), cls.BoundedSemaphore(1),
                       cls.Lock(), cls.RLock(), cls.Condition(), cls.Event()]
        started.set()
        assert conn.recv_bytes() == b"foo"
        with pytest.raises(OSError):
            os.fstat(w)
        stop.wait(5)

    def _check_fds(self, pid, w):
        """List all the open files and check no extra files are presents.

        Return a list of open named semaphores
        """
        import subprocess
        try:
            out = subprocess.check_output(["lsof", "-a", "-Fftn",
                                           "-p", "{}".format(pid),
                                           "-d", "^txt,^cwd,^rtd"])
            lines = out.decode().split("\n")[1:-1]
        except FileNotFoundError:
            print("lsof does not exist on this plateform. Skip open files"
                  "check.")
            return []

        n_pipe = 0
        named_sem = []
        for fd, t, name in zip(lines[::3], lines[1::3], lines[2::3]):

            # Check if fd is a standard IO file. For python2.7, stdin is set
            # to /dev/null during `Process._boostrap`. For other version, stdin
            # should be closed.
            is_std = (fd in ["f1", "f2"])
            if sys.version_info[:2] < (3, 3):
                if sys.platform != "darwin":
                    is_std |= (fd == "f0" and name == "n/dev/null")
                else:
                    is_std |= (name == "n/dev/null")

            # Check if fd is a pipe
            is_pipe = (t in ["tPIPE", "tFIFO"])
            n_pipe += is_pipe

            # Check if fd is open for the rng. This can happen on different
            # plateform and depending of the python version.
            is_rng = (name == "n/dev/urandom")

            # Check if fd is a semaphore or an open library. Store all the
            # named semaphore
            is_mem = (fd in ["fmem", "fDEL"])
            if sys.platform == "darwin":
                is_mem |= "n/loky-" in name
            if is_mem and "n/dev/shm/sem." in name:
                named_sem += [name[1:]]

            # no other files should be opened at this stage in the process
            assert (is_pipe or is_std or is_rng or is_mem)

        # there should be one pipe for communication with main process
        # and the semaphore tracker pipe and the Connection pipe
        assert n_pipe == 3, ("Some pipes were not properly closed during the "
                             "child process setup.")

        # assert that the writable part of the Pipe (not passed to child),
        # have been properly closed.
        assert len(set("f{}".format(w)).intersection(lines)) == 0

        return named_sem

    def test_sync_object_handleling(self):
        """Check the correct handeling of semaphores and pipes with loky

        We use a Pipe object to check the stated of file descriptors in parent
        and child. To make sure there is no interference in the fd numbers, we
        use high number fd, so newly created fd should be inferior.

        To ensure we have the right number of fd in the child Process, we used
        `lsof` as it is compatible with Unix systems.
        Different behaviors are observed with the open fds, in particular:
        - python2.7 and 3.4 have an open fd for /dev/urandom.
        - python2.7 links stdin to /dev/null even if it is closed beforehand.
        """

        # TODO generate high numbered multiprocessing.Pipe directly
        # -> can be used on windows
        r, w = self._high_number_Pipe()

        tmp_fname = "/tmp/foobar" if sys.platform != "win32" else ".foobar"
        with open(tmp_fname, "w"):
            # Process creating semaphore and pipes before stopping
            started, stop = self.Event(), self.Event()
            p = self.Process(target=self._test_sync_object_handleling,
                             args=(started, stop, r, w.fileno()))
            named_sem = []
            try:

                p.start()
                assert started.wait(1), "The process took too long to start"
                r.close()
                w.send_bytes(b"foo")
                if sys.platform != "win32":
                    named_sem = self._check_fds(p.pid, w.fileno())

            finally:
                stop.set()
                p.join()

                # ensure that Pipe->r was closed when the child process exited
                with pytest.raises(IOError):
                    w.send_bytes(b"foo")
                w.close()

                if sys.platform == "linux":
                    # On linux, check that the named semaphores created in the
                    # child process have been unlinked when it terminated.
                    pid = str(os.getpid())
                    for sem in named_sem:
                        if pid not in sem:
                            assert not os.path.exists(sem), (
                                "Some named semaphore are not properly cleaned"
                                " up")

                assert p.exitcode == 0

    @pytest.mark.skipif(parallel_sum is None,
                        reason="cython is not installed on this system.")
    def test_compatibility_openmp(self):
        # Use openMP before launching subprocesses. With fork backend, some fds
        # are nto correctly clean up, causing a freeze. No freeze should be
        # detected with loky.
        parallel_sum(10)
        p = self.Process(target=parallel_sum, args=(10,))
        p.start()
        p.join()
        assert p.exitcode == 0


def wait_for_handle(handle, timeout):
    from loky.backend.connection import wait
    if timeout is not None and timeout < 0.0:
        timeout = None
    return wait([handle], timeout)
