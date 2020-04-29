import os
import sys
import time
import psutil
import pytest
import signal
import pickle
import platform
import socket
import multiprocessing as mp
from tempfile import mkstemp

from loky.backend import get_context
from loky.backend.compat import wait
from loky.backend.context import START_METHODS
from loky.backend.utils import recursive_terminate

from .utils import TimingWrapper, check_subprocess_call
from .utils import with_parallel_sum, _run_openmp_parallel_sum

if sys.version_info < (3, 3):
    FileNotFoundError = NameError


if not hasattr(socket, "socketpair"):
    def socketpair():
        s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s1.bind((socket.gethostname(), 8080))
        s1.listen(1)
        s2.connect((socket.gethostname(), 8080))
        conn, addr = s1.accept()
        return conn, s2

    socket.socketpair = socketpair

DELTA = 0.1
ctx_loky = get_context("loky")
HAVE_SEND_HANDLE = (sys.platform == "win32" or
                    (hasattr(socket, 'CMSG_LEN') and
                     hasattr(socket, 'SCM_RIGHTS') and
                     hasattr(socket.socket, 'sendmsg')))
HAVE_FROM_FD = hasattr(socket, "fromfd")


class TestLokyBackend:
    # loky processes
    Process = staticmethod(ctx_loky.Process)
    current_process = staticmethod(mp.current_process)
    active_children = staticmethod(mp.active_children)

    # interprocess communication objects
    Pipe = staticmethod(ctx_loky.Pipe)
    Manager = staticmethod(ctx_loky.Manager)

    # synchronization primitives
    Lock = staticmethod(ctx_loky.Lock)
    RLock = staticmethod(ctx_loky.RLock)
    Semaphore = staticmethod(ctx_loky.Semaphore)
    BoundedSemaphore = staticmethod(ctx_loky.BoundedSemaphore)
    Condition = staticmethod(ctx_loky.Condition)
    Event = staticmethod(ctx_loky.Event)
    Queue = staticmethod(ctx_loky.Queue)
    SimpleQueue = staticmethod(ctx_loky.SimpleQueue)

    @classmethod
    def teardown_class(cls):
        """Clean up the test environment from any remaining subprocesses.
        """
        for child_process in cls.active_children():
            recursive_terminate(child_process)

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
    def _test_process(cls, q, sq, *args, **kwds):
        current = cls.current_process()
        q.put(args, timeout=1)
        sq.put(args)

        q.put(kwds, timeout=1)
        q.put(current.name, timeout=1)
        q.put(bytes(current.authkey))
        q.put(current.pid)

    @pytest.mark.parametrize("context_name", ["loky", "loky_init_main"])
    def test_process(self, capsys, context_name):
        """behavior of Process variables and functional connection objects
        """
        import contextlib

        @contextlib.contextmanager
        def no_mgr():
            yield None

        with capsys.disabled() if sys.version_info[:2] == (3, 3) else no_mgr():
            if sys.version_info[:2] == (3, 3):
                import logging
                logger = mp.util.get_logger()
                logger.setLevel(5)
                formatter = logging.Formatter(
                    mp.util.DEFAULT_LOGGING_FORMAT)
                handler = logging.StreamHandler()
                handler.setFormatter(formatter)
                old_handler = logger.handlers[0]
                logger.handlers[0] = handler

            q = self.Queue()
            sq = self.SimpleQueue()
            args = (q, sq, 1, 2)
            kwargs = {'hello': 23, 'bye': 2.54}
            name = 'TestLokyProcess'
            ctx = get_context(context_name)
            p = ctx.Process(
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

            # Make sure we do not break security
            with pytest.raises(TypeError):
                pickle.dumps(p.authkey)

            # Make sure we detect bad pickling
            with pytest.raises(RuntimeError):
                pickle.dumps(q)

            p.start()

            assert p.exitcode is None
            assert p.is_alive()
            assert p in self.active_children()

            assert q.get() == args[2:]
            assert sq.get() == args[2:]

            assert q.get() == kwargs
            assert q.get() == p.name
            assert q.get() == current.authkey
            assert q.get() == p.pid

            p.join()

            assert p.exitcode == 0
            assert not p.is_alive()
            assert p not in self.active_children()

            if sys.version_info[:2] == (3, 3):
                logger.handlers[0] = old_handler

    @classmethod
    def _test_connection(cls, conn):
        """Make sure a connection object is functional"""
        if hasattr(conn, "get"):
            conn = conn.get()
        if hasattr(conn, "accept"):
            msg = conn.recv(2)
            conn.send(msg)
        else:
            msg = conn.recv_bytes()
            conn.send_bytes(msg)
        conn.close()

    @pytest.mark.skipif(
        sys.platform == "win32" and sys.version_info[:2] < (3, 3),
        reason="socket are not picklable with python2.7 and vanilla"
        " ForkingPickler on windows")
    def test_socket(self):
        """sockets can be pickled at spawn and are able to send/recv"""
        server, client = socket.socketpair()

        p = self.Process(target=self._test_connection, args=(server,))
        p.start()

        client.settimeout(5)

        msg = b'42'
        client.send(msg)
        assert client.recv(2) == msg

        p.join()
        assert p.exitcode == 0

        client.close()
        server.close()

    @pytest.mark.skipif(not HAVE_SEND_HANDLE or not HAVE_FROM_FD,
                        reason="This system cannot send handle between process"
                        ". Connections object should be shared at spawning.")
    def test_socket_queue(self):
        """sockets can be pickled in a queue and are able to send/recv"""
        q = self.SimpleQueue()

        p = self.Process(target=self._test_connection, args=(q,))
        p.start()

        server, client = socket.socketpair()
        q.put(server)

        msg = b'42'
        client.settimeout(5)
        client.send(msg)
        assert client.recv(2) == msg

        p.join()
        assert p.exitcode == 0

        client.close()
        server.close()

    def test_connection(self):
        """connections can be pickled at spawn and are able to send/recv"""
        parent_connection, child_connection = self.Pipe(duplex=True)

        p = self.Process(target=self._test_connection,
                         args=(child_connection,))
        p.start()

        msg = b'42'
        parent_connection.send(msg)
        assert parent_connection.recv() == msg

        p.join()
        assert p.exitcode == 0
        parent_connection.close()
        child_connection.close()

    @pytest.mark.skipif(not HAVE_SEND_HANDLE,
                        reason="This system cannot send handle between. "
                        "Connections object should be shared at spawning.")
    def test_connection_queue(self):
        """connections can be pickled in a queue and are able to send/recv"""
        q = self.SimpleQueue()
        p = self.Process(target=self._test_connection, args=(q,))
        p.start()

        parent_connection, child_connection = self.Pipe(duplex=True)
        q.put(child_connection)

        msg = b'42'
        parent_connection.send(msg)
        assert parent_connection.recv() == msg

        p.join()
        assert p.exitcode == 0
        parent_connection.close()
        child_connection.close()

    @staticmethod
    def _test_child_env(key, queue):
        import os

        queue.put(os.environ.get(key, 'not set'))

    @pytest.mark.xfail(sys.version_info < (3, 6) and sys.platform == "win32",
                       reason="Can randomly fail with python < 3.6 under windows.")
    def test_child_env_process(self):
        import os

        key = 'loky_child_env_process'
        value = 'loky works'
        out_queue = self.SimpleQueue()
        try:
            # Test that the environment variable is correctly copied in the
            # child process.
            os.environ[key] = value
            p = self.Process(target=self._test_child_env,
                             args=(key, out_queue))
            p.start()
            child_var = out_queue.get()
            p.join()

            assert child_var == value

            # Test that the environment variable is correctly overwritted by
            # using the `env` argument in Process.
            new_value = 'loky rocks'
            p = self.Process(target=self._test_child_env,
                             args=(key, out_queue), env={key: new_value})
            p.start()
            child_var = out_queue.get()
            p.join()

            assert child_var == new_value, p.env
        finally:
            del os.environ[key]

    @classmethod
    def _test_terminate(cls, event):
        # Notify the main process that child process started
        event.set()
        time.sleep(100)

    def test_terminate(self):

        manager = self.Manager()
        event = manager.Event()

        p = self.Process(target=self._test_terminate, args=(event,))
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
        event.wait(5)

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
        manager.shutdown()

        # XXX sometimes get p.exitcode == 0 on Windows ...
        # assert p.exitcode == -signal.SIGTERM

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
                    target=cls._test_recursion, args=(wconn, l + [i])
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
        event.wait(30.0)

    def test_sentinel(self):
        event = self.Event()
        p = self.Process(target=self._test_sentinel, args=(event,))
        with pytest.raises(ValueError):
            p.sentinel
        p.start()
        # Cast long to int for 64-bit Python 2.7 under Windows
        sentinel = int(p.sentinel)
        assert not wait_for_handle(sentinel, timeout=0.0)
        event.set()
        p.join()
        assert p.exitcode == 0
        assert wait_for_handle(sentinel, timeout=1)

    @classmethod
    def _test_wait_sentinel(cls):
        from signal import SIGTERM
        time.sleep(.1)
        os.kill(os.getpid(), SIGTERM)

    def test_wait_sentinel(self):
        p = self.Process(target=self._test_wait_sentinel)
        with pytest.raises(ValueError):
            p.sentinel
        p.start()
        # Cast long to int for 64-bit Python 2.7 under Windows
        sentinel = int(p.sentinel)
        assert isinstance(sentinel, int)
        assert not wait([sentinel], timeout=0.0)
        assert wait([sentinel], timeout=5), (p.exitcode)
        expected_code = 15 if sys.platform == 'win32' else -15
        p.join()  # force refresh of p.exitcode
        assert p.exitcode == expected_code

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
    def _test_sync_object_handling(cls, started, stop, conn, w):
        """Check validity of parents args and Create semaphores to clean up

        started, stop: Event
            make sure the main Process use lsof when this Process is setup
        conn: Connection
            an open pipe that should be closed at exit
        w: int
            fileno of the writable end of the Pipe, it should be closed
        """
        to_clean_up = [cls.Semaphore(0), cls.BoundedSemaphore(1),  # noqa: F841
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
        except (FileNotFoundError, OSError):
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

        # there should be:
        # - one pipe for communication with main process
        # - loky's resource_tracker pipe
        # - the Connection pipe
        # - additionally, on posix + Python 3.8: multiprocessing's
        #   resource_tracker pipe
        if sys.version_info >= (3, 8) and os.name == 'posix':
            n_expected_pipes = 4
        else:
            n_expected_pipes = 3
        msg = ("Some pipes were not properly closed during the child process "
               "setup.")
        assert n_pipe == n_expected_pipes, msg

        # assert that the writable part of the Pipe (not passed to child),
        # have been properly closed.
        assert len(set("f{}".format(w)).intersection(lines)) == 0

        return named_sem

    @pytest.mark.skipif(
        platform.python_implementation() == "PyPy" and
        sys.version_info[:3] <= (3, 5, 3),
        reason="early PyPy versions leak a file descriptor, see "
               "https://bitbucket.org/pypy/pypy/issues/3021")
    def test_sync_object_handling(self):
        """Check the correct handling of semaphores and pipes with loky

        We use a Pipe object to check the stated of file descriptors in parent
        and child. To make sure there is no interference in the fd numbers, we
        use high number fd, so newly created fd should be inferior.

        To ensure we have the right number of fd in the child Process, we used
        `lsof` as it is compatible with Unix systems.
        Different behaviors are observed with the open fds, in particular:
        - python2.7 and 3.4 have an open fd for /dev/urandom.
        - python2.7 links stdin to /dev/null even if it is closed beforehand.
        """

        # TODO generate high numbered mp.Pipe directly
        # -> can be used on windows
        r, w = self._high_number_Pipe()

        tmp_fname = "/tmp/foobar" if sys.platform != "win32" else ".foobar"
        with open(tmp_fname, "w"):
            # Process creating semaphore and pipes before stopping
            started, stop = self.Event(), self.Event()
            p = self.Process(target=self._test_sync_object_handling,
                             args=(started, stop, r, w.fileno()))
            named_sem = []
            try:

                p.start()
                assert started.wait(5), "The process took too long to start"
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

    @with_parallel_sum
    def test_compatibility_openmp(self):
        # Use openMP before launching subprocesses. With fork backend, some fds
        # are nto correctly clean up, causing a freeze. No freeze should be
        # detected with loky.
        _run_openmp_parallel_sum(10)
        p = self.Process(target=_run_openmp_parallel_sum, args=(100,))
        p.start()
        p.join()
        assert p.exitcode == 0

    @pytest.mark.parametrize("run_file", [True, False])
    def test_interactively_define_process_no_main(self, run_file):
        # check that the init_main_module parameter works properly
        # when using -c option, we don't need the safeguard if __name__ ..
        # and thus test LokyProcess without the extra argument. For running
        # a script, it is necessary to use init_main_module=False.
        code = '\n'.join([
            'from loky.backend.process import LokyProcess',
            'p = LokyProcess(target=id, args=(1,), ',
            '                init_main_module={})'.format(not run_file),
            'p.start()',
            'p.join()',
            'msg = "LokyProcess failed to load without safeguard"',
            'assert p.exitcode == 0, msg',
            'print("ok")'
        ])
        cmd = [sys.executable]
        try:
            if run_file:
                fid, filename = mkstemp(suffix="_joblib.py")
                os.close(fid)
                with open(filename, mode='wb') as f:
                    f.write(code.encode('ascii'))
                cmd += [filename]
            else:
                cmd += ["-c", code]
            check_subprocess_call(cmd, stdout_regex=r'ok', timeout=10)
        finally:
            if run_file:
                os.unlink(filename)

    def test_interactively_define_process_fail_main(self):
        # check that the default behavior of the LokyProcess is correct
        code = '\n'.join([
            'from loky.backend.process import LokyProcess',
            'p = LokyProcess(target=id, args=(1,),',
            '                init_main_module=True)',
            'p.start()',
            'p.join()',
            'msg = "LokyProcess succeed without safeguards"',
            'assert p.exitcode != 0, msg'
        ])
        fid, filename = mkstemp(suffix="_joblib.py")
        os.close(fid)
        try:
            with open(filename, mode='wb') as f:
                f.write(code.encode('ascii'))
            stdout, stderr = check_subprocess_call([sys.executable, filename],
                                                   timeout=10)
            if sys.platform == "win32":
                assert "RuntimeError:" in stderr
            else:
                assert "RuntimeError:" in stdout
        finally:
            os.unlink(filename)

    def test_loky_get_context(self):
        # check the behavior of get_context
        ctx_default = get_context()
        assert ctx_default.get_start_method() == "loky"

        ctx_loky = get_context("loky")
        assert ctx_loky.get_start_method() == "loky"

        ctx_loky_init_main = get_context("loky_init_main")
        assert ctx_loky_init_main.get_start_method() == "loky_init_main"

        with pytest.raises(ValueError):
            get_context("not_available")

    def test_interactive_contex_no_main(self):
        # Ensure that loky context is working properly
        code = '\n'.join([
            'from loky.backend import get_context',
            'ctx = get_context()',
            'assert ctx.get_start_method() == "loky"',
            'p = ctx.Process(target=id, args=(1,))',
            'p.start()',
            'p.join()',
            'msg = "loky context failed to load without safeguard"',
            'assert p.exitcode == 0, msg',
            'print("ok")'
        ])
        try:
            fid, filename = mkstemp(suffix="_joblib.py")
            os.close(fid)
            with open(filename, mode='wb') as f:
                f.write(code.encode('ascii'))
            check_subprocess_call([sys.executable, filename],
                                  stdout_regex=r'ok', timeout=10)
        finally:
            os.unlink(filename)


def wait_for_handle(handle, timeout):
    from loky.backend.compat import wait
    if timeout is not None and timeout < 0.0:
        timeout = None
    return wait([handle], timeout)


def _run_nested_delayed(depth, delay, event):
    if depth > 0:
        p = ctx_loky.Process(target=_run_nested_delayed,
                             args=(depth - 1, delay, event))
        p.start()
        p.join()
    else:
        event.set()

    time.sleep(delay)


@pytest.mark.parametrize("use_psutil", [True, False])
def test_recursive_terminate(use_psutil):
    event = ctx_loky.Event()
    p = ctx_loky.Process(target=_run_nested_delayed, args=(4, 1000, event))
    p.start()

    # Wait for all the processes to be launched
    if not event.wait(30):
        recursive_terminate(p, use_psutil=use_psutil)
        raise RuntimeError("test_recursive_terminate was not able to launch "
                           "all nested processes.")

    children = psutil.Process(pid=p.pid).children(recursive=True)
    recursive_terminate(p, use_psutil=use_psutil)

    # The process can take some time finishing so we should wait up to 5s
    gone, alive = psutil.wait_procs(children, timeout=5)
    msg = "Should be no descendant left but found:\n{}"
    assert len(alive) == 0, msg.format(alive)


def _test_default_subcontext(queue):
    if sys.version_info >= (3, 3):
        start_method = mp.get_start_method()
    else:
        from loky.backend.context import _DEFAULT_START_METHOD
        start_method = _DEFAULT_START_METHOD

    queue.put(start_method)


@pytest.mark.parametrize('method', START_METHODS)
def test_default_subcontext(method):
    code = """if True:
        import sys

        from loky.backend.context import get_context, set_start_method
        from tests.test_loky_backend import _test_default_subcontext

        set_start_method('{method}')
        ctx = get_context()
        assert ctx.get_start_method() == '{method}'

        queue = ctx.SimpleQueue()
        p = ctx.Process(target=_test_default_subcontext, args=(queue,))
        p.start()
        p.join()
        start_method = queue.get()
        assert start_method == '{method}', start_method

        try:
            set_start_method('loky')
        except RuntimeError:
            pass
        else:
            raise AssertionError("Did not raise RuntimeError when resetting"
                                 "start_method without force")

        set_start_method(None, force=True)
        ctx = get_context()
        assert ctx.get_start_method() == 'loky'
    """.format(method=method)

    cmd = [sys.executable, "-c", code]
    check_subprocess_call(cmd, timeout=10)

    ctx_default = get_context()
    assert ctx_default.get_start_method() == "loky"
