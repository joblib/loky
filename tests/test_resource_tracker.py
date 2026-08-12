"""Tests for the ResourceTracker class"""

import ctypes
import errno
import gc
import os
import pytest
import re
import signal
import subprocess
import sys
import time
import warnings
import weakref
from ctypes import wintypes

from loky import ProcessPoolExecutor
import loky.backend.resource_tracker as resource_tracker
from loky.backend.context import get_context

from .utils import resource_unlink, create_resource, resource_exists


def _resource_unlink(name, rtype):
    resource_tracker._CLEANUP_FUNCS[rtype](name)


def get_rtracker_fd():
    resource_tracker.ensure_running()
    return resource_tracker._resource_tracker._fd


class _RecordingWinapi:
    """Stand-in for _winapi so the win32 paths can be tested on any platform"""

    def __init__(self, create_process=None):
        self.waited = []
        self.closed = []
        self._create_process = create_process

    def WaitForSingleObject(self, handle, timeout):
        self.waited.append((handle, timeout))

    def CloseHandle(self, handle):
        self.closed.append(handle)

    def CreateProcess(self, *args):
        return self._create_process


def _make_stopped_tracker(handle=42):
    tracker = resource_tracker.ResourceTracker()
    r, w = os.pipe()
    os.close(r)
    tracker._fd, tracker._pid, tracker._proc_handle = w, 123456, handle
    return tracker, w


class TestResourceTracker:
    @pytest.mark.parametrize("rtype", ["file", "folder", "semlock"])
    def test_resource_utils(self, rtype):
        # Check that the resouce utils work as expected in the main process
        if sys.platform == "win32" and rtype == "semlock":
            pytest.skip("no semlock on windows")
        name = create_resource(rtype)
        assert resource_exists(name, rtype)
        resource_unlink(name, rtype)
        assert not resource_exists(name, rtype)

    def test_child_retrieves_resource_tracker(self):

        # First simple fd retrieval check (see #200)
        # checking fd only work on posix for now
        if sys.platform != "win32":
            try:
                parent_rtracker_fd = get_rtracker_fd()
                executor = ProcessPoolExecutor(max_workers=2)
                child_rtracker_fd = executor.submit(get_rtracker_fd).result()

                assert child_rtracker_fd == parent_rtracker_fd
            finally:
                executor.shutdown()

        # Register a resource in the parent process, and un-register it in the
        # child process. If the two processes do not share the same
        # resource_tracker, a cache KeyError should be printed in stderr.
        cmd = """if 1:
        import os, sys

        from loky import ProcessPoolExecutor
        from loky.backend import resource_tracker
        from tempfile import NamedTemporaryFile


        tmpfile = NamedTemporaryFile(delete=False)
        tmpfile.close()
        filename = tmpfile.name
        resource_tracker.VERBOSE = True

        resource_tracker.register(filename, "file")

        def maybe_unlink(name, rtype):
            # resource_tracker.maybe_unlink is actually a bound method of the
            # ResourceTracker. We need a custom wrapper to avoid object
            # serialization.
            from loky.backend import resource_tracker
            resource_tracker.maybe_unlink(name, rtype)

        print(filename)
        e = ProcessPoolExecutor(1)
        e.submit(maybe_unlink, filename, "file").result()
        e.shutdown()
        """

        p = subprocess.run(
            [sys.executable, "-E", "-c", cmd],
            capture_output=True,
            text=True,
        )
        filename = p.stdout.strip()

        pattern = f"decremented refcount of file {filename}"
        assert pattern in p.stderr
        assert "leaked" not in p.stderr

        pattern = f"KeyError: '{filename}'"
        assert pattern not in p.stderr

    # The following four tests are inspired from cpython _test_multiprocessing
    @pytest.mark.parametrize("rtype", ["file", "folder", "semlock"])
    def test_resource_tracker(self, rtype):
        #
        # Check that killing process does not leak named resources
        #
        if (sys.platform == "win32") and rtype == "semlock":
            pytest.skip("no semlock on windows")

        cmd = f"""if 1:
            import time, os, tempfile, sys
            from loky.backend import resource_tracker
            from utils import create_resource

            for _ in range(2):
                rname = create_resource("{rtype}")
                resource_tracker.register(rname, "{rtype}")
                # give the resource_tracker time to register the new resource
                time.sleep(0.5)
                sys.stdout.write(f"{{rname}}\\n")
                sys.stdout.flush()
            time.sleep(10)
        """
        env = {**os.environ, "PYTHONPATH": os.path.dirname(__file__)}
        p = subprocess.Popen(
            [sys.executable, "-c", cmd],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            text=True,
        )
        name1 = p.stdout.readline().rstrip()
        name2 = p.stdout.readline().rstrip()

        # subprocess holding a reference to lock1 is still alive, so this call
        # should succeed
        _resource_unlink(name1, rtype)
        p.terminate()
        p.wait()

        # wait for the resource_tracker to cleanup the leaked resources
        time.sleep(2.0)

        with pytest.raises(OSError) as ctx:
            _resource_unlink(name2, rtype)
        # docs say it should be ENOENT, but OSX seems to give EINVAL
        assert ctx.value.errno in (errno.ENOENT, errno.EINVAL)
        err = p.stderr.read()
        p.stderr.close()
        p.stdout.close()

        expected = f"resource_tracker: There appear to be 2 leaked {rtype}"
        assert re.search(expected, err) is not None

        # resource 1 is still registered, but was destroyed externally: the
        # tracker is expected to complain.
        if sys.platform == "win32":
            errno_map = {"file": 2, "folder": 3}
            expected = (
                f"resource_tracker: {re.escape(name1)}: "
                f"(WindowsError\\(({errno_map[rtype]})|FileNotFoundError)"
            )
        else:
            expected = (
                f"resource_tracker: {re.escape(name1)}: "
                f"(OSError\\({errno.ENOENT}|FileNotFoundError)"
            )
        assert re.search(expected, err) is not None

    @pytest.mark.parametrize("rtype", ["file", "folder", "semlock"])
    def test_resource_tracker_refcounting(self, rtype):
        if sys.platform == "win32" and rtype == "semlock":
            pytest.skip("no semlock on windows")

        cmd = f"""if 1:
        import os
        import tempfile
        import time
        from loky.backend import resource_tracker
        from utils import resource_unlink, create_resource, resource_exists

        resource_tracker.VERBOSE = True

        try:
            name = create_resource("{rtype}")
            assert resource_exists(name, "{rtype}")

            from loky.backend.resource_tracker import _resource_tracker
            _resource_tracker.register(name, "{rtype}")
            _resource_tracker.register(name, "{rtype}")

            # Forget all information about the resource, but do not try to
            # remove it
            _resource_tracker.unregister(name, "{rtype}")
            time.sleep(1)
            assert resource_exists(name, "{rtype}")

            _resource_tracker.register(name, "{rtype}")
            _resource_tracker.register(name, "{rtype}")
            _resource_tracker.maybe_unlink(name, "{rtype}")
            time.sleep(1)
            assert resource_exists(name, "{rtype}")

            _resource_tracker.maybe_unlink(name, "{rtype}")
            for _ in range(100):
                if not resource_exists(name, "{rtype}"):
                    break
                time.sleep(.1)
            else:
                raise AssertionError(f"{{name}} was not unlinked in time")
        finally:
            try:
                if resource_exists(name, "{rtype}"):
                    resource_unlink(name, "{rtype}")
            except NameError:
                # "name" is not defined because create_resource has failed
                pass
        """

        env = {**os.environ, "PYTHONPATH": os.path.dirname(__file__)}
        p = subprocess.run(
            [sys.executable, "-c", cmd], capture_output=True, env=env
        )
        assert p.returncode == 0, p.stderr

    def check_resource_tracker_death(self, signum, should_die):
        # bpo-31310: if the semaphore tracker process has died, it should
        # be restarted implicitly.
        from loky.backend.resource_tracker import _resource_tracker

        pid = _resource_tracker._pid
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _resource_tracker.ensure_running()
        pid = _resource_tracker._pid

        os.kill(pid, signum)
        time.sleep(1.0)  # give it time to die

        ctx = get_context("loky")
        with warnings.catch_warnings(record=True) as all_warn:
            warnings.simplefilter("always")

            # remove unrelated MacOS warning messages first
            warnings.filterwarnings(
                "ignore", message="semaphore are broken on OSX"
            )

            sem = ctx.Semaphore()
            sem.acquire()
            sem.release()
            wr = weakref.ref(sem)
            # ensure `sem` gets collected, which triggers communication with
            # the resource_tracker
            del sem
            gc.collect()
            assert wr() is None
            if should_die:
                assert len(all_warn) == 1
                the_warn = all_warn[0]
                assert issubclass(the_warn.category, UserWarning)
                assert "resource_tracker: process died" in str(
                    the_warn.message
                )
            else:
                assert len(all_warn) == 0, [w.message for w in all_warn]

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Limited signal support on Windows"
    )
    def test_resource_tracker_sigint(self):
        # Catchable signal (ignored by resource tracker)
        self.check_resource_tracker_death(signal.SIGINT, False)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Limited signal support on Windows"
    )
    def test_resource_tracker_sigterm(self):
        # Catchable signal (ignored by resource tracker)
        self.check_resource_tracker_death(signal.SIGTERM, False)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Limited signal support on Windows"
    )
    def test_resource_tracker_sigkill(self):
        # Uncatchable signal.
        self.check_resource_tracker_death(signal.SIGKILL, True)

    def test_resource_tracker_keeps_process_handle(self):
        # The pid alone is not enough to reap the tracker safely on Windows
        resource_tracker.ensure_running()
        handle = resource_tracker._resource_tracker._proc_handle
        if sys.platform == "win32":
            assert handle is not None
        else:
            assert handle is None

    @pytest.mark.skipif(
        sys.platform == "win32", reason="posix-only teardown path"
    )
    def test_resource_tracker_del_already_reaped(self, monkeypatch):
        # An already reaped tracker raises ChildProcessError from os.waitpid,
        # which must not escape the destructor. Still needed on 3.12, whose
        # _stop_locked has an unguarded waitpid (see joblib/joblib#1708).
        base = resource_tracker._ResourceTracker
        if not hasattr(base, "__del__"):
            pytest.skip("this Python version has no ResourceTracker.__del__")

        def raising_del(self):
            raise ChildProcessError(errno.ECHILD, "No child processes")

        monkeypatch.setattr(base, "__del__", raising_del)
        resource_tracker.ResourceTracker().__del__()

    def test_resource_tracker_del_without_base_del(self, monkeypatch):
        # Nothing to override before cpython grew a destructor (gh-88887)
        monkeypatch.delattr(
            resource_tracker._ResourceTracker, "__del__", raising=False
        )
        tracker, w = _make_stopped_tracker()
        tracker.__del__()

        assert tracker._fd == w
        os.close(w)

    def test_resource_tracker_stop_win32_waits_on_handle(self, monkeypatch):
        # The point of the fix: wait on the CreateProcess handle, bounded,
        # rather than on the pid, which os.waitpid reinterprets as a handle
        # value naming some unrelated object.
        winapi = _RecordingWinapi()
        monkeypatch.setattr(resource_tracker, "_winapi", winapi, raising=False)
        tracker, w = _make_stopped_tracker()
        tracker._stop_win32()

        timeout = resource_tracker._WIN32_STOP_TIMEOUT_MS
        assert winapi.waited == [(42, timeout)]
        assert winapi.closed == [42]
        assert tracker._fd is None
        assert tracker._pid is None
        assert tracker._proc_handle is None
        with pytest.raises(OSError):
            # the "alive" fd must have been closed to stop the tracker
            os.close(w)

    # a child that inherited the tracker has no handle of its own to close
    @pytest.mark.parametrize("handle", [42, None])
    def test_resource_tracker_relaunch_closes_handle(
        self, monkeypatch, handle
    ):
        # A tracker that died and gets relaunched must not leak its handle
        winapi = _RecordingWinapi()
        monkeypatch.setattr(resource_tracker, "_winapi", winapi, raising=False)
        monkeypatch.setattr(os, "name", "nt")
        tracker, w = _make_stopped_tracker(handle=handle)
        with pytest.warns(UserWarning, match="died unexpectedly"):
            tracker._teardown_dead_process()

        assert winapi.closed == ([] if handle is None else [handle])
        assert tracker._proc_handle is None

    def test_spawnv_passfds_keeps_the_process_handle(self, monkeypatch):
        # Only the thread handle is closed; the process handle is returned
        winapi = _RecordingWinapi(create_process=(42, 43, 123456, 0))
        monkeypatch.setattr(resource_tracker, "_winapi", winapi, raising=False)
        monkeypatch.setattr(sys, "platform", "win32")

        assert resource_tracker.spawnv_passfds("exe", ["exe"], []) == (
            123456,
            42,
        )
        assert winapi.closed == [43]

    def test_resource_tracker_del_does_not_reap_by_pid_on_win32(
        self, monkeypatch
    ):
        # Non-regression test: the inherited teardown ends in os.waitpid, which
        # on Windows takes a process handle, so passing the pid waits on an
        # unrelated object and either fails or blocks forever. It must not be
        # reached there.
        base = resource_tracker._ResourceTracker
        if not hasattr(base, "__del__"):
            pytest.skip("this Python version has no ResourceTracker.__del__")

        reaped = []
        monkeypatch.setattr(base, "__del__", lambda self: reaped.append(True))
        monkeypatch.setattr(sys, "platform", "win32")

        tracker = resource_tracker.ResourceTracker()
        r, w = os.pipe()
        os.close(r)
        tracker._fd, tracker._pid = w, 123456
        tracker.__del__()

        assert not reaped, "the destructor reaped the tracker by pid"
        assert tracker._fd is None and tracker._pid is None
        with pytest.raises(OSError):
            # the "alive" fd must have been closed to stop the tracker
            os.close(w)

    @pytest.mark.skipif(
        sys.platform != "win32", reason="win32-only teardown path"
    )
    def test_resource_tracker_del_survives_a_foreign_pid_value(self):
        # End-to-end regression test for #642 against a real running tracker.
        # os.waitpid takes a process handle on Windows, so _pid is
        # reinterpreted as a handle value in our own table; CI hit that
        # collision by chance. Force it with a handle that is live but lacks
        # SYNCHRONIZE, which is what turns the inherited teardown into
        # PermissionError. A value naming a live never-signalled object would
        # hang instead, so this is the variant that is safe to assert on.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, os.getpid()
        )
        assert handle, ctypes.WinError(ctypes.get_last_error())

        try:
            tracker = resource_tracker.ResourceTracker()
            tracker.ensure_running()
            tracker._pid = int(handle)
            # Must not raise PermissionError, and must not reap by pid
            tracker.__del__()

            assert tracker._fd is None
            assert tracker._proc_handle is None
        finally:
            kernel32.CloseHandle(handle)

    def test_loky_process_inherit_multiprocessing_resource_tracker(self):
        cmd = """if 1:
        from loky import get_reusable_executor
        from multiprocessing.shared_memory import SharedMemory

        def mp_rtracker_getfd():
            from multiprocessing.resource_tracker import (
                _resource_tracker as mp_resource_tracker
            )
            return mp_resource_tracker._fd


        if __name__ == '__main__':
            executor = get_reusable_executor(max_workers=1)
            # warm up
            f = executor.submit(id, 1).result()

            # loky forces the creation of the resource tracker at process
            # creation so that loky processes can inherit its file descriptor.
            parent_fd = mp_rtracker_getfd()
            child_fd = executor.submit(mp_rtracker_getfd).result()
            assert child_fd == parent_fd

            # non-regression test for #242: unlinking in a loky process a
            # shared_memory segment tracked by multiprocessing and created its
            # parent should not generate warnings.
            shm = SharedMemory(create=True, size=10)
            f = executor.submit(shm.unlink).result()

        """
        p = subprocess.run(
            [sys.executable, "-c", cmd], capture_output=True, text=True
        )
        assert not p.stdout, p.stdout
        assert not p.stderr, p.stderr
