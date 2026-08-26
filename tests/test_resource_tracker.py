"""Tests for the ResourceTracker class"""

import base64
import errno
import gc
import json
import multiprocessing
import os
import pytest
import re
import signal
import subprocess
import sys
import time
import warnings
import weakref

from loky import ProcessPoolExecutor
import loky.backend.resource_tracker as resource_tracker
from loky.backend.context import get_context

from .utils import resource_unlink, create_resource, resource_exists


def _resource_unlink(name, rtype):
    resource_tracker._CLEANUP_FUNCS[rtype](name)


def get_rtracker_fd():
    resource_tracker.ensure_running()
    return resource_tracker._resource_tracker._fd


def get_rtracker_fd_identity():
    stat = os.fstat(get_rtracker_fd())
    return stat.st_dev, stat.st_ino


class _RecordingWinapi:
    """Stand-in for _winapi so Windows paths can be tested everywhere."""

    INFINITE = 0xFFFFFFFF
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 258

    def __init__(
        self,
        create_process=None,
        wait_result=WAIT_OBJECT_0,
        exit_code=0,
    ):
        self.waited = []
        self.closed = []
        self._create_process = create_process
        self._wait_result = wait_result
        self._exit_code = exit_code

    def WaitForSingleObject(self, handle, timeout):
        self.waited.append((handle, timeout))
        return self._wait_result

    def GetExitCodeProcess(self, handle):
        return self._exit_code

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
        resource_tracker.ensure_running()
        handle = resource_tracker._resource_tracker._proc_handle
        assert (handle is not None) == (sys.platform == "win32")

    @pytest.mark.skipif(
        not hasattr(os, "register_at_fork"),
        reason="os.register_at_fork unavailable",
    )
    def test_resource_tracker_at_fork_callback_is_registered(self):
        # Ensure that `_resource_tracker._after_fork_in_child` is called to
        # perform the fd handling specified when the `_fork_intent` module-level
        # variable from the multiprocessing module of the stdlib (rather than
        # the loky vendored copy) is set to False.
        #
        # After a raw os.fork(), the child's ResourceTracker._pid must be None.
        # That only happens if os.register_at_fork registered
        # _resource_tracker._after_fork_in_child
        resource_tracker.ensure_running()
        assert resource_tracker._resource_tracker._pid is not None

        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            child_pid = resource_tracker._resource_tracker._pid
            os.write(write_fd, str(child_pid).encode("ascii"))
            os.close(write_fd)
            os._exit(0)

        os.close(write_fd)
        try:
            child_pid = os.read(read_fd, 32)
        finally:
            os.close(read_fd)
            os.waitpid(pid, 0)

        assert child_pid == b"None"

    @pytest.mark.skipif(
        "fork" not in multiprocessing.get_all_start_methods(),
        reason="fork start method unavailable",
    )
    def test_multiprocessing_fork_preserves_resource_tracker_fd(self):
        # Check that multiprocessing's `preserve_fd=True` is respected.
        parent_fd_identity = get_rtracker_fd_identity()
        ctx = multiprocessing.get_context("fork")

        with ctx.Pool(1) as pool:
            child_fd_identity = pool.apply(get_rtracker_fd_identity)

        assert child_fd_identity == parent_fd_identity

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Windows-specific test"
    )
    def test_resource_tracker_stop_win32_waits_on_handle(self):
        winapi = _RecordingWinapi(exit_code=7)
        tracker, w = _make_stopped_tracker()

        timeout_in_seconds = 0.5
        timeout_in_milliseconds = 1000 * timeout_in_seconds
        tracker._stop_locked(
            wait_for_single_object=winapi.WaitForSingleObject,
            get_exit_code_process=winapi.GetExitCodeProcess,
            close_handle=winapi.CloseHandle,
            wait_timeout_code=winapi.WAIT_TIMEOUT,
            wait_timeout=timeout_in_seconds,
        )

        assert winapi.waited == [(42, timeout_in_milliseconds)]
        assert winapi.closed == [42]
        assert tracker._fd is None
        assert tracker._pid is None
        assert tracker._exitcode == 7
        assert tracker._proc_handle is None
        with pytest.raises(OSError):
            os.close(w)

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Windows-specific test"
    )
    def test_resource_tracker_stop_win32_timeout(self):
        winapi = _RecordingWinapi(wait_result=_RecordingWinapi.WAIT_TIMEOUT)
        tracker, _ = _make_stopped_tracker()

        tracker._stop_locked(
            wait_for_single_object=winapi.WaitForSingleObject,
            get_exit_code_process=winapi.GetExitCodeProcess,
            close_handle=winapi.CloseHandle,
            wait_timeout_code=winapi.WAIT_TIMEOUT,
            wait_timeout=0.5,
        )

        assert tracker._pid is None
        assert tracker._exitcode is None
        assert tracker._waitpid_timed_out
        assert tracker._proc_handle is None
        assert winapi.closed == [42]

    @pytest.mark.parametrize("handle", [42, None])
    def test_resource_tracker_relaunch_closes_handle(
        self, monkeypatch, handle
    ):
        winapi = _RecordingWinapi()
        monkeypatch.setattr(resource_tracker, "_winapi", winapi, raising=False)
        monkeypatch.setattr(os, "name", "nt")
        monkeypatch.setattr(sys, "platform", "win32")
        tracker, _ = _make_stopped_tracker(handle=handle)

        with pytest.warns(UserWarning, match="died unexpectedly"):
            tracker._teardown_dead_process()

        assert winapi.closed == ([] if handle is None else [handle])
        assert tracker._proc_handle is None

    def test_spawnv_passfds_keeps_process_handle(self, monkeypatch):
        winapi = _RecordingWinapi(create_process=(42, 43, 123456, 0))
        monkeypatch.setattr(resource_tracker, "_winapi", winapi, raising=False)
        monkeypatch.setattr(sys, "platform", "win32")

        assert resource_tracker.spawnv_passfds("exe", ["exe"], []) == (
            123456,
            42,
        )
        assert winapi.closed == [43]

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


def test_shutdown_cleans_resources_once_and_folders_last(tmp_path):
    folder = tmp_path / "tracked-folder"
    folder.mkdir()
    filename = folder / "tracked-file"
    filename.touch()

    cmd = "\n".join(
        [
            "from loky.backend import resource_tracker",
            "",
            f"resource_tracker.register({str(folder)!r}, 'folder')",
            f"resource_tracker.register({str(filename)!r}, 'file')",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", cmd],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not folder.exists()
    assert not filename.exists()
    assert "FileNotFoundError" not in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="pass_fds is POSIX-only")
def test_shutdown_cleanup_failure_sets_exit_code():
    read_fd, write_fd = os.pipe()
    cmd = "from loky.backend.resource_tracker import main; " f"main({read_fd})"
    process = subprocess.Popen(
        [sys.executable, "-c", cmd],
        pass_fds=(read_fd,),
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(read_fd)
    try:
        os.write(write_fd, b"REGISTER:test-resource:noop\n")
    finally:
        os.close(write_fd)

    _, stderr = process.communicate()

    assert process.returncode == 2, stderr


def test_decode_json_message_with_newline_in_name():
    name = "folder/name\nwith-newline"
    encoded_name = base64.urlsafe_b64encode(name.encode("utf-8")).decode(
        "ascii"
    )
    message = json.dumps(
        {
            "cmd": "REGISTER",
            "rtype": "file",
            "base64_name": encoded_name,
        }
    ).encode("ascii")

    assert resource_tracker._decode_message(message) == (
        "REGISTER",
        "file",
        name,
    )
