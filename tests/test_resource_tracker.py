"""Tests for the ResourceTracker class"""
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

from loky import ProcessPoolExecutor
import loky.backend.resource_tracker as resource_tracker
from loky.backend.context import get_context

from .utils import resource_unlink, create_resource, resource_exists


def _resource_unlink(name, rtype):
    resource_tracker._CLEANUP_FUNCS[rtype](name)


def get_rtracker_pid():
    resource_tracker.ensure_running()
    return resource_tracker._resource_tracker._pid


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
        parent_rtracker_pid = get_rtracker_pid()
        executor = ProcessPoolExecutor(max_workers=2)
        child_rtracker_pid = executor.submit(get_rtracker_pid).result()

        # First simple pid retrieval check (see #200)
        assert child_rtracker_pid == parent_rtracker_pid

        # Register a resource in the parent process, and un-register it in the
        # child process. If the two processes do not share the same
        # resource_tracker, a cache KeyError should be printed in stderr.
        cmd = '''if 1:
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
        '''
        try:
            p = subprocess.run([sys.executable, '-E', '-c', cmd],
                               capture_output=True,
                               text=True)
            filename = p.stdout.strip()

            pattern = f"decremented refcount of file {filename}"
            assert pattern in p.stderr
            assert "leaked" not in p.stderr

            pattern = f"KeyError: '{filename}'"
            assert pattern not in p.stderr

        finally:
            executor.shutdown()

    # The following four tests are inspired from cpython _test_multiprocessing
    @pytest.mark.parametrize("rtype", ["file", "folder", "semlock"])
    def test_resource_tracker(self, rtype):
        #
        # Check that killing process does not leak named resources
        #
        if (sys.platform == "win32") and rtype == "semlock":
            pytest.skip("no semlock on windows")

        cmd = f'''if 1:
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
        '''
        env = {**os.environ, 'PYTHONPATH': os.path.dirname(__file__)}
        p = subprocess.Popen([sys.executable, '-c', cmd],
                             stderr=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             env=env,
                             text=True)
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

        expected = f'resource_tracker: There appear to be 2 leaked {rtype}'
        assert re.search(expected, err) is not None

        # resource 1 is still registered, but was destroyed externally: the
        # tracker is expected to complain.
        if sys.platform == "win32":
            errno_map = {'file': 2, 'folder': 3}
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

        cmd = f'''if 1:
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
        '''

        env = {**os.environ, 'PYTHONPATH': os.path.dirname(__file__)}
        p = subprocess.run([sys.executable, '-c', cmd],
                           capture_output=True,
                           env=env)
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
                "ignore", message='semaphore are broken on OSX')

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
                    the_warn.message)
            else:
                assert len(all_warn) == 0

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Limited signal support on Windows")
    def test_resource_tracker_sigint(self):
        # Catchable signal (ignored by resource tracker)
        self.check_resource_tracker_death(signal.SIGINT, False)

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Limited signal support on Windows")
    def test_resource_tracker_sigterm(self):
        # Catchable signal (ignored by resource tracker)
        self.check_resource_tracker_death(signal.SIGTERM, False)

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Limited signal support on Windows")
    def test_resource_tracker_sigkill(self):
        # Uncatchable signal.
        self.check_resource_tracker_death(signal.SIGKILL, True)

    @pytest.mark.skipif(sys.version_info < (3, 8),
                        reason="SharedMemory introduced in Python 3.8")
    def test_loky_process_inherit_multiprocessing_resource_tracker(self):
        cmd = '''if 1:
        from loky import get_reusable_executor
        from multiprocessing.shared_memory import SharedMemory
        from multiprocessing.resource_tracker import (
            _resource_tracker as mp_resource_tracker
        )

        def mp_rtracker_getattrs():
            from multiprocessing.resource_tracker import (
                _resource_tracker as mp_resource_tracker
            )
            return mp_resource_tracker._fd, mp_resource_tracker._pid


        if __name__ == '__main__':
            executor = get_reusable_executor(max_workers=1)
            # warm up
            f = executor.submit(id, 1).result()

            # loky forces the creation of the resource tracker at process
            # creation so that loky processes can inherit its file descriptor.
            fd, pid = executor.submit(mp_rtracker_getattrs).result()
            assert fd == mp_resource_tracker._fd
            assert pid == mp_resource_tracker._pid

            # non-regression test for #242: unlinking in a loky process a
            # shared_memory segment tracked by multiprocessing and created its
            # parent should not generate warnings.
            shm = SharedMemory(create=True, size=10)
            f = executor.submit(shm.unlink).result()

        '''
        p = subprocess.run([sys.executable, '-c', cmd],
                           capture_output=True, text=True)
        assert not p.stdout
        assert not p.stderr
