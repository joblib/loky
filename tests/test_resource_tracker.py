"""Tests for the ResourceTracker class"""
import errno
import gc
import io
import os
import pytest
import re
import signal
import sys
import time
import warnings
import weakref

from loky import ProcessPoolExecutor
import loky.backend.resource_tracker as resource_tracker
from loky.backend.semlock import sem_unlink
from loky.backend.context import get_context


def _resource_unlink(name, rtype):
    resource_tracker._CLEANUP_FUNCS[rtype](name)

def get_rtracker_pid():
    resource_tracker.ensure_running()
    return resource_tracker._resource_tracker._pid

class TestResourceTracker:
    def test_child_retrieves_resource_tracker(self):
        parent_rtracker_pid = get_rtracker_pid()
        executor = ProcessPoolExecutor(max_workers=2)
        child_rtracker_pid = executor.submit(get_rtracker_pid).result()

        # First simple pid retrieval check (see #200)
        assert child_rtracker_pid == parent_rtracker_pid

        # Register a resource in the parent process, and un-register it in the
        # child process. If the two processes do not share the same
        # resource_tracker, a cache KeyError should be printed in stderr.
        import subprocess
        semlock_name = 'loky-mysemaphore'
        cmd = '''if 1:
        import os, sys

        from loky import ProcessPoolExecutor
        from loky.backend import resource_tracker
        from loky.backend.semlock import SemLock

        resource_tracker.VERBOSE=True
        semlock_name = "{}"

        # We don't need to create the semaphore as registering / unregistering
        # operations simply add / remove entries from a cache, but do not
        # manipulate the actual semaphores.
        resource_tracker.register(semlock_name, "semlock")

        def unregister(name, rtype):
            # resource_tracker.unregister is actually a bound method of the
            # ResourceTracker. We need a custom wrapper to avoid object
            # serialization.
            from loky.backend import resource_tracker
            resource_tracker.unregister(semlock_name, rtype)

        e = ProcessPoolExecutor(1)
        e.submit(unregister, semlock_name, "semlock").result()
        e.shutdown()
        '''
        try:
            p = subprocess.Popen(
                [sys.executable, '-E', '-c', cmd.format(semlock_name)],
                stderr=subprocess.PIPE)
            p.wait()

            err = p.stderr.read().decode('utf-8')
            p.stderr.close()

            assert re.search("unregister %s" % semlock_name, err) is not None
            assert re.search("KeyError: '%s'" % semlock_name, err) is None

        finally:
            executor.shutdown()

    # The following four tests are inspired from cpython _test_multiprocessing
    @pytest.mark.parametrize("rtype", ["folder", "semlock"])
    def test_resource_tracker(self, rtype):
        #
        # Check that killing process does not leak named resources
        #
        if (sys.platform == "win32") and rtype == "semlock":
            # no semlock on windows
            return

        import subprocess
        cmd = '''if 1:
            import time, os, tempfile
            from loky.backend.semlock import SemLock
            from loky.backend import resource_tracker

            def create_resource(rtype):
                if rtype == "folder":
                    return tempfile.mkdtemp()
                elif rtype == "semlock":
                    name = "/loky-%i-%s" % (os.getpid(), next(SemLock._rand))
                    lock = SemLock(1, 1, 1, name)
                    return name
                else:
                    raise ValueError(
                        "Resource type {{}} not understood".format(rtype))


            # close manually the read end of the pipe in the child process
            # because pass_fds does not exist for python < 3.2
            os.close({r})

            for _ in range(2):
                rname = create_resource("{rtype}")
                resource_tracker.register(rname, "{rtype}")
                os.write({w}, rname.encode("ascii") + b"\\n")
            time.sleep(10)
        '''
        r, w = os.pipe()

        if sys.version_info[:2] >= (3, 2):
            fd_kws = {'pass_fds': [w, r]}
        else:
            fd_kws = {'close_fds': False}
        p = subprocess.Popen([sys.executable,
                             '-E', '-c', cmd.format(r=r, w=w, rtype=rtype)],
                             stderr=subprocess.PIPE,
                             **fd_kws)
        os.close(w)
        with io.open(r, 'rb', closefd=True) as f:
            name1 = f.readline().rstrip().decode('ascii')
            name2 = f.readline().rstrip().decode('ascii')

        # subprocess holding a reference to lock1 is still alive, so this call
        # should succeed
        _resource_unlink(name1, rtype)
        p.terminate()
        p.wait()
        time.sleep(2.0)
        with pytest.raises(OSError) as ctx:
            _resource_unlink(name2, rtype)
        # docs say it should be ENOENT, but OSX seems to give EINVAL
        assert ctx.value.errno in (errno.ENOENT, errno.EINVAL)
        err = p.stderr.read().decode('utf-8')
        p.stderr.close()
        expected = ('resource_tracker: There appear to be 2 leaked {}'.format(
                    rtype))
        assert re.search(expected, err) is not None

        # lock1 is still registered, but was destroyed externally: the tracker
        # is expected to complain.
        expected = ("resource_tracker: %s: (OSError\\(%d|"
                    "FileNotFoundError)" % (name1, errno.ENOENT))
        assert re.search(expected, err) is not None

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
            # in python < 3.3 , the race condition described in bpo-33613 still
            # exists, as this fixe requires signal.pthread_sigmask
            time.sleep(1.0)
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
                        "Limited signal support on Windows")
    def test_resource_tracker_sigint(self):
        # Catchable signal (ignored by resource tracker)
        self.check_resource_tracker_death(signal.SIGINT, False)

    @pytest.mark.skipif(sys.platform == "win32",
                        "Limited signal support on Windows")
    def test_resource_tracker_sigterm(self):
        # Catchable signal (ignored by resource tracker)
        self.check_resource_tracker_death(signal.SIGTERM, False)

    @pytest.mark.skipif(sys.platform == "win32",
                        "Limited signal support on Windows")
    @pytest.mark.skipif(sys.version_info[0] < 3,
                        reason="warnings.catch_warnings limitation")
    def test_resource_tracker_sigkill(self):
        # Uncatchable signal.
        self.check_resource_tracker_death(signal.SIGKILL, True)
