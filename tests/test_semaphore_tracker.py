"""Tests for the SemaphoreTracker class"""
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
import loky.backend.semaphore_tracker as semaphore_tracker
from loky.backend.semlock import sem_unlink
from loky.backend.context import get_context


def get_sem_tracker_pid():
    semaphore_tracker.ensure_running()
    return semaphore_tracker._semaphore_tracker._pid


class TestSemaphoreTracker:
    def tests_child_retrieves_semaphore_tracker(self):
        # Worker processes created with loky should retrieve the
        # semaphore_tracker of their parent. This is tested by an equality
        # check on the tracker's process id.
        parent_sem_tracker_pid = get_sem_tracker_pid()
        executor = ProcessPoolExecutor(max_workers=2)
        child_sem_tracker_pid = executor.submit(get_sem_tracker_pid).result()
        assert child_sem_tracker_pid == parent_sem_tracker_pid

    # The following four tests are inspired from cpython _test_multiprocessing
    def test_semaphore_tracker(self):
        #
        # Check that killing process does not leak named semaphores
        #
        import subprocess
        cmd = '''if 1:
            import time, os
            from loky.backend.synchronize import Lock

            # close manually the read end of the pipe in the child process
            # because pass_fds does not exist for python < 3.2
            os.close(%d)

            lock1 = Lock()
            lock2 = Lock()
            os.write(%d, lock1._semlock.name.encode("ascii") + b"\\n")
            os.write(%d, lock2._semlock.name.encode("ascii") + b"\\n")
            time.sleep(10)
        '''
        r, w = os.pipe()

        if sys.version_info[:2] >= (3, 2):
            fd_kws = {'pass_fds': [w, r]}
        else:
            fd_kws = {'close_fds': False}
        p = subprocess.Popen([sys.executable,
                             '-E', '-c', cmd % (r, w, w)],
                             stderr=subprocess.PIPE,
                             **fd_kws)
        os.close(w)
        with io.open(r, 'rb', closefd=True) as f:
            name1 = f.readline().rstrip().decode('ascii')
            name2 = f.readline().rstrip().decode('ascii')

        # subprocess holding a reference to lock1 is still alive, so this call
        # should succeed
        sem_unlink(name1)
        p.terminate()
        p.wait()
        time.sleep(2.0)
        with pytest.raises(OSError) as ctx:
            sem_unlink(name2)
        # docs say it should be ENOENT, but OSX seems to give EINVAL
        assert ctx.value.errno in (errno.ENOENT, errno.EINVAL)
        err = p.stderr.read().decode('utf-8')
        p.stderr.close()
        expected = ('semaphore_tracker: There appear to be 2 leaked '
                    'semaphores')
        assert re.search(expected, err) is not None

        # lock1 is still registered, but was destroyed externally: the tracker
        # is expected to complain.
        expected = ("semaphore_tracker: %s: (OSError\\(%d|"
                    "FileNotFoundError)" % (name1, errno.ENOENT))
        assert re.search(expected, err) is not None

    def check_semaphore_tracker_death(self, signum, should_die):
        # bpo-31310: if the semaphore tracker process has died, it should
        # be restarted implicitly.
        from loky.backend.semaphore_tracker import _semaphore_tracker
        # _semaphore_tracker.ensure_running()
        pid = _semaphore_tracker._pid
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _semaphore_tracker.ensure_running()
            # in python2.7, the race condition described in bpo-33613 still
            # exists.
            # TODO: prevent race condition in python2.7?
            time.sleep(1.0)
        pid = _semaphore_tracker._pid

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
            # the semaphore tracker
            del sem
            gc.collect()
            assert wr() is None
            if should_die:
                assert len(all_warn) == 1
                the_warn = all_warn[0]
                assert issubclass(the_warn.category, UserWarning)
                assert "semaphore_tracker: process died" in str(
                    the_warn.message)
            else:
                print(all_warn)
                assert len(all_warn) == 0

    def test_semaphore_tracker_sigint(self):
        # Catchable signal (ignored by semaphore tracker)
        self.check_semaphore_tracker_death(signal.SIGINT, False)

    def test_semaphore_tracker_sigterm(self):
        # Catchable signal (ignored by semaphore tracker)
        self.check_semaphore_tracker_death(signal.SIGTERM, False)

    def test_semaphore_tracker_sigkill(self):
        # Uncatchable signal.
        self.check_semaphore_tracker_death(signal.SIGKILL, True)
