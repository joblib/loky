import os
import sys
import time
import pytest
import signal
import threading

from loky.backend import get_context
from .utils import TimingWrapper

loky_context = get_context("loky")

DELTA = 0.1
TIMEOUT1 = .1
TIMEOUT2 = .3

if sys.version_info < (3, 3):
    FileExistsError = OSError
    FileNotFoundError = OSError


@pytest.mark.skipif(sys.platform == "win32", reason="UNIX test")
def test_semlock_failure():
    from loky.backend.semlock import SemLock, sem_unlink
    name = "loky-test-semlock"
    sl = SemLock(0, 1, 1, name=name)

    with pytest.raises(FileExistsError):
        SemLock(0, 1, 1, name=name)
    sem_unlink(sl.name)

    with pytest.raises(FileNotFoundError):
        SemLock._rebuild(None, 0, 0, name)


def assert_sem_value_equal(sem, value):
    try:
        assert sem.get_value() == value
    except NotImplementedError:
        pass


def assert_timing_almost_equal(t1, t2=0):
    assert abs(t1 - t2) < 1e-1


class TestLock():

    def test_lock(self):
        lock = loky_context.Lock()
        assert lock.acquire()
        assert not lock.acquire(False)
        assert lock.release() is None
        with pytest.raises(ValueError) as excinfo:
            lock.release()

        assert "released too many times" in str(excinfo.value)

    def test_rlock(self):
        lock = loky_context.RLock()
        assert lock.acquire()
        assert lock.acquire()
        assert lock.acquire()
        assert lock.release() is None
        assert lock.release() is None
        assert lock.release() is None
        with pytest.raises(AssertionError) as excinfo:
            lock.release()

        assert "not owned by thread" in str(excinfo.value)

    def test_lock_context(self):
        with loky_context.Lock():
            pass


class TestSemaphore():

    def _test_semaphore(self, sem):
        assert_sem_value_equal(sem, 2)
        assert sem.acquire()
        assert_sem_value_equal(sem, 1)
        assert sem.acquire()
        assert_sem_value_equal(sem, 0)
        assert not sem.acquire(False)
        assert_sem_value_equal(sem, 0)
        assert sem.release() is None
        assert_sem_value_equal(sem, 1)
        assert sem.release() is None
        assert_sem_value_equal(sem, 2)

    def test_semaphore(self):
        sem = loky_context.Semaphore(2)
        self._test_semaphore(sem)
        assert sem.release() is None
        assert_sem_value_equal(sem, 3)
        assert sem.release() is None
        assert_sem_value_equal(sem, 4)

    @pytest.mark.skipif(sys.platform == "darwin",
                        reason="OSX have borken `get_value`")
    def test_bounded_semaphore(self):
        sem = loky_context.BoundedSemaphore(2)
        self._test_semaphore(sem)
        with pytest.raises(ValueError):
            sem.release()
        assert_sem_value_equal(sem, 2)

    def test_timeout(self):

        sem = loky_context.Semaphore(0)
        acquire = TimingWrapper(sem.acquire)

        assert not acquire(False)
        assert_timing_almost_equal(acquire.elapsed)

        assert not acquire(False, None)
        assert_timing_almost_equal(acquire.elapsed)

        assert not acquire(False, TIMEOUT1)
        assert_timing_almost_equal(acquire.elapsed)

        assert not acquire(True, TIMEOUT1)
        assert_timing_almost_equal(acquire.elapsed, TIMEOUT1)

        assert not acquire(True, TIMEOUT2)
        assert_timing_almost_equal(acquire.elapsed, TIMEOUT2)


class TestCondition():

    @classmethod
    def _test_notify(cls, cond, sleeping, woken, timeout=None):
        cond.acquire()
        sleeping.release()
        cond.wait(timeout)
        woken.release()
        cond.release()

    def check_invariant(self, cond):
        # this is only supposed to succeed when there are no sleepers
        try:
            sleepers = (cond._sleeping_count.get_value() -
                        cond._woken_count.get_value())
            sleepers == 0
            cond._wait_semaphore.get_value() == 0
        except NotImplementedError:
            pass

    def test_notify(self):
        cond = loky_context.Condition()
        sleeping = loky_context.Semaphore(0)
        woken = loky_context.Semaphore(0)

        p = loky_context.Process(target=self._test_notify,
                                 args=(cond, sleeping, woken))
        p.daemon = True
        p.start()

        p = threading.Thread(target=self._test_notify,
                             args=(cond, sleeping, woken))
        p.daemon = True
        p.start()

        # wait for both children to start sleeping
        sleeping.acquire()
        sleeping.acquire()

        # check no process/thread has woken up
        time.sleep(DELTA)
        assert_sem_value_equal(woken, 0)

        # wake up one process/thread
        cond.acquire()
        cond.notify()
        cond.release()

        # check one process/thread has woken up
        time.sleep(DELTA)
        assert_sem_value_equal(woken, 1)

        # wake up another
        cond.acquire()
        cond.notify()
        cond.release()

        # check other has woken up
        time.sleep(DELTA)
        assert_sem_value_equal(woken, 2)

        # check state is not mucked up
        self.check_invariant(cond)
        p.join()

    @pytest.mark.xfail(sys.platform != "win32" and
                       sys.version_info[:2] <= (3, 3),
                       reason="The test if not robust enough. See issue#74")
    def test_notify_all(self):
        cond = loky_context.Condition()
        sleeping = loky_context.Semaphore(0)
        woken = loky_context.Semaphore(0)

        # start some threads/processes which will timeout
        for i in range(3):
            p = loky_context.Process(target=self._test_notify,
                                     args=(cond, sleeping, woken, TIMEOUT1))
            p.daemon = True
            p.start()

            t = threading.Thread(target=self._test_notify,
                                 args=(cond, sleeping, woken, TIMEOUT1))
            t.daemon = True
            t.start()

        # wait for them all to sleep
        for i in range(6):
            sleeping.acquire()

        # check they have all timed out
        for i in range(6):
            woken.acquire()
        assert_sem_value_equal(woken, 0)

        # check state is not mucked up
        self.check_invariant(cond)

        # start some more threads/processes
        for i in range(3):
            p = loky_context.Process(target=self._test_notify,
                                     args=(cond, sleeping, woken))
            p.daemon = True
            p.start()

            t = threading.Thread(target=self._test_notify,
                                 args=(cond, sleeping, woken))
            t.daemon = True
            t.start()

        # wait for them to all sleep
        for i in range(6):
            sleeping.acquire()

        # check no process/thread has woken up
        time.sleep(DELTA)
        assert_sem_value_equal(woken, 0)

        # wake them all up
        cond.acquire()
        cond.notify_all()
        cond.release()

        # check they have all woken
        for i in range(50):
            try:
                if woken.get_value() == 6:
                    break
            except NotImplementedError:
                break
            time.sleep(DELTA)
        assert_sem_value_equal(woken, 6)

        # check state is not mucked up
        self.check_invariant(cond)

    def test_timeout(self):
        cond = loky_context.Condition()
        wait = TimingWrapper(cond.wait)
        cond.acquire()
        res = wait(TIMEOUT1)
        cond.release()

        assert not res
        assert abs(wait.elapsed - TIMEOUT1) < 1e-1

    @classmethod
    def _test_waitfor_f(cls, cond, state):
        with cond:
            state.release()
            cond.notify()
            result = cond.wait_for(lambda: state.get_value() == 5)
            if not result or state.get_value() != 5:
                sys.exit(1)

    @pytest.mark.skipif(sys.platform == "win32" and
                        sys.version_info[:2] < (3, 3),
                        reason="Condition.wait_for was introduced in 3.3 and "
                        "we do not overload win32 Condition")
    def test_waitfor(self):
        # based on test in test/lock_tests.py
        cond = loky_context.Condition()
        state = loky_context.Semaphore(0)
        try:
            state.get_value()
        except NotImplementedError:
            pytest.skip(msg="`sem_get_value not implemented")

        p = loky_context.Process(target=self._test_waitfor_f,
                                 args=(cond, state))
        p.daemon = True
        p.start()

        with cond:
            result = cond.wait_for(lambda: state.get_value() == 1)
            assert result
            assert state.get_value() == 1

        for i in range(4):
            time.sleep(0.01)
            with cond:
                state.release()
                cond.notify()

        p.join(5)
        assert not p.is_alive()
        assert p.exitcode == 0

    @classmethod
    def _test_wait_result(cls, c, pid):
        with c:
            c.notify()
        time.sleep(1)
        if pid is not None:
            os.kill(pid, signal.SIGINT)

    @pytest.mark.skipif(sys.platform == "win32" and
                        sys.version_info[:2] < (3, 3),
                        reason="Condition.wait always returned None before 3.3"
                        " and we do not overload win32 Condition")
    def test_wait_result(self):
        if sys.platform != 'win32':
            pid = os.getpid()
        else:
            pid = None

        c = loky_context.Condition()
        with c:
            assert not c.wait(0)
            assert not c.wait(0.1)

            p = loky_context.Process(target=self._test_wait_result,
                                     args=(c, pid))
            p.start()

            assert c.wait(10)
            if pid is not None:
                with pytest.raises(KeyboardInterrupt):
                    c.wait(10)
            p.join()


class TestEvent():

    @classmethod
    def _test_event(cls, event):
        time.sleep(TIMEOUT1)
        event.set()

    def test_event(self):
        event = loky_context.Event()
        wait = TimingWrapper(event.wait)

        assert not event.is_set()

        assert not wait(0.0)
        assert_timing_almost_equal(wait.elapsed)
        assert not wait(TIMEOUT1)
        assert_timing_almost_equal(wait.elapsed, TIMEOUT1)

        event.set()

        assert event.is_set()
        assert wait()
        assert_timing_almost_equal(wait.elapsed)
        assert wait(TIMEOUT1)
        assert_timing_almost_equal(wait.elapsed)

        event.clear()

        assert not event.is_set()

        p = loky_context.Process(target=self._test_event, args=(event,))
        p.daemon = True
        p.start()
        assert wait()
        p.join()
