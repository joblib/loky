import time
import sys

import pytest

from loky.reusable_executor import get_reusable_thread_executor
from loky.thread import BrokenThreadPool

# Some tests here are probably not useful anymore, as I integrate the resuable
# thread executor into the already existing test suite
# TODO: move the unnecessary tests


def dummy_initializer():
    pass


def failing_initializer():
    raise ValueError


def failing_function(x):
    raise AttributeError


class TestReusableThreadExecutor:
    def test_submit(self):
        executor = get_reusable_thread_executor(max_workers=4)
        future = executor.submit(pow, 2, 8)
        assert 256 == future.result()
        executor.shutdown()

    def test_executor_reuse(self):

        original_executor = get_reusable_thread_executor(max_workers=2)

        reused_executor = get_reusable_thread_executor(max_workers=1)

        # changing the number of worker will not create a new a new executor,
        # only modify the current one
        assert reused_executor == original_executor
        reused_executor.shutdown()

        new_executor = get_reusable_thread_executor(max_workers=1)
        assert new_executor != reused_executor

        new_executor_with_initializer = get_reusable_thread_executor(
                max_workers=1, initializer=dummy_initializer, initargs=tuple())

        assert new_executor_with_initializer != new_executor

    def test_reusable_thread_executor_resize(self):
        """Test reusable_executor resizing"""

        original_executor = get_reusable_thread_executor(max_workers=2)

        # threads are created lazily (one by new task received until the number
        # of threads reaches max_workers)
        original_executor.map(id, range(2))

        n_alive_threads = sum(t.is_alive() for t in original_executor._threads)
        assert n_alive_threads == 2

        reused_executor = get_reusable_thread_executor(max_workers=1)

        # make sure this executor is the same as the previous one
        assert reused_executor == original_executor

        # wait for the thread to exit
        time.sleep(0.01)

        n_alive_threads = sum(t.is_alive() for t in reused_executor._threads)
        assert n_alive_threads == 1

    def test_initializer_failed(self):
        executor = get_reusable_thread_executor(
                max_workers=2, initializer=failing_initializer,
                initargs=tuple())
        with pytest.raises(BrokenThreadPool):
            f = executor.submit(id, 2)
            f.result()

    def test_task_failed(self):
        executor = get_reusable_thread_executor(max_workers=2)
        f = executor.submit(time.sleep, 2)
        del executor._shutdown
        f.result()
