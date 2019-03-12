"""Tests for the SemaphoreTracker class"""
from loky import ProcessPoolExecutor
import loky.backend.semaphore_tracker as semaphore_tracker


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
