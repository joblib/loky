import threading
import time

from loky import process_executor
from loky.backend import get_context

from ._executor_mixin import ExecutorMixin, TIMEOUT
from ._test_process_executor import (
    AsCompletedTests,
    ExecutorShutdownTest,
    ExecutorTest,
    WaitTests,
    sleep_and_write,
)


def test_dispatch_during_worker_startup(monkeypatch, tmp_path):
    context = get_context("loky")
    process_factory = context.Process
    process_count = 0
    second_process_starting = threading.Event()
    continue_starting = threading.Event()

    class DelayedStartProcess:
        def __init__(self, process):
            object.__setattr__(self, "_process", process)

        def __getattr__(self, name):
            return getattr(self._process, name)

        def __setattr__(self, name, value):
            setattr(self._process, name, value)

        def start(self):
            second_process_starting.set()
            if not continue_starting.wait(TIMEOUT):
                raise RuntimeError("timed out while delaying worker startup")
            self._process.start()

    def delayed_process_factory(*args, **kwargs):
        nonlocal process_count
        process_count += 1
        process = process_factory(*args, **kwargs)
        if process_count == 2:
            return DelayedStartProcess(process)
        return process

    monkeypatch.setattr(context, "Process", delayed_process_factory)

    executor = process_executor.ProcessPoolExecutor(
        max_workers=2, context=context
    )
    marker = tmp_path / "task-started"
    future = []
    submit_error = []

    def submit():
        try:
            future.append(executor.submit(sleep_and_write, 0, marker, "done"))
        except BaseException as exc:
            submit_error.append(exc)

    submit_thread = threading.Thread(target=submit)
    submit_thread.start()
    try:
        assert second_process_starting.wait(TIMEOUT)

        deadline = time.monotonic() + TIMEOUT
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "task did not run while workers were starting"

        continue_starting.set()
        submit_thread.join(TIMEOUT)
        assert not submit_thread.is_alive()
        assert not submit_error
        assert future[0].result(timeout=TIMEOUT) is None
        assert len(executor._processes) == 2
    finally:
        continue_starting.set()
        submit_thread.join(TIMEOUT)
        executor.shutdown(wait=True, kill_workers=bool(submit_error))


class ProcessPoolLokyMixin(ExecutorMixin):
    # Makes sure that the context is defined
    executor_type = process_executor.ProcessPoolExecutor
    context = get_context("loky")


class TestsProcessPoolLokyShutdown(ProcessPoolLokyMixin, ExecutorShutdownTest):
    def _prime_executor(self):
        pass


class TestsProcessPoolLokyWait(ProcessPoolLokyMixin, WaitTests):
    pass


class TestsProcessPoolLokyAsCompleted(ProcessPoolLokyMixin, AsCompletedTests):
    pass


class TestsProcessPoolLokyExecutor(ProcessPoolLokyMixin, ExecutorTest):
    pass
