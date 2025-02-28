from loky import process_executor
from loky.backend import get_context

from ._executor_mixin import ExecutorMixin

from ._test_process_executor import (
    AsCompletedTests,
    ExecutorShutdownTest,
    ExecutorTest,
    WaitTests,
)


class ProcessPoolSpawnMixin(ExecutorMixin):
    executor_type = process_executor.ProcessPoolExecutor
    context = get_context("spawn")


class TestsProcessPoolSpawnShutdown(
    ProcessPoolSpawnMixin, ExecutorShutdownTest
):
    def _prime_executor(self):
        pass


class TestsProcessPoolSpawnWait(ProcessPoolSpawnMixin, WaitTests):
    pass


class TestsProcessPoolSpawnAsCompleted(
    ProcessPoolSpawnMixin, AsCompletedTests
):
    pass


class TestsProcessPoolSpawnExecutor(ProcessPoolSpawnMixin, ExecutorTest):
    pass
