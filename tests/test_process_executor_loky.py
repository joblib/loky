from loky import process_executor
from loky.backend import get_context
from ._executor_mixin import ExecutorMixin

from ._test_process_executor import WaitTests
from ._test_process_executor import ExecutorTest
from ._test_process_executor import AsCompletedTests
from ._test_process_executor import ExecutorShutdownTest


class ProcessPoolLokyMixin(ExecutorMixin):
    # Makes sure that the context is defined
    executor_type = process_executor.ProcessPoolExecutor
    context = get_context("loky")


class TestsProcessPoolLokyShutdown(ProcessPoolLokyMixin,
                                   ExecutorShutdownTest):
    def _prime_executor(self):
        pass


class TestsProcessPoolLokyWait(ProcessPoolLokyMixin,
                               WaitTests):
    pass


class TestsProcessPoolLokyAsCompleted(ProcessPoolLokyMixin,
                                      AsCompletedTests):
    pass


class TestsProcessPoolLokyExecutor(ProcessPoolLokyMixin,
                                   ExecutorTest):
    pass
