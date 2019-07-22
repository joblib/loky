from loky import process_executor
from loky.backend import get_context
from ._executor_mixin import ExecutorMixin

from ._test_process_executor import WaitTests
from ._test_process_executor import ProcessExecutorTest
from ._test_process_executor import AsCompletedTests
from ._test_process_executor import ProcessExecutorShutdownTest


class ProcessPoolLokyMixin(ExecutorMixin):
    # Makes sure that the context is defined
    executor_type = process_executor.ProcessPoolExecutor
    context = get_context("loky")


class TestsProcessPoolLokyShutdown(ProcessPoolLokyMixin,
                                   ProcessExecutorShutdownTest):
    def _prime_executor(self):
        pass


class TestsProcessPoolLokyWait(ProcessPoolLokyMixin,
                               WaitTests):
    pass


class TestsProcessPoolLokyAsCompleted(ProcessPoolLokyMixin,
                                      AsCompletedTests):
    pass


class TestsProcessPoolLokyExecutor(ProcessPoolLokyMixin,
                                   ProcessExecutorTest):
    pass
