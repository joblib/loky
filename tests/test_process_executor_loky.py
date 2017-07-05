import sys

from loky import process_executor
from loky.backend import get_context
from ._executor_mixin import ExecutorMixin


class ProcessPoolLokyMixin(ExecutorMixin):
    # Makes sure that the context is defined
    executor_type = process_executor.ProcessPoolExecutor
    context = get_context("loky")

from ._test_process_executor import ExecutorShutdownTest

class TestsProcessPoolLokyShutdown(ProcessPoolLokyMixin,
                                   ExecutorShutdownTest):
    def _prime_executor(self):
        pass

from ._test_process_executor import WaitTests

class TestsProcessPoolLokyWait(ProcessPoolLokyMixin,
                               WaitTests):
    pass

from ._test_process_executor import AsCompletedTests

class TestsProcessPoolLokyAsCompleted(ProcessPoolLokyMixin,
                                      AsCompletedTests):
    pass

from ._test_process_executor import ExecutorTest

class TestsProcessPoolLokyExecutor(ProcessPoolLokyMixin,
                                   ExecutorTest):
    pass
