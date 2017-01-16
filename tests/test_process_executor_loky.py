import sys

from loky import process_executor
from loky.backend import LokyContext
from ._executor_mixin import ExecutorMixin


if (sys.version_info[:2] > (3, 3) and sys.platform != "win32") or (
        sys.version_info[:2] <= (3, 3)):

    class ProcessPoolLokyMixin(ExecutorMixin):
        # Makes sure that the context is defined
        executor_type = process_executor.ProcessPoolExecutor
        context = LokyContext()

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
