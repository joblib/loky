import sys

from loky import process_executor
from loky.backend import get_context
from ._executor_mixin import ProcessExecutorMixin


if (sys.version_info[:2] > (3, 3)
        and not hasattr(sys, "pypy_version_info")):
    class ProcessPoolSpawnMixin(ProcessExecutorMixin):
        executor_type = process_executor.ProcessPoolExecutor
        context = get_context('spawn')

    from ._test_process_executor import ProcessExecutorShutdownTest

    class TestsProcessPoolSpawnShutdown(ProcessPoolSpawnMixin,
                                        ProcessExecutorShutdownTest):
        def _prime_executor(self):
            pass

    from ._test_process_executor import WaitTests

    class TestsProcessPoolSpawnWait(ProcessPoolSpawnMixin, WaitTests):
        pass

    from ._test_process_executor import AsCompletedTests

    class TestsProcessPoolSpawnAsCompleted(ProcessPoolSpawnMixin,
                                           AsCompletedTests):
        pass

    from ._test_process_executor import ProcessExecutorTest

    class TestsProcessPoolSpawnExecutor(
            ProcessPoolSpawnMixin, ProcessExecutorTest):
        pass

    from ._test_process_executor import ProcessExecutorTest
