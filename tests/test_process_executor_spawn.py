import sys
import multiprocessing as mp

from loky import process_executor
from ._executor_mixin import ExecutorMixin


if sys.version_info[:2] > (3, 3):
    class ProcessPoolSpawnMixin(ExecutorMixin):
        executor_type = process_executor.ProcessPoolExecutor
        context = mp.get_context('spawn')

    from ._test_process_executor import ExecutorShutdownTest

    class TestsProcessPoolSpawnShutdown(ProcessPoolSpawnMixin,
                                        ExecutorShutdownTest):
        def _prime_executor(self):
            pass

    from ._test_process_executor import WaitTests

    class TestsProcessPoolSpawnWait(ProcessPoolSpawnMixin, WaitTests):
        pass

    from ._test_process_executor import AsCompletedTests

    class TestsProcessPoolSpawnAsCompleted(ProcessPoolSpawnMixin,
                                           AsCompletedTests):
        pass

    from ._test_process_executor import ExecutorTest

    class TestsProcessPoolSpawnExecutor(ProcessPoolSpawnMixin, ExecutorTest):
        pass

    from ._test_process_executor import ExecutorTest
