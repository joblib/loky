import sys
import multiprocessing as mp

from loky import process_executor
from ._executor_mixin import ExecutorMixin


if sys.version_info[:2] > (3, 3) and sys.platform != "win32":
    class ProcessPoolForkMixin(ExecutorMixin):
        executor_type = process_executor.ProcessPoolExecutor
        context = mp.get_context('fork')

    from ._test_process_executor import ExecutorShutdownTest

    class TestsProcessPoolForkShutdown(ProcessPoolForkMixin,
                                       ExecutorShutdownTest):
        def _prime_executor(self):
            pass

    from ._test_process_executor import WaitTests

    class TestsProcessPoolForkWait(ProcessPoolForkMixin, WaitTests):
        pass

    from ._test_process_executor import AsCompletedTests

    class TestsProcessPoolForkAsCompleted(ProcessPoolForkMixin,
                                          AsCompletedTests):
        pass

    from ._test_process_executor import ExecutorTest

    class TestsProcessPoolForkExecutor(ProcessPoolForkMixin, ExecutorTest):
        pass

    from ._test_process_executor import ExecutorTest
